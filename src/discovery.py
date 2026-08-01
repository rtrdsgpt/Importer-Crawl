"""
Phase 1: Search & Discovery.

Given a product and a target country, generate a set of search queries
designed to surface importer/distributor/buyer companies, run them against
a web search engine (DuckDuckGo via the `ddgs` library), and produce a
deduplicated list of candidate URLs for the scraping stage (Phase 2).

Usage:
    python src/discovery.py --product "Ceramic Tiles" --country "Germany"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlparse

from ddgs import DDGS
from ddgs.exceptions import DDGSException
from dotenv import load_dotenv

load_dotenv()

# Domains that are almost never a genuine importer's own website. We still
# want to see them show up in search results (they're useful signal), but
# they get tagged so downstream stages can decide how much weight to give
# them rather than treating them like a company site.
NOISE_DOMAINS = {
    "facebook.com", "instagram.com", "twitter.com", "x.com", "youtube.com",
    "pinterest.com", "tiktok.com", "reddit.com", "wikipedia.org",
    "amazon.com", "amazon.de", "ebay.com", "quora.com", "medium.com",
    "etsy.com", "zazzle.com",
    # market-research report sites: describe a market, aren't importers
    "mordorintelligence.com", "marketsandmarkets.com", "grandviewresearch.com",
    "statista.com",
}

# B2B trade directories / lead-gen marketplaces. These list many importers
# on one page (useful as a lead source) but the domain itself is never the
# importer's own website, so it must not be scored/contacted as a company.
DIRECTORY_DOMAINS = {
    "europages.com", "europages.co.uk", "kompass.com", "ec21.com",
    "tradeindia.com", "indiamart.com", "yellowpages.com", "dnb.com",
    "crunchbase.com", "wlw.de", "wlw.com", "wer-liefert-was.de",
    "made-in-china.com", "globalspec.com", "thomasnet.com",
    "b2byellowpages.com", "trademo.com", "go4worldbusiness.com",
    "volza.com", "tradekey.com", "exporthub.com", "bloombiz.com",
    "turkishexporter.net", "accio.com", "ensun.io", "connect2india.com",
    "exportbusinessmart.com", "importgenius.com", "panjiva.com",
}

LINKEDIN_DOMAINS = {"linkedin.com"}

# Query templates. {product} and {country} are filled in at runtime.
# Mixing intents (importer / distributor / buyer / wholesaler / trader)
# widens recall since companies describe themselves differently.
QUERY_TEMPLATES = [
    '{product} importers in {country}',
    '{product} distributors {country}',
    '{product} wholesale suppliers {country}',
    '{product} buyers {country} company',
    'import {product} {country} company contact',
    '{product} trading company {country}',
    '{product} importer directory {country}',
    '"{product}" import "{country}" -export',
    # Trade fairs are one of the highest-signal sources for B2B importers:
    # exhibitor/visitor lists are companies that actively buy or sell the
    # product in that market.
    '{product} trade fair exhibitors {country}',
    '{product} trade show buyers {country}',
    # site:-scoped queries against major multi-country B2B directories,
    # for higher precision than the generic directory query above.
    '{product} importers {country} site:europages.com',
    '{product} suppliers {country} site:kompass.com',
    # LinkedIn: we only ever want the company page URL from search results
    # (for the "Contact LinkedIn" field) -- never fetch/scrape the page
    # itself, since that requires login and violates LinkedIn's ToS.
    'site:linkedin.com/company "{product}" {country} import',
]


@dataclass
class Candidate:
    url: str
    domain: str
    title: str
    snippet: str
    query: str
    source_type: str  # "website" | "directory" | "noise"


def build_queries(product: str, country: str) -> list[str]:
    return [t.format(product=product, country=country) for t in QUERY_TEMPLATES]


LOCALIZATION_SYSTEM_PROMPT = (
    "You are a market-research assistant helping an Indian exporter find "
    "search queries that a local business buyer would actually type."
)

LOCALIZATION_USER_PROMPT = """\
An Indian exporter of "{product}" wants to find importers, distributors, \
wholesalers, and buyers of "{product}" in {country}.

Write 5 web search queries a local business speaker in {country} would \
realistically type into a search engine to find such companies, in the \
primary business language of {country}. Use natural, realistic phrasing \
and terminology a local buyer/importer would use for "{product}" -- do \
not just translate the English word for word.

Respond with ONLY a JSON array of 5 strings, no markdown fences, no \
commentary. Example shape: ["query one", "query two", ...]
"""


def generate_localized_queries(
    product: str, country: str, model: str = "llama-3.3-70b-versatile",
) -> list[str]:
    """Asks an LLM (via Groq) for search queries in the target country's
    business language. English-only queries under-represent genuine local
    importers, whose sites and self-descriptions are in the local language,
    while English-language exporter/manufacturer SEO content from third
    countries dominates English search results instead.

    Best-effort: returns [] if no GROQ_API_KEY is set or the call fails,
    so discovery still works without localization.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("  ! no GROQ_API_KEY found; skipping localized queries")
        return []

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": LOCALIZATION_SYSTEM_PROMPT},
                {"role": "user", "content": LOCALIZATION_USER_PROMPT.format(
                    product=product, country=country)},
            ],
            max_tokens=2048,
            temperature=0.3,
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("model returned empty content")
        content = re.sub(r"^```(?:json)?|```$", "", content.strip(), flags=re.MULTILINE).strip()
        match = re.search(r"\[[\s\S]*\]", content)
        if not match:
            raise ValueError("no JSON array found in model output")
        queries = [str(q).strip() for q in json.loads(match.group(0)) if str(q).strip()]
        print(f"  localized queries: {queries}")
        return queries
    except Exception as exc:  # noqa: BLE001 - localization is best-effort, never fatal
        print(f"  ! localized query generation failed ({exc}); continuing with English queries only")
        return []


def _matches(domain: str, known_domains: set[str]) -> bool:
    """True if domain equals or is a subdomain of one of known_domains."""
    return any(domain == d or domain.endswith(f".{d}") for d in known_domains)


def classify_domain(domain: str) -> str:
    domain = domain.lower().removeprefix("www.")
    if _matches(domain, LINKEDIN_DOMAINS):
        return "linkedin"
    if _matches(domain, NOISE_DOMAINS):
        return "noise"
    if _matches(domain, DIRECTORY_DOMAINS):
        return "directory"
    return "website"


def search_query(ddgs: DDGS, query: str, max_results: int, retries: int = 3) -> list[dict]:
    """Run a single search query with basic retry/backoff for rate limits."""
    for attempt in range(1, retries + 1):
        try:
            return list(ddgs.text(query, max_results=max_results))
        except DDGSException as exc:
            if attempt == retries:
                print(f"  ! giving up on query {query!r}: {exc}")
                return []
            wait = 2 ** attempt
            print(f"  ! search error ({exc}); retrying in {wait}s...")
            time.sleep(wait)
    return []


def discover(
    product: str,
    country: str,
    max_results_per_query: int = 8,
    delay_seconds: float = 1.0,
    localize: bool = False,
) -> list[Candidate]:
    """Run all query templates and return deduplicated candidates."""
    queries = build_queries(product, country)
    if localize:
        queries += generate_localized_queries(product, country)
    seen_keys: set[str] = set()
    candidates: list[Candidate] = []

    with DDGS() as ddgs:
        for i, query in enumerate(queries, start=1):
            print(f"[{i}/{len(queries)}] searching: {query}")
            results = search_query(ddgs, query, max_results_per_query)

            for r in results:
                url = r.get("href") or r.get("url") or ""
                if not url:
                    continue
                parsed = urlparse(url)
                domain = parsed.netloc.lower().removeprefix("www.")
                if not domain:
                    continue

                # Regular company sites: one candidate per domain. LinkedIn
                # is one domain hosting many distinct companies, so dedupe
                # by domain+path there instead, or every company after the
                # first LinkedIn hit would be dropped.
                if _matches(domain, LINKEDIN_DOMAINS):
                    dedup_key = f"{domain}{parsed.path.rstrip('/')}"
                else:
                    dedup_key = domain
                if dedup_key in seen_keys:
                    continue
                seen_keys.add(dedup_key)

                candidates.append(Candidate(
                    url=url,
                    domain=domain,
                    title=(r.get("title") or "").strip(),
                    snippet=(r.get("body") or "").strip(),
                    query=query,
                    source_type=classify_domain(domain),
                ))

            time.sleep(delay_seconds)  # be polite, avoid rate limiting

    return candidates


def save_candidates(candidates: list[Candidate], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in candidates], f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover candidate importer URLs.")
    parser.add_argument("--product", required=True, help='e.g. "Ceramic Tiles"')
    parser.add_argument("--country", required=True, help='e.g. "Germany"')
    parser.add_argument("--max-per-query", type=int, default=8,
                         help="Max search results to fetch per query template.")
    parser.add_argument("--delay", type=float, default=1.0,
                         help="Delay in seconds between search queries.")
    parser.add_argument("--output", default=None,
                         help="Output JSON path (default: data/candidates_<product>_<country>.json)")
    parser.add_argument("--localize", action="store_true",
                         help="Also generate search queries in the target country's business "
                              "language via Groq (requires GROQ_API_KEY).")
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else Path(
        f"data/candidates_{args.product.lower().replace(' ', '_')}_"
        f"{args.country.lower().replace(' ', '_')}.json"
    )

    candidates = discover(
        product=args.product,
        country=args.country,
        max_results_per_query=args.max_per_query,
        delay_seconds=args.delay,
        localize=args.localize,
    )

    save_candidates(candidates, output_path)

    website_count = sum(1 for c in candidates if c.source_type == "website")
    directory_count = sum(1 for c in candidates if c.source_type == "directory")
    noise_count = sum(1 for c in candidates if c.source_type == "noise")

    print(f"\nFound {len(candidates)} unique candidates "
          f"({website_count} website, {directory_count} directory, {noise_count} noise)")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
