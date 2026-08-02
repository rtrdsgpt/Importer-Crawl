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
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlparse

import requests
from ddgs import DDGS
from ddgs.exceptions import DDGSException
from dotenv import load_dotenv

load_dotenv()

# Domains that are almost never a genuine importer's own website. We still
# want to see them show up in search results (they're useful signal), but
# they get tagged so downstream stages can decide how much weight to give
# them rather than treating them like a company site.
#
# facebook.com is deliberately NOT here -- real businesses (including many
# smaller importers/wholesalers) run their primary online presence as a
# Facebook Page or sell via Facebook Marketplace, so it gets the same
# never-fetch-but-use-the-snippet treatment as LinkedIn instead (see
# SOCIAL_DOMAINS below). Instagram/Twitter/etc. stay pure noise -- rarely
# a company's *primary* or only presence, and even less likely to expose
# structured contact info in a search snippet.
NOISE_DOMAINS = {
    "instagram.com", "twitter.com", "x.com", "youtube.com",
    "pinterest.com", "tiktok.com", "reddit.com", "wikipedia.org",
    "amazon.com", "amazon.de", "ebay.com", "quora.com", "medium.com",
    "etsy.com", "zazzle.com",
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

# Market research report publishers. A report titled e.g. "{product} Market
# in {country}" often names real "key players"/"competitive landscape"
# companies active in that specific market -- high-signal, since a paid
# analyst firm already did the work of identifying who's actually in this
# market, unlike a generic directory listing. These were previously
# classified as noise and silently skipped/never scraped, which threw that
# signal away. Unlike trade directories (dominated by a few regional
# players -- europages for Europe, indiamart for India -- which would bias
# query-time site: searches toward one region), the major market-research
# publishers are global/English-language regardless of which country's
# market they're reporting on, so a fixed domain list here doesn't carry
# the same regional bias.
REPORT_DOMAINS = {
    "mordorintelligence.com", "marketsandmarkets.com", "grandviewresearch.com",
    "statista.com", "imarcgroup.com", "fortunebusinessinsights.com",
    "alliedmarketresearch.com", "precedenceresearch.com",
    "futuremarketinsights.com", "verifiedmarketresearch.com",
    "expertmarketresearch.com", "gminsights.com",
    "transparencymarketresearch.com", "researchandmarkets.com",
    "marketresearchfuture.com", "coherentmarketinsights.com",
    "databridgemarketresearch.com", "businessresearchinsights.com",
}

# Login-gated platforms we never fetch directly (ToS), but where a search
# engine's title/snippet is still real evidence -- see scraper.py's
# snippet_only handling and rank_schema.py's eligible_pages().
SOCIAL_DOMAINS = {"linkedin.com", "facebook.com"}

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
    # Market research reports on "{product} market in {country}" often name
    # real "key players"/"competitive landscape" companies -- mined for
    # names in Phase 3 (see mine_directories.py) the same way directory
    # pages are.
    '{product} market {country} key players',
    '{product} market {country} competitive landscape',
    # No hardcoded site:-scoped directory queries here on purpose -- a fixed
    # list would inevitably mean picking well-known European/global
    # directories (europages.com, kompass.com), wasting query budget on
    # every non-European search. generate_localized_queries() (--localize)
    # asks per-run which directories are actually relevant to *this*
    # country instead.
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
    source_type: str  # "website" | "directory" | "report" | "social" | "noise"


def build_queries(product: str, country: str) -> list[str]:
    return [t.format(product=product, country=country) for t in QUERY_TEMPLATES]


LOCALIZATION_SYSTEM_PROMPT = (
    "You are a market-research assistant helping an Indian exporter find "
    "search queries and trade directories that a local business buyer "
    "would actually use."
)

LOCALIZATION_USER_PROMPT = """\
An Indian exporter of "{product}" wants to find importers, distributors, \
wholesalers, and buyers of "{product}" in {country}.

1. Write 5 web search queries a local business speaker in {country} would \
realistically type into a search engine to find such companies, in the \
primary business language of {country}. Use natural, realistic phrasing \
and terminology a local buyer/importer would use for "{product}" -- do \
not just translate the English word for word.

2. Name 1-3 real, well-known B2B trade/import directory websites that are \
actually used by buyers in {country} specifically (regional or national \
directories, chambers of commerce, or global directories with strong \
presence in that market) -- not just the biggest global names by default. \
Give bare domains only (e.g. "example.com"), no URLs. If you don't know \
of any directory genuinely relevant to {country}, return an empty list \
rather than guessing a generic one.

3. If you know a plausible HS (Harmonized System) customs classification \
code for "{product}" -- even just the 4-6 digit heading, not full \
precision -- include ONE extra query in the "queries" list built around \
it (e.g. "HS code 6907 ceramic tiles importers {country}" or "HS 090230 \
import data {country}"). Customs/trade-statistics sites index by HS code \
rather than product name, so this can surface real importer records a \
name-only search misses entirely. Skip this if you're not reasonably \
confident of the code -- a wrong code is worse than no query.

Respond with ONLY a JSON object, no markdown fences, no commentary:
{{"queries": ["query one", "query two", ...], "directories": ["example.com", ...]}}
"""


def generate_localized_queries(
    product: str, country: str, provider: str = "groq", model: str | None = None,
) -> list[str]:
    """Asks an LLM for (a) search queries in the target country's business
    language, and (b) trade directories actually relevant to that specific
    country, and returns both as ready-to-run query strings.

    Two problems this solves at once: English-only queries under-represent
    genuine local importers (their sites and self-descriptions are in the
    local language), and a fixed site:-scoped query list would otherwise
    have to hardcode which directories to target -- which inevitably means
    picking well-known European/global ones (europages.com, kompass.com)
    that are irrelevant, and wasted query budget, for a search targeting
    e.g. South Asia. Asking per-run for directories relevant to *this*
    country avoids baking in that bias.

    Best-effort: returns [] if the chosen provider's API key isn't set or
    the call fails, so discovery still works without either. Uses
    rank_engine.get_raw_completion(), so any of the 6 providers supported
    for ranking works here too -- one provider config for the whole
    pipeline, not a separate constrained set for auxiliary steps.
    """
    import rank_engine as rnk

    config = rnk.PROVIDER_CONFIGS[provider]
    # Only the first key -- the env var may hold a comma-separated list for
    # rank_companies()'s key-rotation, but this is a single best-effort call,
    # not worth the complexity of rotating here too.
    api_keys = rnk.parse_api_keys(os.environ.get(config["env_var"]) or config.get("placeholder_key"))
    if not api_keys:
        print(f"  ! no {config['env_var']} found; skipping localized queries")
        return []
    api_key = api_keys[0]

    content = rnk.get_raw_completion(
        provider, model, api_key, LOCALIZATION_SYSTEM_PROMPT,
        LOCALIZATION_USER_PROMPT.format(product=product, country=country),
        max_tokens=2048, temperature=0.3,
    )
    if not content:
        print("  ! localized query generation failed; continuing with English queries only")
        return []

    try:
        content = re.sub(r"^```(?:json)?|```$", "", content.strip(), flags=re.MULTILINE).strip()
        match = re.search(r"\{[\s\S]*\}", content)
        if not match:
            raise ValueError("no JSON object found in model output")
        data = json.loads(match.group(0))
        queries = [str(q).strip() for q in data.get("queries", []) if str(q).strip()]
        directories = [str(d).strip() for d in data.get("directories", []) if str(d).strip()]
        directory_queries = [f'{product} importers {country} site:{d}' for d in directories]
        print(f"  localized queries: {queries}")
        if directory_queries:
            print(f"  country-relevant directory queries: {directory_queries}")
        return queries + directory_queries
    except Exception as exc:  # noqa: BLE001 - localization is best-effort, never fatal
        print(f"  ! failed to parse localized queries ({exc}); continuing with English queries only")
        return []


SUPPLEMENT_SYSTEM_PROMPT = (
    "You are a market-research assistant helping an Indian exporter find "
    "more importer/distributor leads after an initial search came back thin."
)

SUPPLEMENT_USER_PROMPT = """\
An Indian exporter of "{product}" is searching for importers, distributors, \
wholesalers, and buyers of "{product}" in {country}. A first round of \
search queries only turned up {found} plausible company candidates, which \
is too few -- the queries already tried are listed below, so don't repeat \
them or make trivial rephrasings of them.

QUERIES ALREADY TRIED:
{tried}

Write {n} NEW web search queries that approach this from genuinely \
different angles than what's already been tried, prioritizing:
- Alternate terminology/synonyms for "{product}" a buyer might use instead \
of the literal product name
- Specific, real trade fairs, expos, or industry associations for this \
product category in or near {country}
- Local/regional trade directories, chambers of commerce, or B2B platforms \
specific to {country} (not global generic ones)
- Buyer-side phrasing not yet tried (e.g. procurement, sourcing, tender, \
wholesale purchase, stockist)
- If you're reasonably confident of an HS (Harmonized System) customs code \
for "{product}" and no prior query used one, a query built around it (e.g. \
"HS code 6907 import statistics {country}") -- customs/trade-data sites \
index by code, not product name, and are a different source entirely from \
a company-name search. Skip this angle if unsure of the code.

Respond with ONLY a JSON array of {n} strings, no markdown fences, no \
commentary.
"""


def generate_supplementary_queries(
    product: str, country: str, already_tried: list[str], found_count: int,
    n: int = 6, provider: str = "groq", model: str | None = None,
) -> list[str]:
    """Asks an LLM for additional, genuinely different search queries when
    an initial round found too few candidates -- prioritizing alternate
    terminology, trade fairs, and local directories rather than more
    generic variations of what's already been tried. Best-effort: returns
    [] on any failure so discovery still completes with what it has.
    """
    import rank_engine as rnk

    config = rnk.PROVIDER_CONFIGS[provider]
    api_keys = rnk.parse_api_keys(os.environ.get(config["env_var"]) or config.get("placeholder_key"))
    if not api_keys:
        print(f"  ! no {config['env_var']} found; skipping supplementary queries")
        return []
    api_key = api_keys[0]

    content = rnk.get_raw_completion(
        provider, model, api_key, SUPPLEMENT_SYSTEM_PROMPT,
        SUPPLEMENT_USER_PROMPT.format(
            product=product, country=country, found=found_count, n=n,
            tried="\n".join(f"- {q}" for q in already_tried),
        ),
        max_tokens=2048, temperature=0.4,
    )
    if not content:
        print("  ! supplementary query generation failed")
        return []

    try:
        content = re.sub(r"^```(?:json)?|```$", "", content.strip(), flags=re.MULTILINE).strip()
        match = re.search(r"\[[\s\S]*\]", content)
        if not match:
            raise ValueError("no JSON array found in model output")
        queries = [str(q).strip() for q in json.loads(match.group(0)) if str(q).strip()]
        print(f"  supplementary queries: {queries}")
        return queries
    except Exception as exc:  # noqa: BLE001 - best-effort, never fatal
        print(f"  ! failed to parse supplementary queries ({exc})")
        return []


def _matches(domain: str, known_domains: set[str]) -> bool:
    """True if domain equals or is a subdomain of one of known_domains."""
    return any(domain == d or domain.endswith(f".{d}") for d in known_domains)


def classify_domain(domain: str) -> str:
    domain = domain.lower().removeprefix("www.")
    if _matches(domain, SOCIAL_DOMAINS):
        return "social"
    if _matches(domain, NOISE_DOMAINS):
        return "noise"
    if _matches(domain, DIRECTORY_DOMAINS):
        return "directory"
    if _matches(domain, REPORT_DOMAINS):
        return "report"
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


def run_queries(
    queries: list[str],
    max_results_per_query: int = 8,
    delay_seconds: float = 1.0,
    seen_keys: set[str] | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> list[Candidate]:
    """Runs a list of search queries and returns deduplicated candidates.
    Shared by discover() (English/localized query templates) and
    mine_directories.py (per-company-name queries from directory mining) --
    both need the same dedup/classification logic. `seen_keys` can be
    pre-seeded with domains already known from a prior run, so a second
    call (e.g. directory mining after initial discovery) won't re-add them.
    `on_progress`, if given, is called with the same status strings that
    get printed -- lets a UI (e.g. Streamlit) mirror progress without
    scraping stdout.
    """
    seen_keys = set() if seen_keys is None else seen_keys
    candidates: list[Candidate] = []

    with DDGS() as ddgs:
        for i, query in enumerate(queries, start=1):
            msg = f"[{i}/{len(queries)}] searching: {query}"
            print(msg, flush=True)
            if on_progress:
                on_progress(msg)
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
                # and Facebook are each one domain hosting many distinct
                # companies, so dedupe by domain+path there instead, or
                # every company after the first hit on that domain would
                # be dropped.
                if _matches(domain, SOCIAL_DOMAINS):
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


def discover(
    product: str,
    country: str,
    max_results_per_query: int = 8,
    delay_seconds: float = 1.0,
    localize: bool = False,
    localize_provider: str = "groq",
    min_candidates: int = 0,
    max_expansion_rounds: int = 2,
    on_progress: Callable[[str], None] | None = None,
) -> list[Candidate]:
    """Run all query templates and return deduplicated candidates.

    If `min_candidates` > 0 and the initial round of queries turns up fewer
    than that many genuine (website/social) candidates, asks an LLM for
    supplementary queries -- alternate terminology, trade fairs, local
    directories -- and runs those too, up to `max_expansion_rounds` times.
    Stops early if a round adds no new queries (LLM unavailable/failed) so
    this never turns into an unbounded retry loop.
    """
    queries = build_queries(product, country)
    if localize:
        queries += generate_localized_queries(product, country, provider=localize_provider)

    seen_keys: set[str] = set()
    candidates = run_queries(
        queries, max_results_per_query, delay_seconds, seen_keys=seen_keys, on_progress=on_progress
    )
    tried_queries = list(queries)

    for round_num in range(1, max_expansion_rounds + 1):
        if min_candidates <= 0:
            break
        genuine_count = sum(1 for c in candidates if c.source_type in ("website", "social"))
        if genuine_count >= min_candidates:
            break

        msg = (
            f"only {genuine_count} genuine candidate(s) found (want >= {min_candidates}); "
            f"expanding search (round {round_num}/{max_expansion_rounds})..."
        )
        print(msg, flush=True)
        if on_progress:
            on_progress(msg)

        new_queries = generate_supplementary_queries(
            product, country, tried_queries, genuine_count, provider=localize_provider
        )
        if not new_queries:
            break  # LLM unavailable or failed -- nothing more to try
        tried_queries += new_queries

        candidates += run_queries(
            new_queries, max_results_per_query, delay_seconds, seen_keys=seen_keys, on_progress=on_progress
        )

    return candidates


def save_candidates(candidates: list[Candidate], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in candidates], f, indent=2, ensure_ascii=False)


def main() -> None:
    import rank_engine as rnk  # local import: avoids loading every provider SDK

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
                              "language via an LLM (see --localize-provider).")
    parser.add_argument("--localize-provider", default="groq",
                         choices=list(rnk.PROVIDER_CONFIGS),
                         help="Which provider to use for --localize (default: groq).")
    parser.add_argument("--min-candidates", type=int, default=0,
                         help="If the initial search yields fewer than this many genuine "
                              "(website/social) candidates, ask an LLM for supplementary "
                              "queries (alt terminology, trade fairs, local directories) and "
                              "search again. 0 disables expansion (default: 0).")
    parser.add_argument("--max-expansion-rounds", type=int, default=2,
                         help="Max number of supplementary query rounds when --min-candidates "
                              "is set (default: 2).")
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
        localize_provider=args.localize_provider,
        min_candidates=args.min_candidates,
        max_expansion_rounds=args.max_expansion_rounds,
    )

    save_candidates(candidates, output_path)

    website_count = sum(1 for c in candidates if c.source_type == "website")
    directory_count = sum(1 for c in candidates if c.source_type == "directory")
    report_count = sum(1 for c in candidates if c.source_type == "report")
    social_count = sum(1 for c in candidates if c.source_type == "social")
    noise_count = sum(1 for c in candidates if c.source_type == "noise")

    print(f"\nFound {len(candidates)} unique candidates "
          f"({website_count} website, {directory_count} directory, "
          f"{report_count} report, {social_count} social, {noise_count} noise)")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
