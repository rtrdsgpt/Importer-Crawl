"""
Phase 3: Directory & Report Lead Mining.

Two kinds of scraped pages get discarded at Phase 4 even though their text
often *lists* real company names -- the page itself is never a company, so
eligible_pages() filters both out:

  - Directory/marketplace pages (europages, kompass, tradewheel, bloombiz, ...)
  - Market research report pages (mordorintelligence, grandviewresearch, ...)
    -- a report titled "{product} Market in {country}" often names real
    "key players"/"competitive landscape" companies active in that specific
    market, which is high-signal since an analyst firm already did the work
    of identifying who's actually in this market.

This stage:

  1. Asks an LLM to extract company names mentioned in each page's text as
     buyers/importers/distributors/market participants for {product} in
     {country}.
  2. Runs a new, targeted DDGS search per extracted name (via
     discovery.run_queries) to find that company's own website -- the
     same "search for a canonical source, never trust the listing alone"
     pattern used for LinkedIn URLs.
  3. Merges the newly found candidates into the existing candidates file.

The new candidates still need to go through scraper.py (to get real
contact info) and rank_*.py (to be judged) like any other candidate --
this stage only expands the candidate pool.

Usage:
    export GROQ_API_KEY=gsk_...   # or any provider rank_engine.py supports, via --provider
    python src/mine_directories.py --candidates data/candidates_ceramic_tiles_germany.json \\
        --scraped data/scraped_ceramic_tiles_germany.json \\
        --product "Ceramic Tiles" --country "Germany" --provider groq
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

import discovery as disc

load_dotenv()

MAX_PAGES_TO_MINE = 15  # bound cost: directory/report pages can be numerous per run
MAX_TEXT_CHARS_IN_PROMPT = 4000

SYSTEM_PROMPT = (
    "You extract real company names from B2B directory/marketplace listing "
    "pages and market research report pages. You only report names that are "
    "explicitly present in the text -- you never invent or guess a company "
    "name that isn't there."
)

USER_PROMPT_TEMPLATE = """\
The following is scraped text from a web page -- either a B2B directory/\
marketplace listing, or a market research report -- about "{product}" \
buyers/importers/distributors/market participants in or near "{country}".

Extract the names of real, specific companies mentioned in this text as \
buyers, importers, distributors, exhibitors, or "key players"/"competitive \
landscape" participants active in the "{product}" market in or near \
"{country}" -- not the directory/report publisher itself, not generic \
category labels, not manufacturers/exporters based elsewhere that are only \
mentioned as global players with no stated connection to {country} (unless \
the text is ambiguous about which side they're on or where they operate, \
in which case include them and let a later stage judge).

TEXT:
\"\"\"
{text_content}
\"\"\"

Respond with ONLY a JSON array of company name strings (no markdown fences, \
no commentary). If no real company names are present, respond with [].
"""


def extract_company_names(
    pages_to_mine: list[dict], product: str, country: str,
    provider: str = "groq", model: str | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> list[str]:
    import os

    import rank_engine as rnk

    config = rnk.PROVIDER_CONFIGS[provider]
    api_keys = rnk.parse_api_keys(os.environ.get(config["env_var"]) or config.get("placeholder_key"))
    if not api_keys:
        print(f"  ! no {config['env_var']} found; skipping directory/report mining")
        return []
    api_key = api_keys[0]

    names: set[str] = set()
    pages = pages_to_mine[:MAX_PAGES_TO_MINE]
    for i, page in enumerate(pages, start=1):
        text = (page.get("text_content") or "").strip()
        if len(text) < 100:
            continue
        kind = "report" if page.get("source_type") == "report" else "directory"
        msg = f"[{i}/{len(pages)}] mining {kind} page: {page['url']}"
        print(msg, flush=True)
        if on_progress:
            on_progress(msg)
        content = rnk.get_raw_completion(
            provider, model, api_key, SYSTEM_PROMPT,
            USER_PROMPT_TEMPLATE.format(
                product=product, country=country, text_content=text[:MAX_TEXT_CHARS_IN_PROMPT]),
            max_tokens=1024, temperature=0.1,
        )
        if not content:
            continue
        try:
            content = re.sub(r"^```(?:json)?|```$", "", content.strip(), flags=re.MULTILINE).strip()
            match = re.search(r"\[[\s\S]*\]", content)
            if not match:
                continue
            page_names = [str(n).strip() for n in json.loads(match.group(0)) if str(n).strip()]
            if page_names:
                found_msg = f"  found: {page_names}"
                print(found_msg, flush=True)
                if on_progress:
                    on_progress(found_msg)
            names.update(page_names)
        except Exception as exc:  # noqa: BLE001 - best-effort, one bad page shouldn't stop the rest
            print(f"  ! failed to parse names from this page ({exc}); skipping")

    return sorted(names)


def mine_leads(
    candidates: list[dict], scraped_pages: list[dict], product: str, country: str,
    max_results_per_query: int = 3, delay_seconds: float = 1.0,
    provider: str = "groq", model: str | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> list[dict]:
    """Returns the merged candidate list (original + newly mined), deduped."""
    pages_to_mine = [
        p for p in scraped_pages
        if p.get("source_type") in ("directory", "report")
        and p.get("status") in ("success", "snippet_only")
    ]
    print(f"Mining {len(pages_to_mine)} directory/report pages for company names...")
    names = extract_company_names(
        pages_to_mine, product, country, provider=provider, model=model, on_progress=on_progress,
    )
    print(f"\nExtracted {len(names)} candidate company names: {names}")

    if not names:
        return candidates

    queries = [f'"{name}" official website' for name in names]
    seen_keys = {c["domain"] for c in candidates if c.get("domain")}
    new_candidates = disc.run_queries(
        queries, max_results_per_query=max_results_per_query,
        delay_seconds=delay_seconds, seen_keys=seen_keys, on_progress=on_progress,
    )
    print(f"\nFound {len(new_candidates)} new candidate URLs from directory/report mining")

    return candidates + [asdict(c) for c in new_candidates]


def main() -> None:
    import rank_engine as rnk  # local import: avoids loading every provider SDK

    parser = argparse.ArgumentParser(
        description="Mine company leads from scraped directory and market research report pages.")
    parser.add_argument("--candidates", required=True, help="Path to candidates_*.json (updated in place)")
    parser.add_argument("--scraped", required=True, help="Path to scraped_*.json (read-only)")
    parser.add_argument("--product", required=True)
    parser.add_argument("--country", required=True)
    parser.add_argument("--max-per-query", type=int, default=3)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--provider", default="groq",
                         choices=list(rnk.PROVIDER_CONFIGS),
                         help="Which provider to use for name extraction (default: groq).")
    parser.add_argument("--model", default=None, help="Defaults to the provider's default model")
    args = parser.parse_args()

    candidates_path = Path(args.candidates)
    candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
    scraped_pages = json.loads(Path(args.scraped).read_text(encoding="utf-8"))

    merged = mine_leads(
        candidates, scraped_pages, product=args.product, country=args.country,
        max_results_per_query=args.max_per_query, delay_seconds=args.delay,
        provider=args.provider, model=args.model,
    )

    with candidates_path.open("w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(f"\n{len(merged) - len(candidates)} new candidates added "
          f"({len(candidates)} -> {len(merged)} total)")
    print(f"Updated {candidates_path}")
    print("Re-run scraper.py on this file to fetch the newly discovered candidates.")


if __name__ == "__main__":
    main()
