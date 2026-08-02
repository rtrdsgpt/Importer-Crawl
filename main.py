"""
Run the full importer discovery pipeline end-to-end from the command line:
Discovery -> Scraping -> [Directory & Report Mining] -> Ranking -> [Validation].

This is the CLI equivalent of clicking "Run Discovery Engine" in app.py --
same phases, same incremental checkpointing (a result file after every
qualifying judgment, not just at the end), same early exit if a provider
hits an unrecoverable daily rate limit -- just without the browser UI.
Each phase's own script under src/ can still be run individually for finer
control (e.g. re-ranking already-scraped data with a different provider);
this is for the common case of "just run the whole thing."

Usage:
    export GROQ_API_KEY=gsk_...   # or whichever provider's key
    python main.py --product "Ceramic Tiles" --country "Germany" --provider groq
    python main.py --product "Ceramic Tiles" --country "Germany" --provider groq \\
        --localize --mine-directories --validate --top-n 15
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import discovery as disc
import mine_directories as miner
import rank_engine as rnk
import rank_schema as rc
import scraper as scr
import validate as val

DATA_DIR = Path(__file__).parent / "data"


def slugify(text: str) -> str:
    return text.lower().replace(" ", "_")


def save_json(path: Path, data: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def run_pipeline(
    product: str, country: str, provider: str, model: str | None, api_key: str,
    max_per_query: int = 8, delay: float = 1.0, top_n: int = 10, min_score: int = 40,
    localize: bool = True, mine_dirs: bool = False, validate: bool = True,
    use_map_lookup: bool = True, min_candidates: int = 15, max_expansion_rounds: int = 2,
) -> list[dict]:
    slug = f"{slugify(product)}_{slugify(country)}"
    DATA_DIR.mkdir(exist_ok=True)
    config = rnk.PROVIDER_CONFIGS[provider]
    model = model or config["default_model"]

    print(f"\n=== Phase 1: Discovery ({product} / {country}) ===")
    candidates = disc.discover(
        product, country, max_results_per_query=max_per_query,
        delay_seconds=delay, localize=localize, localize_provider=provider,
        min_candidates=min_candidates, max_expansion_rounds=max_expansion_rounds,
    )
    candidates_dicts = [asdict(c) for c in candidates]
    candidates_path = DATA_DIR / f"candidates_{slug}.json"
    save_json(candidates_path, candidates_dicts)
    print(f"Found {len(candidates_dicts)} candidates -> {candidates_path}")

    print("\n=== Phase 2: Scraping ===")
    scraped = scr.scrape_all(candidates_dicts, delay_seconds=delay)
    scraped_dicts = [asdict(p) for p in scraped]
    scraped_path = DATA_DIR / f"scraped_{slug}.json"
    save_json(scraped_path, scraped_dicts)
    print(f"Scraped {len(scraped_dicts)} pages -> {scraped_path}")

    if mine_dirs:
        print("\n=== Phase 3: Directory & Report Mining ===")
        merged_candidates = miner.mine_leads(
            candidates_dicts, scraped_dicts, product, country,
            max_results_per_query=3, delay_seconds=delay, provider=provider,
        )
        # Only scrape the new tail -- merged_candidates is candidates_dicts
        # with new leads appended, so re-scraping the whole thing would
        # waste time and provider quota re-fetching pages we already have.
        new_candidates_only = merged_candidates[len(candidates_dicts):]
        if new_candidates_only:
            newly_scraped = scr.scrape_all(new_candidates_only, delay_seconds=delay)
            scraped_dicts = scraped_dicts + [asdict(p) for p in newly_scraped]
            candidates_dicts = merged_candidates
            save_json(candidates_path, candidates_dicts)
            save_json(scraped_path, scraped_dicts)
        print(f"Added {len(new_candidates_only)} new leads -> {candidates_path}, {scraped_path}")

    print(f"\n=== Phase 4: Ranking (provider={provider}, model={model}) ===")
    judge_fn = rnk.build_judge_fn(provider, model, api_key)
    results_path = DATA_DIR / f"results_{slug}.json"
    ranked = rc.rank_companies(
        scraped_dicts, product=product, country=country, judge_fn=judge_fn,
        top_n=top_n, min_score=min_score, delay_seconds=delay,
        checkpoint_path=results_path,  # saved after every qualifying result, not just at the end
    )
    rc.save_ranked(ranked, results_path)
    ranked_dicts = [c.model_dump() for c in ranked]
    print(f"Ranked {len(ranked_dicts)} genuine importer(s) -> {results_path}")

    if validate:
        print("\n=== Phase 5: Validation ===")
        validated = val.validate_all(
            ranked_dicts, scraped_dicts, country, use_map_lookup=use_map_lookup,
        )
        validated_path = DATA_DIR / f"validated_{slug}.json"
        save_json(validated_path, validated)
        print(f"Validated {len(validated)} companies -> {validated_path}")
        return validated

    return ranked_dicts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full importer discovery pipeline end-to-end.")
    parser.add_argument("--product", required=True, help='e.g. "Ceramic Tiles"')
    parser.add_argument("--country", required=True, help='e.g. "Germany"')
    parser.add_argument("--provider", default="groq", choices=list(rnk.PROVIDER_CONFIGS),
                         help="LLM provider for ranking (and localization/directory mining "
                              "if enabled). Default: groq (free).")
    parser.add_argument("--model", default=None, help="Defaults to the provider's default model")
    parser.add_argument("--api-key", default=None, help="Defaults to the provider's env var")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--min-score", type=int, default=40)
    parser.add_argument("--max-per-query", type=int, default=8)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--localize", action="store_true", default=True,
                         help="Generate extra search queries in the target country's "
                              "business language (default: on).")
    parser.add_argument("--no-localize", action="store_false", dest="localize")
    parser.add_argument("--mine-directories", action="store_true",
                         help="Extract company leads from scraped directory and market "
                              "research report pages (default: off).")
    parser.add_argument("--validate", action="store_true", default=True,
                         help="Run free country-presence validation on the results (default: on).")
    parser.add_argument("--no-validate", action="store_false", dest="validate")
    parser.add_argument("--no-map-lookup", action="store_true",
                         help="Skip the OSM Nominatim geocoding check during validation.")
    parser.add_argument("--min-candidates", type=int, default=15,
                         help="If discovery's initial queries yield fewer than this many "
                              "genuine (website/social) candidates, ask the LLM for "
                              "supplementary queries and search again. 0 disables this "
                              "(default: 15).")
    parser.add_argument("--max-expansion-rounds", type=int, default=2,
                         help="Max supplementary query rounds when --min-candidates is set "
                              "(default: 2).")
    args = parser.parse_args()

    config = rnk.PROVIDER_CONFIGS[args.provider]
    api_key = args.api_key or os.environ.get(config["env_var"])
    if not api_key:
        raise SystemExit(
            f"No API key found for provider {args.provider!r}. "
            f"Set {config['env_var']} in your environment or .env, or pass --api-key."
        )

    run_pipeline(
        args.product, args.country, args.provider, args.model, api_key,
        max_per_query=args.max_per_query, delay=args.delay,
        top_n=args.top_n, min_score=args.min_score,
        localize=args.localize, mine_dirs=args.mine_directories,
        validate=args.validate, use_map_lookup=not args.no_map_lookup,
        min_candidates=args.min_candidates, max_expansion_rounds=args.max_expansion_rounds,
    )


if __name__ == "__main__":
    main()
