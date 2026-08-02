"""
Phase 5: Deterministic Country-Presence Validation.

The LLM ranking stage (Phase 4) can be fooled by confident-sounding text --
it has no independent way to confirm a company actually operates in the
target country. This stage layers free, deterministic checks on top of the
LLM's judgment rather than replacing it:

  - phone country calling code (+49 for Germany, etc.)
  - country-code TLD (.de for Germany, etc.)
  - country/city name mentioned in the page's own text
  - OSM Nominatim geocoding match (free, no API key -- see
    https://operations.osmfoundation.org/policies/nominatim/ for the usage
    policy this respects: descriptive User-Agent, max 1 req/sec)

Each ranked company gets a `country_signals` breakdown and a
`validation_confidence` count (0-4). Nothing here overrides the LLM's
relevance_score -- this is supplementary evidence for a human reviewer,
not an automatic filter, since a legitimate importer can still fail every
heuristic (e.g. a generic .com domain with a toll-free number).

Usage:
    python src/validate.py --input data/results_ceramic_tiles_germany.json \\
        --country "Germany"
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlparse

import requests

# Minimal lookup tables for common export destinations. Not exhaustive --
# unmapped countries just skip those two specific checks (phone/TLD) and
# fall back to the text-mention and geocoding signals instead.
COUNTRY_CALLING_CODES = {
    "germany": "49", "france": "33", "united kingdom": "44", "uk": "44",
    "united states": "1", "usa": "1", "united arab emirates": "971", "uae": "971",
    "italy": "39", "spain": "34", "netherlands": "31", "belgium": "32",
    "switzerland": "41", "austria": "43", "poland": "48", "sweden": "46",
    "saudi arabia": "966", "australia": "61", "canada": "1", "japan": "81",
    "south korea": "82", "singapore": "65", "brazil": "55", "mexico": "52",
    "south africa": "27", "turkey": "90", "china": "86", "india": "91",
}

COUNTRY_TLDS = {
    "germany": ".de", "france": ".fr", "united kingdom": ".uk", "uk": ".uk",
    "united arab emirates": ".ae", "uae": ".ae", "italy": ".it", "spain": ".es",
    "netherlands": ".nl", "belgium": ".be", "switzerland": ".ch", "austria": ".at",
    "poland": ".pl", "sweden": ".se", "saudi arabia": ".sa", "australia": ".au",
    "canada": ".ca", "japan": ".jp", "south korea": ".kr", "singapore": ".sg",
    "brazil": ".br", "mexico": ".mx", "south africa": ".za", "turkey": ".tr",
    "china": ".cn", "india": ".in",
}

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
NOMINATIM_USER_AGENT = "ExporterCrawlBot/0.1 (research project; country-presence validation)"
NOMINATIM_DELAY_SECONDS = 1.1  # Nominatim usage policy: max 1 request/second


def check_phone_country_code(phone: str | None, country: str) -> bool | None:
    code = COUNTRY_CALLING_CODES.get(country.strip().lower())
    if not code or not phone:
        return None
    digits = "".join(ch for ch in phone if ch.isdigit())
    return digits.startswith(code) or digits.startswith(f"00{code}")


def check_domain_tld(website: str, country: str) -> bool | None:
    tld = COUNTRY_TLDS.get(country.strip().lower())
    if not tld:
        return None
    domain = urlparse(website).netloc.lower().removeprefix("www.")
    return domain.endswith(tld)


def check_text_mentions_country(text_content: str, match_reason: str, country: str) -> bool:
    haystack = f"{text_content} {match_reason}".lower()
    return country.strip().lower() in haystack


def check_found_on_map(company_name: str, country: str) -> bool | None:
    """Free OSM Nominatim geocoding lookup -- no API key required."""
    try:
        resp = requests.get(
            NOMINATIM_URL,
            params={"q": f"{company_name}, {country}", "format": "json", "limit": 1},
            headers={"User-Agent": NOMINATIM_USER_AGENT},
            timeout=10,
        )
        if resp.status_code != 200:
            return None
        results = resp.json()
        return len(results) > 0
    except requests.RequestException:
        return None


def validate_company(company: dict, country: str, scraped_text_by_url: dict[str, str],
                      use_map_lookup: bool) -> dict:
    text_content = scraped_text_by_url.get(company["website"], "")

    signals = {
        "phone_country_code": check_phone_country_code(company.get("contact_phone"), country),
        "domain_tld": check_domain_tld(company["website"], country),
        "text_mentions_country": check_text_mentions_country(
            text_content, company.get("match_reason", ""), country),
        "found_on_map": check_found_on_map(company["company_name"], country) if use_map_lookup else None,
    }
    if use_map_lookup:
        time.sleep(NOMINATIM_DELAY_SECONDS)

    # Count only signals that ran and came back positive; None means "not
    # applicable" (e.g. no calling-code mapping for this country) and isn't
    # held against the company either way.
    confidence = sum(1 for v in signals.values() if v is True)
    checked = sum(1 for v in signals.values() if v is not None)

    return {
        **company,
        "country_signals": signals,
        "validation_confidence": f"{confidence}/{checked}" if checked else "unavailable",
    }


def validate_all(ranked: list[dict], scraped_pages: list[dict], country: str,
                  use_map_lookup: bool = True,
                  on_progress: Callable[[str], None] | None = None) -> list[dict]:
    scraped_text_by_url = {p["url"]: p.get("text_content", "") for p in scraped_pages}
    validated = []
    for i, company in enumerate(ranked, start=1):
        msg = f"[{i}/{len(ranked)}] validating: {company['company_name']}"
        print(msg, flush=True)
        if on_progress:
            on_progress(msg)
        result = validate_company(company, country, scraped_text_by_url, use_map_lookup)
        status_msg = f"  -> {result['validation_confidence']} signals matched"
        print(status_msg, flush=True)
        if on_progress:
            on_progress(status_msg)
        validated.append(result)
    return validated


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate ranked companies' country presence.")
    parser.add_argument("--input", required=True, help="Path to results_*.json from rank_*.py")
    parser.add_argument("--scraped", default=None,
                         help="Path to scraped_*.json (default: swap results_ -> scraped_)")
    parser.add_argument("--country", required=True)
    parser.add_argument("--no-map-lookup", action="store_true",
                         help="Skip the OSM Nominatim geocoding check (faster, fewer network calls)")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    input_path = Path(args.input)
    ranked = json.loads(input_path.read_text(encoding="utf-8"))

    scraped_path = Path(args.scraped) if args.scraped else Path(
        str(input_path).replace("results_", "scraped_"))
    scraped_pages = json.loads(scraped_path.read_text(encoding="utf-8")) if scraped_path.exists() else []

    validated = validate_all(ranked, scraped_pages, args.country, use_map_lookup=not args.no_map_lookup)

    output_path = Path(args.output) if args.output else Path(
        str(input_path).replace("results_", "validated_"))
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(validated, f, indent=2, ensure_ascii=False)

    print(f"\nValidated {len(validated)} companies")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
