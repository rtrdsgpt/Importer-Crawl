"""
Phase 4: Interface & Output.

Streamlit dashboard that runs the full pipeline (Discovery -> Scraping ->
[Directory Mining] -> Ranking -> [Validation]) end-to-end for a given
product/country, or browses previously saved results without re-running
anything.

Usage:
    streamlit run src/app.py
"""

from __future__ import annotations

import csv
import io
import json
import os
from dataclasses import asdict
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

import discovery as disc
import mine_directories as miner
import rank_engine as rnk
import rank_schema as rc
import scraper as scr
import validate as val

load_dotenv()

DATA_DIR = Path("data")

# Friendly display labels for rank_engine.py's PROVIDER_CONFIGS keys.
PROVIDER_LABELS = {
    "groq": "Groq (free, recommended)",
    "gemini": "Google Gemini (free)",
    "openai": "OpenAI",
    "claude": "Anthropic Claude",
    "hf": "Hugging Face (free, weaker reasoning)",
}
LABEL_TO_PROVIDER = {v: k for k, v in PROVIDER_LABELS.items()}

REQUIRED_COLUMNS = [
    ("company_name", "Company Name"),
    ("website", "Website"),
    ("relevance_score", "Relevance Score"),
    ("match_reason", "Match Reason"),
    ("contact_email", "Contact Email"),
    ("contact_phone", "Contact Phone"),
    ("contact_linkedin", "Contact LinkedIn"),
    ("sources_used", "Sources Used"),
]


def slugify(text: str) -> str:
    return text.lower().replace(" ", "_")


def rows_for_display(companies: list[dict]) -> list[dict]:
    rows = []
    for c in companies:
        row = {label: c.get(key) for key, label in REQUIRED_COLUMNS}
        row["Sources Used"] = "; ".join(c.get("sources_used") or [])
        if "validation_confidence" in c:
            row["Validation Confidence"] = c["validation_confidence"]
            signals = c.get("country_signals") or {}
            row["Country Signals"] = "; ".join(
                f"{k}={v}" for k, v in signals.items() if v is not None
            )
        rows.append(row)
    return rows


def to_csv_bytes(rows: list[dict]) -> bytes:
    if not rows:
        return b""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().encode("utf-8")


def render_results(companies: list[dict], key_prefix: str) -> None:
    if not companies:
        st.warning("No genuine importers met the relevance threshold for this run.")
        return
    rows = rows_for_display(companies)
    st.dataframe(rows, use_container_width=True, hide_index=True)
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download CSV", data=to_csv_bytes(rows), file_name="importer_results.csv",
            mime="text/csv", key=f"{key_prefix}_csv",
        )
    with col2:
        st.download_button(
            "Download JSON", data=json.dumps(companies, indent=2, ensure_ascii=False),
            file_name="importer_results.json", mime="application/json", key=f"{key_prefix}_json",
        )


def run_pipeline(
    product: str, country: str, provider_label: str, model: str, api_key: str,
    max_per_query: int, delay: float, top_n: int, min_score: int,
    localize: bool, mine_dirs: bool, aux_provider: str, validate: bool, use_map_lookup: bool,
) -> list[dict]:
    slug = f"{slugify(product)}_{slugify(country)}"
    DATA_DIR.mkdir(exist_ok=True)

    with st.status("Phase 1: Discovering candidate companies...", expanded=True) as status:
        log = status.write
        candidates = disc.discover(
            product, country, max_results_per_query=max_per_query,
            delay_seconds=delay, localize=localize, localize_provider=aux_provider, on_progress=log,
        )
        status.update(label=f"Phase 1 done: {len(candidates)} unique candidates found")
    candidates_dicts = [asdict(c) for c in candidates]

    with st.status("Phase 2: Scraping candidate pages...", expanded=True) as status:
        log = status.write
        scraped = scr.scrape_all(candidates_dicts, delay_seconds=delay, on_progress=log)
        status.update(label=f"Phase 2 done: {len(scraped)} pages scraped")
    scraped_dicts = [asdict(p) for p in scraped]

    if mine_dirs:
        with st.status("Phase 2.5: Mining directory pages for more leads...", expanded=True) as status:
            merged_candidates = miner.mine_leads(
                candidates_dicts, scraped_dicts, product, country,
                max_results_per_query=3, delay_seconds=delay, provider=aux_provider,
            )
            new_count = len(merged_candidates) - len(candidates_dicts)
            status.update(label=f"Phase 2.5 done: {new_count} new leads found; re-scraping...")
            if new_count > 0:
                scraped = scr.scrape_all(merged_candidates, delay_seconds=delay, on_progress=status.write)
                scraped_dicts = [asdict(p) for p in scraped]
            status.update(label=f"Phase 2.5 done: {new_count} new leads added and scraped")

    with st.status(f"Phase 3: Ranking with {provider_label}...", expanded=True) as status:
        log = status.write
        judge_fn = rnk.build_judge_fn(LABEL_TO_PROVIDER[provider_label], model, api_key)
        ranked = rc.rank_companies(
            scraped_dicts, product=product, country=country, judge_fn=judge_fn,
            top_n=top_n, min_score=min_score, delay_seconds=delay, on_progress=log,
        )
        status.update(label=f"Phase 3 done: {len(ranked)} genuine importers ranked")
    ranked_dicts = [c.model_dump() for c in ranked]

    rc.save_ranked(ranked, DATA_DIR / f"results_{slug}.json")

    if validate:
        with st.status("Phase 3.5: Validating country presence...", expanded=True) as status:
            log = status.write
            validated = val.validate_all(
                ranked_dicts, scraped_dicts, country, use_map_lookup=use_map_lookup, on_progress=log,
            )
            status.update(label=f"Phase 3.5 done: {len(validated)} companies validated")
        with (DATA_DIR / f"validated_{slug}.json").open("w", encoding="utf-8") as f:
            json.dump(validated, f, indent=2, ensure_ascii=False)
        return validated

    return ranked_dicts


def main() -> None:
    st.set_page_config(page_title="Importer Discovery Engine", page_icon="\U0001f50e", layout="wide")
    st.title("AI-Powered Importer Discovery Engine")
    st.caption("Discover and rank genuine importer companies for an Indian exporter entering a foreign market.")

    with st.sidebar:
        st.header("Search")
        product = st.text_input("Product", value="Ceramic Tiles")
        country = st.text_input("Target Country", value="Germany")

        st.header("LLM Provider")
        provider_label = st.selectbox("Provider", list(PROVIDER_LABELS.values()))
        provider_config = rnk.PROVIDER_CONFIGS[LABEL_TO_PROVIDER[provider_label]]
        model = st.text_input("Model", value=provider_config["default_model"])
        default_key = os.environ.get(provider_config["env_var"], "")
        api_key = st.text_input(
            f"{provider_config['env_var']}", value=default_key, type="password",
            help="Pre-filled from your local .env if present.",
        )

        st.header("Options")
        localize = st.checkbox("Localize search queries", value=True,
                                help="Generates extra search queries in the target country's "
                                     "business language.")
        mine_dirs = st.checkbox("Mine directory pages for extra leads", value=False,
                                 help="Extracts company names from scraped B2B directory pages "
                                      "and searches for their real websites. Adds significant runtime.")
        aux_provider_labels = {k: v for k, v in PROVIDER_LABELS.items() if k in ("groq", "gemini", "openai")}
        aux_provider_label = st.selectbox(
            "Auxiliary provider (localization & directory mining)",
            list(aux_provider_labels.values()), disabled=not (localize or mine_dirs),
            help="These two steps only support Groq/Gemini/OpenAI (OpenAI-compatible APIs), "
                 "independent of the ranking provider chosen above.",
        )
        aux_provider = {v: k for k, v in aux_provider_labels.items()}[aux_provider_label]
        validate = st.checkbox("Validate country presence (free)", value=True,
                                help="Deterministic checks: phone country code, country-code "
                                     "TLD, text mentions, and an OSM Nominatim map lookup.")
        use_map_lookup = st.checkbox("  Include OSM map lookup", value=True, disabled=not validate,
                                      help="Adds ~1.1s per company (Nominatim rate limit). "
                                           "Uncheck for a faster, offline-only validation pass.")

        with st.expander("Advanced"):
            top_n = st.slider("Max results to return", 1, 20, 10)
            min_score = st.slider("Minimum relevance score", 0, 100, 40)
            max_per_query = st.number_input("Max search results per query", 3, 100, 8)
            delay = st.number_input("Delay between requests (s)", 0.0, 5.0, 1.0, step=0.5)

        run_clicked = st.button("Run Discovery Engine", type="primary", use_container_width=True)

    tab_results, tab_browse = st.tabs(["Results", "Browse Saved Results"])

    with tab_results:
        if run_clicked:
            if not api_key:
                st.error(f"Please provide a {provider_config['env_var']} value in the sidebar.")
            else:
                results = run_pipeline(
                    product, country, provider_label, model, api_key,
                    int(max_per_query), float(delay), top_n, min_score,
                    localize, mine_dirs, aux_provider, validate, use_map_lookup,
                )
                st.session_state["last_results"] = results
                st.session_state["last_query"] = f"{product} / {country}"

        if "last_results" in st.session_state:
            st.subheader(f"Results: {st.session_state['last_query']}")
            render_results(st.session_state["last_results"], key_prefix="run")
        elif not run_clicked:
            st.info("Configure a search in the sidebar and click **Run Discovery Engine**.")

    with tab_browse:
        st.subheader("Previously saved results")
        saved_files = sorted(DATA_DIR.glob("validated_*.json")) + sorted(DATA_DIR.glob("results_*.json"))
        if not saved_files:
            st.info("No saved result files found in data/.")
        else:
            chosen = st.selectbox("Choose a saved file", saved_files, format_func=lambda p: p.name)
            if chosen:
                companies = json.loads(chosen.read_text(encoding="utf-8"))
                st.caption(f"{len(companies)} companies in {chosen.name}")
                render_results(companies, key_prefix="browse")


if __name__ == "__main__":
    main()
