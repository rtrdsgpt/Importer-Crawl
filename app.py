"""
Phase 6: Interface & Output.

Streamlit dashboard that runs the full pipeline (Discovery -> Scraping ->
[Directory Mining] -> Ranking -> [Validation]) end-to-end for a given
product/country, or browses previously saved results without re-running
anything.

Usage:
    streamlit run app.py
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import sys
from dataclasses import asdict
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent / "src"))

import discovery as disc
import mine_directories as miner
import rank_engine as rnk
import rank_schema as rc
import scraper as scr
import validate as val

load_dotenv()

DATA_DIR = Path(__file__).parent / "data"

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


def save_json(path: Path, data: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


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


PROGRESS_RE = re.compile(r"^\[(\d+)/(\d+)\]")


def make_progress_logger(status):
    """Every pipeline stage prints/on_progress's a "[i/N] doing thing" line
    per item -- this wraps status.write() to also drive a real st.progress
    bar off that same "[i/N]" prefix, so each phase gets both a live text
    log and a numeric progress bar without changing every pipeline module's
    callback signature again. The bar is created immediately (not lazily on
    first match) so it renders above the per-item log lines, not after the
    first one -- call this right before starting the loop it tracks."""
    bar = status.progress(0.0)

    def log(msg: str) -> None:
        status.write(msg)
        match = PROGRESS_RE.match(msg)
        if match:
            current, total = int(match.group(1)), int(match.group(2))
            bar.progress(min(current / total, 1.0))

    return log


def run_pipeline(
    product: str, country: str, provider_label: str, model: str, api_key: str,
    max_per_query: int, delay: float, top_n: int, min_score: int,
    localize: bool, mine_dirs: bool, validate: bool, use_map_lookup: bool,
) -> dict:
    slug = f"{slugify(product)}_{slugify(country)}"
    DATA_DIR.mkdir(exist_ok=True)
    provider = LABEL_TO_PROVIDER[provider_label]

    # Discovery, Scraping, and Ranking always run; Directory Mining and
    # Validation are optional -- only count enabled phases so the overall
    # bar actually reaches 100% instead of stalling on skipped phases.
    total_phases = 3 + int(mine_dirs) + int(validate)
    phases_done = 0
    st.caption("Overall pipeline progress")
    overall_bar = st.progress(0.0)

    def advance_overall(label: str) -> None:
        nonlocal phases_done
        phases_done += 1
        overall_bar.progress(phases_done / total_phases, text=f"{label} ({phases_done}/{total_phases} phases)")

    with st.status("Phase 1: Discovering candidate companies...", expanded=True) as status:
        status.write(
            "Searching the web for companies that look like importers, distributors, "
            "wholesalers, or buyers of this product in the target country. Runs a set of "
            "English query variations" + (
                f", plus queries generated by {provider_label} in the target country's "
                "business language (local buyers often describe themselves in their own "
                "language, which English-only search misses)." if localize else "."
            )
        )
        candidates = disc.discover(
            product, country, max_results_per_query=max_per_query,
            delay_seconds=delay, localize=localize, localize_provider=provider, on_progress=make_progress_logger(status),
        )
        status.update(label=f"Phase 1 done: {len(candidates)} unique candidates found")
    advance_overall("Phase 1 (Discovery) done")
    candidates_dicts = [asdict(c) for c in candidates]
    candidates_path = DATA_DIR / f"candidates_{slug}.json"
    save_json(candidates_path, candidates_dicts)
    st.caption(f"Saved {len(candidates_dicts)} candidates to `{candidates_path}` -- available now, "
               f"don't need to wait for the rest of the pipeline.")
    st.download_button(
        f"Download candidates so far ({len(candidates_dicts)})",
        data=json.dumps(candidates_dicts, indent=2, ensure_ascii=False),
        file_name="candidates.json", mime="application/json", key="candidates_json_live",
    )

    with st.status("Phase 2: Scraping candidate pages...", expanded=True) as status:
        status.write(
            "Fetching each candidate's page to extract clean text, emails, phones, and "
            "LinkedIn links. Pages disallowed by robots.txt are never fetched -- only their "
            "search-engine snippet is kept, clearly flagged as lower-confidence. LinkedIn "
            "itself is never fetched either (against its ToS), but a company's LinkedIn "
            "title/snippet is still kept as low-confidence evidence -- useful for a company "
            "whose only real presence is a LinkedIn page, not a separate website."
        )
        scraped = scr.scrape_all(candidates_dicts, delay_seconds=delay, on_progress=make_progress_logger(status))
        status.update(label=f"Phase 2 done: {len(scraped)} pages scraped")
    advance_overall("Phase 2 (Scraping) done")
    scraped_dicts = [asdict(p) for p in scraped]
    scraped_path = DATA_DIR / f"scraped_{slug}.json"
    save_json(scraped_path, scraped_dicts)
    st.caption(f"Saved {len(scraped_dicts)} scraped pages to `{scraped_path}` -- available now.")
    st.download_button(
        f"Download scraped pages so far ({len(scraped_dicts)})",
        data=json.dumps(scraped_dicts, indent=2, ensure_ascii=False),
        file_name="scraped.json", mime="application/json", key="scraped_json_live",
    )

    if mine_dirs:
        with st.status("Phase 3: Mining directory pages for more leads...", expanded=True) as status:
            status.write(
                f"B2B directory pages (europages, Kompass, etc.) often list many company "
                f"names in one page. Asking {provider_label} to extract those names, then "
                "searching for each company's own website so it gets the same verification "
                "as every other candidate."
            )
            merged_candidates = miner.mine_leads(
                candidates_dicts, scraped_dicts, product, country,
                max_results_per_query=3, delay_seconds=delay, provider=provider,
                on_progress=make_progress_logger(status),
            )
            # mine_leads returns candidates_dicts + newly found ones appended --
            # only scrape the new tail, then merge into (not replace) what
            # Phase 2 already scraped. Re-scraping everything here would waste
            # time and LLM-provider quota on pages we already have.
            new_candidates_only = merged_candidates[len(candidates_dicts):]
            new_count = len(new_candidates_only)
            status.update(label=f"Phase 3 done: {new_count} new leads found; scraping those...")
            if new_count > 0:
                newly_scraped = scr.scrape_all(
                    new_candidates_only, delay_seconds=delay, on_progress=make_progress_logger(status),
                )
                scraped_dicts = scraped_dicts + [asdict(p) for p in newly_scraped]
                candidates_dicts = merged_candidates
                save_json(candidates_path, candidates_dicts)
                save_json(scraped_path, scraped_dicts)
                status.write(f"Updated `{candidates_path}` and `{scraped_path}` with the new leads.")
            status.update(label=f"Phase 3 done: {new_count} new leads added and scraped")
        advance_overall("Phase 3 (Directory Mining) done")

    with st.status(f"Phase 4: Ranking with {provider_label}...", expanded=True) as status:
        results_path = DATA_DIR / f"results_{slug}.json"
        status.write(
            "Asking the LLM to judge each scraped company: is it a genuine importer/"
            "distributor/wholesaler/buyer (not a manufacturer, exporter, or directory), "
            "how relevant is it, and what's the evidence? Contact details the model proposes "
            "are checked against what was actually found on the page -- nothing invented is kept. "
            f"Results save to `{results_path}` after every qualifying company, not just at the "
            f"end -- check that file directly if you want to watch it fill in live."
        )
        judge_fn = rnk.build_judge_fn(provider, model, api_key)
        ranked = rc.rank_companies(
            scraped_dicts, product=product, country=country, judge_fn=judge_fn,
            top_n=top_n, min_score=min_score, delay_seconds=delay, on_progress=make_progress_logger(status),
            checkpoint_path=results_path,
        )
        status.update(label=f"Phase 4 done: {len(ranked)} genuine importers ranked")
    advance_overall("Phase 4 (Ranking) done")
    ranked_dicts = [c.model_dump() for c in ranked]

    rc.save_ranked(ranked, results_path)

    if validate:
        with st.status("Phase 5: Validating country presence...", expanded=True) as status:
            status.write(
                "Layering free, deterministic checks on top of the LLM's judgment: does the "
                "phone number's country code match, does the domain use the country's TLD, "
                "does the page text mention the country, and does an OpenStreetMap lookup "
                "find the company there? This doesn't change the ranking -- it's supplementary "
                "evidence for you to weigh."
            )
            validated = val.validate_all(
                ranked_dicts, scraped_dicts, country, use_map_lookup=use_map_lookup, on_progress=make_progress_logger(status),
            )
            status.update(label=f"Phase 5 done: {len(validated)} companies validated")
        advance_overall("Phase 5 (Validation) done")
        with (DATA_DIR / f"validated_{slug}.json").open("w", encoding="utf-8") as f:
            json.dump(validated, f, indent=2, ensure_ascii=False)
        final = validated
    else:
        final = ranked_dicts

    return {
        "final": final,
        "candidates": candidates_dicts,
        "scraped": scraped_dicts,
    }


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
                                     "business language, using the same provider selected above.")
        mine_dirs = st.checkbox("Mine directory pages for extra leads", value=False,
                                 help="Extracts company names from scraped B2B directory pages "
                                      "and searches for their real websites, using the same "
                                      "provider selected above. Adds significant runtime.")
        validate = st.checkbox("Validate country presence", value=True,
                                help="Deterministic checks: phone country code, country-code "
                                     "TLD, text mentions, and an OSM Nominatim map lookup.")
        use_map_lookup = st.checkbox("  Include OSM map lookup", value=True, disabled=not validate,
                                      help="Adds ~1.1s per company (Nominatim rate limit). "
                                           "Uncheck for a faster, offline-only validation pass.")

        with st.expander("Advanced"):
            top_n = st.slider("Max results to return", 1, 100, 10)
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
                run_output = run_pipeline(
                    product, country, provider_label, model, api_key,
                    int(max_per_query), float(delay), top_n, min_score,
                    localize, mine_dirs, validate, use_map_lookup,
                )
                st.session_state["last_run"] = run_output
                st.session_state["last_query"] = f"{product} / {country}"

        if "last_run" in st.session_state:
            run_output = st.session_state["last_run"]
            st.subheader(f"Results: {st.session_state['last_query']}")
            render_results(run_output["final"], key_prefix="run")

            with st.expander("Intermediate outputs (candidates, scraped pages)"):
                st.caption(
                    "Everything the pipeline found along the way -- useful for debugging "
                    "why a company was or wasn't included in the final results."
                )
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        f"Download candidates ({len(run_output['candidates'])})",
                        data=json.dumps(run_output["candidates"], indent=2, ensure_ascii=False),
                        file_name="candidates.json", mime="application/json", key="candidates_json",
                    )
                with col2:
                    st.download_button(
                        f"Download scraped pages ({len(run_output['scraped'])})",
                        data=json.dumps(run_output["scraped"], indent=2, ensure_ascii=False),
                        file_name="scraped.json", mime="application/json", key="scraped_json",
                    )
        elif not run_clicked:
            st.info(
                "Configure a search in the sidebar and click **Run Discovery Engine**.\n\n"
                "The pipeline runs in stages, each shown live as it happens: "
                "**1. Discovery** (web search for candidate companies) -> "
                "**2. Scraping** (fetch each page, extract contacts) -> "
                "*3. Directory mining* (optional: pull more leads out of B2B directory "
                "pages) -> **4. Ranking** (an LLM judges genuine-importer role and "
                "relevance) -> *5. Validation* (optional: free country-presence checks). "
                "Expect this to take a few minutes depending on how many candidates are found."
            )

    with tab_browse:
        st.subheader("Previously saved files")
        st.caption(
            "Every stage's output lands in data/ as it's produced, not just the final "
            "results -- candidates (Phase 1), scraped pages (Phase 2), and results/"
            "validated (Phase 4/5) are all browsable here, including from runs that "
            "didn't finish (checkpointed results survive an interrupted run)."
        )
        # Final-results files render as the ranked-company table (render_results);
        # candidates/scraped have a different schema (Candidate/ScrapedPage, not
        # RankedCompany) so they get a generic table + JSON download instead.
        final_files = sorted(DATA_DIR.glob("validated_*.json")) + sorted(DATA_DIR.glob("results_*.json"))
        raw_files = sorted(DATA_DIR.glob("candidates_*.json")) + sorted(DATA_DIR.glob("scraped_*.json"))
        saved_files = final_files + raw_files
        if not saved_files:
            st.info("No saved files found in data/ yet -- run a search first.")
        else:
            chosen = st.selectbox("Choose a saved file", saved_files, format_func=lambda p: p.name)
            if chosen:
                data = json.loads(chosen.read_text(encoding="utf-8"))
                st.caption(f"{len(data)} entries in {chosen.name}")
                if chosen in final_files:
                    render_results(data, key_prefix="browse")
                else:
                    st.dataframe(data, use_container_width=True, hide_index=True)
                    st.download_button(
                        "Download JSON", data=json.dumps(data, indent=2, ensure_ascii=False),
                        file_name=chosen.name, mime="application/json", key="browse_raw_json",
                    )


if __name__ == "__main__":
    main()
