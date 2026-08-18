"""
MCP server exposing importer discovery and ranking as tools for an agent
to call directly, rather than only through main.py/app.py/api.py.

Two tools, matching Phases 1+2 and Phase 4 of the pipeline (see README's
Architecture section):

  - search-importers: runs Discovery (web search) + Scraping and returns
    scraped candidate pages. Internally, Discovery's adaptive query
    expansion (--min-candidates) is itself a small bounded-retry agentic
    loop -- see README's "Agentic Framing" section.
  - rank-candidates: runs Phase 4's LLM judging over a set of scraped
    pages (e.g. search-importers' own output, or a caller-supplied set)
    and returns the ranked, hallucination-guarded results.

Composable: an agent calls search-importers, optionally filters/augments
the pages, then calls rank-candidates -- or brings its own pages straight
to rank-candidates and skips discovery/scraping entirely.

Usage:
    export GROQ_API_KEY=gsk_...
    python mcp_server.py
    # or point an MCP-aware client (e.g. Claude Desktop) at this script
    # over stdio.
"""

from __future__ import annotations

import os
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from mcp.server.mcpserver import MCPServer

import discovery as disc
import rank_engine as rnk
import rank_schema as rc
import scraper as scr

mcp_app = MCPServer(
    name="exporter-crawl",
    instructions=(
        "Tools for finding and ranking importer/distributor/buyer companies "
        "abroad for an Indian exporter, given a product and target country. "
        "Call search-importers first to gather scraped candidate pages, then "
        "rank-candidates to judge and rank them -- or supply your own pages "
        "straight to rank-candidates."
    ),
)


@mcp_app.tool(
    name="search-importers",
    description=(
        "Search the web for candidate importer/distributor/buyer companies of a "
        "product in a target country, then scrape each candidate page for text "
        "and contact details. Returns scraped pages ready for rank-candidates."
    ),
)
def search_importers(
    product: str,
    country: str,
    max_per_query: int = 8,
    localize: bool = False,
    min_candidates: int = 0,
) -> dict:
    """
    Args:
        product: e.g. "Ceramic Tiles"
        country: target market, e.g. "Germany"
        max_per_query: max search results to fetch per query template
        localize: also generate search queries in the target country's
            business language via an LLM (needs a provider API key set in
            the server's environment)
        min_candidates: if the initial search yields fewer than this many
            genuine (website/social) candidates, ask an LLM for
            supplementary queries and search again (bounded retries -- see
            discovery.discover()'s max_expansion_rounds). 0 disables this.
    """
    candidates = disc.discover(
        product, country, max_results_per_query=max_per_query,
        delay_seconds=1.0, localize=localize, min_candidates=min_candidates,
    )
    candidates_dicts = [asdict(c) for c in candidates]
    scraped = scr.scrape_all(candidates_dicts, delay_seconds=1.0)
    scraped_dicts = [asdict(p) for p in scraped]
    return {
        "candidate_count": len(candidates_dicts),
        "scraped_count": len(scraped_dicts),
        "pages": scraped_dicts,
    }


@mcp_app.tool(
    name="rank-candidates",
    description=(
        "Judge and rank a set of scraped candidate pages (as returned by "
        "search-importers) by how likely each is a genuine importer/distributor/"
        "wholesaler/buyer of a product in a target country. Returns the top-N "
        "ranked companies with hallucination-guarded contact details."
    ),
)
def rank_candidates(
    pages: list[dict],
    product: str,
    country: str,
    provider: str = "groq",
    model: str | None = None,
    top_n: int = 10,
    min_score: int = 40,
) -> list[dict]:
    """
    Args:
        pages: scraped pages, e.g. search-importers' "pages" output
        product: e.g. "Ceramic Tiles"
        country: target market, e.g. "Germany"
        provider: LLM provider to judge with -- one of PROVIDER_CONFIGS
            (groq, gemini, openai, claude, hf, ollama)
        model: defaults to the provider's default model
        top_n: max companies to return
        min_score: minimum relevance_score (0-100) to keep
    """
    if provider not in rnk.PROVIDER_CONFIGS:
        raise ValueError(f"unknown provider {provider!r}; choose one of {list(rnk.PROVIDER_CONFIGS)}")
    config = rnk.PROVIDER_CONFIGS[provider]
    model = model or config["default_model"]
    api_key = os.environ.get(config["env_var"]) or config.get("placeholder_key")
    if not api_key:
        raise ValueError(
            f"No API key found for provider {provider!r}. Set {config['env_var']} "
            "in the MCP server's environment."
        )

    judge_fn = rnk.build_judge_fn(provider, model, api_key)
    ranked = rc.rank_companies(
        pages, product=product, country=country, judge_fn=judge_fn,
        top_n=top_n, min_score=min_score, delay_seconds=0.5,
    )
    return [c.model_dump() for c in ranked]


if __name__ == "__main__":
    mcp_app.run()
