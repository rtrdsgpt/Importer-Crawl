"""
Shared logic for Phase 3 (LLM Reasoning, Filtering & Ranking).

Provider-specific scripts (rank_hf.py, rank_openai.py, rank_claude.py) each
implement just the API call for their LLM and reuse everything else from
here: the prompt, the pydantic schemas, hallucination-guarded contact
validation, and the rank/sort/save pipeline.

Only pages that were actually scraped successfully (source_type "website",
status "success") are evaluated -- we don't guess at a company from a
search snippet alone, since that risks fabricating details rather than
grounding them in real page content.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Callable, Literal, Optional

from pydantic import BaseModel, Field

MAX_PAGE_CHARS_IN_PROMPT = 4000  # keep prompts small/cheap and within context limits

GENUINE_ROLES = {"importer", "distributor", "wholesaler", "buyer", "trading_company"}

SYSTEM_PROMPT = (
    "You are a meticulous trade analyst helping an Indian exporter evaluate "
    "potential importer companies abroad. You only make claims grounded in "
    "the page content you are given -- you never invent contact details or "
    "facts that aren't present in the text. You pay close attention to "
    "whether a company is describing itself as a SELLER (manufacturer, "
    "producer, exporter -- offering/producing/supplying the product to "
    "others) versus a BUYER (importer, distributor, wholesaler -- sourcing "
    "or purchasing the product from suppliers). These are opposite roles "
    "and must not be confused."
)

USER_PROMPT_TEMPLATE = """\
TASK: Evaluate whether the following company is a genuine IMPORTER, \
DISTRIBUTOR, WHOLESALER, or BUYER of "{product}" that operates in \
"{country}" -- i.e. a company an Indian exporter of {product} could \
realistically approach as a customer in {country}.

Score LOW (and set company_role accordingly) if the company is instead:
- A manufacturer or exporter of {product} itself (a competitor/supplier, not a buyer) \
-- watch for seller-side language like "we offer/produce/manufacture/supply X to \
dealers/architects/customers", which indicates a SELLER, not a buyer
- A trade directory, marketplace, or market-research/news site, not an actual company
- In an unrelated industry, or not actually active in {country}
- Too vague/thin on this page to support a confident judgment

If the evidence is genuinely ambiguous or thin, use a low-to-mid score and say so \
plainly in match_reason -- do not assign a high score to a weakly-supported guess.

COMPANY DATA
URL: {url}
Page title: {page_title}
Meta description: {meta_description}
Emails found on page: {emails}
Phones found on page: {phones}
LinkedIn links found on page: {linkedin_links}
Found via search query: "{query}"
Search result snippet: {snippet}

PAGE CONTENT (truncated):
\"\"\"
{text_content}
\"\"\"

Respond with ONLY a single JSON object (no markdown fences, no commentary) \
matching exactly this schema:
{{
  "company_name": string,
  "company_role": one of ["importer", "distributor", "wholesaler", "buyer", \
"trading_company", "manufacturer", "exporter", "marketplace_directory", "irrelevant"],
  "relevance_score": integer from 0 to 100,
  "match_reason": string (2-3 sentences citing specific evidence from the page),
  "contact_email": string from the "Emails found on page" list above, or null,
  "contact_phone": string from the "Phones found on page" list above, or null,
  "contact_linkedin": string from the "LinkedIn links found on page" list above, or null
}}
"""


class LLMJudgment(BaseModel):
    company_name: str
    company_role: Literal[
        "importer", "distributor", "wholesaler", "buyer", "trading_company",
        "manufacturer", "exporter", "marketplace_directory", "irrelevant",
    ]
    relevance_score: int = Field(ge=0, le=100)
    match_reason: str
    contact_email: Optional[str] = None
    contact_phone: Optional[str] = None
    contact_linkedin: Optional[str] = None


class RankedCompany(BaseModel):
    company_name: str
    website: str
    relevance_score: int = Field(ge=0, le=100)
    match_reason: str
    contact_email: Optional[str] = None
    contact_phone: Optional[str] = None
    contact_linkedin: Optional[str] = None
    sources_used: list[str]
    company_role: str  # kept for transparency, beyond the required 8 fields


def extract_json_object(text: str | None) -> dict:
    if not text:
        raise ValueError("model returned empty content (no text to parse)")
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("no JSON object found in model output")
    return json.loads(match.group(0))


def build_prompt(page: dict, product: str, country: str) -> str:
    return USER_PROMPT_TEMPLATE.format(
        product=product,
        country=country,
        url=page["url"],
        page_title=page.get("page_title") or "(none)",
        meta_description=page.get("meta_description") or "(none)",
        emails=page.get("emails") or [],
        phones=page.get("phones") or [],
        linkedin_links=page.get("linkedin_links") or [],
        query=page.get("query", ""),
        snippet=page.get("search_snippet", ""),
        text_content=(page.get("text_content") or "")[:MAX_PAGE_CHARS_IN_PROMPT],
    )


def eligible_pages(pages: list[dict]) -> list[dict]:
    return [p for p in pages if p.get("source_type") == "website" and p.get("status") == "success"]


def to_ranked_company(page: dict, judgment: LLMJudgment) -> RankedCompany:
    # Guard against hallucinated contacts: only trust values the scraper
    # actually found on the page, never let the LLM invent one.
    def validated(value: str | None, allowed: list[str]) -> str | None:
        if value and value in allowed:
            return value
        return allowed[0] if allowed else None

    return RankedCompany(
        company_name=judgment.company_name,
        website=page["url"],
        relevance_score=judgment.relevance_score,
        match_reason=judgment.match_reason,
        contact_email=validated(judgment.contact_email, page.get("emails") or []),
        contact_phone=validated(judgment.contact_phone, page.get("phones") or []),
        contact_linkedin=validated(judgment.contact_linkedin, page.get("linkedin_links") or []),
        sources_used=[page["url"], page.get("query", "")],
        company_role=judgment.company_role,
    )


JudgeFn = Callable[[dict, str, str], Optional[LLMJudgment]]


def rank_companies(
    pages: list[dict], product: str, country: str, judge_fn: JudgeFn,
    top_n: int, min_score: int, delay_seconds: float,
) -> list[RankedCompany]:
    """Runs judge_fn(page, product, country) over every eligible page,
    filters to genuine buyer-side roles above min_score, and returns the
    top_n ranked by relevance_score."""
    pages_to_judge = eligible_pages(pages)
    print(f"Evaluating {len(pages_to_judge)} scraped company pages...")

    ranked: list[RankedCompany] = []
    for i, page in enumerate(pages_to_judge, start=1):
        print(f"[{i}/{len(pages_to_judge)}] judging: {page['url']}")
        judgment = judge_fn(page, product, country)
        if judgment is None:
            time.sleep(delay_seconds)
            continue

        print(f"  -> role={judgment.company_role} score={judgment.relevance_score}")
        if judgment.company_role in GENUINE_ROLES and judgment.relevance_score >= min_score:
            ranked.append(to_ranked_company(page, judgment))

        time.sleep(delay_seconds)

    ranked.sort(key=lambda c: c.relevance_score, reverse=True)
    return ranked[:top_n]


def save_ranked(ranked: list[RankedCompany], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump([c.model_dump() for c in ranked], f, indent=2, ensure_ascii=False)


def print_summary(ranked: list[RankedCompany], min_score: int, output_path: Path) -> None:
    print(f"\nRanked {len(ranked)} genuine importer(s) (min_score={min_score})")
    for c in ranked:
        print(f"  {c.relevance_score:3d}  {c.company_name}  ({c.website})")
    print(f"Saved to {output_path}")


def default_output_path(input_path: Path, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    return Path(str(input_path).replace("scraped_", "results_"))
