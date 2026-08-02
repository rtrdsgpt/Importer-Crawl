"""
Shared logic for Phase 4 (LLM Reasoning, Filtering & Ranking).

rank_engine.py implements the per-provider API calls (OpenAI-compatible,
Anthropic, Hugging Face) and reuses everything else from here: the prompt,
the pydantic schemas, hallucination-guarded contact validation, and the
rank/sort/save pipeline.

Only "website" and "social" (LinkedIn/Facebook) pages that were either
fully scraped ("success") or never fetched but have a search-engine
snippet ("snippet_only", explicitly flagged as low-confidence in the
prompt -- covers robots.txt-disallowed pages and social platforms alike)
are evaluated -- we don't guess at a company from nothing, since that
risks fabricating details rather than grounding them in real evidence.
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

Use this rubric for relevance_score so scores are comparable across companies:
- 0-10: Wrong role entirely (manufacturer/exporter/directory/irrelevant), or no \
real evidence either way
- 11-30: Wrong role, but with some tangential connection to {product} or {country}
- 31-50: Plausible buyer-side role, but evidence is indirect, thin, or inferred \
rather than stated
- 51-70: Genuine buyer-side role with reasonable direct evidence (e.g. page \
describes sourcing/stocking/distributing {product})
- 71-90: Strong, clear, explicit evidence of a genuine buyer-side role active in \
{country}
- 91-100: Unambiguous, well-documented match with multiple corroborating details \
(role, country, product all explicitly confirmed on the page)

COMPANY DATA
URL: {url}
Page title: {page_title}
Meta description: {meta_description}
Emails found on page: {emails}
Phones found on page: {phones}
LinkedIn links found on page: {linkedin_links}
Found via search query: "{query}"
Search result snippet: {snippet}

{content_confidence_note}
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


def first_sentence(text: str) -> str:
    """First sentence of match_reason, for a compact one-line log summary --
    the full reason is still kept in the saved output."""
    match = re.match(r"\s*[^.!?]*[.!?]", text)
    return match.group(0).strip() if match else text.strip()


class HardRateLimitError(Exception):
    """Raised by a provider's judge_fn when a rate-limit error looks like a
    long/daily quota exhaustion (e.g. "tokens per day" limits) rather than a
    transient per-minute one. Retrying with backoff is pointless when the
    quota won't clear for minutes to hours -- rank_companies() catches this
    and stops the whole run early instead of cycling every remaining page
    through the same futile retries."""


def extract_json_object(text: str | None) -> dict:
    if not text:
        raise ValueError("model returned empty content (no text to parse)")
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("no JSON object found in model output")
    return json.loads(match.group(0))


ROBOTS_SNIPPET_NOTE = (
    "NOTE: This page's robots.txt disallows automated access, so it was never "
    "fetched. All you have is the search engine's own title and snippet below "
    "-- not the actual page. Treat this as thin, low-confidence evidence: do "
    "not assign a high relevance_score on a search snippet alone, and say "
    "explicitly in match_reason that this judgment is based only on a search "
    "snippet, not the full page.\n"
)

LINKEDIN_SNIPPET_NOTE = (
    "NOTE: This is a LinkedIn company page. LinkedIn is never fetched directly "
    "(login-walled, against its ToS to scrape), so all you have is the search "
    "engine's title and snippet below -- not the actual page. This may be the "
    "company's only real online presence (no separate website), which is "
    "itself plausible for a smaller trading/import business. Treat this as "
    "thin, low-confidence evidence: do not assign a high relevance_score on a "
    "search snippet alone, and say explicitly in match_reason that this "
    "judgment is based only on a LinkedIn search snippet, not the full page.\n"
)

FACEBOOK_SNIPPET_NOTE = (
    "NOTE: This is a Facebook Page or Marketplace listing. Facebook is never "
    "fetched directly (login-walled, against its ToS to scrape), so all you "
    "have is the search engine's title and snippet below -- not the actual "
    "page. Many smaller importers/wholesalers/traders run their primary "
    "presence entirely on Facebook rather than a dedicated website, so "
    "don't discount this just because it's a social platform -- but do "
    "treat it as thin, low-confidence evidence: do not assign a high "
    "relevance_score on a search snippet alone, and say explicitly in "
    "match_reason that this judgment is based only on a Facebook search "
    "snippet, not the full page.\n"
)


def build_prompt(page: dict, product: str, country: str) -> str:
    content_confidence_note = ""
    if page.get("status") == "snippet_only":
        if page.get("source_type") == "social":
            content_confidence_note = (
                LINKEDIN_SNIPPET_NOTE if "linkedin.com" in page.get("url", "") else FACEBOOK_SNIPPET_NOTE
            )
        else:
            content_confidence_note = ROBOTS_SNIPPET_NOTE
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
        content_confidence_note=content_confidence_note,
        text_content=(page.get("text_content") or "")[:MAX_PAGE_CHARS_IN_PROMPT],
    )


def eligible_pages(pages: list[dict]) -> list[dict]:
    return [
        p for p in pages
        if p.get("status") in ("success", "snippet_only")
        and p.get("source_type") in ("website", "social")
    ]


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
    on_progress: Callable[[str], None] | None = None,
    checkpoint_path: Path | None = None,
    next_judge_fn: Callable[[], Optional[JudgeFn]] | None = None,
) -> list[RankedCompany]:
    """Runs judge_fn(page, product, country) over every eligible page,
    filters to genuine buyer-side roles above min_score, and returns the
    top_n ranked by relevance_score.

    If checkpoint_path is given, the current best-known results are written
    to disk after every single qualifying judgment (not just at the end) --
    a run that takes hours and gets interrupted, rate-limited into the
    ground, or crashes partway through still leaves real results on disk
    instead of nothing.

    If next_judge_fn is given, a HardRateLimitError (daily quota, not a
    transient per-minute limit) doesn't stop the run -- it calls
    next_judge_fn() for a judge_fn built on the next API key and retries
    the same page, so one key running out mid-run doesn't waste every
    remaining page. Only once next_judge_fn() itself returns None (no more
    keys) does the run stop early as before."""
    pages_to_judge = eligible_pages(pages)

    # eligible_pages() only keeps source_type=="website" pages that were
    # actually scraped -- "directory" pages (europages, kompass, etc.) are
    # excluded on purpose, since the directory listing itself is never the
    # company; noise/failed/linkedin pages have nothing to judge either way.
    # Spelling out the breakdown here means "N scraped but M judged" is
    # self-explanatory in the log instead of looking like a bug.
    directory_count = sum(1 for p in pages if p.get("source_type") == "directory")
    other_excluded = len(pages) - len(pages_to_judge) - directory_count
    print(
        f"Evaluating {len(pages_to_judge)} of {len(pages)} scraped pages "
        f"({directory_count} directory pages excluded -- not companies themselves; "
        f"{other_excluded} failed/skipped/noise pages excluded -- nothing to judge)"
    )

    ranked: list[RankedCompany] = []
    give_up = False
    for i, page in enumerate(pages_to_judge, start=1):
        msg = f"[{i}/{len(pages_to_judge)}] judging: {page['url']}"
        print(msg, flush=True)
        if on_progress:
            on_progress(msg)

        judgment = None
        while True:
            try:
                judgment = judge_fn(page, product, country)
                break
            except HardRateLimitError as exc:
                new_judge_fn = next_judge_fn() if next_judge_fn is not None else None
                if new_judge_fn is not None:
                    judge_fn = new_judge_fn
                    rotate_msg = (
                        f"  ! hard rate limit (daily quota) hit on this key; switching to "
                        f"the next API key and retrying this page"
                    )
                    print(rotate_msg, flush=True)
                    if on_progress:
                        on_progress(rotate_msg)
                    continue
                stop_msg = (
                    f"! hard rate limit (daily quota, not transient) hit on page {i}/{len(pages_to_judge)}"
                    f"{' -- no more API keys to switch to' if next_judge_fn is not None else ''}: {exc}\n"
                    f"  stopping here rather than retrying every remaining page -- "
                    f"{len(ranked)} result(s) already found are checkpointed and returned as-is"
                )
                print(stop_msg, flush=True)
                if on_progress:
                    on_progress(stop_msg)
                give_up = True
                break
        if give_up:
            break
        if judgment is None:
            time.sleep(delay_seconds)
            continue

        status_msg = (
            f"  -> role={judgment.company_role} score={judgment.relevance_score}"
            f" -- {first_sentence(judgment.match_reason)}"
        )
        print(status_msg, flush=True)
        if on_progress:
            on_progress(status_msg)
        if judgment.company_role in GENUINE_ROLES and judgment.relevance_score >= min_score:
            ranked.append(to_ranked_company(page, judgment))
            if checkpoint_path:
                checkpoint = sorted(ranked, key=lambda c: c.relevance_score, reverse=True)[:top_n]
                save_ranked(checkpoint, checkpoint_path)

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
