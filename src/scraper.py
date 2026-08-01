"""
Phase 2: Web Scraping & Content Extraction.

Reads the candidate URLs produced by discovery.py, fetches each page, and
extracts clean text plus structured signals (emails, phones, LinkedIn links)
for the LLM reasoning stage (Phase 3).

Primary path: requests + BeautifulSoup (fast, no extra infra).
Fallback: Jina Reader (https://r.jina.ai/<url>) when the static fetch comes
back with too little text -- a cheap way to handle JS-heavy pages without
pulling in a headless browser.

LinkedIn URLs are never fetched here (see discovery.py) -- they pass through
untouched so Phase 3 can use them directly as the "Contact LinkedIn" field.

Usage:
    python src/scraper.py --input data/candidates_ceramic_tiles_germany.json
"""

from __future__ import annotations

import argparse
import json
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib import robotparser
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

USER_AGENT = (
    "Mozilla/5.0 (compatible; ExporterCrawlBot/0.1; "
    "+https://github.com/; research project, respects robots.txt)"
)
REQUEST_HEADERS = {"User-Agent": USER_AGENT, "Accept-Language": "en;q=0.9,*;q=0.5"}

JINA_READER_PREFIX = "https://r.jina.ai/"
MIN_TEXT_LEN_BEFORE_FALLBACK = 300  # below this, assume JS-rendered/empty page
MAX_TEXT_CHARS = 8000  # cap what we keep per page to bound LLM context later

EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
PHONE_RE = re.compile(r"\+?\d[\d ().\-]{7,}\d")

# Emails that come from template boilerplate, tracking pixels, or example
# placeholders rather than a real company inbox.
EMAIL_DOMAIN_BLOCKLIST = {
    "example.com", "yourdomain.com", "domain.com", "email.com",
    "sentry.io", "wixpress.com", "godaddy.com", "schema.org",
    "your-company.com", "company.com",
}

NON_CONTENT_TAGS = ["script", "style", "noscript", "svg", "nav", "footer", "header"]


def _get_with_hard_timeout(url: str, timeout: float, **kwargs) -> requests.Response:
    """requests' own `timeout=` only bounds each individual read, not the
    call's total wall-clock time -- a server that trickles bytes slowly (or
    a DNS/TCP-level stall) can block far past the configured timeout. Run
    the request in a daemon thread and enforce a real wall-clock cap with
    Thread.join(); a thread that's still stuck when we give up is abandoned
    (daemon=True keeps it from blocking process exit)."""
    box: dict = {}

    def target() -> None:
        try:
            box["response"] = requests.get(url, timeout=timeout, **kwargs)
        except requests.RequestException as exc:
            box["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout + 3)  # small buffer over requests' own timeout
    if thread.is_alive():
        raise requests.Timeout(f"hard timeout after {timeout + 3}s (stuck below requests' own timeout)")
    if "error" in box:
        raise box["error"]
    return box["response"]


@dataclass
class ScrapedPage:
    url: str
    domain: str
    source_type: str
    title: str
    search_snippet: str
    status: str  # "success" | "failed" | "skipped_linkedin" | "skipped_noise"
                 # | "robots_disallowed" | "snippet_only"
    error: str | None = None
    used_jina_fallback: bool = False
    page_title: str = ""
    meta_description: str = ""
    text_content: str = ""
    emails: list[str] = field(default_factory=list)
    phones: list[str] = field(default_factory=list)
    linkedin_links: list[str] = field(default_factory=list)


class RobotsCache:
    """Caches robots.txt parsers per domain so we only fetch each once."""

    def __init__(self, timeout: float = 5.0):
        self._parsers: dict[str, robotparser.RobotFileParser | None] = {}
        self._timeout = timeout

    def allowed(self, url: str) -> bool:
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        rp = self._parsers.get(origin, "__unset__")
        if rp == "__unset__":
            rp = self._fetch(origin)
            self._parsers[origin] = rp
        if rp is None:
            return True  # fail open: couldn't fetch/parse robots.txt
        return rp.can_fetch(USER_AGENT, url)

    def _fetch(self, origin: str) -> robotparser.RobotFileParser | None:
        try:
            resp = _get_with_hard_timeout(f"{origin}/robots.txt", self._timeout,
                                           headers=REQUEST_HEADERS)
            if resp.status_code >= 400:
                return None
            rp = robotparser.RobotFileParser()
            rp.parse(resp.text.splitlines())
            return rp
        except requests.RequestException:
            return None


def extract_text_and_meta(html: str, base_url: str) -> tuple[str, str, str, list[str]]:
    """Returns (page_title, meta_description, clean_text, linkedin_links)."""
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(NON_CONTENT_TAGS):
        tag.decompose()

    page_title = (soup.title.string or "").strip() if soup.title and soup.title.string else ""

    meta_description = ""
    meta_tag = soup.find("meta", attrs={"name": "description"})
    if meta_tag and meta_tag.get("content"):
        meta_description = meta_tag["content"].strip()

    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()[:MAX_TEXT_CHARS]

    linkedin_links = []
    for a in soup.find_all("a", href=True):
        href = urljoin(base_url, a["href"])
        if "linkedin.com/company" in href.lower() and href not in linkedin_links:
            linkedin_links.append(href)

    return page_title, meta_description, text, linkedin_links


def extract_contacts(html: str) -> tuple[list[str], list[str]]:
    soup = BeautifulSoup(html, "lxml")

    emails = set(EMAIL_RE.findall(html))
    for a in soup.find_all("a", href=True):
        if a["href"].lower().startswith("mailto:"):
            addr = a["href"].split(":", 1)[1].split("?")[0].strip()
            if addr:
                emails.add(addr)
    emails = {
        e for e in emails
        if e.split("@")[-1].lower() not in EMAIL_DOMAIN_BLOCKLIST
        and not e.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"))
    }

    phones = set()
    for a in soup.find_all("a", href=True):
        if a["href"].lower().startswith("tel:"):
            num = a["href"].split(":", 1)[1].strip()
            if num:
                phones.add(num)
    body_text = soup.get_text(separator=" ", strip=True)
    for m in PHONE_RE.findall(body_text):
        m = m.strip()
        digits = re.sub(r"\D", "", m)
        if not (8 <= len(digits) <= 15):  # implausible phone-number length
            continue
        if re.match(r"^\d{4}\s*[-–]?\s*\d{4}$", m):  # year range, e.g. "2025 2026"
            continue
        if re.search(r"[-–]\s*\d+\.\d+$", m):  # trailing decimal, e.g. stat figures
            continue
        if sum(1 for tok in m.split() if len(tok) == 1) >= 3:  # enumerated single digits
            continue
        phones.add(m)

    return sorted(emails), sorted(phones)


def fetch_html(url: str, timeout: float) -> requests.Response:
    return _get_with_hard_timeout(url, timeout, headers=REQUEST_HEADERS, allow_redirects=True)


def fetch_via_jina(url: str, timeout: float) -> str | None:
    try:
        resp = _get_with_hard_timeout(f"{JINA_READER_PREFIX}{url}", timeout, headers=REQUEST_HEADERS)
        if resp.status_code == 200 and len(resp.text.strip()) > MIN_TEXT_LEN_BEFORE_FALLBACK:
            return resp.text.strip()[:MAX_TEXT_CHARS]
    except requests.RequestException:
        pass
    return None


def scrape_one(candidate: dict, robots: RobotsCache, timeout: float) -> ScrapedPage:
    url = candidate["url"]
    source_type = candidate.get("source_type", "website")
    base = ScrapedPage(
        url=url,
        domain=candidate.get("domain", ""),
        source_type=source_type,
        title=candidate.get("title", ""),
        search_snippet=candidate.get("snippet", ""),
        status="failed",
    )

    if source_type == "linkedin":
        base.status = "skipped_linkedin"
        return base
    if source_type == "noise":
        base.status = "skipped_noise"
        return base

    if not robots.allowed(url):
        # We never fetch a robots.txt-disallowed page ourselves -- but the
        # search engine already crawled it under its own identity and gave
        # us a title/snippet in Phase 1. Treating that as thin, low-
        # confidence evidence (like we do for LinkedIn) is a legitimate,
        # policy-respecting middle ground between "fetch it anyway" and
        # "discard everything we know about this candidate".
        if base.title or base.search_snippet:
            base.status = "snippet_only"
            base.text_content = (
                f"[Search result title]: {base.title}\n"
                f"[Search result snippet]: {base.search_snippet}"
            ).strip()
        else:
            base.status = "robots_disallowed"
        return base

    fetch_error: str | None = None
    resp = None
    try:
        resp = fetch_html(url, timeout)
        if resp.status_code >= 400:
            fetch_error = f"HTTP {resp.status_code}"
            resp = None
    except requests.RequestException as exc:
        fetch_error = f"request failed: {exc}"

    page_title = meta_description = ""
    text = ""
    emails: list[str] = []
    phones: list[str] = []
    linkedin_links: list[str] = []

    if resp is not None:
        content_type = resp.headers.get("Content-Type", "")
        if "text/html" not in content_type and content_type != "":
            base.error = f"non-HTML content-type: {content_type}"
            return base
        page_title, meta_description, text, linkedin_links = extract_text_and_meta(resp.text, url)
        emails, phones = extract_contacts(resp.text)

    # Fall back to Jina Reader when the direct fetch was blocked/failed
    # outright, or when it "succeeded" but returned near-empty text
    # (typical of JS-rendered pages a static fetch can't execute).
    if resp is None or len(text) < MIN_TEXT_LEN_BEFORE_FALLBACK:
        jina_text = fetch_via_jina(url, timeout)
        if jina_text:
            text = jina_text
            base.used_jina_fallback = True
        elif resp is None:
            base.error = fetch_error
            return base

    base.status = "success"
    base.page_title = page_title
    base.meta_description = meta_description
    base.text_content = text
    base.emails = emails
    base.phones = phones
    base.linkedin_links = linkedin_links
    return base


def scrape_all(candidates: list[dict], timeout: float = 10.0,
                delay_seconds: float = 1.0) -> list[ScrapedPage]:
    robots = RobotsCache()
    results = []
    for i, candidate in enumerate(candidates, start=1):
        print(f"[{i}/{len(candidates)}] scraping: {candidate['url']}", flush=True)
        page = scrape_one(candidate, robots, timeout)
        print(f"  -> {page.status}"
              f"{' (jina fallback)' if page.used_jina_fallback else ''}", flush=True)
        results.append(page)
        if page.status not in ("skipped_linkedin", "skipped_noise"):
            time.sleep(delay_seconds)
    return results


def save_scraped(pages: list[ScrapedPage], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(p) for p in pages], f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape candidate URLs into clean text.")
    parser.add_argument("--input", required=True, help="Path to candidates_*.json from discovery.py")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--timeout", type=float, default=10.0, help="Per-request timeout (s)")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between requests (s)")
    args = parser.parse_args()

    input_path = Path(args.input)
    candidates = json.loads(input_path.read_text(encoding="utf-8"))

    output_path = Path(args.output) if args.output else Path(
        str(input_path).replace("candidates_", "scraped_")
    )

    pages = scrape_all(candidates, timeout=args.timeout, delay_seconds=args.delay)
    save_scraped(pages, output_path)

    counts: dict[str, int] = {}
    for p in pages:
        counts[p.status] = counts.get(p.status, 0) + 1

    print(f"\nScraped {len(pages)} candidates: {counts}")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
