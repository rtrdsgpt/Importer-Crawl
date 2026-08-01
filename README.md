# Importer Crawl

An AI-powered pipeline that discovers and ranks the N most relevant importer
companies for an Indian exporter entering a foreign market, given a
**Product** and a **Target Country**.

Example: `Product: Ceramic Tiles | Target Country: Germany`

For each company found, the pipeline returns:

- Company Name
- Website
- Relevance Score (0–100)
- Match Reason
- Contact Email
- Contact Phone
- Contact LinkedIn
- Sources Used

## Table of Contents

- [Architecture](#architecture)
- [Design Decisions](#design-decisions)
- [Data Sources](#data-sources)
- [Ranking Methodology](#ranking-methodology)
- [Assumptions & Limitations](#assumptions--limitations)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Sample Results](#sample-results)

---

## Architecture

The pipeline is a sequence of file-based stages under `src/`. Each stage
reads the previous stage's JSON output and writes its own, so any stage can
be re-run independently and every intermediate artifact is inspectable.

```
Product + Country
        |
        v
Phase 1  discovery.py         --> data/candidates_<product>_<country>.json
        |  (web search, optionally localized to the target country's
        |   business language)
        v
Phase 2  scraper.py           --> data/scraped_<product>_<country>.json
        |  (fetch pages, extract clean text/contacts, respect robots.txt)
        v
Phase 2.5 (optional)
        mine_directories.py   --> merges new leads into candidates_*.json
        |  (extract company names from directory pages already scraped,
        |   search for their real websites, feed back into Phase 2)
        v
Phase 3  rank_engine.py       --> data/results_<product>_<country>.json
        |  --provider {hf,openai,groq,gemini,claude}
        |  (LLM judges genuine-importer role + relevance score per company)
        v
Phase 3.5 (optional)
        validate.py           --> data/validated_<product>_<country>.json
        |  (deterministic country-presence checks layered on top)
        v
Phase 4  app.py (Streamlit)
         (runs the whole pipeline from a browser, or browses saved results)
```

### Module map

| File | Phase | Responsibility |
|---|---|---|
| `src/discovery.py` | 1 | Query generation (English + localized), DuckDuckGo search via `ddgs`, domain classification, dedup |
| `src/scraper.py` | 2 | `requests` + BeautifulSoup scraping, Jina Reader fallback for JS-heavy pages, robots.txt enforcement, contact extraction |
| `src/mine_directories.py` | 2.5 | LLM extraction of company names from directory pages, follow-up search per name |
| `src/rank_schema.py` | 3 | Shared prompt, Pydantic schemas, hallucination-guarded contact validation, ranking/sorting — used by every provider |
| `src/rank_engine.py` | 3 | Single CLI (`--provider {hf,openai,groq,gemini,claude}`) dispatching to the right SDK (OpenAI-compatible client for OpenAI/Groq/Gemini, `anthropic` for Claude, `huggingface_hub` for HF), sharing `rank_schema.py` |
| `src/validate.py` | 3.5 | Deterministic country-presence signals (phone code, TLD, text mention, free OSM geocoding) |
| `src/app.py` | 4 | Streamlit dashboard: runs the full pipeline live, or browses saved results, with CSV/JSON export |

---

## Design Decisions

**File-based pipeline, not a single monolithic script.** Each phase is a
standalone CLI script that reads/writes JSON. This makes every intermediate
result inspectable and re-runnable (e.g. re-rank already-scraped data with
a different LLM provider without re-scraping), and keeps each stage's
failure modes isolated.

**Multiple LLM providers behind a shared interface.** `rank_schema.py`
holds the prompt, the Pydantic output schema, the hallucination guard, and
the ranking/filtering logic once; `rank_engine.py` picks a `--provider`
(`hf`, `openai`, `groq`, `gemini`, `claude`) and dispatches to the right
SDK — an OpenAI-compatible client for OpenAI/Groq/Gemini, `anthropic` for
Claude, `huggingface_hub` for HF — reusing the same prompt/schema either
way. `rank_engine.get_raw_completion()` exposes the same 5-provider dispatch
as a generic text-completion call, so the auxiliary steps (query
localization in `discovery.py`, directory-name extraction in
`mine_directories.py`) support the exact same provider set as ranking —
one provider choice covers the whole pipeline, not a separate constrained
list for auxiliary steps. This was necessary in practice — during
development the free Hugging Face tier ran out of credits and its 7B
model confidently misclassified a German tile *manufacturer*
(`agrob-buchtal.de`) as a "buyer" at relevance score 85, which is exactly
the class of error the ranking stage exists to prevent. Being able to
switch providers (Groq's `openai/gpt-oss-120b`, Gemini, Claude, OpenAI)
without rewriting the pipeline logic was essential, not a nice-to-have.

**Domain classification at discovery time.** Every discovered URL is
tagged `website` / `directory` / `noise` / `linkedin` based on its domain
(see `NOISE_DOMAINS`, `DIRECTORY_DOMAINS`, `LINKEDIN_DOMAINS` in
`discovery.py`). This determines how each URL is treated downstream:
directories are never scored as if they were a company; noise domains
(Pinterest, Etsy, market-research report sites, etc.) are skipped before
ever making a network request; LinkedIn is never fetched at all.

**LinkedIn is never scraped.** LinkedIn requires login for nearly all
content and actively fights automated access; scraping it violates its
ToS regardless of `robots.txt`. Instead, a dedicated search query
(`site:linkedin.com/company ...`) captures each company's LinkedIn URL
directly from search-engine results — the URL is used as-is for the
"Contact LinkedIn" field, and the page itself is never fetched.

**`robots.txt` is respected, not negotiated around.** Where it disallows
access, the scraper does not fetch the page — but it also doesn't throw
that candidate away entirely. The search engine already crawled it under
its own identity and gave us a title/snippet; that gets passed to the LLM
as clearly-labeled, low-confidence evidence (the `snippet_only` status —
same treatment as LinkedIn). Sites using active bot-detection
(Cloudflare/Vercel challenge pages, WAF 403s) are left alone entirely —
even where `robots.txt` technically permits access, defeating a bot
challenge is a form of access-control circumvention this project
deliberately avoids. See [Assumptions & Limitations](#assumptions--limitations).

**Hallucination-guarded contact extraction.** The scraper extracts
emails/phones/LinkedIn links from each page independently via regex and
`mailto:`/`tel:` parsing. When the LLM proposes a contact value in Phase 3,
it's checked against that independently-extracted list — a value the LLM
invents that wasn't actually found on the page is silently dropped rather
than trusted. This is the main defense against the LLM fabricating a
plausible-looking but fake email or phone number.

**Deterministic validation layered on top of the LLM, not replacing it.**
Phase 3.5 doesn't re-score or re-rank anything — it adds a
`country_signals` breakdown and a `validation_confidence` count as
supplementary evidence for a human reviewer. A legitimate importer can
still fail every heuristic (generic `.com` domain, toll-free number), so
this is corroboration, not a filter.

**Hard wall-clock timeouts on every network call.** `requests`' own
`timeout=` parameter only bounds each individual read, not a call's total
wall-clock time — a server trickling bytes slowly (or a DNS/TCP-level
stall) can block far past the configured timeout. Every HTTP call in
`scraper.py` goes through `_get_with_hard_timeout()`, which runs the
request in a daemon thread and enforces a real wall-clock cap via
`Thread.join()`. This was found empirically: a single `robots.txt` fetch
to a directory site once stalled for ~3 minutes despite a 5s timeout.

---

## Data Sources

- **Web search** — DuckDuckGo, via the `ddgs` Python library. No API key,
  no rate-limit cost. 13 query templates per run mixing importer /
  distributor / wholesaler / buyer / trading-company intent, plus
  trade-fair-exhibitor queries and directory-scoped (`site:`) queries.
- **Localized search queries** — an LLM (Groq) generates additional
  queries in the target country's primary business language. English-only
  queries under-represent genuine local importers (their sites and
  self-descriptions are in the local language) while English-language
  exporter/manufacturer SEO content from third countries dominates
  English results instead. Empirically this roughly doubled useful
  candidate count on the Ceramic Tiles / Germany test run.
- **Company websites** — scraped directly (`requests` + BeautifulSoup),
  with a Jina Reader (`r.jina.ai`) fallback for JS-rendered pages a
  static fetch can't execute.
- **B2B trade directories** — europages, Kompass, TradeWheel, Volza,
  ExportHub, wer-liefert-was (wlw), etc. Used two ways: as direct
  candidates (tagged `directory`, never scored as a company), and as a
  lead source for Phase 2.5 directory mining.
- **Trade fair exhibitor pages** — one of the highest-signal sources for
  genuine B2B buyers, since exhibitor/visitor lists are companies
  actively engaged with the product category in that market.
- **LinkedIn company URLs** — sourced via search only (`site:linkedin.com/company`
  queries), never scraped.
- **OpenStreetMap Nominatim** — free geocoding (no API key) used in Phase
  3.5 to check whether a company name resolves to a real location in the
  target country.

---

## Ranking Methodology

For every successfully scraped `website`-type page (plus `snippet_only`
pages, with reduced confidence), the LLM is asked to judge:

1. **`company_role`** — one of `importer`, `distributor`, `wholesaler`,
   `buyer`, `trading_company` (genuine, buyer-side roles) or
   `manufacturer`, `exporter`, `marketplace_directory`, `irrelevant`
   (excluded). The prompt explicitly calls out seller-side language
   ("we offer/produce/manufacture/supply X to dealers/architects/customers")
   as a signal for the latter group — this exact confusion (a manufacturer
   being scored as a buyer) was the concrete bug that shaped this prompt.
2. **`relevance_score`** (0–100) — how strong a match this company is.
3. **`match_reason`** — 2–3 sentences citing specific evidence from the
   page. The model is instructed to use a low-to-mid score and say so
   explicitly when evidence is thin, rather than guess confidently.
4. **Contact fields** — email/phone/LinkedIn, constrained to what the
   scraper actually found on the page (see hallucination guard above).

Only companies with a genuine buyer-side role **and** `relevance_score >=
min_score` (default 40) survive. Survivors are sorted by `relevance_score`
descending and truncated to the top N (default 10) — "quality over
quantity" is enforced at this filtering step, not just aspirationally.

If Phase 3.5 validation is run, each surviving company additionally gets:

- `phone_country_code` — does the contact phone's calling code match the
  target country?
- `domain_tld` — does the website use the target country's ccTLD?
- `text_mentions_country` — does the scraped page text or match reason
  mention the target country?
- `found_on_map` — does an OSM Nominatim search for the company name +
  country return a result?

---

## Assumptions & Limitations

- **"Genuine importer" is defined narrowly.** Only `importer`,
  `distributor`, `wholesaler`, `buyer`, and `trading_company` roles count.
  A manufacturer or exporter of the *same product* is explicitly excluded
  even if highly relevant to the search — they're a competitor to the
  Indian exporter, not a customer.
- **A company's own page is the source of truth for contacts.** Contact
  info mentioned only in a directory listing about a company (not on the
  company's own site) is not surfaced, to avoid propagating stale or
  third-party-mediated data as if it were verified.
- **`robots.txt` disallow means no scrape, full stop** — even for the one
  page in a disallow-everything policy, and even though `robots.txt` is
  not strictly legally binding. Sites using active bot-detection
  (challenge pages, aggressive WAF blocking) are treated the same way:
  left alone, not fought. This is a deliberate ethical/legal boundary,
  not a technical limitation — see the scraper failure counts below.
- **English + one localized query set is not exhaustive recall.** Some
  genuine importers will simply not be found, especially in markets with
  multiple regional languages or highly fragmented B2B ecosystems.
- **Free-tier LLMs vary a lot in reasoning quality.** The free Hugging
  Face 7B model (`Qwen2.5-7B-Instruct`) was noticeably worse at the
  buyer-vs-seller distinction than Groq's Llama-3.3-70B or Claude/GPT —
  see the multi-provider design decision above. Results will differ
  meaningfully depending on which provider/model is selected.
- **Bot-protected sources are accepted as out of reach.** On the Ceramic
  Tiles / Germany test run, 7 of 67 candidates failed to scrape due to
  active bot-detection (Vercel security checkpoints, WAF 403s) even where
  `robots.txt` technically allowed access. No attempt is made to defeat
  these — see the ToS discussion in Design Decisions.
- **Directory-mined leads are lower-confidence by construction.** A
  company name extracted from a directory listing is re-searched to find
  its own website (same verification standard as everything else), but
  the initial extraction step trusts the LLM to read the directory page
  correctly — it's an LLM extraction task, not a deterministic one.

---

## Setup Instructions

### Prerequisites

- Python 3.11+
- At least one LLM provider API key (see below) — the Groq and Google
  Gemini free tiers are enough to run the whole pipeline at no cost.

### Install

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Configure API keys

Copy `.env.example` to `.env` and fill in whichever provider(s) you plan
to use — none are required to be all filled in:

```bash
cp .env.example .env
```

```
HF_TOKEN=...              # https://huggingface.co/settings/tokens
OPENAI_API_KEY=...        # https://platform.openai.com/api-keys
ANTHROPIC_API_KEY=...     # https://console.anthropic.com
GEMINI_API_KEY=...        # https://aistudio.google.com/apikey (free)
GROQ_API_KEY=...          # https://console.groq.com/keys (free)
```

The optional localized-query generation (Phase 1) and directory mining
(Phase 2.5) steps use whichever provider you pick for ranking — no
separate key is needed for them.

---

## Usage

### Option A: Streamlit dashboard (recommended)

```bash
streamlit run src/app.py
```

Opens at `http://localhost:8501`. Enter a product and country, pick an
LLM provider, toggle localization/directory-mining/validation, and click
**Run Discovery Engine**. Results can be exported as CSV or JSON. The
**Browse Saved Results** tab loads any previous run from `data/` without
re-running anything.

### Option B: CLI, phase by phase

```bash
# Phase 1: Discovery (add --localize for target-language queries; --localize-provider
# defaults to groq, also accepts hf/openai/gemini/claude)
python src/discovery.py --product "Ceramic Tiles" --country "Germany" --localize

# Phase 2: Scraping
python src/scraper.py --input data/candidates_ceramic_tiles_germany.json

# Phase 2.5 (optional): Directory lead mining -- then re-run Phase 2 to scrape the new leads
# (--provider defaults to groq, also accepts hf/openai/gemini/claude)
python src/mine_directories.py \
    --candidates data/candidates_ceramic_tiles_germany.json \
    --scraped data/scraped_ceramic_tiles_germany.json \
    --product "Ceramic Tiles" --country "Germany"
python src/scraper.py --input data/candidates_ceramic_tiles_germany.json

# Phase 3: Ranking (--provider: hf, openai, groq, gemini, or claude)
python src/rank_engine.py --input data/scraped_ceramic_tiles_germany.json \
    --product "Ceramic Tiles" --country "Germany" --provider groq --top-n 10 --min-score 40

# Phase 3.5 (optional): Country-presence validation
python src/validate.py --input data/results_ceramic_tiles_germany.json --country "Germany"
```

---

## Sample Results

_Pending — see note in conversation. `data/results_ceramic_tiles_germany.json`
needs a fresh run against the localized/expanded candidate set before this
section can cite real numbers, and the assignment requires results for
three product–country combinations, not one._
