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

- [Usage](#usage)
- [Architecture](#architecture)
- [Design Decisions](#design-decisions)
- [Data Sources](#data-sources)
- [Ranking Methodology](#ranking-methodology)
- [Assumptions & Limitations](#assumptions--limitations)
- [Sample Results](#sample-results)

---

## Usage

### Setup

**Prerequisites:** Python 3.11+, and at least one LLM provider API key —
the Groq and Google Gemini free tiers are enough to run the whole pipeline
at no cost. Alternatively, `--provider ollama` needs no key at all: install
[Ollama](https://ollama.com), run `ollama serve`, pull a model
(`ollama pull gemma4:e4b`), and rank with unlimited local volume — no
rate limits, no daily quota, no cost, at the expense of your own machine's
compute and a smaller model's judgment quality vs. a hosted one.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in whichever provider(s) you plan
to use (none are required to all be filled in):

```bash
cp .env.example .env
```

```
HF_TOKEN=...              # https://huggingface.co/settings/tokens
OPENAI_API_KEY=...        # https://platform.openai.com/api-keys
ANTHROPIC_API_KEY=...     # https://console.anthropic.com
GEMINI_API_KEY=...        # https://aistudio.google.com/apikey (free)
GROQ_API_KEY=...          # https://console.groq.com/keys (free)
# --provider ollama needs no key -- just a local `ollama serve` running.
```

The optional localized-query generation and directory-mining steps use
whichever provider you pick for ranking — no separate key needed for them.

### Option A: `main.py` (simplest — one command, whole pipeline)

```bash
python main.py --product "Ceramic Tiles" --country "Germany" --provider groq
```

Runs Discovery → Scraping → Ranking end-to-end and writes every stage's
output to `data/`. Add `--mine-directories` to also pull extra leads out of
scraped B2B directory pages, or `--no-validate` to skip the free
country-presence checks (on by default). Results save incrementally as
they're found (`data/results_<product>_<country>.json`), so an
interrupted or rate-limited run still leaves real output on disk instead
of nothing. `python main.py --help` for the full flag list.

### Option B: Streamlit dashboard

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Enter a product and country, pick an
LLM provider, toggle localization/directory-mining/validation, and click
**Run Discovery Engine** — every stage streams live progress (a log and a
progress bar) plus an overall pipeline progress bar. Results, and the
intermediate candidates/scraped-pages data, are downloadable as CSV/JSON.
The **Browse Saved Results** tab loads any previous run from `data/`
without re-running anything.

### Option C: CLI, phase by phase

Useful for inspecting or re-running a single stage (e.g. re-rank
already-scraped data with a different provider without re-scraping).

```bash
# Phase 1: Discovery (add --localize for target-language queries; --localize-provider
# defaults to groq, also accepts hf/openai/gemini/claude/ollama)
python src/discovery.py --product "Ceramic Tiles" --country "Germany" --localize

# Phase 2: Scraping
python src/scraper.py --input data/candidates_ceramic_tiles_germany.json

# Phase 3 (optional): Directory & report lead mining -- then re-run Phase 2 to scrape the new leads
# (--provider defaults to groq, also accepts hf/openai/gemini/claude/ollama)
python src/mine_directories.py \
    --candidates data/candidates_ceramic_tiles_germany.json \
    --scraped data/scraped_ceramic_tiles_germany.json \
    --product "Ceramic Tiles" --country "Germany"
python src/scraper.py --input data/candidates_ceramic_tiles_germany.json

# Phase 4: Ranking (--provider: hf, openai, groq, gemini, claude, or ollama)
python src/rank_engine.py --input data/scraped_ceramic_tiles_germany.json \
    --product "Ceramic Tiles" --country "Germany" --provider groq --top-n 10 --min-score 40

# Phase 5 (optional): Country-presence validation
python src/validate.py --input data/results_ceramic_tiles_germany.json --country "Germany"
```

---

## Architecture

The pipeline is a sequence of file-based stages under `src/`, orchestrated
either by `main.py` (CLI) or `app.py` (Streamlit) at the project root.
Each stage reads the previous stage's JSON output and writes its own, so
any stage can be re-run independently and every intermediate artifact is
inspectable.

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
Phase 3 (optional)
        mine_directories.py   --> merges new leads into candidates_*.json
        |  (extract company names from directory AND market-research-report
        |   pages already scraped, search for their real websites, feed
        |   back into Phase 2)
        v
Phase 4  rank_engine.py       --> data/results_<product>_<country>.json
        |  --provider {hf,openai,groq,gemini,claude,ollama}
        |  (LLM judges genuine-importer role + relevance score per company;
        |   checkpointed to disk after every qualifying result, not just
        |   at the end)
        v
Phase 5 (optional)
        validate.py           --> data/validated_<product>_<country>.json
        |  (deterministic country-presence checks layered on top)
        v
main.py (CLI) or app.py (Streamlit)
         (orchestrates every phase above end-to-end; app.py also browses
         saved results without re-running anything)
```

### Module map

| File | Phase | Responsibility |
|---|---|---|
| `main.py` | orchestrator | CLI entry point: runs every phase end-to-end for a product/country in one command |
| `app.py` | orchestrator (6) | Streamlit dashboard: same orchestration as `main.py`, live in a browser, with CSV/JSON export and a saved-results browser |
| `src/discovery.py` | 1 | Query generation (English + localized), DuckDuckGo search via `ddgs`, domain classification, dedup |
| `src/scraper.py` | 2 | `requests` + BeautifulSoup scraping, `pypdf` for local PDF text extraction, Jina Reader fallback for JS-heavy pages and PDFs with no text layer, robots.txt enforcement, contact extraction |
| `src/mine_directories.py` | 3 (optional) | LLM extraction of company names from directory and market-research-report pages, follow-up search per name |
| `src/rank_schema.py` | 4 | Shared prompt, Pydantic schemas, hallucination-guarded contact validation, ranking/sorting/checkpointing — used by every provider |
| `src/rank_engine.py` | 4 | Single CLI (`--provider {hf,openai,groq,gemini,claude,ollama}`) dispatching to the right SDK (OpenAI-compatible client for OpenAI/Groq/Gemini/Ollama, `anthropic` for Claude, `huggingface_hub` for HF), sharing `rank_schema.py` |
| `src/validate.py` | 5 (optional) | Deterministic country-presence signals (phone code, TLD, text mention, free OSM geocoding) |

---

## Design Decisions

**File-based pipeline, not a single monolithic script.** Each phase is a
standalone CLI script that reads/writes JSON. This makes every intermediate
result inspectable and re-runnable (e.g. re-rank already-scraped data with
a different LLM provider without re-scraping), and keeps each stage's
failure modes isolated. `main.py`/`app.py` sit on top as thin orchestrators
for the common "just run the whole thing" case.

**Multiple LLM providers behind a shared interface.** `rank_schema.py`
holds the prompt, the Pydantic output schema, the hallucination guard, and
the ranking/filtering logic once; `rank_engine.py` picks a `--provider`
(`hf`, `openai`, `groq`, `gemini`, `claude`, `ollama`) and dispatches to
the right SDK — an OpenAI-compatible client for OpenAI/Groq/Gemini/Ollama,
`anthropic` for Claude, `huggingface_hub` for HF — reusing the same
prompt/schema either way. `ollama` needs no API key at all: it's a local
model server (`ollama serve` + `ollama pull <model>`) that happens to
speak the same OpenAI-compatible chat-completions format, so it drops
into the exact same code path as the cloud providers — unlimited local
volume with no rate limits or daily quota, at the cost of your own
machine's compute and a smaller model's judgment quality vs. a hosted
one. `rank_engine.get_raw_completion()` exposes the same 6-provider
dispatch as a generic text-completion call, so the auxiliary steps (query
localization in `discovery.py`, directory-name extraction in
`mine_directories.py`) support the exact same provider set as ranking —
one provider choice covers the whole pipeline, not a separate constrained
list for auxiliary steps. This was necessary in practice — during
development the free Hugging Face tier ran out of credits and its 7B
model confidently misclassified a German tile *manufacturer*
(`agrob-buchtal.de`) as a "buyer" at relevance score 85, which is exactly
the class of error the ranking stage exists to prevent. Being able to
switch providers (Groq's `openai/gpt-oss-120b`, Gemini, Claude, OpenAI,
or a local Ollama model) without rewriting the pipeline logic was
essential, not a nice-to-have.

**Incremental checkpointing, not save-at-the-end.** Phase 4 ranking can run
for hours across hundreds of pages, and a single provider outage,
interruption, or exhausted daily quota partway through used to mean the
entire run's output was lost. `rank_companies()` now writes the current
best-known results to disk after *every* qualifying judgment, so the
results file always reflects real progress, not just a completed run.

**Fail fast on unrecoverable rate limits, rotate keys if available.** A
per-minute rate limit is worth retrying with backoff; a daily-quota rate
limit is not — retrying still fails identically on every one of the
remaining pages, burning wall-clock time for nothing.
`rank_engine.is_hard_rate_limit()` pattern-matches provider error messages
for daily/quota language (seen in practice on Groq's "tokens per day"
limit) and raises a distinct `HardRateLimitError`. If only one API key is
configured, `rank_companies()` catches this and stops the run immediately,
returning whatever's already been checkpointed instead of grinding through
certain-to-fail retries on everything left. If more than one key is
configured (see below), it instead switches to the next key and retries
the *same* page, so one free-tier key running dry partway through a long
run doesn't waste every remaining page — it only stops for real once every
configured key has hit its daily quota.

**Multiple API keys, round-robin on quota exhaustion.** Free-tier daily
token quotas (e.g. Groq) are the most common practical wall a long
discovery run hits. Any provider's env var (`GROQ_API_KEY`, etc.) accepts
a comma-separated list of keys — `GROQ_API_KEY=key_one,key_two` — and
`rank_engine.build_key_rotator()` builds a judge function per key on
demand, handed to `rank_companies()` as `next_judge_fn`. A single key
still behaves exactly as before (`next_judge_fn` immediately returns
`None`, so there's nothing to opt into or configure specially). The
Streamlit app's API key field and `main.py`/`rank_engine.py`'s `--api-key`
flag accept the same comma-separated format.

**Domain classification at discovery time.** Every discovered URL is
tagged `website` / `directory` / `report` / `noise` / `social` based on its
domain (see `NOISE_DOMAINS`, `DIRECTORY_DOMAINS`, `REPORT_DOMAINS`,
`SOCIAL_DOMAINS` in `discovery.py`). This determines how each URL is
treated downstream: directories and reports are scraped normally in Phase
2 but never scored as if they were a company in Phase 4 -- instead they're
candidate input for Phase 3's mining step, which extracts real company
names mentioned on the page (buyers/importers/distributors for a
directory; "key players"/"competitive landscape" for a report) and
searches for each one's own site. Noise domains (Pinterest, Etsy, etc.)
are skipped before ever making a network request -- there's no company
name worth mining out of a Pinterest board. LinkedIn and Facebook are
never fetched at all, but unlike pure noise, their search-result
title/snippet is still kept as low-confidence evidence -- a real business
(especially a smaller importer/wholesaler) can have its primary or only
presence on one of these rather than a dedicated website, and discarding
those entirely would lose real leads, not just noise.

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
`mailto:`/`tel:` parsing. When the LLM proposes a contact value in Phase 4,
it's checked against that independently-extracted list — a value the LLM
invents that wasn't actually found on the page is silently dropped rather
than trusted. This is the main defense against the LLM fabricating a
plausible-looking but fake email or phone number.

**An explicit scoring rubric, not a free-floating 0–100.** Early runs
showed scores clustering on arbitrary low round numbers (5, 10, 15, 20)
with no clear separation between "wrong role" and "right role, weak
evidence." The prompt now spells out what each band means (0–10 wrong
role/no evidence, 11–30 wrong role with tangential relevance, 31–50
plausible but indirect evidence, 51–70 direct evidence, 71–90 strong
explicit evidence, 91–100 unambiguous multi-signal match), and ranking
calls use `temperature=0` wherever the provider allows it (Claude's
current models reject a non-default temperature entirely, so the rubric
is the only determinism lever there).

**Deterministic validation layered on top of the LLM, not replacing it.**
Phase 5 doesn't re-score or re-rank anything — it adds a
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
  no rate-limit cost. 11 fixed English query templates per run mixing
  importer / distributor / wholesaler / buyer / trading-company intent,
  plus trade-fair-exhibitor queries. No directories are hardcoded into
  these -- see below.
- **Localized search queries and country-relevant directories** — an LLM
  generates additional queries in the target country's primary business
  language, *and* is separately asked which B2B trade directories are
  actually relevant to that specific country, building `site:`-scoped
  queries against those rather than a fixed list. A hardcoded directory
  list would inevitably mean defaulting to well-known European/global
  names (europages.com, kompass.com) regardless of target market --
  wasted query budget for a search targeting, say, South Asia, with no
  equivalent boost from a locally-relevant directory. English-only
  queries under-represent genuine local importers (their sites and
  self-descriptions are in the local language) while English-language
  exporter/manufacturer SEO content from third countries dominates
  English results instead. Empirically this roughly doubled useful
  candidate count on the Ceramic Tiles / Germany test run.
- **Company websites** — scraped directly (`requests` + BeautifulSoup),
  with a Jina Reader (`r.jina.ai`) fallback for JS-rendered pages a static
  fetch can't parse.
- **PDFs** (trade reports, catalogs, company profiles) — extracted locally
  via `pypdf` directly from the fetched bytes, no external service
  involved. Only falls back to Jina Reader if the PDF has no extractable
  text layer at all (scanned/image-only). Previously Jina Reader was the
  *only* path for any non-HTML content, which meant a slow or failing
  Jina call lost a normal, digitally-generated PDF's content entirely even
  though local extraction handles that case in milliseconds.
- **B2B trade directories** — europages, Kompass, TradeWheel, Volza,
  ExportHub, wer-liefert-was (wlw), etc. Used two ways: as direct
  candidates (tagged `directory`, never scored as a company), and as a
  lead source for Phase 3 mining.
- **Market research reports** — Mordor Intelligence, Grand View Research,
  IMARC, Fortune Business Insights, and similar analyst-firm publishers
  (tagged `report`, see `REPORT_DOMAINS` in `discovery.py`). A report
  titled "{product} Market in {country}" often names real "key players"/
  "competitive landscape" companies specifically active in that market —
  high-signal, since an analyst firm already did the work of identifying
  who's actually in this market, rather than a generic global directory
  listing. These were previously classified as pure noise and skipped
  before ever being scraped; like directories, they're now scraped
  normally and fed into Phase 3 mining instead. Global/English-language
  regardless of which country's market they cover, so — unlike trade
  directories — a fixed domain list here doesn't carry the same regional
  bias risk.
- **Trade fair exhibitor pages** — one of the highest-signal sources for
  genuine B2B buyers, since exhibitor/visitor lists are companies
  actively engaged with the product category in that market.
- **LinkedIn and Facebook company URLs** — sourced via search only
  (`site:linkedin.com/company`, and Facebook Pages/Marketplace results that
  turn up organically). Neither platform's own page is ever fetched
  (login-walled, against their ToS), but the search engine's title/snippet
  is kept as low-confidence evidence — useful for a company whose only real
  presence is a LinkedIn page or Facebook Page, not a separate website. This
  matters more for smaller importers/wholesalers/traders than it might seem:
  many run their entire public presence on one of these platforms with no
  dedicated site at all, and dropping them as "noise" (Facebook's original
  classification) silently excluded genuine leads.
- **Adaptive query expansion** — if the initial round of queries (English +
  localized + directory `site:` queries) turns up fewer than a configurable
  minimum number of genuine (website/social) candidates, an LLM is asked for
  new queries that deliberately avoid repeating what's already been tried —
  prioritizing alternate product terminology, trade fairs/expos/industry
  associations, and local directories or chambers of commerce specific to
  the target country. This repeats for a capped number of rounds (default 2)
  and stops early if the threshold is met or the LLM has nothing new to
  suggest, so it never turns into an unbounded retry loop. Configurable via
  `--min-candidates`/`--max-expansion-rounds` on `main.py`/`discovery.py`, or
  the "Min genuine candidates before expanding search" control in the
  Streamlit app's Advanced section.
- **OpenStreetMap Nominatim** — free geocoding (no API key) used in Phase 5
  to check whether a company name resolves to a real location in the
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
2. **`relevance_score`** (0–100) — scored against an explicit rubric (see
   Design Decisions) rather than a free-floating number, for comparability
   across companies and providers.
3. **`match_reason`** — 2–3 sentences citing specific evidence from the
   page. The model is instructed to use a low-to-mid score and say so
   explicitly when evidence is thin, rather than guess confidently.
4. **Contact fields** — email/phone/LinkedIn, constrained to what the
   scraper actually found on the page (see hallucination guard above).

Only companies with a genuine buyer-side role **and** `relevance_score >=
min_score` (default 40) survive. Survivors are sorted by `relevance_score`
descending and truncated to the top N (default 10) — "quality over
quantity" is enforced at this filtering step, not just aspirationally.
Every stage of this filtering is visible in the log/UI: how many pages
were scraped vs. judged (directory pages are excluded from judging on
purpose — see Design Decisions), and a one-sentence summary of *why* next
to every score as it's produced.

If Phase 5 validation is run, each surviving company additionally gets:

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
- **Free-tier LLMs vary a lot in reasoning quality, and have real quota
  limits.** The free Hugging Face 7B model (`Qwen2.5-7B-Instruct`) was
  noticeably worse at the buyer-vs-seller distinction than Groq's
  `gpt-oss-120b` or Claude/GPT — see the multi-provider design decision
  above. Groq's free tier also has a hard daily token quota; a large run
  can exhaust it mid-ranking (handled via checkpointing + fail-fast, not
  avoided entirely). Results will differ meaningfully depending on which
  provider/model is selected.
- **Bot-protected sources are accepted as out of reach.** Some candidates
  fail to scrape due to active bot-detection (Vercel security checkpoints,
  WAF 403s) even where `robots.txt` technically allowed access. No attempt
  is made to defeat these — see the ToS discussion in Design Decisions.
- **Directory-mined leads are lower-confidence by construction.** A
  company name extracted from a directory listing is re-searched to find
  its own website (same verification standard as everything else), but
  the initial extraction step trusts the LLM to read the directory page
  correctly — it's an LLM extraction task, not a deterministic one.

---

## Sample Results

_Pending — three product–country combinations are required for the final
submission; sample runs will be added here once available._
