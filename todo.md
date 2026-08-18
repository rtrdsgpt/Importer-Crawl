# Exporter Crawl — TODO

Lead-gen pipeline finding importers for Indian exporters. Already well-engineered (multi-provider
LLM ranking, checkpointing, hallucination guards, ethical scraping via robots.txt) — just needs
the productionization layer, not a redesign. Note: also has a natural Supply-Chain/Trade angle for
the FinTech/Supply-Chain CV, not just the Agentic one. See `Project Plan.md` (Projects root)
section 6.

## 1. API layer
- [ ] FastAPI wrapper: `POST /discover` (product + target country → job id), job-status/results
      endpoints — currently only Streamlit (`app.py`) + CLI (`main.py`)

## 2. Testing
- [ ] pytest for `src/rank_schema.py`'s hallucination-guarded contact validation
- [ ] pytest for `src/discovery.py`'s domain classifier (`website`/`directory`/`report`/`noise`/
      `social`)
- [ ] pytest for `src/validate.py`'s deterministic country-presence checks (phone country code,
      ccTLD, text mention, OSM geocoding)

## 3. MLOps
- [ ] Dockerfile (none currently)
- [ ] GitHub Actions CI — lint + the pytest suite above (currently only `.github/dependabot.yml`
      exists, no actual CI workflow)
- [ ] Basic tracing: structured logs or OpenTelemetry spans per pipeline phase (Discovery → Scrape
      → Directory/Report Mining → LLM Ranking → Country Validation) — the phase boundaries already
      exist in code, just need spans hung on them

## 4. Agentic / MCP framing
- [ ] Reframe the existing adaptive query-expansion step (in `src/discovery.py`) explicitly as a
      bounded-retry agentic loop in docs/README — the logic already exists, this is a framing +
      possibly small refactor task
- [ ] Consider MCP-exposing discovery/ranking as tools (`search-importers`, `rank-candidates`) —
      cheap since the underlying logic already exists
