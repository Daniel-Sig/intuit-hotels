# Review Intelligence Assistant: progress notes

Working notes for the Senior MLE home assignment (see the PDF in this folder; it is git-ignored).
Last updated 2026-10-16. Latest commit: `e907d0a`. Work since then (LLM planner/answerer, citations, judge, balanced retrieval) is not committed.

## Where things stand

| Assignment item | Status |
|---|---|
| Task 1: system design | **Done**: `DESIGN.md` (2-page summary) and `DESIGN_DETAILS.md` (full rationale) |
| Task 2: FastAPI prototype | **Done**, including the LLM answer (`app/llm.py`) and LLM planner (`app/planner.py`) |
| Task 3: logging, 2+ metrics, automated tests | **Done** (14 metrics, JSON logs, 83 offline tests) |
| Task 3: quality evaluation and regression plan | **Done as a plan**: `EVAL_AND_MONITORING.md` (golden set not built; the assignment only asks to propose) |
| Monitoring/alerting plan | **Done**: `EVAL_AND_MONITORING.md` |
| Bonus: 515K to 50M scaling | **Done**: `SCALING.md` |
| Consistent error bodies | **Done**: all errors are `{"error", "detail"}` |
| README with setup/run/test | **Done**: `README.md`; `main.py` builds everything; requirements pinned; `.env.example` |
| Architecture diagram | **Done**: `docs/architecture.pdf` (also `.png`, `.svg`; `docs/make_architecture.py` regenerates the SVG) |
| LLM | Local `qwen2.5:3b` via Ollama for test runs (`LLM_BASE_URL=http://localhost:11434/v1 LLM_MODEL=qwen2.5:3b LLM_TIMEOUT=120`); Gemini free tier also works (`GEMINI_API_KEY`, ~20 requests/day per model). The assignment names 4o-mini: swap via `LLM_BASE_URL`/`LLM_MODEL`/key |
| Golden set | `eval/golden.csv` (40 questions) + `scripts/run_golden.py`: deterministic per-layer checks, `--no-llm` for CI, baseline gate. 100% on every layer in both modes (small, self-written set) |
| Claim check | `app/verifier.py`: second LLM pass removes unsupported answer sentences; `VERIFY_CLAIMS=0` disables |
| Topic stats | For topic questions with no number asked, share of mentions in positive vs negative comments over ALL matching reviews, plus plain-language summaries |
| Evaluation tooling | `scripts/run_questions.py` (CSV of questions to answers, citations, evidence) and `scripts/judge_answers.py` (LLM-as-judge, `app/judge.py`). In production the judge would be a stronger model |
| Retrieval quality | Evidence balanced between positive and negative text when no side is requested; numbers in answers must appear in the stats/evidence; operation/nationality/sentiment/months from rules, not the LLM |

## How to run it

Python 3.11 venv at `.venv` (3.13 was not installed; we chose 3.11). The older empty `venv` folder can be deleted.
Interpreter: `C:\Users\sigda\PycharmProjects\review_intellegence_assistant\.venv\Scripts\python.exe`

```
.venv\Scripts\python.exe main.py                      # download, clean + split, build indexes (skips finished steps; --rebuild redoes all)
.venv\Scripts\python.exe -m uvicorn app.api:app       # serve (verified live: /health, /ask, /metrics)
.venv\Scripts\python.exe -m pytest                    # 83 tests, no data or network needed
```

See `README.md` for the full setup, LLM options and evaluation commands.
`RERANK=0` turns the reranker off. `RELEVANCE_MIN` overrides the off-topic threshold.

## Dataset facts (checked on the real data)

- Kaggle `jiashenliu/515k-hotel-reviews-data-in-europe`: 515,738 rows, 17 columns, no all-null columns, 526 exact duplicates (dropped).
- **Only 6 cities:** Amsterdam, Barcelona, London, Milan, Paris, Vienna. **Rome is not in the data**, even though an assignment example asks about it. The system must decline it.
- Reviews span 2015-08-04 to 2017-08-03.
- The dataset's own `Review_Total_*_Word_Counts` columns do not match the text; word counts are computed from the text.
- We first downloaded `datafiniti/hotel-reviews` by mistake (US-only, no reviewer nationality); it is not used.

## Design decisions and why

- **Hybrid retrieval:** exact-match metadata filters (city, nationality, month, hotel, sentiment) plus BM25 on the text. Fine-tuning rejected: no citations, and updates need retraining.
- **BM25 instead of embeddings:** embedding ~570K chunks on a laptop CPU was estimated at hours; BM25 builds in ~13 s and queries in ~7 ms. Cost: no synonym matching. Mitigations: cross-encoder rerank, and (later) LLM query expansion.
- **Chronological 70/15/15 split** (cut on date boundaries so a day is never divided): train 360,211 rows to 2017-01-02, val to 2017-04-23, test to 2017-08-03. **Only train is indexed.** Val and test are held out for evaluation. Consequence: May-Aug 2017 is not searchable, so seasonal questions rely on 2016 data. Index everything for a production version.
- **One BM25 shard per (city, sentiment)** = 12 shards, 474,888 chunks (a review's positive and negative text are separate chunks; chunks under 5 words are dropped). Shard scores use separate IDF stats, so cross-shard ranking is approximate.
- **Dedupe at query time, not index time**, so identical text from different nationalities/months stays filterable.
- **Reranker:** `Xenova/ms-marco-MiniLM-L-6-v2` via fastembed on the top 50 candidates. Scores only the first 80 words per chunk (a 370-word chunk had pushed rerank to 2 s; now ~330 ms).
- **Aggregates** (counts, averages, mention counts, group-by) are computed with pandas over *all* matching reviews, not the top-k. A closed menu of 3 operations (`mean_score`, `count`, `mention_count`) plus optional `group_by`. The LLM must never write code or SQL.
- **Grounding:** `insufficient_evidence` is set for `unsupported_city`, `too_few_results` (<3 matches) and `low_relevance` (best rerank score below -1.0, only when the question has a specific topic). Flagged responses return no evidence or stats.

## Code map

```
main.py                 download, clean + split, build indexes
app/dataset.py          Kaggle download into data/raw
app/preprocess.py       clean(): dedupe, review_id, city/country, month/year, trip type, traveler group
app/data_prep.py        chronological split + prepare()
app/bm25_index.py       chunking, build_index(), ReviewSearch (filters, dedupe)
app/rerank.py           cross-encoder Reranker
app/query_parser.py     rule-based question -> filters + topic (stand-in for the LLM parser)
app/router.py           rule-based question -> aggregate operation (stand-in for the LLM router)
app/stats.py            StatsEngine + Operation (the closed menu)
app/service.py          AssistantService.ask(): parse -> search -> rerank -> flags -> stats
app/schemas.py          API contract (AskRequest / AskResponse / Evidence / Stats)
app/observability.py    Prometheus metrics, JSON logging, stage timing
app/api.py              FastAPI: POST /ask, GET /health, GET /metrics
tests/                  37 tests on a small synthetic dataset
```

Metrics: `requests_total`, `request_latency_seconds`, `stage_latency_seconds` (parse/bm25/rerank/stats), `insufficient_evidence_total` (by reason), `stats_failures_total`. Logs omit the question text (length + short hash only).

## Measured numbers (this laptop)

- BM25 build 12.6 s; index load 0.5 s; BM25 query ~7 ms; stats 25-55 ms; rerank ~230-390 ms; total `/ask` ~260-440 ms.
- Off-topic check on a small hand-made set: 0/19 sensible questions flagged, 9/11 nonsense caught.

## Known issues and weaknesses

1. **Off-topic misses:** "how do I bake a chocolate cake" (rerank -0.13) and "tell me a joke" (+2.4) pass. A score cutoff cannot fix this; the LLM answer step needs a domain check.
2. **Threshold is provisional:** -1.0 sits in a narrow gap (-0.12 for the lowest sensible query, -1.87 for the highest nonsense one) tuned on ~30 queries. Re-fit on a golden set from the val split.
3. **Open-ended questions** ("what do reviewers from Australia say about Amsterdam?") rank the filtered pool by generic words like "reviewers"/"say", which is arbitrary. Better: a balanced sample of positive and negative reviews, favouring informative ones. Not built.
4. **`mention_count` is approximate:** stem matching over-matches ("friendly" matches "friends"), short stems are risky ("wi fi"), and synonyms are not covered.
5. **Nationality demonyms** ("Australian") are not recognised; only "Australia".
6. **Unsupported cities** are a hand-written list (`UNSUPPORTED_CITIES`); a city not on it (e.g. "Lyon") passes through unfiltered. The LLM parser would fix this properly.
7. **Polarity is mixed** in evidence ("Staff was not as friendly" can come back for "is staff friendly"). The answer step must weigh both sides.
8. **Error body shapes differ:** our 422s use `{"detail": "..."}`, pydantic's use a list.
9. **Reranker off means no relevance check** (only the first two flags apply).
10. The startup warm-up request is counted in the metrics.
11. `_country_city` assumes every UK address is London (true for this dataset; would mislabel other UK cities).
12. Versions are not pinned in `requirements.txt`.

## LLM layer (not built; no credentials yet)

Plan: call 4o-mini through a small client interface, with a fake client for tests.
- **Router/parser call:** function calling with the 3 operations as tools (enums restrict city, group_by, etc.). Output is validated with pydantic, then checked against real dataset values; on any failure fall back to the rule-based parser/router.
- **Answer call:** write a cited answer from the evidence and stats only; verify every cited review ID was in the evidence; refuse when `insufficient_evidence`.
- New metrics to add then: `llm_latency_seconds`, `llm_tokens_total`, `llm_cost_usd_total`, `llm_errors_total`, `llm_fallback_total`, `citation_invalid_total`, `filter_rejected_total`.
- Replace the hand-written `QUESTION_FRAME` list and `UNSUPPORTED_CITIES` with LLM behaviour.

## Next steps

All assignment deliverables exist. Remaining: commit, review the diagram PDF and docs once more, and the later "improvements" section (see memory note: enum-constrained planner output, stronger judge/model, merge LLM calls, explain withheld answers).

## Alerting proposal (for the write-up)

5xx rate above 2% for 5 min; p95 latency above 1 s for 10 min; `insufficient_evidence` rate more than double its 7-day baseline; sustained `citation_invalid_total` growth; daily LLM spend above budget.
