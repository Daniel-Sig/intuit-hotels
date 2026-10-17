# Review Intelligence Assistant: progress notes

Working notes for the Senior MLE home assignment (see the PDF in this folder; it is git-ignored).
Last updated 2026-10-17. All work is committed.

## Where things stand

| Assignment item | Status |
|---|---|
| Task 1: system design | **Done**: `DESIGN.md` |
| Task 2: FastAPI prototype | **Done**, including the LLM answer (`app/llm.py`) and LLM planner (`app/planner.py`) |
| Task 3: logging, 2+ metrics, automated tests | **Done** (14 metrics, JSON logs, 83 offline tests) |
| Task 3: quality evaluation and regression plan | **Done**: plan in `EVAL_AND_MONITORING.md`, plus a working golden set and judge |
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

Fixed since the first prototype: open-ended questions now get a balanced positive/negative sample; demonyms ("Australian") and unsupported cities are handled by the rules and the planner; all errors share `{"error", "detail"}`; `requirements.txt` is pinned.

Still open:
1. **Off-topic questions:** with the LLM planner on, its `in_scope` flag declines them (`off_topic`). In rules-only mode ("how do I bake a chocolate cake", "tell me a joke") some still pass the rerank cutoff.
2. **`RELEVANCE_MIN` (-1.0) is provisional:** it sits in a narrow gap tuned on ~30 queries and has not been re-fit on a larger golden set.
3. **`mention_count` is approximate:** stem matching over-matches ("friendly" matches "friends") and synonyms are not covered.
4. **Polarity is a hint:** a positive comment can contain a complaint, and balanced evidence makes the sample look 50/50, so proportions come from the statistics.
5. **Reranker off (`RERANK=0`) means no relevance check.**
6. The startup warm-up request is counted in the metrics.
7. `_country_city` assumes every UK address is London (true for this dataset).
8. The 3B model sometimes writes a bare number as the whole answer, and an answer that fails a check is withheld without an explanation.

## LLM layer (built)

One OpenAI-compatible client (`app/llm.py`) serves three calls: the planner (`app/planner.py`), the answer (`app/llm.py`) and the claim check (`app/verifier.py`). Planner output is validated against real dataset values and falls back to the rule parser; answers are checked for retrieved citation IDs, grounded numbers and unsupported sentences. LLM metrics: `llm_latency_seconds`, `llm_tokens_total`, `llm_errors_total`, `llm_fallback_total`, `citation_invalid_total`, `ungrounded_number_total`, `claims_removed_total`, `verify_errors_total`, `filter_rejected_total`. Tested with a local `qwen2.5:3b` and Gemini's free tier; `gpt-4o-mini` is supported through the same settings but has not been run.

## Next steps

All assignment deliverables exist. Remaining: revoke the shared Gemini key, optionally run once with `gpt-4o-mini`, and the later "improvements" section (see memory note: enum-constrained planner output, stronger judge/model, merge LLM calls, explain withheld answers).

## Alerting proposal (for the write-up)

5xx rate above 2% for 5 min; p95 latency above 1 s for 10 min; `insufficient_evidence` rate more than double its 7-day baseline; sustained `citation_invalid_total` growth; daily LLM spend above budget.
