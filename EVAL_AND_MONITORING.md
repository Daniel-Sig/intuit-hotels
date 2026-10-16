# Monitoring and evaluation plan

## 1. What is implemented

**Logs:** one JSON line per request (`app.api`): request id, filters applied, result count, `insufficient`, `latency_ms`, per-stage `timings_ms`. The question text is **not** logged, only its length and a short hash, since it may contain personal data. Errors are logged with the stack trace. The request id is returned in `X-Request-ID`.

**Metrics** (`GET /metrics`, Prometheus format):

| Metric | Use |
|---|---|
| `requests_total{endpoint,status}` | traffic and error rate |
| `request_latency_seconds{endpoint}` | p50/p95/p99 |
| `stage_latency_seconds{stage}` (parse, bm25, rerank, stats, llm) | finds the bottleneck |
| `insufficient_evidence_total{reason}` | unsupported_city, too_few_results, low_relevance, off_topic |
| `stats_failures_total` | aggregate errors |
| `llm_latency_seconds`, `llm_tokens_total{kind}`, `llm_errors_total` | LLM health and cost |
| `llm_fallback_total` | planner fell back to rules |
| `citation_invalid_total` | answers dropped for ungrounded citations |
| `ungrounded_number_total` | answers dropped for stating a number not in the stats or evidence |
| `filter_rejected_total{field}` | LLM proposed filter values that do not exist |

**Tests:** 63 offline tests on a small synthetic dataset and fake LLM clients (data cleaning, parsing, search, stats, API contract and error shapes, planner validation and fallback, answer grounding and failure handling).

## 2. Alerts (proposal)

| Alert | Condition | Meaning |
|---|---|---|
| High error rate | 5xx > 2% for 5 min | page |
| Slow | p95 `request_latency` > 1 s for 10 min (retrieval-only) or > 8 s with LLM | page / ticket |
| LLM degraded | `llm_errors_total` rate > 10% for 10 min, or `llm_fallback_total` rising | ticket: service still works in reduced mode |
| Grounding problems | sustained growth of `citation_invalid_total` or `ungrounded_number_total` | ticket: prompt/model regression |
| Behaviour drift | `insufficient_evidence` rate > 2x its 7-day baseline | ticket: data, parser or traffic change |
| Cost | daily LLM tokens or spend above budget | ticket |
| Index missing | `/health` 503 | page |

## 3. Quality evaluation approach (a plan; not yet run)

**Golden set.** About 100-200 questions, built mostly from the **val split** (never indexed, so no leakage) and written to cover each question type: open-ended ("what do guests from Australia say about Amsterdam"), specific-topic ("is staff friendly in Paris"), seasonal ("noisy in London in June"), aggregates (average score, counts, "how many mention breakfast", group-by), and negative cases (a city not in the data such as Rome, off-topic questions, impossible filter combinations). Each item stores the question, the expected filters and operation, whether the system should decline, and for aggregates the expected number computed directly with pandas. Labels are reviewed by a person; an LLM can draft candidates.

**Evaluated layer by layer, so a failure points at its cause:**

| Layer | Metric | Method |
|---|---|---|
| Planner | filter and operation exact-match, correct decline rate | compare to labels (deterministic) |
| Retrieval | recall@k and nDCG of labelled relevant reviews; every returned review satisfies the filters | labels plus a filter check |
| Aggregates | exact match vs pandas | deterministic |
| Answer grounding | share of claims supported by cited reviews; numbers appear in the stats | LLM-as-judge with a rubric, calibrated on a human-labelled sample (citation validity is already enforced in code) |
| Answer usefulness | covers both sides, no overclaiming, correct refusal | judge plus periodic human review |
| Refusals | decline precision and recall on the negative set; false-decline rate on valid questions | labels |

**LLM-as-judge (prototype implemented).** `scripts/run_questions.py` runs a CSV of questions through the real pipeline and `scripts/judge_answers.py` scores each row with an LLM against a fixed rubric: grounded (1-5), numbers correct, relevant (1-5), balanced, and for refusals whether declining was right. The pass/fail verdict is computed in code from those scores, because small judge models contradict their own verdicts. The prototype was run with the same 3B local model that produces the answers, which is lenient and noisy: it passed a wrong decline early on and sometimes gave unconvincing reasons. In production the judge would be a stronger model than the one being judged, calibrated against a human-labelled sample before its scores gate anything. It did already catch a real defect, a fabricated percentage, which led to the deterministic number check.

**Regression testing.** Run the golden set in CI on every change to a prompt, model, index build or threshold. The deterministic layers (planner, retrieval, aggregates, refusals) run offline against a fixed index. LLM-judged layers run on a sample and compare to the last accepted baseline. Gate: no drop beyond tolerance on any layer (for example, filter accuracy and refusal metrics must not fall by more than 2 points, grounding must not fall at all). The relevance threshold (`RELEVANCE_MIN`, currently provisional at -1.0 from about 30 queries) is re-fitted on this set, trading false declines against missed off-topic questions.

**Drift monitoring (production).**
- *Input drift:* distribution of cities, topics and question length, the `insufficient_evidence` rate by reason, and the share of questions that hit the rule-based fallback.
- *Data drift:* review volume and score distributions per city and month at each index refresh; new nationalities or cities (so the "unsupported city" list is data-driven).
- *Quality drift:* weekly replay of the golden set against the live stack, plus a sample of production answers (about 1%) scored by the judge and spot-checked by a person. Optional thumbs up/down from CX agents feeds new golden-set items.
- *Model drift:* provider model changes show up as shifts in golden-set scores, `citation_invalid_total` and token counts, so the model version is pinned and logged.
