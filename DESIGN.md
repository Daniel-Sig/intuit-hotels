# Review Intelligence Assistant: design write-up

CX agents ask questions about 515K European hotel reviews in plain English. The service answers in a few sentences and cites the reviews it used, or says "not enough evidence". Flow: `docs/architecture.pdf`. Monitoring and evaluation: `EVAL_AND_MONITORING.md`. 50M reviews: `SCALING.md`.

## 1. Key decisions and tradeoffs

| Decision | Why | Cost |
|---|---|---|
| **Retrieval, not fine-tuning** | A fine-tuned model can't point to the review behind a claim, can't be stopped from adding outside knowledge, can't apply an exact filter like "Australian guests in June", and can't count 515K reviews. New reviews would mean retraining. Retrieval gives citations, exact filters and updates by re-indexing. | More moving parts. |
| **BM25 search, not embeddings** | Embedding ~570K chunks on a laptop would take hours. BM25 builds in ~13 s and answers in ~7 ms. | No synonyms; a reranker helps. |
| **Split each review into its positive and negative comment; one index per (city, sentiment)** | The dataset already stores them apart. Sentiment becomes an exact filter and we can show both sides. Searching "friendly" over everything mostly returns "staff were not friendly". | A positive comment can still contain a complaint. |
| **Numbers from pandas over all matching reviews** (fixed menu: average score, count, mentions, optional group-by) | Search returns at most 50 chunks, so counting them gives a confident wrong number, and an LLM can't count thousands of reviews. The model never writes code. | Small menu; "mentions" are word-stem matches. |
| **LLM only for fuzzy language** | It reads the question and writes the cited answer. Search, math and exact values are plain code, and every model output is checked, so a weak model lowers quality but can't invent a filter, number or citation. | 2–3 LLM calls per question. |
| **Model (differs from the brief)** | The brief names 4o-mini; no key was available and a free model was allowed. The prototype uses a local `qwen2.5:3b` or Gemini's free tier; switching is three environment variables. | A 3B model is weaker; quality figures are for it. |

## 2. Data flow and serving

**Offline:** CSV → clean → split by time (only the train part is indexed) → chunk → 12 BM25 indexes, plus a parquet of the 360K train reviews for statistics. 

**The train table** feeds only the statistics. It is loaded into memory at startup; per question, pandas filters it with the same filters the search used and computes on that slice (counts, average scores, or how many reviews mention a topic and what share of mentions are positive versus negative). Nothing is precomputed, because any filter can combine with any typed topic. It costs 25–55 ms; at 50M reviews it becomes rollups (`SCALING.md`).

**Keeping it fresh.** The build reruns **daily** if reviews arrive quickly, **weekly** if volume is low. Each run writes the indexes and parquet together into a new versioned folder, must pass the golden set, and replicas switch to it one at a time; rollback is the previous folder. Production indexes the full history, not just the train split.

**Online, per `POST /ask`:**
1. **Plan:** the LLM proposes cities, hotels, topic and scope; code fixes the exact values (section 5). If the LLM is down, a rule-based parser takes over.
2. **Retrieve and rerank:** BM25 in the matching indexes, then a cross-encoder reranks.
3. **Aggregate:** pandas over all matching reviews.
4. **Gate:** unknown city or off-topic, then fewer than 3 matches, then low relevance (skipped for numeric questions, which come from the statistics) return `insufficient_evidence` with a `reason` and no LLM call.
5. **Answer and check:** the LLM sees only the retrieved reviews and the statistics. Code checks the cited IDs and every number; a second LLM call checks each sentence against the evidence and removes unsupported ones. A failed check drops the answer; the evidence is still returned.

**API and serving.** `POST /ask` takes a question and optional filters and returns `answer`, `citations` (ID and first 10 words), `evidence`, `stats`, `insufficient_evidence` and `reason`. Bad input gets a 422; all errors share `{"error", "detail"}`. The service is stateless and loads its index and models at startup. The index is read-only, so replicas are interchangeable; scaling is limited by the LLM provider's rate and cost, not the service.

## 3. Performance, cost and bottlenecks

One desktop (6 cores, 16 GB), one user at a time:

| Stage | Time |
|---|---|
| BM25 / statistics / rerank | ~7 ms / 25–55 ms / 230–390 ms |
| Whole request without the LLM | ~260–440 ms |
| Whole request with the LLM | ~5–6 s (free hosted); 0.5–6 s (local 3B on a GPU) |

The LLM dominates latency and cost. Mitigations: skip it when evidence is insufficient, keep prompts small, retry with a timeout, make the claim check optional, and return retrieval results if the answer fails. Next: stream the answer, merge LLM calls, cache.

## 4. Failure modes

| Failure | Behaviour |
|---|---|
| LLM down, slow, rate-limited, or returns bad JSON or made-up filter values | Planner falls back to rules or drops the bad value; `answer: null`, evidence and stats still returned. |
| Answer cites an unretrieved review, states a number not in the data, or makes an unsupported claim | Answer dropped or the sentence removed; evidence still returned. |
| Prompt injection in a question or review | Text is passed as untrusted data, the model has no tools, and output is checked against the evidence. |
| Unknown city, off-topic, or too few matches | Declined with a reason; no LLM answer call. |

## 5. The subtle logic, and why

**Who decides what.** The model is good at "what is this about" and bad at exact values.
- *Cities and hotels:* kept only if real **and** named in the question (a 3B model added every city to "how many reviews are from Australia?").
- *Nationality, months, sentiment, which number:* rules. The model wrote "Australian" (not a dataset value), turned "at night" into six months, and added unasked sentiment. Sentiment is filtered only for complaints or praise; "is the wifi good?" wants both sides.
- *Topic:* the model's words must be what's left after removing filter phrases; it once invented "food" for "Italian guests".

**Balanced evidence.** With no side requested, 25 candidates come from each polarity and the final list takes half from each. That makes the sample look 50/50 whatever the truth, so proportions come from the statistics, never the sample.

**Two kinds of statistics.** A numeric question runs one menu operation (code picks it); an opinion question ("is staff friendly?") runs a topic breakdown. "Friendly staff" needs both words; "crowded/noisy" means either. Each result gets a plain-language sentence because the model swapped positive and negative when reading raw fields.

**Answer checks.** Every number must appear in the statistics, reviews or question; a whole-number percentage may round a stored decimal ("78%" passes for 78.4, "73.4%" doesn't). The claim check fails open and drops the answer only if something was removed and nothing supported remains.

## 6. Observability and quality

**Logs and metrics.** One JSON log line per request (ID, filters, result count, stage timings; never the question text). Prometheus metrics cover requests, per-stage latency, decline reasons, and LLM latency, tokens, errors, fallbacks and failed checks. **Alerts:** 5xx above 2%, p95 over budget, rising LLM errors or failed checks, doubled decline rate, LLM spend.

**Quality.** A 40-question golden set checks each layer separately (decline, filters, operation, the number recomputed with pandas, evidence) and blocks regressions against a saved baseline; an LLM judge scores a sample of answers; a weekly replay and the decline rate watch for drift.

## 7. Known limitations

"Mentions" are stem matches (they over-match and miss synonyms), and "positive/negative" means which comment a word is in, not measured sentiment. Only the train split is indexed (Aug 2015 – Jan 2017). A 3B model sometimes writes a bare number. The golden set is small and self-written: a regression guard, not proof of quality.
