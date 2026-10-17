# Review Intelligence Assistant: design write-up

CX agents ask questions about 515K European hotel reviews in plain English. The service answers in a few sentences, cites the reviews it used, or says "not enough evidence". The flow is in `docs/architecture.pdf`. Monitoring and evaluation: `EVAL_AND_MONITORING.md`. 50M reviews: `SCALING.md`.

## 1. Key decisions and tradeoffs

| Decision | Why | Cost |
|---|---|---|
| **Retrieval, not fine-tuning** | A fine-tuned model can't point to the review behind a claim, can't be stopped from adding outside knowledge, can't apply an exact filter like "Australian guests in June", and can't count 515K reviews. New reviews would mean retraining. Retrieval gives citations, exact filters and updates by re-indexing. | More moving parts. |
| **BM25 search, not embeddings** | Embedding ~570K chunks on a laptop would take hours. BM25 builds in ~13 s and answers in ~7 ms. | No synonyms; a reranker helps. |
| **Split each review into its positive and negative comment; one index per (city, sentiment)** | The dataset already stores them apart. Sentiment becomes an exact filter, and we can show both sides. Searching "friendly" over everything mostly returns "staff were not friendly". | The label is a hint: a positive comment can contain a complaint. |
| **Numbers come from pandas over all matching reviews** (a fixed menu: average score, count, mentions, optional group-by) | Search returns at most 50 chunks, so counting them gives a confident wrong number, and an LLM can't count thousands of reviews. The model never writes code. | Small menu; "mentions" are word-stem matches. |
| **LLM only for fuzzy language** | It reads the question and writes the cited answer. Search, math and exact values (nationality, months, sentiment, which number to compute) are plain code. Every model output is checked, so a weak model lowers quality but can't invent a filter, number or citation. | 2–3 LLM calls per question. |
| **Model (differs from the brief)** | The brief names 4o-mini; no key was available and a free model was allowed. The prototype uses Gemini's free tier or a local `qwen2.5:3b`. Switching is three environment variables. | Free tiers are small and a 3B model is weaker; quality figures are for it. |

## 2. Data flow and serving

**Offline:** CSV → clean → split by time (only the train part is indexed) → chunk → 12 BM25 indexes, plus a parquet file for statistics. Everything else depends on the question, so it runs online.

**What the train table is for.** The train parquet (360K reviews, ten columns: city, hotel, nationality, month, trip type, score, review date, positive and negative text) feeds only the statistics. It is loaded into memory once at startup. For each question, pandas filters it down to the matching reviews with the same filters the search used, then computes the answer on that slice: counts and average scores, or for opinion questions how many reviews mention the topic and what share of those mentions sit in positive versus negative comments. Nothing is precomputed, because any filter can be combined with any typed topic word. A question costs 25–55 ms, small next to the LLM call; at 50M reviews this becomes precomputed rollups (`SCALING.md`). It is separate from the search indexes because search returns at most 50 chunks, which is too few to count from.

**Keeping it fresh.** New reviews arrive after the build, so the build reruns on a schedule: **daily** if reviews arrive quickly or CX needs recent complaints, **weekly** if volume is low (the cadence follows how fast the data changes, not query traffic). A rebuild takes minutes at this size (BM25 ~13 s). Each run writes the 12 indexes and the parquet together into a new versioned folder, so they always agree. It must pass the golden set, then replicas switch to it one at a time (rolling restart); a bad build is never served and rollback is the previous folder. In production the build indexes the full history, not only the train split. At 50M reviews, the daily job indexes only new reviews (`SCALING.md`).

**Online, per `POST /ask`:**
1. **Plan:** the LLM proposes cities, hotels, topic and scope; code fixes the exact values (section 5). Unknown city or off-topic: declined. LLM down: rule-based parser.
2. **Retrieve and rerank:** BM25 in the matching indexes, then a cross-encoder reranks.
3. **Aggregate:** pandas over all matching reviews.
4. **Gate:** too few matches or poor relevance returns `insufficient_evidence` with a `reason`, with no LLM call.
5. **Answer and check:** the LLM sees only the retrieved reviews and the statistics. Code checks the cited IDs and every number; a second LLM call then checks each sentence against the evidence and removes unsupported ones. If a check fails, the answer is dropped and the evidence is still returned.

**The four example questions.** Australians in Amsterdam: nationality and city filters. Paris staff: a breakdown over all Paris reviews gives the real share of positive mentions. London noise in June: month filter, counting "noisy" or "crowded". Spring in Rome: Rome isn't in the data (six cities), so it's declined and the covered cities are listed.

**API.** `POST /ask` takes `question` plus optional `top_k`, `city`, `nationality`, `months`, `sentiment`. It returns `answer`, `citations` (ID and first 10 words), `evidence`, `stats`, `filters_applied`, `insufficient_evidence`, `reason`, `latency_ms`. Bad input gets a 422; all errors share one `{"error", "detail"}` shape.

**Serving.** One stateless service. It loads its index and models at startup and reports ready on `/health` only after a warm-up. The index is read-only, so replicas are interchangeable; scale them until the LLM provider's rate limit or cost binds. Beyond one machine: `SCALING.md`.

## 3. Performance, cost and bottlenecks

Measured on one desktop (6 cores, 16 GB), one user at a time:

| Stage | Time |
|---|---|
| BM25 / statistics / rerank | ~7 ms / 25–55 ms / 230–390 ms |
| Whole request without the LLM | ~260–440 ms |
| Whole request with the LLM | ~5–6 s (free hosted); 0.5–6 s (local 3B on a GPU) |

The LLM dominates latency and cost. Mitigations: skip it when evidence is insufficient, keep prompts small, retry with a timeout, make the claim check optional, and return the retrieval result if the answer fails. Next: stream the answer, merge LLM calls, cache repeated questions, use a cheaper model.

## 4. Failure modes

| Failure | Behaviour |
|---|---|
| LLM down, slow or rate-limited | Planner falls back to rules; `answer: null`, evidence and stats still returned. |
| LLM returns bad JSON or made-up filter values | Falls back to rules, or the bad value is dropped. |
| Answer cites an unretrieved review, states a number not in the data, or makes an unsupported claim | Answer dropped or the sentence removed; evidence still returned. |
| Prompt injection in a question or review | Text is passed as untrusted data, the model has no tools, and output is checked against the evidence. |
| Unknown city, off-topic, or too few matches | Declined with a reason; no LLM answer call. |

## 5. The subtle logic, and why

Each rule exists because a simpler version failed on real data.

**Who decides what.** The model is good at "what is this about" and bad at exact values.
- *Cities and hotels:* kept only if they're real **and** named in the question. Otherwise a 3B model added every city to "how many reviews are from Australia?" and guessed a city from a hotel name.
- *Nationality, months, sentiment, which number:* decided by rules. The model wrote "Australian" (not a dataset value), turned "at night" into six months, and added sentiment nobody asked for. Sentiment is filtered only for complaints or praise; "is the wifi good?" wants both sides.
- *Topic:* the model's words must be what's left after removing filter phrases. It once invented "food" for "Italian guests".

**Balanced evidence.** With no side requested, 25 candidates come from each polarity and the final list takes half from each. That makes the sample look 50/50 whatever the truth, so proportions always come from the statistics, never the sample.

**Two kinds of statistics.** If a number is asked for, one menu operation runs (code picks it, not the model). For an opinion question ("is staff friendly?") a topic breakdown runs: how many reviews mention it, and what share of mentions sit in positive or negative comments. "Friendly staff" needs both words; "crowded/noisy" means either. Each result also gets a plain-language sentence, because the model swapped positive and negative when reading raw fields.

**Decline order.** Out of scope or unknown city first, then fewer than 3 matches, then low relevance. Relevance is skipped for numeric questions, since that answer comes from the statistics, not the sample.

**Answer checks, cheapest first.** Citations must be retrieved reviews. Every number must appear in the statistics, reviews or question (a whole-number percentage may round a stored decimal: "78%" passes for 78.4, "73.4%" doesn't). Then a second LLM pass removes unsupported sentences. It fails open, and drops the answer only if something was removed and nothing supported remains.

## 6. Observability and quality

**Logs and metrics.** One JSON log line per request (ID, filters, result count, stage timings; never the question text, only its length and a hash). Prometheus metrics at `/metrics` cover requests, latency per stage, decline reasons, and LLM latency, tokens, errors, fallbacks and failed checks. **Alerts:** 5xx above 2%, p95 latency over budget, rising LLM errors or failed checks, decline rate at twice baseline, daily LLM spend over budget.

**Quality.** A 40-question golden set checks each layer separately (decline reason, filters, operation, the number recomputed with pandas, valid evidence) and blocks regressions against a saved baseline. An LLM judge scores answers on a sample. Drift is watched through question mix, decline rates and a weekly golden-set replay. See `EVAL_AND_MONITORING.md`.

## 7. Known limitations

"Mentions" are stem matches (they over-match and miss synonyms), and "positive/negative" means which comment a word is in, not measured sentiment. Only the train split is indexed (Aug 2015 – Jan 2017). A 3B model sometimes writes a bare number or misreads a statistic. The golden set is small and self-written: it guards against regressions but doesn't prove quality.
