# Review Intelligence Assistant: design write-up

CX agents ask natural-language questions about 515K European hotel reviews and get a short answer that cites the reviews it came from, or an explicit "not enough evidence".

## 1. Key decisions and tradeoffs

| Decision | Why | Cost / alternative rejected |
|---|---|---|
| **Hybrid retrieval, no fine-tuning** | Exact metadata filters (city, nationality, month, hotel, sentiment) plus text search. Answers can cite reviews, and new reviews need an index update, not retraining. | Fine-tuning gives no citations and goes stale. |
| **BM25 instead of embeddings** | Embedding ~570K chunks on a laptop CPU was estimated at hours. BM25 builds in ~13 s and queries in ~7 ms. | No synonym matching. Mitigated by a cross-encoder rerank of the top 50 and by the LLM planner choosing the topic words. |
| **One BM25 shard per (city, sentiment)** | 12 shards. Filters become shard selection, and positive and negative text (separate chunks) can be searched apart. | Shards have separate IDF statistics, so cross-shard ranking is approximate. |
| **Numbers computed in pandas, never by the LLM** | Counts, averages and "how many mention X" run over *all* matching reviews, not the top-k. The LLM chooses from a closed menu (`mean_score`, `count`, `mention_count`, optional `group_by`), validated by pydantic. It never writes code or SQL. | The menu is limited. New question types need a new operation. |
| **LLM used in two places** | (1) *Planner*: cities, hotels, topic and a scope check, each validated against the data. (2) *Answerer*: writes a cited answer from the evidence and stats only. Not used for retrieval, ranking or arithmetic. Deterministic rules decide the operation, nationality (including demonyms), sentiment and months: small models set these wrongly, and rules are exact. | Two calls per question add latency and cost. Both degrade safely (below). |
| **Model** | The assignment names 4o-mini. The prototype uses a free option instead, through an OpenAI-compatible client behind a small interface: Gemini (free tier) or a local `qwen2.5:3b` via Ollama (no rate limits; used for the test runs). Switching to 4o-mini is `LLM_MODEL` / `LLM_BASE_URL` / key, no code change. | Gemini's free tier allows only ~20 requests/day per model and often returns 503; a 3B local model is weaker than 4o-mini, which is why the deterministic checks below matter. |
| **Chronological 70/15/15 split, only train indexed** | Keeps val/test clean for evaluation. | May–Aug 2017 is not searchable. A production build indexes everything. |

## 2. Data flow and serving

**Offline (once, minutes on a laptop):** Kaggle CSV → clean (drop 526 duplicates, parse city/country from the address, month/year, trip type, traveler group) → chronological split → chunk each review into positive and negative text → BM25 shards on disk. A parquet of the train split feeds the stats engine.

**Online, per `POST /ask`:**
1. **Plan**: the LLM proposes cities, hotels, topic and whether the question is in scope; every value is checked against the real dataset, and an invented topic word is dropped. Rules add nationality (including "Australian"), months and seasons, explicit praise/complaint sentiment, and the aggregate operation. Unknown cities are declined, not guessed. Out-of-scope questions are declined here. On any LLM failure, the rule-based parser takes over.
2. **Retrieve**: BM25 on the topic within the filtered shards, de-duplicated at query time. If the question does not ask for one side, positive and negative text are searched separately (25 each) and the evidence is balanced half and half after reranking. Otherwise searching "friendly" returns mostly "staff were not friendly" and the answer sees one side. Balanced evidence does not show the true ratio of praise to complaints; proportions must come from the stats.
3. **Rerank**: cross-encoder on the first 80 words of each candidate, keeping `top_k`.
4. **Aggregate**: pandas over all matching reviews.
5. **Gate**: `insufficient_evidence` for an unsupported city, fewer than 3 matches, or a best rerank score below the threshold. Flagged responses return no evidence or numbers and the LLM is not called.
6. **Answer** (LLM): JSON `{answer, cited_review_ids}`. Review text is passed as delimited, untrusted data.

**Grounding and unsupported claims:** the model sees only the retrieved reviews and computed stats; it must cite review IDs; every cited ID is checked against the evidence set, and an answer citing anything else is dropped (the evidence is still returned). Every number in the answer must also appear in the stats, the evidence or the question (an invented "73.4% of guests" is caught and the answer dropped). Numbers come only from the stats engine. The prompt tells it to weigh positive and negative comments and to say so when the evidence does not answer the question.

**API contract:** `POST /ask` takes `question` and optional `top_k`, `city`, `nationality`, `months`, `sentiment` (explicit filters override parsed ones). It returns `answer`, `evidence[]`, `stats`, `filters_applied`, `insufficient_evidence`, `message` and `latency_ms`. `GET /health` and `GET /metrics` are also exposed. All errors share `{"error", "detail"}`.

## 3. Performance and bottlenecks (measured on a laptop, CPU only)

| Stage | Time |
|---|---|
| BM25 query | ~7 ms |
| Stats (pandas) | 25–55 ms |
| Rerank | ~230–390 ms (was 2 s before truncating chunks to 80 words) |
| **LLM answer** | **~5–6 s on the free tier** |
| Retrieval-only `/ask` (no LLM) | ~260–440 ms |

The LLM dominates latency and cost, and it is the only call with external failure modes. Mitigations: skip it entirely when evidence is insufficient; small prompt (≤10 short reviews); `max_retries=1` with a 20 s timeout; the retrieval result is returned even if the answer fails. Next steps if latency matters: stream the answer, merge planner and answerer into one call, cache answers by normalised question + filters.

## 4. Failure modes and mitigations

| Failure | Behaviour |
|---|---|
| LLM down, slow, rate-limited (429/503) | Planner: falls back to rules. Answerer: `answer: null`, evidence and stats still returned. Counted in `llm_errors_total`. |
| LLM returns invalid JSON, invented filter values or invalid operation | Pydantic and dataset validation reject it; fall back to rules (`llm_fallback_total`, `filter_rejected_total`). |
| LLM cites reviews that were not retrieved | Answer dropped (`citation_invalid_total`). |
| Prompt injection inside review text or question | Both are wrapped as untrusted data; the model has no tools and cannot execute anything; output is validated against the evidence. |
| City not in the data (for example Rome) | Declined with the list of covered cities. |
| Off-topic or low-relevance question | Declined before the answer step. A score cutoff alone misses some off-topic questions ("tell me a joke" scored +2.4), which is why the planner also checks scope. |
| Too few matches | `insufficient_evidence` with a message. |
| Stats computation error | Logged and skipped; request still succeeds. |
| Index missing at startup | `/health` and `/ask` return 503. |

## 5. Known limitations

`mention_count` uses stem matching, so it over-matches (for example "friendly" matches "friends") and ignores synonyms. Open-ended questions rank the filtered pool by generic words rather than by informativeness. Ranking across shards is approximate. Seasonal answers rely on Aug 2015 – Jan 2017 data because of the train-only index. Free-tier LLM limits make the demo single-user.

See `EVAL_AND_MONITORING.md` for metrics, alerts and the evaluation plan, and `SCALING.md` for the 50M-review answer.
