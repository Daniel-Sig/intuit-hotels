# Design details

The full-length version of the design write-up, with the complete rationale for chunking and aggregators, the claim check, the API contract and every failure mode. The two-page summary is `DESIGN.md`.

CX agents ask natural-language questions about 515K European hotel reviews and get a short answer that cites the reviews it came from, or an explicit "not enough evidence".

## 1. Key decisions and tradeoffs

| Decision | Why | Cost / alternative rejected |
|---|---|---|
| **Hybrid retrieval, no fine-tuning** | Exact metadata filters (city, nationality, month, hotel, sentiment) plus text search. Answers can cite reviews, and new reviews need an index update, not retraining. | Fine-tuning gives no citations and goes stale. |
| **BM25 instead of embeddings** | Embedding ~570K chunks on a laptop CPU was estimated at hours. BM25 builds in ~13 s and queries in ~7 ms. | No synonym matching. Mitigated by a cross-encoder rerank of the top 50 and by the LLM planner choosing the topic words. |
| **One BM25 shard per (city, sentiment)** | 12 shards. Filters become shard selection, and positive and negative text (separate chunks) can be searched apart. | Shards have separate IDF statistics, so cross-shard ranking is approximate. |
| **Numbers computed in pandas, never by the LLM** | Counts, averages and "how many mention X" run over *all* matching reviews, not the top-k. The LLM chooses from a closed menu (`mean_score`, `count`, `mention_count`, optional `group_by`), validated by pydantic. It never writes code or SQL. | The menu is limited. New question types need a new operation. |
| **LLM used in two places** | (1) *Planner*: cities, hotels, topic and a scope check, each validated against the data. (2) *Answerer*: writes a cited answer from the evidence and stats only. Not used for retrieval, ranking or arithmetic. Deterministic rules decide the operation, nationality (including demonyms), sentiment and months: small models set these wrongly, and rules are exact. | Two calls per question add latency and cost. Both degrade safely (below). |
| **Model (deviation from the brief)** | The brief names 4o-mini. No 4o-mini credentials were available and a free model of any kind was acceptable, so the prototype uses one through an OpenAI-compatible client behind a small interface: Gemini (free tier) or a local `qwen2.5:3b` via Ollama (no rate limits; used for the test runs). Switching to 4o-mini is `LLM_MODEL` / `LLM_BASE_URL` / key, with no code change. | Gemini's free tier allows only ~20 requests/day per model and often returns 503; a 3B local model is weaker than 4o-mini, which is why the deterministic checks below matter. Answer quality figures in this repo are for the 3B model. |

**Where the LLM is used, and why.** It is used where the input is fuzzy language: deciding what a question is about and whether it concerns hotel feedback at all, and turning a handful of retrieved reviews into a few readable sentences. It is deliberately not used where code gives an exact answer: retrieval and ranking (BM25 and a cross-encoder are fast, explainable and let every claim point at a review), arithmetic (pandas over all matching reviews), and extracting exact values such as nationality, months and the aggregate to compute (rules). Everything the model returns is validated before use, so a weak or unavailable model lowers answer quality or removes the answer; it cannot invent a filter, a number or a citation that reaches the user. That property is also what made a small local model usable for testing.
| **Chronological 70/15/15 split, only train indexed** | Keeps val/test clean for evaluation. | May–Aug 2017 is not searchable. A production build indexes everything. |

## 2. Data flow and serving

**Offline (once, minutes on a laptop):** Kaggle CSV → clean (drop 526 duplicates, parse city/country from the address, month/year, trip type, traveler group) → chronological split → chunk each review into positive and negative text → BM25 shards on disk. A parquet of the train split feeds the stats engine.

**Why a review is chunked into its positive and negative text.** Each review in the dataset has two separate fields, what the guest liked and what they disliked. Each becomes its own chunk (360,211 train reviews give 474,888 chunks), carrying the review's metadata (hotel, city, nationality, month, scores) and a `sentiment` label. Reasons:
- **Polarity is kept.** In one combined text, "free wifi" and "the wifi did not work" look the same to a word-matching search. As a label, sentiment becomes an exact filter ("what do guests complain about" searches only negative text), and the index is sharded by city and sentiment so that filter selects shards instead of scanning.
- **It makes balanced answers possible.** Searching "friendly" over everything returned mostly "staff were not friendly". With polarity separate, retrieval can take half its evidence from each side, and the topic breakdown can report how mentions split between the two.
- **Chunks are quotable and rerankable.** A chunk is one self-contained comment, so a citation shows exactly what the guest said, and the cross-encoder scores a coherent piece of text (its first 80 words) rather than a mixed review.
- **Empty fragments stay out.** Placeholders and fragments such as "No Positive", "Nothing" or "N A" carry no information. Chunks under 5 words are dropped, counted from the text because the dataset's own word-count columns do not match it.

The cost is that the label is a hint, not a verdict: a guest's positive comment often contains a caveat ("Loved this hotel but traffic was noisy"), and a chunk shows one half of a review without the other.

**Why numbers come from aggregators, not from the retrieved reviews.** Retrieval returns a sample of at most 50 chunks out of hundreds of thousands, chosen because they match the query words. A question such as "how many Paris reviews mention breakfast?", "what is the average score in Milan in summer?" or "which city is rated highest?" is about the whole population, so counting or averaging the sample would give a confident, wrong figure. A language model cannot read that many reviews and is unreliable at counting and averaging. So these questions are answered by pandas over every review that matches the same filters used for retrieval, in 25 to 55 ms, and the model only restates the result.
- **A closed menu keeps it safe and checkable.** The three operations (`mean_score`, `count`, `mention_count`, optionally grouped by city, nationality, month, trip type or traveler group) are chosen by a deterministic router and validated, so the model never writes code or SQL. Groups under 30 reviews are left out of rankings to avoid noisy winners.
- **Opinion questions get the same treatment.** For "is the staff friendly?" the topic breakdown gives the real proportion across all matching reviews, which ten sampled reviews cannot.
- **Limits.** The menu is small, so a new kind of question needs a new operation. "Mentions" are stem matches, which over-match ("friendly" also hits "friends") and miss synonyms, and "positive" and "negative" mean which comment the word appears in, not a measured sentiment.

**Online, per `POST /ask`:**
1. **Plan**: the LLM proposes cities, hotels, topic and whether the question is in scope; every value is checked against the real dataset, and an invented topic word is dropped. Rules add nationality (including "Australian"), months and seasons, explicit praise/complaint sentiment, and the aggregate operation. Unknown cities are declined, not guessed. Out-of-scope questions are declined here. On any LLM failure, the rule-based parser takes over.
2. **Retrieve**: BM25 on the topic within the filtered shards, de-duplicated at query time. If the question does not ask for one side, positive and negative text are searched separately (25 each) and the evidence is balanced half and half after reranking. Otherwise searching "friendly" returns mostly "staff were not friendly" and the answer sees one side. Balanced evidence does not show the true ratio of praise to complaints; proportions must come from the stats.
3. **Rerank**: cross-encoder on the first 80 words of each candidate, keeping `top_k`.
4. **Aggregate**: pandas over all matching reviews. For a question about a topic that asks for no number ("is staff friendly?"), a topic breakdown is computed over every matching review: how many mention the topic, and what share of mentions sit in the guest's positive comment versus the negative one. Every result also gets a one-sentence plain-language summary, because a small model misreads raw JSON fields (it once swapped positive and negative). The decline reason is returned in `reason`.
5. **Gate**: `insufficient_evidence` for an unsupported city, fewer than 3 matches, or a best rerank score below the threshold. Flagged responses return no evidence or numbers and the LLM is not called.
6. **Answer** (LLM): JSON `{answer, cited_review_ids}`. Review text is passed as delimited, untrusted data. Temperature 0.

**Claim check:** a second LLM pass labels each sentence of the answer supported, unsupported or not-a-claim against the cited reviews and the stats, and removes unsupported sentences (the count is returned in `unsupported_claims_removed`; if nothing supported remains, the answer is dropped). It fails open: if the check itself errors, the answer, which already passed the deterministic checks below, is kept. Disable with `VERIFY_CLAIMS=0`.

**Grounding and unsupported claims:** the model sees only the retrieved reviews and computed stats; it must cite review IDs; every cited ID is checked against the evidence set, and an answer citing anything else is dropped (the evidence is still returned). Every number in the answer must also appear in the stats, the evidence or the question (an invented "73.4% of guests" is caught and the answer dropped). Numbers come only from the stats engine. The prompt tells it to weigh positive and negative comments and to say so when the evidence does not answer the question.

**API contract:** `POST /ask` takes `question` and optional `top_k`, `city`, `nationality`, `months`, `sentiment` (explicit filters override parsed ones). It returns `answer`, `citations[]` (review ID and the first 10 words), `evidence[]`, `stats`, `filters_applied`, `insufficient_evidence`, `reason` (unsupported_city, off_topic, too_few_results or low_relevance), `unsupported_claims_removed`, `message` and `latency_ms`. `GET /health` and `GET /metrics` are also exposed. All errors share `{"error", "detail"}`.

**Serving topology.** One stateless FastAPI service (uvicorn). At startup it loads the 12 BM25 shards (memory-mapped, about 0.5 s), the chunk text and the train-split parquet for statistics, and the 80 MB cross-encoder, then answers a warm-up query before `/health` reports ready. Nothing is written per request, so replicas are interchangeable behind a load balancer. Each holds its own read-only copy of the index, which is built offline and shipped as a versioned artifact, so a new index is a rolling restart. The LLM is an external service reached over HTTP with a timeout and retries. It is the only dependency that can be slow or down, and the service keeps answering without it. `/metrics` is scraped by Prometheus and logs are one JSON line per request. In the prototype all of this runs in one process on a laptop; at larger scale the reranker becomes its own service and the index moves to a search engine (see `SCALING.md`).

## 3. Performance and bottlenecks (measured on a laptop, CPU only)

| Stage | Time |
|---|---|
| BM25 query | ~7 ms |
| Stats (pandas) | 25–55 ms |
| Rerank | ~230–390 ms (was 2 s before truncating chunks to 80 words) |
| **LLM answer** | **~5–6 s on the free tier** |
| Retrieval-only `/ask` (no LLM) | ~260–440 ms |
| Full question, local 3B model on a GPU (planner, answer and claim check) | ~0.5–6 s |

The LLM dominates latency and cost, and it is the only call with external failure modes. Mitigations: skip it entirely when evidence is insufficient; small prompt (≤10 short reviews); 3 retries with backoff and a configurable timeout (`LLM_TIMEOUT`, default 20 s); the retrieval result is returned even if the answer fails. The claim check adds a third call per answered question and can be turned off. Next steps if latency matters: stream the answer, merge planner and answerer into one call, cache answers by normalised question + filters.

## 4. Failure modes and mitigations

| Failure | Behaviour |
|---|---|
| LLM down, slow, rate-limited (429/503) | Planner: falls back to rules. Answerer: `answer: null`, evidence and stats still returned. Counted in `llm_errors_total`. |
| LLM returns invalid JSON, or filter values that are not real or that the question does not state | Invalid JSON falls back to rules (`llm_fallback_total`); unreal or unstated values are dropped (`filter_rejected_total`). The operation always comes from rules. |
| LLM cites reviews that were not retrieved | Answer dropped (`citation_invalid_total`). |
| LLM states a number not in the stats, evidence or question, or a sentence the reviews do not support | Number check drops the answer (`ungrounded_number_total`); claim check removes the sentence (`claims_removed_total`); the evidence is still returned. |
| Prompt injection inside review text or question | Both are wrapped as untrusted data; the model has no tools and cannot execute anything; output is validated against the evidence. |
| City not in the data (for example Rome) | Declined with the list of covered cities. |
| Off-topic or low-relevance question | Declined before the answer step. A score cutoff alone misses some off-topic questions ("tell me a joke" scored +2.4), which is why the planner also checks scope. |
| Too few matches | `insufficient_evidence` with a message. |
| Stats computation error | Logged and skipped; request still succeeds. |
| Index missing at startup | `/health` and `/ask` return 503. |

## 5. Known limitations

`mention_count` uses stem matching, so it over-matches (for example "friendly" matches "friends") and ignores synonyms. Open-ended questions rank the filtered pool by generic words rather than by informativeness. Ranking across shards is approximate. Seasonal answers rely on Aug 2015 – Jan 2017 data because of the train-only index. Free-tier LLM limits make the demo single-user. A 3B model sometimes answers with a bare number and misreads a statistic that a stronger model would not; the checks catch invented figures and citations, not every misreading. The golden set is small and self-written, so it guards against regressions without proving quality.

See `EVAL_AND_MONITORING.md` for metrics, alerts and the evaluation plan, and `SCALING.md` for the 50M-review answer.
