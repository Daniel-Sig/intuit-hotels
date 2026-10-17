# Review Intelligence Assistant

A FastAPI service that answers natural-language questions about 515K European hotel reviews. Every answer cites the reviews it came from, numbers are computed over all matching reviews rather than a sample, and when the data cannot support an answer the service says so instead of guessing.

```
POST /ask {"question": "What do Australian guests say about Amsterdam hotels?"}
```

Design, tradeoffs and diagram: [`DESIGN.md`](DESIGN.md) and [`docs/architecture.pdf`](docs/architecture.pdf). Monitoring and evaluation plan: [`EVAL_AND_MONITORING.md`](EVAL_AND_MONITORING.md). 515K to 50M reviews: [`SCALING.md`](SCALING.md).

## Quick start

Python 3.11. About 1 GB of disk (dataset and indexes ~400 MB, ranking model ~150 MB). Internet is needed once for the dataset and the ranking model.

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python main.py                    # download, clean and split, build the search indexes (a few minutes)
uvicorn app.api:app               # serve on http://localhost:8000
```

`main.py` skips any step whose output already exists; `python main.py --rebuild` redoes all of them. Without an LLM configured the service runs in retrieval-only mode (see below), so it works straight away.

**Kaggle:** the dataset is [`jiashenliu/515k-hotel-reviews-data-in-europe`](https://www.kaggle.com/datasets/jiashenliu/515k-hotel-reviews-data-in-europe). If `kagglehub` asks for credentials, create a Kaggle API token and set `KAGGLE_USERNAME` and `KAGGLE_KEY`, or download `Hotel_Reviews.csv` yourself into `data/raw/` and run `python main.py` again.

## Try it

```bash
curl -X POST localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"question": "How friendly is the staff in Paris hotels?"}'
```

| Endpoint | Purpose |
|---|---|
| `POST /ask` | Ask a question. Body: `question` (3-500 characters), optional `top_k` (1-50), `city`, `nationality`, `months` (1-12), `sentiment` |
| `GET /health` | Ready once the indexes, statistics and ranker have loaded and warmed up |
| `GET /metrics` | Prometheus metrics |
| `GET /docs` | Interactive API docs |

Response fields: `answer` (null in retrieval-only mode, or when a check fails), `citations` (review ID and first 10 words), `evidence`, `stats`, `filters_applied`, `insufficient_evidence`, `reason`, `message`, `latency_ms`. Errors all share `{"error", "detail"}`; bad input gets a 422.

Example questions: *What do Australian guests say about Amsterdam hotels? Is the staff friendly in Paris? What do guests say about noise in London in June? What do guests say about hotels in Rome in spring?* (Rome is not in the data, so it is declined and the six covered cities are listed: Amsterdam, Barcelona, London, Milan, Paris, Vienna.)

## Choosing a language model

The model writes the final answer and helps read the question. Everything else is plain code, so the service answers without it. The client speaks the OpenAI API, so any compatible model works. Copy `.env.example` to `.env` and pick one:

| Option | Settings |
|---|---|
| Nothing | Leave blank: you get evidence and statistics, `answer: null` |
| Local Ollama (free) | `ollama pull qwen2.5:3b`, then `LLM_BASE_URL=http://localhost:11434/v1`, `LLM_MODEL=qwen2.5:3b`, `LLM_TIMEOUT=120` |
| Gemini free tier | `GEMINI_API_KEY=...` (about 20 requests a day per model, so enough only for a quick try) |
| OpenAI `gpt-4o-mini` (the brief's model) | `LLM_BASE_URL=https://api.openai.com/v1`, `LLM_API_KEY=...`, `LLM_MODEL=gpt-4o-mini` |

The prototype was developed and evaluated with `qwen2.5:3b`; a stronger model improves answer wording, not the retrieval or the numbers. Optional switches: `RERANK=0`, `VERIFY_CLAIMS=0` (skip the claim-check LLM pass), `LLM_PLANNER=0` (rule-based question parser instead of the LLM).

## Tests and evaluation

```bash
pytest                                        # 83 tests, offline, no data or network needed (~15 s)
python scripts/run_golden.py --no-llm         # 40-question golden set against the real index, no LLM
python scripts/run_golden.py --no-llm --check eval/golden_baseline.json   # fails if any layer regresses
python scripts/run_questions.py eval/questions.csv eval/results.csv       # questions -> answers (needs an LLM)
python scripts/judge_answers.py eval/results.csv eval/judged.csv          # LLM-as-judge on those answers
```

The golden set checks each layer separately (decline reason, filters, aggregate, the number recomputed independently with pandas, evidence validity). It is small and written by the developer, so treat it as a regression guard, not proof of quality. See [`EVAL_AND_MONITORING.md`](EVAL_AND_MONITORING.md).

## Environment and assumptions

- **Hardware:** built and measured on one desktop (6 cores, 16 GB RAM, no GPU needed for the service; the local LLM used a GPU).
- **Speed:** about 0.3-0.4 s per question without an LLM (BM25 ~7 ms, statistics 25-55 ms, rerank 230-390 ms); 5-6 s with a free hosted model.
- **Data:** only the training split is indexed (the first 70% of reviews by date, Aug 2015 to Jan 2017). The rest is held out for evaluation; a production build would index everything.
- **Cities:** the dataset covers six cities. Questions about others are declined.
- **Not included:** authentication, rate limiting and a persistent store for logs and metrics. Logs are JSON on stdout (the question text is never logged) and metrics are exposed for a Prometheus scraper.
- **First start:** the first run downloads the ranking model (~150 MB) once.

## Layout

```
main.py             build everything: download, clean and split, index
app/api.py          FastAPI app, error handling, metrics endpoint
app/service.py      the pipeline: plan, retrieve, rerank, statistics, gate, answer
app/planner.py      LLM question reader, with validation and a rule-based fallback
app/query_parser.py, router.py   rules for nationality, months, sentiment, which number to compute
app/bm25_index.py   chunking, index build, filtered search
app/stats.py        pandas statistics over all matching reviews
app/llm.py          LLM client, answer checks (citations, numbers)
app/verifier.py     claim check
app/observability.py  JSON logs and Prometheus metrics
scripts/, eval/     evaluation tools, golden set, sample questions
tests/              unit and API tests
docs/               architecture diagram (PDF, PNG, SVG) and its generator
```
