"""Run questions from a CSV through the real pipeline and write the answers to a new column.

    GEMINI_API_KEY=... python scripts/run_questions.py eval/questions.csv eval/results.csv

Waits between questions so the free tier's requests-per-minute cap is not hit (planner + answer = 2 calls each).
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from app.bm25_index import ReviewSearch
from app.data_prep import PROCESSED_DIR
from app.llm import Answerer, client_from_env
from app.observability import LLM_ERRORS, LLM_FALLBACK, CITATION_INVALID
from app.planner import Planner
from app.rerank import Reranker
from app.schemas import AskRequest
from app.service import AssistantService
from app.stats import StatsEngine

PAUSE_SECONDS = int(os.environ.get("PAUSE_SECONDS", "60"))  # 0 for a local model


def counter(c) -> float:
    return c._value.get()


def main(src: str, dst: str) -> None:
    llm = client_from_env()
    if llm is None:
        sys.exit("No LLM configured: set LLM_API_KEY/GEMINI_API_KEY or LLM_BASE_URL")
    search = ReviewSearch()
    service = AssistantService(search, Reranker(), StatsEngine.from_parquet(PROCESSED_DIR / "train.parquet"), Answerer(llm))
    service.planner = Planner(llm, service.cities, service.nationalities, search.values("Hotel_Name"))

    df = pd.read_csv(src)
    rows = []
    for i, q in enumerate(df["question"]):
        if i:
            time.sleep(PAUSE_SECONDS)
        before = (counter(LLM_FALLBACK), counter(LLM_ERRORS), counter(CITATION_INVALID))
        resp = service.ask(AskRequest(question=q))
        fallback, errors, bad_cite = (counter(c) - b for c, b in zip((LLM_FALLBACK, LLM_ERRORS, CITATION_INVALID), before))
        status = "planner_fallback" if fallback else "ok"
        if resp.answer is None and not resp.insufficient_evidence:
            status += "; answer_failed" if (errors - fallback) or bad_cite else "; no_answer"
        rows.append({
            "answer": resp.answer or "",
            "citations": " | ".join(f"{c.review_id}: {c.snippet}" for c in resp.citations),
            "status": status,
            "insufficient_evidence": resp.insufficient_evidence,
            "message": resp.message or "",
            "filters_applied": json.dumps(resp.filters_applied),
            "operation": json.dumps(resp.stats.operation.model_dump()) if resp.stats and resp.stats.operation else "",
            "evidence_ids": ",".join(str(e.review_id) for e in resp.evidence),
            "evidence_text": "\n".join(f"[{e.review_id}] ({e.sentiment}) {e.text}"[:700] for e in resp.evidence),
            "latency_s": round(resp.latency_ms / 1000, 1),
        })
        print(f"[{i + 1}/{len(df)}] {status}: {q}", flush=True)
        pd.concat([df.iloc[: len(rows)].reset_index(drop=True), pd.DataFrame(rows)], axis=1).to_csv(dst, index=False)


if __name__ == "__main__":
    main(*sys.argv[1:3])
