"""Score the answers in a results CSV with an LLM judge and add judge_* columns.

    python scripts/judge_answers.py eval/results_ollama.csv eval/judged.csv

Judge model: JUDGE_MODEL (falls back to LLM_MODEL), same LLM_BASE_URL / key as the app.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from app.judge import KEYS, judge_row
from app.llm import OpenAICompatClient


def main(src: str, dst: str) -> None:
    client = OpenAICompatClient(model=os.environ.get("JUDGE_MODEL") or os.environ.get("LLM_MODEL"))
    df = pd.read_csv(src).fillna("")
    rows = []
    for i, r in df.iterrows():
        rows.append(judge_row(client, r["question"], r["answer"], r.get("evidence_text", ""), r["operation"], r["message"]))
        print(f"[{i + 1}/{len(df)}] {rows[-1]['verdict']}: {r['question']}", flush=True)
    out = pd.concat([df, pd.DataFrame(rows).add_prefix("judge_")], axis=1)
    out.to_csv(dst, index=False)
    passed = sum(r["verdict"] == "pass" for r in rows)
    print(f"pass {passed}/{len(rows)}; wrote {dst}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
