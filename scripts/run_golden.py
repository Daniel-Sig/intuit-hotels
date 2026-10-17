"""Run the golden set and report pass rates per layer.

    python scripts/run_golden.py --no-llm                     # rules only: fast, deterministic, for CI
    LLM_BASE_URL=... LLM_MODEL=... python scripts/run_golden.py   # LLM planner on
    python scripts/run_golden.py --answers                    # also generate answers (slow; checks citations exist)
    python scripts/run_golden.py --no-llm --check eval/golden_baseline.json     # exit 1 on a regression
    python scripts/run_golden.py --no-llm --save-baseline eval/golden_baseline.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from app.bm25_index import ReviewSearch
from app.data_prep import PROCESSED_DIR
from app.golden import LAYERS, check_case, load_cases, regressions, summarize
from app.llm import Answerer, client_from_env
from app.planner import Planner
from app.rerank import Reranker
from app.schemas import AskRequest
from app.service import AssistantService
from app.stats import StatsEngine
from app.verifier import ClaimVerifier

TRUTH_COLUMNS = ["city", "Reviewer_Nationality", "review_month", "trip_type", "traveler_group", "Reviewer_Score",
                 "Positive_Review", "Negative_Review"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="eval/golden.csv")
    ap.add_argument("--no-llm", action="store_true", help="rule-based planner only; skips cases that need an LLM")
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--answers", action="store_true", help="also generate answers (needs an LLM)")
    ap.add_argument("--report", default="eval/golden_report.json")
    ap.add_argument("--check", help="baseline JSON; exit 1 if a gated layer regresses")
    ap.add_argument("--save-baseline", help="write this run's pass rates as the baseline")
    args = ap.parse_args()

    llm = None if args.no_llm else client_from_env()
    if llm is None and not args.no_llm:
        print("no LLM configured; running rules only (same as --no-llm)", file=sys.stderr)
    if args.answers and llm is None:
        sys.exit("--answers needs an LLM")

    train = PROCESSED_DIR / "train.parquet"
    search = ReviewSearch()
    service = AssistantService(
        search, None if args.no_rerank else Reranker(), StatsEngine.from_parquet(train),
        Answerer(llm, ClaimVerifier(llm)) if (llm and args.answers) else None,
    )
    if llm:
        service.planner = Planner(llm, service.cities, service.nationalities, search.values("Hotel_Name"))
    truth_df = pd.read_parquet(train, columns=TRUTH_COLUMNS)

    cases = [c for c in load_cases(args.cases) if llm or not c.requires_llm]
    results = []
    for case in cases:
        resp = service.ask(AskRequest(question=case.question))
        checks = check_case(case, resp, truth_df, service.cities, answers=args.answers)
        results.append({
            "id": case.id, "category": case.category, "question": case.question, "checks": checks,
            "latency_ms": resp.latency_ms,
            "got": {"reason": resp.reason, "filters": resp.filters_applied,
                    "operation": resp.stats.operation.model_dump(exclude={"rows"}) if resp.stats and resp.stats.operation else None},
            "expected": {"decline": case.expect_decline, "filters": case.expect_filters, "operation": case.expect_operation},
        })
        failed = [k for k, v in checks.items() if v is False]
        print(f"{case.id} {'FAIL ' + ','.join(failed) if failed else 'ok'}: {case.question}", flush=True)

    summary = summarize(results)
    print(f"\n{len(cases)} cases ({'LLM planner' if llm else 'rules only'})")
    for layer in LAYERS:
        s = summary[layer]
        if s["total"]:
            print(f"  {layer:10s} {s['passed']:>3}/{s['total']:<3} {s['rate']:.0%}")
    Path(args.report).write_text(json.dumps({"summary": summary, "cases": results}, indent=2, default=str), encoding="utf-8")

    if args.save_baseline:
        Path(args.save_baseline).write_text(json.dumps({k: v["rate"] for k, v in summary.items()}, indent=2), encoding="utf-8")
    if args.check:
        bad = regressions(summary, json.loads(Path(args.check).read_text(encoding="utf-8")))
        for line in bad:
            print("REGRESSION", line)
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
