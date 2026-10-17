"""Golden-set helpers on synthetic data, plus a rules-only regression run against the real index when present."""
import json
from pathlib import Path

import pandas as pd
import pytest

from app.data_prep import PROCESSED_DIR
from app.golden import Case, load_cases, normalize_filters, regressions, summarize, truth

ROOT = Path(__file__).resolve().parent.parent


def test_normalize_filters_ignores_order_and_a_filter_that_covers_every_city():
    cities = ["A", "B"]
    assert normalize_filters({"city": ["B", "A"], "review_month": [3, 1]}, cities) == {"review_month": [1, 3]}
    assert normalize_filters({"city": ["A"], "extra": 1}, cities) == {"city": ["A"]}


def test_truth_is_computed_directly_from_the_data(clean_df):
    assert truth(clean_df, {"metric": "count", "filters": {"city": ["Paris"]}}) == {None: 3.0}
    assert truth(clean_df, {"metric": "mention_count", "prefix": "friend", "filters": {"city": ["Paris"]}}) == {None: 2.0}
    mean = truth(clean_df, {"metric": "mean_score", "filters": {"city": ["Paris"]}})[None]
    assert mean == round(float(clean_df[clean_df.city == "Paris"]["Reviewer_Score"].mean()), 2)


def test_regressions_flags_only_drops_beyond_tolerance():
    summary = {k: {"rate": 1.0} for k in ("decline", "filters", "operation", "numeric", "evidence")}
    summary["filters"]["rate"] = 0.9
    assert regressions(summary, {"filters": 1.0, "decline": 1.0}) == ["filters: 0.900 < baseline 1.000"]
    assert regressions(summary, {"filters": 0.91}) == []  # within the 0.02 tolerance


def test_summarize_counts_only_applicable_layers():
    results = [{"checks": {"decline": True, "filters": False, "operation": None, "numeric": None,
                           "evidence": None, "balanced": None, "answered": None}}]
    s = summarize(results)
    assert s["filters"] == {"passed": 0, "total": 1, "rate": 0.0} and s["operation"]["total"] == 0


def test_golden_file_is_well_formed():
    cases = load_cases(ROOT / "eval" / "golden.csv")
    assert len(cases) >= 30 and len({c.id for c in cases}) == len(cases)
    assert any(c.expect_decline for c in cases) and any(c.numeric for c in cases)
    assert all(isinstance(c, Case) for c in cases)


@pytest.mark.skipif(
    not (PROCESSED_DIR / "train.parquet").exists() or not (ROOT / "data" / "bm25").exists(),
    reason="needs the real processed data and BM25 index (see README)",
)
def test_rules_only_golden_run_has_not_regressed():
    """Deterministic layers on the real index, no LLM and no reranker. Fails if a layer drops below the baseline."""
    from app.bm25_index import ReviewSearch
    from app.golden import check_case
    from app.schemas import AskRequest
    from app.service import AssistantService
    from app.stats import StatsEngine

    train = PROCESSED_DIR / "train.parquet"
    service = AssistantService(ReviewSearch(), None, StatsEngine.from_parquet(train))
    truth_df = pd.read_parquet(train)
    results = []
    for case in (c for c in load_cases(ROOT / "eval" / "golden.csv") if not c.requires_llm):
        resp = service.ask(AskRequest(question=case.question))
        results.append({"checks": check_case(case, resp, truth_df, service.cities, answers=False)})
    baseline = json.loads((ROOT / "eval" / "golden_baseline.json").read_text(encoding="utf-8"))
    assert regressions(summarize(results), baseline) == []
