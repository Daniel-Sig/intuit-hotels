"""Golden-set evaluation: labelled questions checked layer by layer, with no LLM judging.

Each case says what the pipeline should do (decline or not, which filters, which aggregate, which number). The
number is recomputed here with plain pandas, independently of `StatsEngine`, so a bug in the stats code shows up.
Layers are scored separately so a failure points at its cause. Cases only an LLM can pass are marked
`requires_llm` and skipped when the run has no LLM.
"""
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from app.schemas import AskResponse

LAYERS = ("decline", "filters", "operation", "numeric", "evidence", "balanced", "answered")
GATED = ("decline", "filters", "operation", "numeric", "evidence")  # "balanced"/"answered" are reported only
FILTER_KEYS = ("city", "Reviewer_Nationality", "review_month", "Hotel_Name", "sentiment")
GROUP_COLS = {"city": "city", "nationality": "Reviewer_Nationality", "month": "review_month",
              "trip_type": "trip_type", "traveler_group": "traveler_group"}
EVIDENCE_FIELD = {"city": "city", "Reviewer_Nationality": "reviewer_nationality", "review_month": "review_month",
                  "Hotel_Name": "hotel", "sentiment": "sentiment"}


@dataclass
class Case:
    id: str
    question: str
    category: str
    expect_decline: str | None = None
    expect_filters: dict = field(default_factory=dict)
    expect_operation: dict | None = None
    numeric: dict | None = None
    requires_llm: bool = False


def load_cases(path: str | Path) -> list[Case]:
    cases = []
    for row in pd.read_csv(path, dtype=str, keep_default_na=False).to_dict("records"):
        cases.append(Case(
            id=row["id"], question=row["question"], category=row["category"],
            expect_decline=row["expect_decline"] or None,
            expect_filters=json.loads(row["expect_filters"] or "{}"),
            expect_operation=json.loads(row["expect_operation"]) if row["expect_operation"] else None,
            numeric=json.loads(row["numeric"]) if row["numeric"] else None,
            requires_llm=row["requires_llm"] == "1",
        ))
    return cases


def normalize_filters(filters: dict, all_cities: list[str]) -> dict:
    """Comparable form: only the checked keys, sorted lists, and "every city" counts as no city filter."""
    out = {}
    for key in FILTER_KEYS:
        value = filters.get(key)
        if value in (None, [], ""):
            continue
        if isinstance(value, list):
            value = sorted(value)
            if key == "city" and value == sorted(all_cities):
                continue
        out[key] = value
    return out


def truth(df: pd.DataFrame, spec: dict) -> dict:
    """Ground truth {group: value} (group None when ungrouped) computed directly from the data."""
    f = spec.get("filters", {})
    sub = df
    for key, col in (("city", "city"), ("nationality", "Reviewer_Nationality"), ("months", "review_month")):
        if key in f:
            sub = sub[sub[col].isin(f[key])]
    metric = spec["metric"]

    def value(g: pd.DataFrame) -> float:
        if metric == "mean_score":
            return round(float(g["Reviewer_Score"].mean()), 2)
        if metric == "count":
            return float(len(g))
        pattern = re.compile(r"\b" + re.escape(spec["prefix"]), re.IGNORECASE)  # words starting with `prefix`
        return float((g["Positive_Review"].str.contains(pattern) | g["Negative_Review"].str.contains(pattern)).sum())

    if spec.get("group_by"):
        col = GROUP_COLS[spec["group_by"]]
        return {(int(k) if col == "review_month" else str(k)): value(g) for k, g in sub.groupby(col) if len(g) >= 30}
    return {None: value(sub)}


def _close(metric: str, got: float, want: float) -> bool:
    if metric == "mean_score":
        return abs(got - want) <= 0.011
    if metric == "count":
        return got == want
    return abs(got - want) <= max(5, 0.03 * want)  # mention counts: stem matching vs plain regex differ slightly


def _numeric_ok(resp: AskResponse, spec: dict, df: pd.DataFrame) -> bool:
    op = resp.stats.operation if resp.stats else None
    if op is None:
        return False
    metric = spec["metric"]
    want = truth(df, spec)
    key = {"mean_score": "value", "count": "reviews", "mention_count": "mentions"}[metric]
    rows = {r["group"]: r[key] for r in op.rows}
    if not rows:
        return False
    if not spec.get("group_by"):
        return _close(metric, rows.get(None, float("nan")), want[None])
    return all(g in want and _close(metric, v, want[g]) for g, v in rows.items())


def _operation_ok(resp: AskResponse, expected: dict | None) -> bool:
    op = resp.stats.operation if resp.stats else None
    if expected is None:
        return op is None
    return (op is not None and op.name == expected["name"] and op.group_by == expected.get("group_by")
            and (expected.get("term") is None or (op.term or "").lower() == expected["term"].lower()))


def _evidence_ok(resp: AskResponse) -> bool:
    for e in resp.evidence:
        for key, value in resp.filters_applied.items():
            if key in EVIDENCE_FIELD:
                allowed = value if isinstance(value, list) else [value]
                if getattr(e, EVIDENCE_FIELD[key]) not in allowed:
                    return False
    return True


def check_case(case: Case, resp: AskResponse, df: pd.DataFrame, all_cities: list[str], answers: bool) -> dict:
    """{layer: True/False/None}; None means the layer does not apply to this case."""
    out = dict.fromkeys(LAYERS)
    out["decline"] = resp.reason == case.expect_decline
    if case.expect_decline is None:
        out["filters"] = normalize_filters(resp.filters_applied, all_cities) == normalize_filters(case.expect_filters, all_cities)
        out["operation"] = _operation_ok(resp, case.expect_operation)
        if case.numeric:
            out["numeric"] = _numeric_ok(resp, case.numeric, df)
        if resp.evidence:
            out["evidence"] = _evidence_ok(resp)
            if "sentiment" not in resp.filters_applied and len(resp.evidence) >= 2:
                out["balanced"] = len({e.sentiment for e in resp.evidence}) == 2
        if answers and not resp.insufficient_evidence:
            out["answered"] = bool(resp.answer) and bool(resp.citations)
    return out


def summarize(results: list[dict]) -> dict:
    """Per-layer pass rate over the cases the layer applied to."""
    summary = {}
    for layer in LAYERS:
        scored = [r["checks"][layer] for r in results if r["checks"][layer] is not None]
        summary[layer] = {"passed": sum(scored), "total": len(scored), "rate": round(sum(scored) / len(scored), 3) if scored else None}
    return summary


def regressions(summary: dict, baseline: dict, tolerance: float = 0.02) -> list[str]:
    """Gated layers whose pass rate fell more than `tolerance` below the baseline."""
    bad = []
    for layer in GATED:
        old, new = baseline.get(layer), summary[layer]["rate"]
        if old is not None and new is not None and new < old - tolerance:
            bad.append(f"{layer}: {new:.3f} < baseline {old:.3f}")
    return bad
