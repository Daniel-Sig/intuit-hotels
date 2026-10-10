"""Aggregates over the full set of reviews matching the filters (not just the top-k evidence).

The operations form a closed menu. Anything that picks an operation, whether the rule-based
router or an LLM later, can only name one of these, so there is no free-form code execution.
"""
import re
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, field_validator

from app.bm25_index import MIN_WORDS, STEMMER
from app.schemas import OperationResult, Stats

GROUP_COLS = {
    "city": "city",
    "nationality": "Reviewer_Nationality",
    "month": "review_month",
    "trip_type": "trip_type",
    "traveler_group": "traveler_group",
}
MIN_GROUP_SIZE = 30  # smaller groups give noisy averages, so they are left out of rankings
TOP_GROUPS = 10
COLUMNS = [
    "city", "Reviewer_Nationality", "review_month", "trip_type", "traveler_group", "Hotel_Name",
    "Reviewer_Score", "Positive_Review", "Negative_Review", "review_date",
]
FILTER_COLUMNS = {"city", "Reviewer_Nationality", "review_month", "Hotel_Name"}


def mention_pattern(term: str) -> re.Pattern:
    """Regex matching words that start with the stem of each word in `term`.

    Same stemmer as the search index, so "noise" also finds "noisy" and "breakfast" finds
    "breakfasts". It can over-match (the stem of "friendly" is "friend", so "friends" counts).
    """
    stems = [re.escape(STEMMER.stemWord(w)) for w in re.findall(r"[a-z0-9']+", term.lower())]
    return re.compile(r"\b" + r"\W+".join(rf"{s}\w*" for s in stems), re.IGNORECASE)


class Operation(BaseModel):
    """One entry from the menu. Validated whether it came from rules or an LLM."""

    name: Literal["mean_score", "count", "mention_count"]
    group_by: Literal["city", "nationality", "month", "trip_type", "traveler_group"] | None = None
    term: str | None = None

    @field_validator("term")
    @classmethod
    def clean_term(cls, v: str | None) -> str | None:
        v = (v or "").strip()
        return v[:40] or None

    def model_post_init(self, _):
        if self.name == "mention_count" and not self.term:
            raise ValueError("mention_count needs a term")


class StatsEngine:
    def __init__(self, df: pd.DataFrame):
        self.df = df[COLUMNS].copy()
        self.df["has_positive"] = self.df["Positive_Review"].str.split().str.len() >= MIN_WORDS
        self.df["has_negative"] = self.df["Negative_Review"].str.split().str.len() >= MIN_WORDS
        self.scope = f"{df['review_date'].min().date()} to {df['review_date'].max().date()}"

    @classmethod
    def from_parquet(cls, path: Path) -> "StatsEngine":
        return cls(pd.read_parquet(path, columns=COLUMNS))

    def _subset(self, filters: dict) -> pd.DataFrame:
        mask = pd.Series(True, index=self.df.index)
        for col, value in filters.items():
            if col in FILTER_COLUMNS:  # `sentiment` only applies to text chunks, not whole reviews
                mask &= self.df[col].isin(value if isinstance(value, (list, tuple, set)) else [value])
        return self.df[mask]

    def compute(self, filters: dict, op: Operation | None = None) -> Stats:
        sub = self._subset(filters)
        stats = Stats(
            reviews_matched=len(sub),
            avg_reviewer_score=round(float(sub["Reviewer_Score"].mean()), 2) if len(sub) else None,
            with_positive_comment_rate=round(float(sub["has_positive"].mean()), 3) if len(sub) else None,
            with_negative_comment_rate=round(float(sub["has_negative"].mean()), 3) if len(sub) else None,
            data_scope=f"train split, reviews from {self.scope}",
        )
        if op is not None and len(sub):
            stats.operation = self._run(sub, op, filters.get("sentiment"))
        return stats

    def _run(self, sub: pd.DataFrame, op: Operation, sentiment: str | None) -> OperationResult:
        group_col = GROUP_COLS[op.group_by] if op.group_by else None
        if op.name == "mention_count":
            pattern = mention_pattern(op.term)
            cols = {"positive": ["Positive_Review"], "negative": ["Negative_Review"]}.get(
                sentiment, ["Positive_Review", "Negative_Review"]
            )
            sub = sub.assign(
                _hit=pd.concat([sub[c].str.contains(pattern) for c in cols], axis=1).any(axis=1)
            )

        if group_col is None:
            groups = [(None, sub)]
        else:
            groups = [(k, g) for k, g in sub.groupby(group_col, dropna=True)]

        rows = []
        for key, g in groups:
            row = {"group": None if key is None else (int(key) if group_col == "review_month" else str(key)), "reviews": len(g)}
            if op.name == "mean_score":
                row["value"] = round(float(g["Reviewer_Score"].mean()), 2)
            elif op.name == "mention_count":
                row["mentions"] = int(g["_hit"].sum())
                row["share"] = round(float(g["_hit"].mean()), 3)
            rows.append(row)

        if group_col is not None:
            if op.name != "count":
                rows = [r for r in rows if r["reviews"] >= MIN_GROUP_SIZE]
            sort_key = {"mean_score": "value", "count": "reviews", "mention_count": "mentions"}[op.name]
            rows = sorted(rows, key=lambda r: r[sort_key], reverse=True)[:TOP_GROUPS]
        return OperationResult(name=op.name, group_by=op.group_by, term=op.term, rows=rows)
