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
from app.schemas import OperationResult, Stats, TopicBreakdown

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
    def stem(word: str) -> str:
        s = STEMMER.stemWord(word)
        # The stemmer turns a final "y" into "i" ("noisy" -> "noisi"), which would miss "noise": drop that letter.
        # A word that really ends in "i" ("wifi") keeps it, or "wif" would also match "wife".
        return s.removesuffix("i") if word.endswith("y") and s.endswith("i") else s

    stems = [re.escape(stem(w)) for w in re.findall(r"[a-z0-9']+", term.lower())]
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


def _summary(op: Operation, rows: list[dict]) -> str:
    """One plain sentence per result, best group first, so a model (or a person) does not have to read the rows."""

    def line(r: dict) -> str:
        name = "all" if r["group"] is None else r["group"]
        if op.name == "mean_score":
            return f"{r['value']} ({r['reviews']:,} reviews)" if r["group"] is None else f"{name} {r['value']} ({r['reviews']:,} reviews)"
        if op.name == "count":
            return f"{r['reviews']:,} reviews" if r["group"] is None else f"{name} {r['reviews']:,} reviews"
        counted = f"{r['mentions']:,} of {r['reviews']:,} reviews ({r['share']:.1%})"
        return counted if r["group"] is None else f"{name}: {counted}"

    what = {"mean_score": "Average reviewer score", "count": "Review count",
            "mention_count": f"Reviews mentioning '{op.term}'"}[op.name]
    if not rows:
        return f"{what}: no group has enough reviews."
    if op.group_by is None:
        return f"{what}: {line(rows[0])}."
    order = "highest first" if op.name == "mean_score" else "most first"
    return f"{what} by {op.group_by}, {order}: " + "; ".join(line(r) for r in rows) + "."


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

    def _topic_breakdown(self, sub: pd.DataFrame, topic: str, any_of: bool = False) -> TopicBreakdown | None:
        """Mentions of `topic` in the positive vs negative comment.

        By default every topic word must appear (any order): "friendly staff". With `any_of`, one is enough, for
        alternatives such as "crowded/noisy".
        """
        words = re.findall(r"[a-z0-9']+", topic.lower())[:4]
        if not words or sub.empty:
            return None
        patterns = [mention_pattern(w) for w in words]

        def hits(col: str) -> pd.Series:
            out = pd.Series(not any_of, index=sub.index)
            for p in patterns:
                out = (out | sub[col].str.contains(p)) if any_of else (out & sub[col].str.contains(p))
            return out

        pos, neg = hits("Positive_Review"), hits("Negative_Review")
        n_pos, n_neg = int(pos.sum()), int(neg.sum())
        n_any, term = int((pos | neg).sum()), (" or " if any_of else " ").join(words)
        pos_share = round(n_pos / (n_pos + n_neg), 3) if n_pos + n_neg else None
        neg_share = round(1 - pos_share, 3) if pos_share is not None else None
        summary = (
            f"{n_any:,} of {len(sub):,} reviews ({n_any / len(sub):.1%}) mention '{term}'. Of the mentions, "
            f"{n_pos:,} ({pos_share:.1%}) are in the guest's POSITIVE comment (what they liked) and "
            f"{n_neg:,} ({neg_share:.1%}) in the NEGATIVE comment (what they disliked)."
            if pos_share is not None else f"No reviews mention '{term}'."
        )
        return TopicBreakdown(
            term=term, reviews_in_scope=len(sub), reviews_mentioning=n_any, mention_rate=round(n_any / len(sub), 3),
            mentions_in_positive_comment=n_pos, mentions_in_negative_comment=n_neg,
            positive_share_of_mentions=pos_share, negative_share_of_mentions=neg_share, summary=summary,
        )

    def compute(self, filters: dict, op: Operation | None = None, topic: str = "", any_of: bool = False) -> Stats:
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
        elif op is None and topic:
            stats.topic_breakdown = self._topic_breakdown(sub, topic, any_of)
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
        return OperationResult(name=op.name, group_by=op.group_by, term=op.term, rows=rows, summary=_summary(op, rows))
