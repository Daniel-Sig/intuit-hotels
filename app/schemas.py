"""API contract."""
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500, description="Natural-language question about guest reviews.")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of evidence reviews to return.")

    # Optional explicit filters. When given they override what is parsed from the question.
    city: str | None = None
    nationality: str | None = None
    months: list[int] | None = Field(default=None, description="Review months, 1-12.")
    sentiment: Literal["positive", "negative"] | None = None

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 3:
            raise ValueError("question must contain at least 3 non-space characters")
        return v

    @field_validator("months")
    @classmethod
    def valid_months(cls, v: list[int] | None) -> list[int] | None:
        if v is not None and (not v or any(m < 1 or m > 12 for m in v)):
            raise ValueError("months must be a non-empty list of integers between 1 and 12")
        return v


class Evidence(BaseModel):
    review_id: int
    text: str
    sentiment: Literal["positive", "negative"]
    hotel: str
    city: str
    reviewer_nationality: str
    review_month: int
    review_year: int
    reviewer_score: float
    bm25_score: float
    rerank_score: float | None = None


class Citation(BaseModel):
    review_id: int
    snippet: str = Field(description="First 10 words of the review, followed by '...'.")


class OperationResult(BaseModel):
    name: Literal["mean_score", "count", "mention_count"]
    group_by: str | None = None
    term: str | None = None
    rows: list[dict] = Field(description="One row per group (or one row if ungrouped); each has `reviews` plus the metric.")


class Stats(BaseModel):
    """Computed over every review matching the filters, not just the returned evidence."""

    reviews_matched: int
    avg_reviewer_score: float | None
    with_positive_comment_rate: float | None
    with_negative_comment_rate: float | None
    data_scope: str
    operation: OperationResult | None = None


class AskResponse(BaseModel):
    # Null until the LLM answer layer is plugged in; the evidence is what the answer will be grounded on.
    answer: str | None = None
    citations: list[Citation] = Field(default_factory=list, description="Reviews the answer cites, in citation order.")
    evidence: list[Evidence]
    stats: Stats | None = None
    filters_applied: dict
    insufficient_evidence: bool
    message: str | None = Field(default=None, description="Explains an empty or insufficient result.")
    latency_ms: float


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
