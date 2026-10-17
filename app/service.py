"""Question -> filters -> BM25 candidates -> (optional) rerank -> evidence."""
import logging
import os
import re
import time

import pandas as pd

from app.bm25_index import ReviewSearch
from app.llm import Answerer
from app.observability import INSUFFICIENT, LLM_ERRORS, LLM_FALLBACK, STATS_FAILURES, stage
from app.planner import Plan, Planner
from app.query_parser import QueryParser
from app.rerank import Reranker
from app.router import route
from app.schemas import AskRequest, AskResponse, Evidence, Stats
from app.stats import StatsEngine

log = logging.getLogger(__name__)

CANDIDATES = 50  # BM25 candidates handed to the reranker
MIN_EVIDENCE = 3  # fewer than this and we say there isn't enough to answer
# Cross-encoder score below which even the best candidate is treated as off-topic. Provisional: set from a
# small hand-made set of sensible vs nonsense questions (see README), to be re-fit on a golden set from val.
RELEVANCE_MIN = float(os.environ.get("RELEVANCE_MIN", "-1.0"))


class UnknownFilterValue(ValueError):
    """An explicit filter value that does not exist in the dataset."""


class AssistantService:
    def __init__(self, search: ReviewSearch, reranker: Reranker | None = None, stats: StatsEngine | None = None,
                 answerer: Answerer | None = None, planner: Planner | None = None):
        self.answerer = answerer
        self.planner = planner
        self.search = search
        self.reranker = reranker
        self.stats = stats
        self.cities = search.values("city")
        self.nationalities = search.values("Reviewer_Nationality")
        self.parser = QueryParser(self.cities, self.nationalities, search.values("Hotel_Name"))

    def _filters(self, req: AskRequest, parsed_filters: dict) -> dict:
        """Parsed filters, overridden by anything the caller set explicitly."""
        filters = dict(parsed_filters)
        if req.city:
            if req.city not in self.cities:
                raise UnknownFilterValue(f"city {req.city!r} not in dataset; available: {self.cities}")
            filters["city"] = [req.city]
        if req.nationality:
            if req.nationality not in self.nationalities:
                raise UnknownFilterValue(f"nationality {req.nationality!r} not in dataset")
            filters["Reviewer_Nationality"] = [req.nationality]
        if req.months:
            filters["review_month"] = sorted(set(req.months))
        if req.sentiment:
            filters["sentiment"] = req.sentiment
        return filters

    def ask(self, req: AskRequest) -> AskResponse:
        start = time.perf_counter()
        with stage("parse"):
            plan = self._plan(req.question)
            parsed = plan.parsed
            filters = self._filters(req, parsed.filters)

        if not plan.in_scope:
            INSUFFICIENT.labels("off_topic").inc()
            return self._response([], filters, start, "This doesn't look like a question about hotel guest reviews.",
                                  reason="off_topic")

        if parsed.unsupported and "city" not in filters:
            INSUFFICIENT.labels("unsupported_city").inc()
            return self._response(
                [], filters, start,
                f"No reviews for {', '.join(parsed.unsupported)}; the dataset covers {', '.join(self.cities)}.",
                reason="unsupported_city",
            )

        search_kwargs = dict(filters)
        city = search_kwargs.pop("city", None)
        sentiment = search_kwargs.pop("sentiment", None)
        # Search on the topic alone when there is one; an open-ended question ("what do X say about Y")
        # has none, so it falls back to the full leftover text and is not relevance-checked.
        search_query = parsed.topic or parsed.query
        # With no side requested, search positive and negative text separately. Otherwise one polarity can crowd
        # out the other (searching "friendly" mostly returns "staff were not friendly") and the answer sees one side.
        balanced = sentiment is None
        with stage("bm25"):
            if balanced:
                candidates = pd.concat(
                    [self.search.search(search_query, k=CANDIDATES // 2, city=city, sentiment=s, **search_kwargs)
                     for s in ("positive", "negative")], ignore_index=True)
            else:
                candidates = self.search.search(search_query, k=CANDIDATES, city=city, sentiment=sentiment, **search_kwargs)

        if self.reranker is not None and not candidates.empty:
            with stage("rerank"):
                ranked = self.reranker.rerank(search_query, candidates, top_n=len(candidates))
            top = self._pick(ranked, req.top_k, balanced)
            # Not applied when a number was asked for: that answer comes from the stats, and the evidence is context.
            if parsed.topic and plan.operation is None and ranked["rerank_score"].max() < RELEVANCE_MIN:
                INSUFFICIENT.labels("low_relevance").inc()
                return self._response(
                    [], filters, start,
                    "The reviews found don't appear relevant to this question; "
                    "it may not be about hotel guest feedback.",
                    reason="low_relevance",
                )
        else:
            top = self._pick(candidates.sort_values("score", ascending=False).reset_index(drop=True), req.top_k, balanced)                 if not candidates.empty else candidates

        evidence = [self._evidence(row) for row in top.itertuples(index=False)]
        operation = plan.operation
        # A topic that matched nothing and asked for no number would only get whole-pool stats: misleading.
        stats = None if (candidates.empty and parsed.topic and operation is None) else self._stats(filters, operation, parsed.topic, any_of=bool(re.search(r"/|\bor\b", req.question)))
        message = None
        # Judged on everything that matched, not on what top_k happened to return.
        insufficient = len(candidates) < MIN_EVIDENCE
        if insufficient:
            INSUFFICIENT.labels("too_few_results").inc()
            message = f"Only {len(candidates)} matching review(s) found; not enough to answer reliably."
        response = self._response(evidence, filters, start, message, stats, insufficient,
                                  "too_few_results" if insufficient else None)
        if self.answerer is not None and not insufficient:
            if result := self.answerer.answer(req.question, evidence, stats):
                response.answer, response.citations = result.text, result.citations
                response.unsupported_claims_removed = result.removed_claims
            response.latency_ms = round((time.perf_counter() - start) * 1000, 1)
        return response

    @staticmethod
    def _pick(ranked: pd.DataFrame, k: int, balanced: bool) -> pd.DataFrame:
        """Top `k` of an already best-first `ranked` frame. When `balanced`, up to half come from each polarity
        (the other side tops up if one runs short), then the result is put back in score order."""
        if not balanced:
            return ranked.head(k)
        half = -(-k // 2)
        chosen = pd.concat([ranked[ranked["sentiment"] == s].head(half) for s in ("positive", "negative")])
        if len(chosen) < k:
            chosen = pd.concat([chosen, ranked.drop(chosen.index).head(k - len(chosen))])
        by = "rerank_score" if "rerank_score" in chosen else "score"
        return chosen.sort_values(by, ascending=False).reset_index(drop=True)

    def _plan(self, question: str) -> Plan:
        """LLM plan when configured; the rule-based parser/router on any failure or when there is no LLM."""
        if self.planner is not None:
            try:
                return self.planner.plan(question)
            except Exception:
                LLM_ERRORS.inc()
                LLM_FALLBACK.inc()
                log.exception("llm planner failed; using rule-based parser")
        return Plan(self.parser.parse(question), route(question))

    def _stats(self, filters: dict, operation, topic: str = "", any_of: bool = False) -> Stats | None:
        """Aggregates must never fail the request: on error, log and answer without them."""
        if self.stats is None:
            return None
        try:
            with stage("stats"):
                return self.stats.compute(filters, operation, topic, any_of)
        except Exception:
            STATS_FAILURES.inc()
            log.exception("stats computation failed")
            return None

    @staticmethod
    def _evidence(row) -> Evidence:
        return Evidence(
            review_id=int(row.review_id), text=row.text, sentiment=row.sentiment, hotel=row.Hotel_Name,
            city=row.city, reviewer_nationality=row.Reviewer_Nationality, review_month=int(row.review_month),
            review_year=int(row.review_year), reviewer_score=float(row.Reviewer_Score),
            bm25_score=float(row.score),
            rerank_score=float(row.rerank_score) if hasattr(row, "rerank_score") else None,
        )

    @staticmethod
    def _response(
        evidence: list[Evidence], filters: dict, start: float, message: str | None, stats: Stats | None = None,
        insufficient: bool = True, reason: str | None = None,
    ) -> AskResponse:
        return AskResponse(
            evidence=evidence,
            stats=stats,
            filters_applied=filters,
            insufficient_evidence=insufficient,
            reason=reason,
            message=message,
            latency_ms=round((time.perf_counter() - start) * 1000, 1),
        )
