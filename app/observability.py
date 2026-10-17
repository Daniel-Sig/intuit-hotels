"""Metrics (Prometheus), structured JSON logging, and per-request context."""
import contextvars
import json
import logging
import time
from contextlib import contextmanager

from prometheus_client import Counter, Histogram

# 5 ms .. 5 s: spans a fast BM25 lookup up to a slow rerank/LLM call
BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0)

REQUESTS = Counter("requests_total", "HTTP requests", ["endpoint", "status"])
REQUEST_LATENCY = Histogram("request_latency_seconds", "End-to-end request latency", ["endpoint"], buckets=BUCKETS)
STAGE_LATENCY = Histogram("stage_latency_seconds", "Latency of one pipeline stage", ["stage"], buckets=BUCKETS)
INSUFFICIENT = Counter("insufficient_evidence_total", "Answers flagged as insufficient evidence", ["reason"])
STATS_FAILURES = Counter("stats_failures_total", "Aggregate computations that failed and were skipped")
LLM_LATENCY = Histogram("llm_latency_seconds", "LLM answer call latency", buckets=(0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0))
LLM_TOKENS = Counter("llm_tokens_total", "LLM tokens used", ["kind"])
LLM_ERRORS = Counter("llm_errors_total", "LLM calls that failed or returned unusable output")
CITATION_INVALID = Counter("citation_invalid_total", "LLM answers dropped for citing reviews not in the evidence")
UNGROUNDED_NUMBER = Counter("ungrounded_number_total", "LLM answers dropped for stating a number not in the stats or evidence")
CLAIMS_REMOVED = Counter("claims_removed_total", "Answer sentences dropped by the claim check as unsupported")
VERIFY_ERRORS = Counter("verify_errors_total", "Claim checks that failed; the answer was kept")
LLM_FALLBACK = Counter("llm_fallback_total", "Questions planned by the rule-based parser because the LLM planner failed")
FILTER_REJECTED = Counter("filter_rejected_total", "LLM-proposed filter values dropped for not existing in the data", ["field"])

# Shared with the worker thread FastAPI runs sync endpoints in; the dict is mutated, not replaced.
request_ctx: contextvars.ContextVar[dict] = contextvars.ContextVar("request_ctx", default={})


@contextmanager
def stage(name: str):
    """Time one pipeline stage into the histogram and the current request's timing log."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        STAGE_LATENCY.labels(name).observe(elapsed)
        request_ctx.get().setdefault("timings_ms", {})[name] = round(elapsed * 1000, 1)


class JsonFormatter(logging.Formatter):
    """One JSON object per line; request id and any `extra={"fields": {...}}` are merged in."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if request_id := request_ctx.get().get("request_id"):
            entry["request_id"] = request_id
        entry.update(getattr(record, "fields", {}))
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


def setup_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    for noisy in ("httpx", "httpcore", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
