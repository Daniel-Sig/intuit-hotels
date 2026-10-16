"""FastAPI app. Run with:  uvicorn app.api:app"""
import hashlib
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.bm25_index import ReviewSearch
from app.data_prep import PROCESSED_DIR
from app.llm import Answerer, client_from_env
from app.observability import REQUEST_LATENCY, REQUESTS, request_ctx, setup_logging
from app.planner import Planner
from app.rerank import Reranker
from app.schemas import AskRequest, AskResponse, ErrorResponse
from app.service import AssistantService, UnknownFilterValue
from app.stats import StatsEngine

load_dotenv()
setup_logging()
log = logging.getLogger("app.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.service = None
    try:
        reranker = Reranker() if os.environ.get("RERANK", "1") != "0" else None
        llm = client_from_env()
        app.state.service = AssistantService(
            ReviewSearch(), reranker, StatsEngine.from_parquet(PROCESSED_DIR / "train.parquet"),
            Answerer(llm) if llm else None,
        )
        svc = app.state.service
        if llm and os.environ.get("LLM_PLANNER", "1") != "0":
            svc.planner = Planner(llm, svc.cities, svc.nationalities, svc.search.values("Hotel_Name"))
        warm = AssistantService(svc.search, reranker, svc.stats)  # no LLM call at startup
        warm.ask(AskRequest(question="friendly staff", top_k=3))  # warm caches/model before traffic
        log.info("service ready (rerank=%s, llm=%s)", reranker is not None, llm.model if llm else None)
    except FileNotFoundError:
        log.exception("BM25 index missing - build it first (see README)")
    yield


app = FastAPI(title="Review Intelligence Assistant", version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def observe(request: Request, call_next):
    """Request id, request count/latency metrics, and the X-Request-ID response header."""
    ctx = {"request_id": uuid.uuid4().hex[:12]}
    request_ctx.set(ctx)
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["X-Request-ID"] = ctx["request_id"]
        return response
    finally:
        route = request.scope.get("route")
        endpoint = route.path if route else "unmatched"  # path template, not raw URL: keeps label cardinality bounded
        REQUESTS.labels(endpoint, str(status)).inc()
        REQUEST_LATENCY.labels(endpoint).observe(time.perf_counter() - start)


@app.exception_handler(RequestValidationError)
async def invalid_request(request: Request, exc: RequestValidationError):
    """Same {"error", "detail"} body as every other error; detail names each bad field."""
    detail = "; ".join(f"{'.'.join(str(p) for p in e['loc'][1:]) or 'body'}: {e['msg']}" for e in exc.errors())
    return JSONResponse(status_code=422, content=ErrorResponse(error="invalid_request", detail=detail).model_dump())


@app.exception_handler(HTTPException)
async def http_error(request: Request, exc: HTTPException):
    code = {422: "unknown_filter_value", 503: "index_unavailable"}.get(exc.status_code, "http_error")
    return JSONResponse(status_code=exc.status_code, content=ErrorResponse(error=code, detail=str(exc.detail)).model_dump())


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    log.exception("unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content=ErrorResponse(error="internal_error").model_dump())


@app.get("/metrics", include_in_schema=False)
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
def health(request: Request):
    ready = request.app.state.service is not None
    return JSONResponse(status_code=200 if ready else 503, content={"status": "ok" if ready else "index_unavailable"})


@app.post(
    "/ask",
    response_model=AskResponse,
    responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
def ask(req: AskRequest, request: Request):
    service: AssistantService | None = request.app.state.service
    if service is None:
        raise HTTPException(status_code=503, detail="Search index is not available.")
    try:
        response = service.ask(req)
    except UnknownFilterValue as e:
        raise HTTPException(status_code=422, detail=str(e))
    log.info(
        "ask",
        extra={"fields": {
            # Question text is not logged (may contain personal data): length and a short hash are enough to correlate.
            "question_len": len(req.question),
            "question_hash": hashlib.sha1(req.question.encode()).hexdigest()[:10],
            "filters": response.filters_applied,
            "results": len(response.evidence),
            "insufficient": response.insufficient_evidence,
            "latency_ms": response.latency_ms,
            "timings_ms": request_ctx.get().get("timings_ms", {}),
        }},
    )
    return response
