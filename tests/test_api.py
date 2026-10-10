"""API tests. The client is used without a `with` block, so the real-index lifespan never runs;
each test installs a service built on the tiny synthetic index instead."""
import pytest
from fastapi.testclient import TestClient

from app.api import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def ready(client, make_service):
    app.state.service = make_service()
    yield client
    app.state.service = None


def ask(client, **body):
    return client.post("/ask", json=body)


def test_ask_returns_filtered_evidence_and_stats(ready, monkeypatch):
    monkeypatch.setattr("app.service.MIN_EVIDENCE", 2)  # the tiny corpus has only two matching Paris chunks
    r = ask(ready, question="Is the staff friendly in Paris?")
    assert r.status_code == 200
    body = r.json()
    assert body["filters_applied"] == {"city": ["Paris"]}
    assert body["evidence"] and {e["city"] for e in body["evidence"]} == {"Paris"}
    assert body["stats"]["reviews_matched"] == 3
    assert body["insufficient_evidence"] is False
    assert r.headers["x-request-id"]


def test_explicit_filters_override_the_question(ready):
    body = ask(ready, question="noisy rooms in Paris", city="London", sentiment="negative").json()
    assert body["filters_applied"]["city"] == ["London"]
    assert {e["city"] for e in body["evidence"]} == {"London"}


def test_unsupported_city_is_flagged_not_answered(ready):
    body = ask(ready, question="Is spring a good time to visit Rome?").json()
    assert body["insufficient_evidence"] is True and body["evidence"] == []
    assert "Rome" in body["message"]


def test_low_relevance_is_flagged(client, make_service):
    app.state.service = make_service(rerank_score=-8.0)
    body = ask(client, question="quantum computing breakthroughs").json()
    assert body["insufficient_evidence"] is True and body["evidence"] == [] and body["stats"] is None


def test_open_ended_question_skips_relevance_check(client, make_service):
    app.state.service = make_service(rerank_score=-8.0)
    body = ask(client, question="What do reviewers from Australia say about hotels in Amsterdam?").json()
    # Whatever retrieval finds, it must not be rejected as "irrelevant": there is no topic to be irrelevant to.
    assert "relevant" not in (body["message"] or "")
    assert body["filters_applied"] == {"city": ["Amsterdam"], "Reviewer_Nationality": ["Australia"]}
    assert body["stats"]["reviews_matched"] == 1


def test_unmatched_topic_gets_no_whole_pool_stats(ready):
    assert ask(ready, question="quantum computing breakthroughs").json()["stats"] is None
    # ...but a question that asks for a number still gets it, even with no matching text.
    body = ask(ready, question="How many reviews are from Australia?").json()
    assert body["stats"]["operation"]["rows"][0]["reviews"] == 3


def test_top_k_below_minimum_is_not_insufficient(ready):
    body = ask(ready, question="staff", top_k=1).json()
    assert len(body["evidence"]) == 1 and body["insufficient_evidence"] is False


@pytest.mark.parametrize("body", [
    {"question": "x"},
    {"question": "ok question", "top_k": 500},
    {"question": "ok question", "months": [13]},
    {"question": "ok question", "sentiment": "neutral"},
    {},
])
def test_invalid_input_is_rejected(ready, body):
    assert ask(ready, **body).status_code == 422


def test_unknown_explicit_city_is_422(ready):
    r = ask(ready, question="breakfast", city="Atlantis")
    assert r.status_code == 422 and "Atlantis" in r.json()["detail"]


def test_service_unavailable_returns_503(client):
    app.state.service = None
    assert ask(client, question="breakfast please").status_code == 503
    assert client.get("/health").status_code == 503


def test_health_and_metrics(ready):
    assert ready.get("/health").json() == {"status": "ok"}
    ask(ready, question="breakfast in Amsterdam")
    text = ready.get("/metrics").text
    for name in ("requests_total", "request_latency_seconds", "stage_latency_seconds", "insufficient_evidence_total"):
        assert name in text
