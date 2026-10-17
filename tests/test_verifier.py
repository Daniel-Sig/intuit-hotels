"""Claim-level check with fake LLMs: unsupported sentences are removed, failures keep the answer."""
import json
import re

from app.llm import Answerer
from app.schemas import AskRequest
from app.verifier import ClaimVerifier, split_claims
from tests.test_llm import FakeLLM


def _answer_two_sentences(user: str) -> str:
    ids = [int(i) for i in re.findall(r"review_id=(\d+)", user)]
    return json.dumps({"answer": f"Staff are friendly [{ids[0]}]. Most guests also love the pool [{ids[1]}].",
                       "cited_review_ids": ids[:2]})


def _service(make_service, verdicts):
    service = make_service()
    service.answerer = Answerer(FakeLLM(_answer_two_sentences), ClaimVerifier(FakeLLM(verdicts)))
    return service


def _v(*labels):
    return json.dumps({"verdicts": [{"claim": i, "verdict": v} for i, v in enumerate(labels, 1)]})


def test_split_claims():
    assert split_claims("One thing [123]. Another [456, 789]! Third?") == ["One thing [123].", "Another [456, 789]!", "Third?"]


def test_unsupported_sentence_is_removed_and_its_citation_dropped(make_service):
    resp = _service(make_service, _v("supported", "unsupported")).ask(AskRequest(question="is the staff friendly?"))
    assert resp.answer.startswith("Staff are friendly") and "pool" not in resp.answer
    assert resp.unsupported_claims_removed == 1
    assert len(resp.citations) == 1


def test_answer_is_dropped_when_nothing_is_supported(make_service):
    resp = _service(make_service, _v("unsupported", "unsupported")).ask(AskRequest(question="is the staff friendly?"))
    assert resp.answer is None and resp.evidence


def test_not_a_claim_sentences_are_kept(make_service):
    resp = _service(make_service, _v("supported", "not_a_claim")).ask(AskRequest(question="is the staff friendly?"))
    assert "pool" in resp.answer and resp.unsupported_claims_removed == 0


def test_verifier_failure_keeps_the_answer(make_service):
    for bad in ("not json", json.dumps({"verdicts": [{"claim": 1, "verdict": "supported"}]}), _v("supported", "maybe")):
        resp = _service(make_service, bad).ask(AskRequest(question="is the staff friendly?"))
        assert resp.answer and resp.unsupported_claims_removed == 0


def test_answer_is_kept_when_the_verifier_calls_every_sentence_not_a_claim(make_service):
    resp = _service(make_service, _v("not_a_claim", "not_a_claim")).ask(AskRequest(question="is the staff friendly?"))
    assert resp.answer and resp.unsupported_claims_removed == 0
