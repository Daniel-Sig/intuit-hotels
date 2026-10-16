"""LLM answer step with a fake client: grounding, citation checks and failure handling."""
import json

from app.llm import Answerer
from app.schemas import AskRequest


class FakeLLM:
    def __init__(self, reply=None, error: Exception | None = None):
        self.reply, self.error, self.calls = reply, error, []

    def complete(self, system, user):
        self.calls.append(user)
        if self.error:
            raise self.error
        return self.reply(user) if callable(self.reply) else self.reply


def _cite_first(user: str) -> str:
    import re
    ids = [int(i) for i in re.findall(r"review_id=(\d+)", user)]
    return json.dumps({"answer": "Staff are friendly.", "cited_review_ids": ids[:1]})


def test_answer_added_when_citations_valid(make_service):
    service = make_service()
    llm = FakeLLM(_cite_first)
    service.answerer = Answerer(llm)
    resp = service.ask(AskRequest(question="is the staff friendly?"))
    assert resp.answer == "Staff are friendly."
    assert len(resp.citations) == 1 and resp.citations[0].snippet.endswith("...")
    cited = next(e for e in resp.evidence if e.review_id == resp.citations[0].review_id)
    assert resp.citations[0].snippet == " ".join(cited.text.split()[:10]) + "..."
    assert "<evidence>" in llm.calls[0]


def test_invalid_citation_drops_answer_but_keeps_evidence(make_service):
    service = make_service()
    service.answerer = Answerer(FakeLLM(json.dumps({"answer": "x", "cited_review_ids": [999999]})))
    resp = service.ask(AskRequest(question="is the staff friendly?"))
    assert resp.answer is None and resp.evidence and resp.citations == []


def test_snippet_is_first_ten_words():
    from app.llm import snippet
    assert snippet("one two three four five six seven eight nine ten eleven twelve") == (
        "one two three four five six seven eight nine ten...")


def test_llm_failure_or_garbage_does_not_fail_request(make_service):
    for llm in (FakeLLM(error=TimeoutError("boom")), FakeLLM("not json")):
        service = make_service()
        service.answerer = Answerer(llm)
        resp = service.ask(AskRequest(question="is the staff friendly?"))
        assert resp.answer is None and resp.evidence


def test_llm_not_called_when_insufficient(make_service):
    service = make_service()
    llm = FakeLLM(_cite_first)
    service.answerer = Answerer(llm)
    resp = service.ask(AskRequest(question="what about hotels in Rome?"))
    assert resp.insufficient_evidence and resp.answer is None and not llm.calls


def test_ungrounded_numbers():
    from app.llm import ungrounded_numbers
    src = '{"mentions": 8563, "reviews": 41607, "share": 0.206, "value": 8.55} Staff 5 stars'
    assert ungrounded_numbers("8,563 of 41607 reviews, about 20.6%, avg 8.55 [123, 456]", src, 3) == []
    assert ungrounded_numbers("3 reviews said so, rated 5 [review_id=999]", src, 3) == []
    assert ungrounded_numbers("73.4% of comments, 8.9 average", src, 3) == ["73.4", "8.9"]
    assert ungrounded_numbers("about 4000 reviews", src, 3) == ["4000"]


def test_answer_with_invented_number_is_dropped(make_service):
    import re
    def reply(user):
        ids = [int(i) for i in re.findall(r"review_id=(\d+)", user)]
        return json.dumps({"answer": "73.4% of guests praise the staff [%d]" % ids[0], "cited_review_ids": ids[:1]})
    service = make_service()
    service.answerer = Answerer(FakeLLM(reply))
    resp = service.ask(AskRequest(question="is the staff friendly?"))
    assert resp.answer is None and resp.citations == [] and resp.evidence
