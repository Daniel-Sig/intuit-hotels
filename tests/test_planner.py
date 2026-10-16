"""LLM planner with a fake client: validation against real values, fallback, scope check."""
import json

from app.planner import Planner
from app.schemas import AskRequest
from tests.test_llm import FakeLLM


def _service(make_service, reply=None, error=None):
    service = make_service()
    service.planner = Planner(FakeLLM(json.dumps(reply) if reply is not None else None, error),
                              service.cities, service.nationalities, service.search.values("Hotel_Name"))
    return service


def test_plan_filters_are_validated_against_dataset(make_service):
    service = _service(make_service, {
        "in_scope": True, "cities": ["paris", "Lyon"], "nationalities": ["Australia", "Atlantis"],
        "months": [6, 13, "x"], "topic": "friendly staff", "operation": None,
    })
    resp = service.ask(AskRequest(question="do Australians like Paris staff"))
    assert resp.filters_applied == {"city": ["Paris"], "Reviewer_Nationality": ["Australia"], "review_month": [6]}


def test_demonym_and_explicit_sentiment_come_from_rules(make_service):
    # The model drops the demonym and invents a sentiment filter; the rules correct both.
    service = _service(make_service, {"in_scope": True, "cities": ["Amsterdam"], "nationalities": [],
                                      "sentiment": "positive", "topic": "breakfast"})
    resp = service.ask(AskRequest(question="what do Australian guests complain about in Amsterdam"))
    assert resp.filters_applied == {"city": ["Amsterdam"], "Reviewer_Nationality": ["Australia"], "sentiment": "negative"}
    resp = service.ask(AskRequest(question="is the breakfast good in Amsterdam"))
    assert "sentiment" not in resp.filters_applied


def test_topic_words_that_only_describe_the_ask_are_stripped(make_service):
    service = _service(make_service, {"in_scope": True, "topic": "complaints", "months": [3, 4, 5, 6, 7, 8, 9]})
    resp = service.ask(AskRequest(question="what do guests complain about in spring"))
    assert resp.filters_applied["review_month"] == [3, 4, 5]  # rule months win over the model's


def test_invented_topic_is_replaced_by_the_rule_based_topic(make_service):
    service = _service(make_service, {"in_scope": True, "cities": ["Paris"], "topic": "food"})
    resp = service.ask(AskRequest(question="what do guests say about Paris"))
    assert not resp.insufficient_evidence or "relevant" not in (resp.message or "")


def test_unknown_city_is_declined(make_service):
    service = _service(make_service, {"in_scope": True, "cities": ["Lyon"], "topic": "breakfast"})
    resp = service.ask(AskRequest(question="breakfast in Lyon"))
    assert resp.insufficient_evidence and "Lyon" in resp.message and not resp.evidence


def test_out_of_scope_question_is_declined(make_service):
    service = _service(make_service, {"in_scope": False})
    resp = service.ask(AskRequest(question="tell me a joke"))
    assert resp.insufficient_evidence and not resp.evidence


def test_operation_comes_from_rules_not_the_llm(make_service):
    # The model asks for a count by month on an opinion question; it is ignored.
    service = _service(make_service, {"in_scope": True, "topic": "friendly staff",
                                      "operation": {"name": "count", "group_by": "month"}})
    assert service.ask(AskRequest(question="is the staff friendly?")).stats.operation is None
    service = _service(make_service, {"in_scope": True, "topic": ""})
    resp = service.ask(AskRequest(question="How many reviews are there per city?"))
    assert resp.stats.operation.name == "count" and resp.stats.operation.group_by == "city"


def test_planner_failure_falls_back_to_rules(make_service):
    for service in (_service(make_service, error=TimeoutError()), _service(make_service, "not json")):
        resp = service.ask(AskRequest(question="is the staff friendly in London?"))
        assert resp.filters_applied.get("city") == ["London"]


def test_unsupported_city_is_declined_as_such_even_if_marked_out_of_scope(make_service):
    service = _service(make_service, {"in_scope": False, "unsupported_cities": ["Rome"], "topic": "spring"})
    resp = service.ask(AskRequest(question="is spring good in Rome"))
    assert resp.insufficient_evidence and "Rome" in resp.message
