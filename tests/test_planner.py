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


def test_unknown_city_is_declined(make_service):
    service = _service(make_service, {"in_scope": True, "cities": ["Lyon"], "topic": "breakfast"})
    resp = service.ask(AskRequest(question="breakfast in Lyon"))
    assert resp.insufficient_evidence and "Lyon" in resp.message and not resp.evidence


def test_out_of_scope_question_is_declined(make_service):
    service = _service(make_service, {"in_scope": False})
    resp = service.ask(AskRequest(question="tell me a joke"))
    assert resp.insufficient_evidence and not resp.evidence


def test_planner_operation_drives_stats(make_service):
    service = _service(make_service, {"in_scope": True, "topic": "", "operation": {"name": "count", "group_by": "city"}})
    resp = service.ask(AskRequest(question="reviews per place"))
    assert resp.stats.operation.name == "count" and resp.stats.operation.group_by == "city"


def test_planner_failure_falls_back_to_rules(make_service):
    for service in (_service(make_service, error=TimeoutError()),
                    _service(make_service, {"in_scope": True, "operation": {"name": "nonsense"}})):
        resp = service.ask(AskRequest(question="is the staff friendly in London?"))
        assert resp.filters_applied.get("city") == ["London"]
