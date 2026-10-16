import json

from app.judge import judge_row
from tests.test_llm import FakeLLM

GOOD = {"grounded": 5, "numbers_correct": "yes", "relevant": 4, "balanced": "n/a",
        "decline_appropriate": "n/a", "reason": "ok"}


def test_judge_returns_rubric_and_sees_all_inputs():
    llm = FakeLLM(json.dumps(GOOD))
    out = judge_row(llm, "is staff friendly?", "Yes [1]", "[1] (positive) friendly staff", "none")
    assert out["verdict"] == "pass" and out["grounded"] == 5
    assert all(x in llm.calls[0] for x in ("is staff friendly?", "Yes [1]", "friendly staff"))


def test_declined_row_shows_the_decline_message():
    llm = FakeLLM(json.dumps({**GOOD, "decline_appropriate": "yes"}))
    judge_row(llm, "hotels in Rome?", "", "", "", declined_message="No reviews for Rome")
    assert "(declined) No reviews for Rome" in llm.calls[0]


def test_judge_failure_is_an_error_row_not_an_exception():
    for llm in (FakeLLM(error=TimeoutError("x")), FakeLLM("not json"), FakeLLM(json.dumps({"verdict": "maybe"}))):
        assert judge_row(llm, "q", "a", "e", "s")["verdict"] == "error"


def test_verdict_is_computed_from_scores_not_the_models_own_verdict():
    contradicting = {**GOOD, "grounded": 1, "verdict": "pass"}
    assert judge_row(FakeLLM(json.dumps(contradicting)), "q", "a", "e", "s")["verdict"] == "fail"
    wrong_numbers = {**GOOD, "numbers_correct": "no"}
    assert judge_row(FakeLLM(json.dumps(wrong_numbers)), "q", "a", "e", "s")["verdict"] == "fail"
    unbalanced = {**GOOD, "balanced": "no"}  # informational only
    assert judge_row(FakeLLM(json.dumps(unbalanced)), "q", "a", "e", "s")["verdict"] == "pass"


def test_declined_rows_pass_only_when_the_decline_was_right():
    right = {"decline_appropriate": "yes", "reason": "ok", "grounded": 1, "relevant": 1}
    wrong = {**right, "decline_appropriate": "no"}
    assert judge_row(FakeLLM(json.dumps(right)), "q", "", "", "", "no Rome")["verdict"] == "pass"
    assert judge_row(FakeLLM(json.dumps(wrong)), "q", "", "", "", "no Rome")["verdict"] == "fail"
