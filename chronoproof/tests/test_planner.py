import json

import pytest

from planner.llm_planner import PlanError, plan, rule_plan
from temporal.engine import Query

# question -> expected query (as a dict)
EXPECTED = {
    "Who raised a hand while S4 was talking?": {
        "select": {"action": "hand_raise"}, "relation": ["concurrent"],
        "to": {"action": "talking", "person": "S4"}},
    "What happened right before S2 started sleeping?": {
        "select": {}, "relation": ["precedes"],
        "to": {"action": "sleeping", "person": "S2"}, "pick": "nearest"},
    "How many people used a phone?": {
        "select": {"action": "phone"}, "aggregate": "count_people"},
    "Who was sleeping while S3 was on the phone?": {
        "select": {"action": "sleeping"}, "relation": ["concurrent"],
        "to": {"action": "phone", "person": "S3"}},
    "How long did S2 sleep?": {
        "select": {"action": "sleeping", "person": "S2"}, "aggregate": "total_duration"},
}


@pytest.mark.parametrize("question,expected", EXPECTED.items())
def test_rules_handle_sample_questions(question, expected):
    assert rule_plan(question) == Query.model_validate(expected)


def test_rules_extra_phrasings():
    assert rule_plan("How many times did S1 raise a hand?").aggregate == "count"
    q = rule_plan("Who was talking after S2 slept?")
    assert q.relation == ["follows"] and q.to.person == "S2" and q.select.action == "talking"
    assert rule_plan("what did s3 do").select.person == "S3"
    assert rule_plan("Show everything in R1C1").select.zone == "R1C1"


def test_rules_reject_nonsense():
    with pytest.raises(PlanError):
        rule_plan("What is the weather like?")
    with pytest.raises(PlanError):
        rule_plan("Who spoke before lunch?")  # reference has no action/person/zone


def test_no_key_uses_rules(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    r = plan("How long did S2 sleep?")
    assert r.source == "rules" and r.query.aggregate == "total_duration"


GOOD = json.dumps(EXPECTED["How long did S2 sleep?"])


def test_llm_valid_output_used():
    r = plan("How long did S2 sleep?", generate=lambda s, u: GOOD)
    assert r.source == "llm" and r.query.select.person == "S2"


def test_llm_code_fences_are_stripped():
    r = plan("How long did S2 sleep?", generate=lambda s, u: f"```json\n{GOOD}\n```")
    assert r.source == "llm"


def test_llm_retried_once_with_error_message():
    calls = []

    def fake(system, user):
        calls.append(user)
        return "not json" if len(calls) == 1 else GOOD

    r = plan("How long did S2 sleep?", generate=fake)
    assert r.source == "llm" and len(calls) == 2
    assert "invalid" in calls[1] and "not json" in calls[1]


def test_llm_with_extra_answer_key_is_rejected_then_fixed():
    bad = json.dumps({**json.loads(GOOD), "answer": "35 seconds"})  # LLM must not answer
    outputs = iter([bad, GOOD])
    r = plan("How long did S2 sleep?", generate=lambda s, u: next(outputs))
    assert r.source == "llm" and r.query.aggregate == "total_duration"


def test_llm_failing_twice_falls_back_to_rules():
    calls = []
    r = plan("How long did S2 sleep?", generate=lambda s, u: calls.append(1) or "nope")
    assert len(calls) == 2
    assert r.source.startswith("rules (llm failed") and r.query.select.person == "S2"


def test_llm_exception_falls_back_to_rules():
    def boom(s, u):
        raise RuntimeError("network down")

    r = plan("How many people used a phone?", generate=boom)
    assert r.source.startswith("rules (llm failed") and r.query.aggregate == "count_people"


def test_fallback_still_fails_on_nonsense():
    with pytest.raises(PlanError):
        plan("What is the weather like?", generate=lambda s, u: "nope")
