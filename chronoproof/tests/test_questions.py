"""The five demo questions with expected answers AND timestamps (end to end, rule planner)."""
import pytest

from ask import ask
from events.behavior_parser import build_events

# question -> (answer fragment, expected [(person, action, start, end)] for events, same for evidence)
CASES = [
    ("Who raised a hand while S4 was talking?",
     "No match", [], [("S4", "talking", 10.0, 20.0)]),
    ("What happened right before S2 started sleeping?",
     "S4 talking (10-20 s)", [("S4", "talking", 10.0, 20.0)], [("S2", "sleeping", 20.0, 55.0)]),
    ("How many people used a phone?",
     "1 person(s)", [("S3", "phone", 40.0, 58.0)], []),
    ("Who was sleeping while S3 was on the phone?",
     "S2 sleeping (20-55 s)", [("S2", "sleeping", 20.0, 55.0)], [("S3", "phone", 40.0, 58.0)]),
    ("How long did S2 sleep?",
     "35 s in total", [("S2", "sleeping", 20.0, 55.0)], []),
]


@pytest.mark.parametrize("question,fragment,events,evidence", CASES)
def test_question(question, fragment, events, evidence, tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    csv = tmp_path / "e.csv"
    build_events("data/perception_2.csv", csv)
    a = ask(question, csv, tmp_path / "proof.json")
    assert fragment in a.answer
    assert [(e.person, e.action, e.t_start, e.t_end) for e in a.events] == events
    assert [(e.person, e.action, e.t_start, e.t_end) for e in a.evidence] == evidence
    assert 0 < a.confidence <= 1
