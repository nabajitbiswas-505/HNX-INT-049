import json
import shutil

import pytest
from pydantic import ValidationError

from ask import ask
from events.behavior_parser import build_events
from planner.llm_planner import PlanError
from proof.proof import Answer, ProofError, build_answer
from proof.verify import main as verify_main, verify_proof
from temporal.engine import Query, QueryResult, run_query


@pytest.fixture()
def csv(tmp_path):
    p = tmp_path / "events.csv"
    build_events("data/perception_2.csv", p)
    return p


def test_answer_without_timestamps_is_rejected():
    with pytest.raises(ValidationError, match="no timestamps"):
        Answer(question="q", answer="35 seconds", confidence=0.9, query={})


def test_build_answer_rejects_empty_result():
    q = Query.model_validate({"select": {"action": "phone"}})
    with pytest.raises(ProofError):
        build_answer("q", q, QueryResult(aggregate="list"))


def test_query_that_matches_nothing_cannot_be_answered(csv, tmp_path):
    # no hand_raise by S4 and nothing to compare against -> no timestamps -> refused
    with pytest.raises(ProofError):
        ask("Did S4 raise a hand?", csv, tmp_path / "p.json")


def test_nobody_answer_keeps_timestamped_evidence(csv, tmp_path):
    a = ask("Who raised a hand while S4 was talking?", csv, tmp_path / "p.json")
    assert a.events == [] and [(e.t_start, e.t_end) for e in a.evidence] == [(10.0, 20.0)]
    assert a.answer.startswith("No match")


def test_proof_file_contents(csv, tmp_path):
    ask("How long did S2 sleep?", csv, tmp_path / "p.json")
    doc = json.loads((tmp_path / "p.json").read_text())
    assert set(doc) >= {"question", "query", "answer", "events_csv", "events_csv_sha256", "planner_source"}
    assert doc["answer"]["events"][0]["t_start"] == 20.0 and doc["answer"]["value"] == 35.0


def test_verify_passes_on_fresh_proof(csv, tmp_path):
    ask("Who was sleeping while S3 was on the phone?", csv, tmp_path / "p.json")
    assert verify_proof(tmp_path / "p.json")
    assert verify_main([str(tmp_path / "p.json")]) == 0


def test_verify_fails_if_csv_changed(csv, tmp_path):
    ask("How long did S2 sleep?", csv, tmp_path / "p.json")
    csv.write_text(csv.read_text().replace("20.0,55.0", "20.0,50.0"))
    rep = verify_proof(tmp_path / "p.json")
    assert not rep and "hash mismatch" in rep.problems[0]
    assert verify_main([str(tmp_path / "p.json")]) == 1


def test_verify_fails_if_stored_timestamp_tampered(csv, tmp_path):
    ask("How long did S2 sleep?", csv, tmp_path / "p.json")
    doc = json.loads((tmp_path / "p.json").read_text())
    doc["answer"]["events"][0]["t_end"] = 40.0
    (tmp_path / "p.json").write_text(json.dumps(doc))
    rep = verify_proof(tmp_path / "p.json")
    assert not rep and any("timestamps differ" in p for p in rep.problems)


def test_verify_fails_if_stored_answer_text_tampered(csv, tmp_path):
    ask("How long did S2 sleep?", csv, tmp_path / "p.json")
    doc = json.loads((tmp_path / "p.json").read_text())
    doc["answer"]["answer"] = "S2 slept 10 s"
    (tmp_path / "p.json").write_text(json.dumps(doc))
    assert "answer text differs" in verify_proof(tmp_path / "p.json").problems


def test_verify_fails_on_garbage_and_missing_csv(csv, tmp_path):
    (tmp_path / "bad.json").write_text("{}")
    assert not verify_proof(tmp_path / "bad.json")
    ask("How long did S2 sleep?", csv, tmp_path / "p.json")
    csv.unlink()
    assert "not found" in verify_proof(tmp_path / "p.json").problems[0]


def test_verify_with_copied_csv(csv, tmp_path):
    ask("How long did S2 sleep?", csv, tmp_path / "p.json")
    copy = tmp_path / "copy.csv"
    shutil.copy(csv, copy)
    assert verify_proof(tmp_path / "p.json", copy)


def test_unanswerable_question_raises_plan_error(csv, tmp_path):
    with pytest.raises(PlanError):
        ask("What is the weather like?", csv, tmp_path / "p.json")
