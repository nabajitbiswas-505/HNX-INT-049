"""Answer schema, answer text, and proof.json writing.

An Answer must carry at least one timestamped event (in `events`, or in
`evidence` for "nobody / none" answers); otherwise it is rejected.
The answer sentence is built from a template here, never by an LLM.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, model_validator

from temporal.engine import Query, QueryResult


class ProofError(ValueError):
    """The answer cannot be backed by timestamped events."""


class EventRef(BaseModel):
    event_id: int
    person: str
    action: str
    t_start: float
    t_end: float


class Answer(BaseModel):
    question: str
    answer: str
    events: list[EventRef] = Field(default_factory=list)    # events that ARE the answer
    evidence: list[EventRef] = Field(default_factory=list)  # events checked / compared against
    confidence: float = Field(ge=0.0, le=1.0)
    value: Optional[float] = None
    query: dict
    proof_file: Optional[str] = None

    @model_validator(mode="after")
    def _needs_timestamps(self):
        if not (self.events or self.evidence):
            raise ValueError("answer has no timestamps (no supporting events)")
        return self


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _ref(e: dict) -> EventRef:
    return EventRef(event_id=e["Event_ID"], person=e["Person_ID"], action=e["Action"],
                    t_start=e["Start_Sec"], t_end=e["End_Sec"])


def _fmt(e: EventRef) -> str:
    return f"{e.person} {e.action} ({e.t_start:g}-{e.t_end:g} s)"


def _sel_text(sel) -> str:
    parts = [p for p in (sel.person, sel.action, sel.zone) if p]
    return " ".join(parts) if parts else "any event"


def _num(x: float) -> str:
    return f"{x:g}"


def compose_text(query: Query, result: QueryResult, events: list[EventRef], evidence: list[EventRef]) -> str:
    listed = "; ".join(_fmt(e) for e in events)
    relative = ""
    if query.to is not None and evidence:
        relative = f", relative to {'; '.join(_fmt(e) for e in evidence)}"

    if query.aggregate == "count":
        return f"{_num(result.value)} event(s): {listed}{relative}" if events else \
            f"0 events{relative}"
    if query.aggregate == "count_people":
        return f"{_num(result.value)} person(s): {listed}{relative}" if events else \
            f"0 people{relative}"
    if query.aggregate == "total_duration":
        return f"{_num(result.value)} s in total: {listed}{relative}" if events else \
            f"0 s{relative}"
    if events:
        return f"{listed}{relative}"
    ref = "; ".join(_fmt(e) for e in evidence)
    return (f"No match: no '{_sel_text(query.select)}' event is "
            f"{'/'.join(query.relation)} to {ref}")


def build_answer(question: str, query: Query, result: QueryResult) -> Answer:
    events = [_ref(e) for e in result.events]
    evidence = [_ref(e) for e in result.reference]
    if not (events or evidence):
        raise ProofError(
            "No supporting events with timestamps were found, so no answer is given "
            "(an answer without timestamps is not accepted).")
    used = result.events + result.reference
    confidence = min(e["Confidence"] for e in used)  # conservative: weakest link
    text = compose_text(query, result, events, evidence)
    return Answer(question=question, answer=text, events=events, evidence=evidence,
                  confidence=round(confidence, 2), value=result.value,
                  query=query.model_dump(mode="json"))


def write_proof(answer: Answer, events_csv: str | Path, proof_path: str | Path,
                planner_source: str = "unknown") -> Path:
    proof_path = Path(proof_path)
    proof_path.parent.mkdir(parents=True, exist_ok=True)
    answer.proof_file = str(proof_path)
    doc = {
        "question": answer.question,
        "query": answer.query,
        "planner_source": planner_source,
        "answer": answer.model_dump(mode="json"),
        "events_csv": str(events_csv),
        "events_csv_sha256": sha256_file(events_csv),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    proof_path.write_text(json.dumps(doc, indent=2))
    return proof_path
