"""Re-run a proof.json and check it reproduces the same answer.

    python -m proof.verify out/proof.json [--csv other_events.csv]

Checks, in order:
 1. the stored answer still passes the Answer schema (has timestamps)
 2. the events CSV hash matches the one recorded (the data was not changed)
 3. re-running the stored query gives the same events, timestamps and value
 4. the answer sentence is identical
Exit code 0 = verified, 1 = failed.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from temporal.engine import Query, QueryError, run_query
from proof.proof import Answer, ProofError, build_answer, sha256_file

TOL = 1e-6


@dataclass
class Report:
    ok: bool
    problems: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.ok


def _same(a, b) -> bool:
    return abs(a - b) <= TOL


def verify_proof(proof_path: str | Path, csv_override: str | Path | None = None) -> Report:
    problems: list[str] = []
    try:
        doc = json.loads(Path(proof_path).read_text())
        stored = Answer.model_validate(doc["answer"])
        query = Query.model_validate(doc["query"])
    except Exception as exc:
        return Report(False, [f"proof file is invalid: {exc}"])

    csv = Path(csv_override or doc["events_csv"])
    if not csv.exists():
        return Report(False, [f"events CSV not found: {csv}"])
    if sha256_file(csv) != doc["events_csv_sha256"]:
        return Report(False, [f"events CSV changed since the proof was made (hash mismatch): {csv}"])

    try:
        fresh = build_answer(stored.question, query, run_query(csv, query))
    except (ProofError, QueryError) as exc:
        return Report(False, [f"re-running the query failed: {exc}"])

    for label in ("events", "evidence"):
        old, new = getattr(stored, label), getattr(fresh, label)
        if [e.event_id for e in old] != [e.event_id for e in new]:
            problems.append(f"{label}: event ids differ ({[e.event_id for e in old]} vs {[e.event_id for e in new]})")
            continue
        for o, n in zip(old, new):
            if not (_same(o.t_start, n.t_start) and _same(o.t_end, n.t_end)):
                problems.append(f"{label}: timestamps differ for event {o.event_id}")
    if (stored.value is None) != (fresh.value is None) or (
            stored.value is not None and not _same(stored.value, fresh.value)):
        problems.append(f"value differs ({stored.value} vs {fresh.value})")
    if stored.answer != fresh.answer:
        problems.append("answer text differs")
    return Report(not problems, problems)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Verify a proof.json")
    p.add_argument("proof")
    p.add_argument("--csv", help="use this events CSV instead of the one recorded in the proof")
    args = p.parse_args(argv)
    report = verify_proof(args.proof, args.csv)
    if report:
        print(f"VERIFIED: {args.proof} reproduces the same answer and timestamps.")
        return 0
    print(f"FAILED: {args.proof}")
    for problem in report.problems:
        print(f"  - {problem}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
