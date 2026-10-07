"""Ask a question: plan -> run query -> build answer -> write proof.json.

    python ask.py "Who was sleeping while S3 was on the phone?"
    python ask.py "How long did S2 sleep?" --proof out/proof.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from planner.llm_planner import PlanError, plan
from proof.proof import Answer, ProofError, build_answer, write_proof
from temporal.engine import QueryError, run_query


def ask(question: str, events_csv="data/generated_events_2.csv",
        proof_path="out/proof.json", generate=None) -> Answer:
    planned = plan(question, generate=generate)
    result = run_query(events_csv, planned.query)
    answer = build_answer(question, planned.query, result)
    write_proof(answer, events_csv, proof_path, planner_source=planned.source)
    return answer


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Temporal question answering over behaviour events")
    p.add_argument("question")
    p.add_argument("--events", default="data/generated_events_2.csv")
    p.add_argument("--proof", default="out/proof.json")
    args = p.parse_args(argv)

    if not Path(args.events).exists():
        print(f"Events file not found: {args.events}\n"
              "Create it with: python -m events.behavior_parser", file=sys.stderr)
        return 2
    try:
        ans = ask(args.question, args.events, args.proof)
    except (PlanError, QueryError, ProofError) as exc:
        print(f"Cannot answer: {exc}", file=sys.stderr)
        return 1

    print(f"Q: {ans.question}")
    print(f"A: {ans.answer}")
    print(f"   confidence: {ans.confidence}")
    for e in ans.events:
        print(f"   event #{e.event_id}: {e.person} {e.action} {e.t_start:g}-{e.t_end:g} s")
    for e in ans.evidence:
        print(f"   compared with #{e.event_id}: {e.person} {e.action} {e.t_start:g}-{e.t_end:g} s")
    print(f"   proof: {ans.proof_file}  (check with: python -m proof.verify {ans.proof_file})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
