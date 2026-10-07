# ChronoProof (HNX26PSI02: Video Understanding & Temporal Reasoning)

Answers questions about **what happened, when, in what order and for how long**
over classroom behaviour events. Every answer carries timestamps and a
`proof.json` that anyone can re-run to check it.

**Core principle:** the LLM only *translates* a question into a JSON query. Order,
overlap, counts and durations are computed in code (`temporal/`), so the system
never lets a language model guess a time.

```
perception_2.csv --> events/behavior_parser.py --> generated_events_2.csv
                                                        |
question --> planner (Gemini or rules) --> query JSON --> temporal/engine.py
                                                        |
                                    answer + timestamps --> proof.json --> proof/verify.py
```

## Setup
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Python 3.10+. `google-genai` is only needed for the Gemini planner.

## Run
```bash
# 1. perception rows -> events
python -m events.behavior_parser                       # writes data/generated_events_2.csv

# 2. ask a question (rule-based planner if no API key)
python ask.py "Who was sleeping while S3 was on the phone?"

# 3. verify the proof
python -m proof.verify out/proof.json
```
Optional Gemini planner: `export GEMINI_API_KEY=...` (and `GEMINI_MODEL=...`; model ids
change, see https://ai.google.dev/gemini-api/docs/models). If the key is missing or the
LLM output is invalid twice, the rule-based planner is used and `planner_source`
in `proof.json` says so.

## Reproduce the results
```bash
python -m pytest -q          # 109 tests
```
Sample input/output (also in `examples/`):

| Question | Answer |
|---|---|
| Who raised a hand while S4 was talking? | No match; checked S4 talking 10-20 s (hand raise was 30-34.5 s) |
| What happened right before S2 started sleeping? | S4 talking, 10-20 s |
| How many people used a phone? | 1: S3 phone, 40-58 s |
| Who was sleeping while S3 was on the phone? | S2 sleeping, 20-55 s |
| How long did S2 sleep? | 35 s (20-55 s) |

## What is in the repo
| Path | Purpose |
|---|---|
| `events/behavior_parser.py` | Threshold rules -> events (phone, hand_raise, sleeping, talking). Confidence = coverage x (0.5 + 0.5 x mean margin past threshold). Reports overlapping behaviours. |
| `temporal/allen.py` | Allen's 13 interval relations with an epsilon tolerance, plus groups `precedes`, `follows`, `concurrent`. |
| `temporal/engine.py` | `events`, `count`, `duration`, `first_after`, `last_before`, `gap`, `where`, and the validated JSON query (`Query`). |
| `planner/llm_planner.py` | Question -> query JSON (Gemini, one retry with the error, then rule-based fallback). |
| `proof/proof.py`, `proof/verify.py` | Answer schema (rejects answers with no timestamps), `proof.json` writer, and the verifier. |
| `ask.py` | Command-line pipeline. |

Time convention: `Start_Sec`/`End_Sec` are the first and last sample where the action
held; `End_Exclusive = End_Sec + sample interval`. Relations use `End_Sec` with
`eps = 0.25 s`.

## Declared resources
pandas, numpy, pydantic, pytest; optional Gemini API via `google-genai`. No datasets or
pre-trained models are bundled: `data/perception_2.csv` is the team's own perception
file, and `data/events_2.csv` is its answer key.

## SCOPE NOTE
**Minimum viable (implemented and tested):** event extraction from perception rows;
Allen-relation query engine; JSON query validation; rule-based and Gemini planners;
timestamped answers; `proof.json` with a CSV hash and `verify` that detects changed
data or tampered answers.

**Known limits (be upfront with judges):**
- Input is **precomputed perception data**, not raw video. Detection, tracking and
  pose estimation are not part of this repo.
- Thresholds were **tuned on one 60 s, 4-person file** and checked against its answer
  key. They have not been validated on other data.
- Times are at the **0.5 s sampling resolution**; there is no sub-sample refinement
  (bisection needs the raw frames).
- Confidence values are computed from the formula above and do **not** equal the
  hand-written values in `events_2.csv` (0.80-0.90); the tests ignore that column.
- The rule-based planner covers common phrasings only. The Gemini path is unit-tested
  with a fake client but **has not been run against the live API in this repo**.
- If several behaviours occur in the same frame only the highest-priority one becomes
  the event; the others are listed in `Overlaps`.

**Stretch (not done):** raw-video perception, re-ID after occlusion, absence
certificates, bisection refinement, annotated evidence clips, Streamlit demo.
