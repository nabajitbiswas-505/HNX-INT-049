"""Question -> query JSON.

The planner only *translates*. It never computes times, counts or durations;
that is done by temporal.engine. Two backends:

1. Gemini (google-genai SDK), used when GEMINI_API_KEY is set. Output is validated
   against the pydantic Query model; on failure it is retried once with the error.
2. A rule-based parser, used when there is no key, or the LLM fails twice.

Env vars: GEMINI_API_KEY, GEMINI_MODEL (default below; model ids change, check
https://ai.google.dev/gemini-api/docs/models).
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Callable, Optional

from pydantic import ValidationError

from temporal.engine import Query, Selector

DEFAULT_MODEL = "gemini-2.5-flash"
Generate = Callable[[str, str], str]  # (system_prompt, user_prompt) -> raw text


class PlanError(ValueError):
    """The question could not be turned into a valid query."""


@dataclass
class PlanResult:
    query: Query
    source: str            # "llm", "rules" or "rules (llm failed: ...)"


SYSTEM_PROMPT = """You translate questions about classroom behaviour events into a JSON query.
Output ONE JSON object and nothing else. NEVER compute or state times, counts or
durations: a program does that. Only describe what to look up.

Schema:
{
  "select":   {"action": <action|null>, "person": <"S1".."S4"|null>, "zone": <e.g. "R1C1"|null>},
  "relation": [<relation names>],        // empty list if the question has no time relation
  "to":       {same shape as select},    // the reference events; required iff relation is non-empty
  "aggregate": "list" | "count" | "count_people" | "total_duration",
  "pick":     "all" | "nearest"          // "nearest" for "right before"/"right after"/"just before"
}
action is one of: phone, hand_raise, sleeping, talking (or null for "anything").
relation names: "concurrent" (while/during/at the same time), "precedes" (before),
follows (after). Optional exact names: before, after, meets, met_by, overlaps,
overlapped_by, starts, started_by, during, contains, finishes, finished_by, equals.
"select" is what the question asks about; "to" is what it is compared against.
Use only these keys.

CRITICAL INSTRUCTION FOR ACTIONS:
Video perception data often contains compound YOLO actions (e.g. "sitting upright, interacting with bench & cell phone"). 
Your output must still map strictly to the core action categories (phone, hand_raise, sleeping, talking) so the downstream engine can perform substring matching.

Examples:

Q: Who was sleeping while S3 was on the phone?
{"select":{"action":"sleeping"},"relation":["concurrent"],"to":{"action":"phone","person":"S3"},"aggregate":"list","pick":"all"}
Q: What happened right before S2 started sleeping?
{"select":{},"relation":["precedes"],"to":{"action":"sleeping","person":"S2"},"aggregate":"list","pick":"nearest"}
Q: How many people used a phone?
{"select":{"action":"phone"},"relation":[],"aggregate":"count_people","pick":"all"}
Q: How long did S2 sleep?
{"select":{"action":"sleeping","person":"S2"},"relation":[],"aggregate":"total_duration","pick":"all"}
"""


# --------------------------------------------------------------------------
# Rule-based backend
# --------------------------------------------------------------------------
_ACTION_PATTERNS = [
    ("hand_raise", r"\b(hands?|raised?|raising)\b"),
    ("sleeping", r"\b(sleep\w*|asleep|nap\w*)\b"),
    ("phone", r"\b(phones?|mobile)\b"),
    ("talking", r"\b(talk\w*|chat\w*|speak\w*|spoke)\b"),
    ("leaning", r"\b(lean\w*)\b"),
    ("moving", r"\b(mov\w*|active\w*)\b"),
]
_CONNECTOR = re.compile(
    r"\b(right before|just before|immediately before|before|"
    r"right after|just after|immediately after|after|"
    r"at the same time as|while|during|when)\b"
)
_GENERIC = re.compile(r"\b(what happened|everything|all events|timeline|list)\b")


def _selector(text: str) -> Selector:
    action, best = None, None
    for name, pat in _ACTION_PATTERNS:
        m = re.search(pat, text)
        if m and (best is None or m.start() < best):
            action, best = name, m.start()
    person = re.search(r"\bs\d+\b", text)
    zone = re.search(r"\br\d+c\d+\b", text)
    return Selector(action=action,
                    person=person.group(0).upper() if person else None,
                    zone=zone.group(0).upper() if zone else None)


def rule_plan(question: str) -> Query:
    q = question.lower().strip().rstrip("?.! ")

    if re.search(r"\bhow many (people|persons|students)\b|\bnumber of (people|persons|students)\b", q):
        aggregate = "count_people"
    elif re.search(r"\bhow many\b|\bnumber of\b", q):
        aggregate = "count"
    elif re.search(r"\bhow long\b|\bduration\b|\btotal time\b|\bhow much time\b", q):
        aggregate = "total_duration"
    else:
        aggregate = "list"

    m = _CONNECTOR.search(q)
    if m is None:
        sel = _selector(q)
        if not (sel.action or sel.person or sel.zone) and not _GENERIC.search(q):
            raise PlanError(f"could not find an action, person or zone in: {question!r}")
        return Query(select=sel, aggregate=aggregate)

    word = m.group(1)
    left, right = q[:m.start()], q[m.end():]
    if word in ("while", "during", "when", "at the same time as"):
        relation = ["concurrent"]
    elif word.endswith("before"):
        relation = ["precedes"]
    else:
        relation = ["follows"]
    pick = "nearest" if word.startswith(("right", "just", "immediately")) else "all"

    ref = _selector(right)
    if not (ref.action or ref.person or ref.zone):
        raise PlanError(f"could not tell what to compare against in: {question!r}")
    return Query(select=_selector(left), relation=relation, to=ref, aggregate=aggregate, pick=pick)


# --------------------------------------------------------------------------
# Gemini backend
# --------------------------------------------------------------------------
def gemini_generate(system: str, user: str) -> str:
    """Call Gemini with JSON output. Needs `pip install google-genai` and GEMINI_API_KEY."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    
    # PERMANENT FIX: Use the multi-turn chat interface to resolve the AFC warning
    chat = client.chats.create(
        model=os.environ.get("GEMINI_MODEL", DEFAULT_MODEL),
        config=types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            temperature=0,
        )
    )
    resp = chat.send_message(user)
    return resp.text or ""


def _parse(raw: str) -> Query:
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
    return Query.model_validate(json.loads(text))


def _llm_plan(question: str, generate: Generate) -> Query:
    user = f"Q: {question}"
    raw = generate(SYSTEM_PROMPT, user)
    try:
        return _parse(raw)
    except (ValueError, ValidationError) as first_error:  # JSONDecodeError is a ValueError
        retry = (f"{user}\n\nYour previous output was invalid:\n{raw}\n\n"
                 f"Error: {first_error}\nReturn corrected JSON only.")
        return _parse(generate(SYSTEM_PROMPT, retry))


def plan(question: str, generate: Optional[Generate] = None) -> PlanResult:
    """Plan a question. `generate` can be injected (tests); otherwise Gemini is used if a key is set."""
    if generate is None and os.environ.get("GEMINI_API_KEY"):
        generate = gemini_generate
    if generate is None:
        return PlanResult(rule_plan(question), "rules")
    try:
        return PlanResult(_llm_plan(question, generate), "llm")
    except Exception as exc:  # network error, bad key, invalid JSON twice, ...
        reason = str(exc).splitlines()[0][:80] if str(exc) else type(exc).__name__
        return PlanResult(rule_plan(question), f"rules (llm failed: {reason})")