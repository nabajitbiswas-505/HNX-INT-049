"""Query engine: all time, order, count and duration logic lives here, in code.

A query is JSON validated by the `Query` model, e.g.

    {"select": {"action": "sleeping"},
     "relation": ["concurrent"],
     "to": {"action": "phone", "person": "S3"}}

meaning "events of A=sleeping that share time with B=phone by S3".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from temporal import allen

ACTIONS = ("phone", "hand_raise", "sleeping", "talking", "leaning", "moving")
Action = Literal["phone", "hand_raise", "sleeping", "talking", "leaning", "moving"]
Aggregate = Literal["list", "count", "count_people", "total_duration"]

EVENT_FIELDS = ["Event_ID", "Person_ID", "Action", "Start_Sec", "End_Sec",
                "Seat_Zone", "Confidence"]


class QueryError(ValueError):
    """Raised for queries that are valid JSON but cannot be answered from the data."""


# --------------------------------------------------------------------------
# Query model
# --------------------------------------------------------------------------
class Selector(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Optional[Action] = None
    person: Optional[str] = None
    zone: Optional[str] = None

    @field_validator("person", "zone")
    @classmethod
    def _upper(cls, v):
        return v.strip().upper() if v else v


class Query(BaseModel):
    model_config = ConfigDict(extra="forbid")
    select: Selector
    relation: list[str] = Field(default_factory=list)
    to: Optional[Selector] = None
    aggregate: Aggregate = "list"
    pick: Literal["all", "nearest"] = "all"
    eps: float = Field(0.25, ge=0)

    @model_validator(mode="after")
    def _check(self):
        if self.relation:
            allen.expand(self.relation)  # raises ValueError on unknown names
        if bool(self.relation) != (self.to is not None):
            raise ValueError("'relation' and 'to' must be given together")
        if self.pick == "nearest" and self.to is None:
            raise ValueError("pick='nearest' needs 'relation' and 'to'")
        return self


@dataclass
class QueryResult:
    aggregate: str
    events: list[dict] = field(default_factory=list)      # the answer events
    reference: list[dict] = field(default_factory=list)   # B events the answer is relative to
    value: Optional[float] = None                         # count / people / seconds


# --------------------------------------------------------------------------
# Event store
# --------------------------------------------------------------------------
def load_events(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in EVENT_FIELDS if c not in df.columns]
    if missing:
        raise QueryError(f"events file {path} is missing columns: {missing}")
    return df


def _to_dicts(df: pd.DataFrame) -> list[dict]:
    out = []
    for r in df[EVENT_FIELDS].to_dict("records"):
        out.append({
            "Event_ID": int(r["Event_ID"]), "Person_ID": str(r["Person_ID"]),
            "Action": str(r["Action"]), "Start_Sec": float(r["Start_Sec"]),
            "End_Sec": float(r["End_Sec"]), "Seat_Zone": str(r["Seat_Zone"]),
            "Confidence": float(r["Confidence"]),
        })
    return out


class EventStore:
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    # ---- basic accessors -------------------------------------------------
    def events(self, action=None, person=None, zone=None, t_min=None, t_max=None) -> pd.DataFrame:
        """Events matching all given filters. t_min/t_max keep events that overlap [t_min, t_max]."""
        d = self.df
        if action:
            # PERMANENT FIX: Substring matching to handle complex YOLO action strings
            d = d[d["Action"].str.contains(action, case=False, na=False)]
        if person:
            d = d[d["Person_ID"] == person]
        if zone:
            d = d[d["Seat_Zone"] == zone]
        if t_min is not None:
            d = d[d["End_Sec"] >= t_min]
        if t_max is not None:
            d = d[d["Start_Sec"] <= t_max]
        return d

    @staticmethod
    def count(events: pd.DataFrame) -> int:
        return int(len(events))

    @staticmethod
    def duration(event) -> float:
        """Seconds between first and last sample of one event (row or dict)."""
        return float(event["End_Sec"] - event["Start_Sec"])

    @staticmethod
    def total_duration(events: pd.DataFrame) -> float:
        return float((events["End_Sec"] - events["Start_Sec"]).sum())

    def first_after(self, t: float, action=None, person=None):
        """Earliest event starting at or after t, or None."""
        d = self.events(action, person)
        d = d[d["Start_Sec"] >= t].sort_values("Start_Sec")
        return None if d.empty else d.iloc[0]

    def last_before(self, t: float, action=None, person=None):
        """Latest event ending at or before t, or None."""
        d = self.events(action, person)
        d = d[d["End_Sec"] <= t].sort_values("End_Sec")
        return None if d.empty else d.iloc[-1]

    @staticmethod
    def gap(a, b) -> float:
        """Seconds of empty time between two events; 0 if they share time."""
        return float(max(0.0, max(a["Start_Sec"], b["Start_Sec"]) - min(a["End_Sec"], b["End_Sec"])))

    # ---- relational query ------------------------------------------------
    def where(self, events_a: pd.DataFrame, events_b: pd.DataFrame,
              relations: list[str], eps: float = 0.25) -> pd.DataFrame:
        """Events from A that have any of `relations` to any event in B.

        Adds columns Relation and Matched_With (B event ids). An event is never
        compared with itself.
        """
        wanted = allen.expand(relations)
        keep, rel_col, with_col = [], [], []
        for ia, a in events_a.iterrows():
            hits, rels = [], []
            for _, b in events_b.iterrows():
                if int(a["Event_ID"]) == int(b["Event_ID"]):
                    continue
                r = allen.relation((a["Start_Sec"], a["End_Sec"]), (b["Start_Sec"], b["End_Sec"]), eps)
                if r in wanted:
                    hits.append(int(b["Event_ID"]))
                    rels.append(r)
            if hits:
                keep.append(ia)
                rel_col.append(rels[0])
                with_col.append(hits)
        out = events_a.loc[keep].copy()
        out["Relation"] = rel_col
        out["Matched_With"] = with_col
        return out

    # ---- JSON query ---------------------------------------------------------
    def _check_selector(self, sel: Selector, label: str) -> None:
        if sel.person and sel.person not in set(self.df["Person_ID"]):
            raise QueryError(f"{label}: unknown person '{sel.person}'. Known: {sorted(set(self.df['Person_ID']))}")
        if sel.zone and sel.zone not in set(self.df["Seat_Zone"]):
            raise QueryError(f"{label}: unknown zone '{sel.zone}'. Known: {sorted(set(self.df['Seat_Zone']))}")

    def run(self, query: Query | dict) -> QueryResult:
        q = query if isinstance(query, Query) else Query.model_validate(query)
        self._check_selector(q.select, "select")
        a = self.events(q.select.action, q.select.person, q.select.zone)
        reference = pd.DataFrame(columns=self.df.columns)

        if q.to is not None:
            self._check_selector(q.to, "to")
            reference = self.events(q.to.action, q.to.person, q.to.zone)
            a = self.where(a, reference, q.relation, q.eps)
            if q.pick == "nearest" and not a.empty:
                gaps = []
                for _, ev in a.iterrows():
                    refs = reference[reference["Event_ID"].isin(ev["Matched_With"])]
                    gaps.append(min(self.gap(ev, r) for _, r in refs.iterrows()))
                a = a[[g == min(gaps) for g in gaps]]
            used = {i for ids in a["Matched_With"] for i in ids} if not a.empty else set()
            reference = reference[reference["Event_ID"].isin(used)] if used else reference.iloc[0:0]
            # If nothing matched, keep the reference candidates as evidence of what was checked.
            if a.empty:
                reference = self.events(q.to.action, q.to.person, q.to.zone)

        a = a.sort_values(["Start_Sec", "Person_ID"])
        res = QueryResult(aggregate=q.aggregate, events=_to_dicts(a), reference=_to_dicts(reference))
        if q.aggregate == "count":
            res.value = float(self.count(a))
        elif q.aggregate == "count_people":
            res.value = float(a["Person_ID"].nunique())
        elif q.aggregate == "total_duration":
            res.value = self.total_duration(a)
        return res


def run_query(events_csv: str | Path, query: Query | dict) -> QueryResult:
    return EventStore(load_events(events_csv)).run(query)