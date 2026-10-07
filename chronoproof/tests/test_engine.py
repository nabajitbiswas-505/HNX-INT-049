import pandas as pd
import pytest
from pydantic import ValidationError

from events.behavior_parser import build_events
from temporal.engine import EventStore, Query, QueryError, load_events, run_query


@pytest.fixture(scope="module")
def csv(tmp_path_factory):
    p = tmp_path_factory.mktemp("d") / "events.csv"
    build_events("data/perception_2.csv", p)
    return p


def ids(res):
    return [(e["Person_ID"], e["Action"], e["Start_Sec"], e["End_Sec"]) for e in res.events]


# ---- the five demo questions, as queries --------------------------------

def test_sleeping_while_s3_on_phone(csv):
    res = run_query(csv, {"select": {"action": "sleeping"}, "relation": ["concurrent"],
                          "to": {"action": "phone", "person": "S3"}})
    assert ids(res) == [("S2", "sleeping", 20.0, 55.0)]
    assert [e["Person_ID"] for e in res.reference] == ["S3"]


def test_count_phone_people(csv):
    res = run_query(csv, {"select": {"action": "phone"}, "aggregate": "count_people"})
    assert res.value == 1 and ids(res) == [("S3", "phone", 40.0, 58.0)]


def test_how_long_did_s2_sleep(csv):
    res = run_query(csv, {"select": {"action": "sleeping", "person": "S2"}, "aggregate": "total_duration"})
    assert res.value == 35.0


def test_right_before_s2_sleeping(csv):
    res = run_query(csv, {"select": {}, "relation": ["precedes"],
                          "to": {"action": "sleeping", "person": "S2"}, "pick": "nearest"})
    assert ids(res) == [("S4", "talking", 10.0, 20.0)]


def test_hand_raise_while_s4_talking_is_nobody(csv):
    res = run_query(csv, {"select": {"action": "hand_raise"}, "relation": ["concurrent"],
                          "to": {"action": "talking", "person": "S4"}})
    assert res.events == []
    # the reference events checked are kept as timestamped evidence
    assert [(e["Start_Sec"], e["End_Sec"]) for e in res.reference] == [(10.0, 20.0)]


# ---- building blocks -----------------------------------------------------

def test_filters_and_count(csv):
    s = EventStore(load_events(csv))
    assert s.count(s.events()) == 4
    assert s.count(s.events(action="phone")) == 1
    assert s.count(s.events(zone="R1C1")) == 1
    assert s.count(s.events(t_min=56, t_max=60)) == 1  # only the phone event reaches 56-60


def test_first_after_and_last_before(csv):
    s = EventStore(load_events(csv))
    assert s.first_after(25.0)["Action"] == "hand_raise"
    assert s.last_before(25.0)["Action"] == "talking"
    assert s.first_after(100.0) is None
    assert s.last_before(1.0) is None


def test_gap_and_duration(csv):
    s = EventStore(load_events(csv))
    talk = s.events(action="talking").iloc[0]
    hand = s.events(action="hand_raise").iloc[0]
    assert s.gap(talk, hand) == 10.0       # 20.0 -> 30.0
    assert s.gap(talk, talk) == 0.0
    assert s.duration(hand) == 4.5


def test_event_not_matched_with_itself(csv):
    res = run_query(csv, {"select": {"action": "phone"}, "relation": ["concurrent"],
                          "to": {"action": "phone"}})
    assert res.events == []


def test_nearest_picks_one_of_many(csv):
    # everything that ends before the phone event starts; nearest = hand_raise (ends 34.5)
    res = run_query(csv, {"select": {}, "relation": ["before", "meets"],
                          "to": {"action": "phone"}, "pick": "nearest"})
    assert [e["Action"] for e in res.events] == ["hand_raise"]


# ---- validation ------------------------------------------------------------

def test_unknown_action_rejected():
    with pytest.raises(ValidationError):
        Query.model_validate({"select": {"action": "dancing"}})


def test_unknown_relation_rejected():
    with pytest.raises(ValidationError):
        Query.model_validate({"select": {}, "relation": ["sometime"], "to": {}})


def test_relation_requires_to_and_vice_versa():
    with pytest.raises(ValidationError):
        Query.model_validate({"select": {}, "relation": ["during"]})
    with pytest.raises(ValidationError):
        Query.model_validate({"select": {}, "to": {"action": "phone"}})


def test_extra_keys_rejected():
    with pytest.raises(ValidationError):
        Query.model_validate({"select": {}, "answer": "35 seconds"})


def test_unknown_person_gives_clear_error(csv):
    with pytest.raises(QueryError, match="unknown person 'S9'"):
        run_query(csv, {"select": {"person": "S9"}})


def test_person_is_case_insensitive(csv):
    res = run_query(csv, {"select": {"person": "s3"}})
    assert [e["Person_ID"] for e in res.events] == ["S3"]


def test_missing_columns_error(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"a": [1]}).to_csv(p, index=False)
    with pytest.raises(QueryError, match="missing columns"):
        load_events(p)
