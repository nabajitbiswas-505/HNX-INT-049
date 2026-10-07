import pandas as pd

from events.behavior_parser import OUTPUT_COLUMNS, build_events, events_from_df

COLS = ["Timestamp_Sec", "Person_ID", "Seat_Zone", "Head_Pitch", "Head_Yaw",
        "Wrist_Above_Shoulder", "Motion_Level", "Phone_Near"]


def make(person="S1", n=40, step=0.5, **overrides_by_index):
    """Quiet baseline rows; overrides_by_index maps sample index -> dict of changes."""
    rows = []
    for i in range(n):
        row = dict(Timestamp_Sec=i * step, Person_ID=person, Seat_Zone="R1C1",
                   Head_Pitch=3, Head_Yaw=0, Wrist_Above_Shoulder="no",
                   Motion_Level=0.3, Phone_Near="no")
        row.update(overrides_by_index.get(f"i{i}", {}))
        rows.append(row)
    return pd.DataFrame(rows, columns=COLS)


def test_matches_answer_key(tmp_path):
    out = tmp_path / "gen.csv"
    got = build_events("data/perception_2.csv", out)
    key = pd.read_csv("data/events_2.csv")
    cols = ["Person_ID", "Action", "Start_Sec", "End_Sec"]
    pd.testing.assert_frame_equal(
        got[cols].reset_index(drop=True), key[cols].reset_index(drop=True), check_dtype=False
    )


def test_empty_input_writes_headers(tmp_path):
    out = tmp_path / "empty.csv"
    empty = tmp_path / "in.csv"
    pd.DataFrame(columns=COLS).to_csv(empty, index=False)
    build_events(empty, out)
    back = pd.read_csv(out)  # must not raise
    assert list(back.columns) == OUTPUT_COLUMNS and back.empty


def test_quiet_input_gives_no_events():
    assert events_from_df(make()).empty


def test_single_frame_hand_raise_is_dropped():
    df = make(i10={"Wrist_Above_Shoulder": "yes"})
    assert events_from_df(df).empty


def test_one_sample_gap_in_phone_is_bridged():
    phone = {"Phone_Near": "yes", "Head_Pitch": 35}
    changes = {f"i{i}": phone for i in range(10, 30) if i != 20}  # gap at i=20
    ev = events_from_df(make(**changes))
    assert len(ev) == 1
    row = ev.iloc[0]
    assert row.Action == "phone" and row.Start_Sec == 5.0 and row.End_Sec == 14.5
    assert row.Confidence < 1.0  # the bridged gap lowers coverage


def test_gap_longer_than_tolerance_splits_event():
    phone = {"Phone_Near": "yes", "Head_Pitch": 35}
    changes = {f"i{i}": phone for i in list(range(4, 12)) + list(range(20, 28))}
    ev = events_from_df(make(n=40, **changes))
    assert len(ev) == 2


def test_overlaps_reported():
    # talking (yaw) while hand is raised: hand_raise wins priority, talking is listed
    chg = {f"i{i}": {"Wrist_Above_Shoulder": "yes", "Head_Yaw": 40} for i in range(5, 12)}
    ev = events_from_df(make(**chg))
    assert ev.iloc[0].Action == "hand_raise"
    assert "talking" in ev.iloc[0].Overlaps


def test_end_exclusive_uses_sample_interval():
    ev = events_from_df(pd.read_csv("data/perception_2.csv"))
    assert ((ev.End_Exclusive - ev.End_Sec) == 0.5).all()
