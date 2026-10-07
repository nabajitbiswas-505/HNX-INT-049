"""Turn per-frame perception rows into timestamped behaviour events.

Input columns : Timestamp_Sec, Person_ID, Seat_Zone, Head_Pitch, Head_Yaw,
                Wrist_Above_Shoulder, Motion_Level, Phone_Near
Output columns: see OUTPUT_COLUMNS.

Thresholds below were tuned on perception_2.csv only (see README scope note).

Time convention
---------------
Start_Sec / End_Sec are the first and last *sample* where the action held, so a
single-sample event has Duration_Sec 0. End_Exclusive = End_Sec + sample interval
is the moment the next sample would have been taken, i.e. the latest the action
could have ended.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HAND_MIN_SEC = 1.0
PHONE_MIN_SEC = 3.0
SLEEP_MIN_SEC = 20.0
TALK_MIN_SEC = 3.0
MIN_DURATION = {
    "phone": PHONE_MIN_SEC,
    "hand_raise": HAND_MIN_SEC,
    "sleeping": SLEEP_MIN_SEC,
    "talking": TALK_MIN_SEC,
}

PHONE_PITCH = 25
SLEEP_PITCH = 40
SLEEP_MOTION = 0.1
TALK_YAW = 25

GAP_TOLERANCE_SEC = 2.0

# Priority order when several conditions hold in the same frame.
ACTION_PRIORITY = ["phone", "hand_raise", "sleeping", "talking"]

OUTPUT_COLUMNS = [
    "Event_ID", "Person_ID", "Action", "Start_Sec", "End_Sec", "End_Exclusive",
    "Duration_Sec", "Seat_Zone", "Confidence", "Overlaps",
]


def add_flags_and_margins(df: pd.DataFrame) -> pd.DataFrame:
    """Add one boolean flag (is_*), one margin (m_*, 0-1) and Raw_Action.

    Flags are independent of each other, so overlapping behaviours can be reported.
    A margin says how far past its threshold the reading is (0 = barely, 1 = far).
    """
    df = df.copy()
    phone_yes = df["Phone_Near"] == "yes"

    df["is_phone"] = phone_yes & (df["Head_Pitch"] >= PHONE_PITCH)
    df["is_hand_raise"] = df["Wrist_Above_Shoulder"] == "yes"
    df["is_sleeping"] = (
        (df["Head_Pitch"] >= SLEEP_PITCH) & (df["Motion_Level"] < SLEEP_MOTION) & ~phone_yes
    )
    df["is_talking"] = df["Head_Yaw"].abs() >= TALK_YAW

    df["m_phone"] = ((df["Head_Pitch"] - PHONE_PITCH) / PHONE_PITCH).clip(0, 1)
    df["m_hand_raise"] = 0.5  # binary signal: no margin to measure, use a neutral value
    df["m_sleeping"] = (
        0.5 * (df["Head_Pitch"] - SLEEP_PITCH) / SLEEP_PITCH
        + 0.5 * (SLEEP_MOTION - df["Motion_Level"]) / SLEEP_MOTION
    ).clip(0, 1)
    df["m_talking"] = ((df["Head_Yaw"].abs() - TALK_YAW) / TALK_YAW).clip(0, 1)

    raw = np.full(len(df), "attentive", dtype=object)
    for act in reversed(ACTION_PRIORITY):  # reversed so the highest priority wins
        raw[df[f"is_{act}"].to_numpy()] = act
    df["Raw_Action"] = raw
    return df


def sample_interval(df: pd.DataFrame) -> float:
    """Median spacing between distinct timestamps (0.5 s for the sample data)."""
    ts = np.sort(df["Timestamp_Sec"].unique())
    return float(np.median(np.diff(ts))) if len(ts) > 1 else 0.0


def events_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """Convert a perception DataFrame into an events DataFrame (always has OUTPUT_COLUMNS)."""
    if df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    dt = sample_interval(df)
    df = add_flags_and_margins(df)
    rows = []

    for person in df["Person_ID"].unique():
        person_df = df[df["Person_ID"] == person].sort_values("Timestamp_Sec")
        seat_zone = person_df["Seat_Zone"].iloc[0]

        for action in ACTION_PRIORITY:
            hits = person_df[person_df["Raw_Action"] == action]
            if hits.empty:
                continue

            times = hits["Timestamp_Sec"].to_numpy()
            split = np.where(np.diff(times) > GAP_TOLERANCE_SEC)[0] + 1
            for group in np.split(times, split):
                start, end = float(group[0]), float(group[-1])
                duration = end - start
                if duration < MIN_DURATION[action]:
                    continue

                interval = person_df[person_df["Timestamp_Sec"].between(start, end)]
                matched = interval[interval["Raw_Action"] == action]

                # Confidence = coverage * (0.5 + 0.5 * mean margin past threshold)
                #   coverage: fraction of frames inside [start, end] that matched
                #             (bridged gaps lower it)
                #   margin  : how clearly the readings exceed the threshold
                coverage = len(matched) / len(interval)
                margin = float(matched[f"m_{action}"].mean())
                confidence = coverage * (0.5 + 0.5 * margin)

                overlaps = [
                    other for other in ACTION_PRIORITY
                    if other != action and bool(interval[f"is_{other}"].any())
                ]
                rows.append({
                    "Person_ID": person,
                    "Action": action,
                    "Start_Sec": start,
                    "End_Sec": end,
                    "End_Exclusive": end + dt,
                    "Duration_Sec": duration,
                    "Seat_Zone": seat_zone,
                    "Confidence": round(float(confidence), 2),
                    "Overlaps": ";".join(overlaps),
                })

    if not rows:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    out = pd.DataFrame(rows).sort_values(["Start_Sec", "Person_ID"]).reset_index(drop=True)
    out.insert(0, "Event_ID", range(1, len(out) + 1))
    return out[OUTPUT_COLUMNS]


def build_events(input_path, output_path) -> pd.DataFrame:
    events = events_from_df(pd.read_csv(input_path))
    events.to_csv(output_path, index=False)  # headers are written even when empty
    return events


if __name__ == "__main__":
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "data/perception_2.csv")
    dst = Path(sys.argv[2] if len(sys.argv) > 2 else "data/generated_events_2.csv")
    result = build_events(src, dst)
    print(f"Wrote {len(result)} events to {dst}")
    print(result.to_string(index=False))
