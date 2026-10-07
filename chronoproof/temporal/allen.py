"""Allen's 13 interval relations with an epsilon tolerance.

An interval is a (start, end) tuple with start <= end. Two time points that differ
by at most `eps` seconds are treated as equal, which absorbs sampling jitter
(the perception data is sampled every 0.5 s).

relation(a, b) returns exactly one name describing how `a` relates to `b`.
"""
from __future__ import annotations

from typing import Callable

Interval = tuple[float, float]

ALL_RELATIONS = [
    "before", "after", "meets", "met_by", "overlaps", "overlapped_by",
    "starts", "started_by", "during", "contains", "finishes", "finished_by", "equals",
]

# Convenient groups used by the query layer.
RELATION_GROUPS: dict[str, list[str]] = {
    "precedes": ["before", "meets"],            # a ends before / as b starts
    "follows": ["after", "met_by"],             # a starts after / as b ends
    "concurrent": [                              # a and b share some time
        "overlaps", "overlapped_by", "starts", "started_by",
        "during", "contains", "finishes", "finished_by", "equals",
    ],
}


def _cmp(x: float, y: float, eps: float) -> int:
    """-1 if x is clearly before y, 0 if they are within eps, +1 if clearly after."""
    if abs(x - y) <= eps:
        return 0
    return -1 if x < y else 1


def _check(a: Interval, b: Interval) -> None:
    for name, (s, e) in (("a", a), ("b", b)):
        if s > e:
            raise ValueError(f"interval {name} has start > end: {(s, e)}")


def relation(a: Interval, b: Interval, eps: float = 0.25) -> str:
    """Return the single Allen relation of a with respect to b."""
    _check(a, b)
    (a0, a1), (b0, b1) = a, b

    end_vs_start = _cmp(a1, b0, eps)   # a's end against b's start
    if end_vs_start < 0:
        return "before"
    if end_vs_start == 0 and _cmp(a0, b0, eps) != 0:
        return "meets"

    start_vs_end = _cmp(a0, b1, eps)   # a's start against b's end
    if start_vs_end > 0:
        return "after"
    if start_vs_end == 0 and _cmp(a1, b1, eps) != 0:
        return "met_by"

    ss, ee = _cmp(a0, b0, eps), _cmp(a1, b1, eps)
    table = {
        (0, 0): "equals",
        (0, -1): "starts",
        (0, 1): "started_by",
        (1, 0): "finishes",
        (-1, 0): "finished_by",
        (1, -1): "during",
        (-1, 1): "contains",
        (-1, -1): "overlaps",
        (1, 1): "overlapped_by",
    }
    return table[(ss, ee)]


def _make(name: str) -> Callable[..., bool]:
    def fn(a: Interval, b: Interval, eps: float = 0.25) -> bool:
        return relation(a, b, eps) == name
    fn.__name__ = name
    fn.__doc__ = f"True if a {name.replace('_', ' ')} b."
    return fn


before = _make("before")
after = _make("after")
meets = _make("meets")
met_by = _make("met_by")
overlaps = _make("overlaps")
overlapped_by = _make("overlapped_by")
starts = _make("starts")
started_by = _make("started_by")
during = _make("during")
contains = _make("contains")
finishes = _make("finishes")
finished_by = _make("finished_by")
equals = _make("equals")


def expand(names: list[str]) -> set[str]:
    """Expand group names (precedes, follows, concurrent) into concrete relations."""
    out: set[str] = set()
    for n in names:
        if n in RELATION_GROUPS:
            out.update(RELATION_GROUPS[n])
        elif n in ALL_RELATIONS:
            out.add(n)
        else:
            raise ValueError(
                f"unknown relation '{n}'. Allowed: {ALL_RELATIONS + list(RELATION_GROUPS)}"
            )
    return out
