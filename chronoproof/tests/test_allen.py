import pytest

from temporal import allen

B = (10.0, 20.0)

# (relation name, an interval `a` that has that relation to B)
CASES = [
    ("before", (1.0, 5.0)),
    ("after", (25.0, 30.0)),
    ("meets", (5.0, 10.0)),
    ("met_by", (20.0, 25.0)),
    ("overlaps", (5.0, 15.0)),
    ("overlapped_by", (15.0, 25.0)),
    ("starts", (10.0, 15.0)),
    ("started_by", (10.0, 25.0)),
    ("during", (12.0, 18.0)),
    ("contains", (5.0, 25.0)),
    ("finishes", (15.0, 20.0)),
    ("finished_by", (5.0, 20.0)),
    ("equals", (10.0, 20.0)),
]


def test_there_are_13_relations():
    assert len(allen.ALL_RELATIONS) == 13
    assert {n for n, _ in CASES} == set(allen.ALL_RELATIONS)


@pytest.mark.parametrize("name,a", CASES)
def test_relation_name(name, a):
    assert allen.relation(a, B, eps=0.0) == name


@pytest.mark.parametrize("name,a", CASES)
def test_named_predicate_true_and_exclusive(name, a):
    assert getattr(allen, name)(a, B, eps=0.0) is True
    for other in allen.ALL_RELATIONS:
        if other != name:
            assert getattr(allen, other)(a, B, eps=0.0) is False


INVERSE = {
    "before": "after", "meets": "met_by", "overlaps": "overlapped_by",
    "starts": "started_by", "during": "contains", "finishes": "finished_by",
    "equals": "equals",
}
INVERSE.update({v: k for k, v in list(INVERSE.items())})


@pytest.mark.parametrize("name,a", CASES)
def test_swapping_arguments_gives_inverse(name, a):
    assert allen.relation(B, a, eps=0.0) == INVERSE[name]


# ---- epsilon edge cases -------------------------------------------------

def test_gap_within_eps_counts_as_meets():
    assert allen.relation((5.0, 9.9), B, eps=0.25) == "meets"


def test_gap_exactly_eps_counts_as_meets():
    assert allen.relation((5.0, 9.75), B, eps=0.25) == "meets"


def test_gap_just_over_eps_is_before():
    assert allen.relation((5.0, 9.7), B, eps=0.25) == "before"


def test_small_overlap_within_eps_counts_as_meets():
    assert allen.relation((5.0, 10.2), B, eps=0.25) == "meets"


def test_starts_within_eps_counts_as_starts():
    assert allen.relation((10.2, 15.0), B, eps=0.25) == "starts"


def test_nearly_equal_intervals_are_equal():
    assert allen.relation((10.1, 19.9), B, eps=0.25) == "equals"


def test_zero_eps_is_strict():
    assert allen.relation((5.0, 10.1), B, eps=0.0) == "overlaps"


def test_point_interval_inside():
    assert allen.relation((15.0, 15.0), B, eps=0.0) == "during"


def test_invalid_interval_raises():
    with pytest.raises(ValueError):
        allen.relation((5.0, 1.0), B)


# ---- groups ---------------------------------------------------------------

def test_expand_groups():
    assert allen.expand(["precedes"]) == {"before", "meets"}
    assert allen.expand(["follows"]) == {"after", "met_by"}
    assert len(allen.expand(["concurrent"])) == 9
    assert allen.expand(["during", "precedes"]) == {"during", "before", "meets"}


def test_expand_unknown_raises():
    with pytest.raises(ValueError):
        allen.expand(["sometime"])


def test_groups_partition_all_relations():
    union = set()
    for g in allen.RELATION_GROUPS.values():
        assert not (union & set(g))
        union |= set(g)
    assert union == set(allen.ALL_RELATIONS)
