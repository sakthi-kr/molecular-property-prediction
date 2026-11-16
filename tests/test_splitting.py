"""Property labels must not influence the fixed molecule-group assignments."""
from __future__ import annotations

import pytest

from molprop.splitting import SplitError, audit_assignments, make_group_split

RATIOS = {"train": 0.8, "validation": 0.1, "test": 0.1}


def test_canonical_duplicates_are_kept_together_and_seed_repeats():
    groups = [f"molecule-{n // 2}" for n in range(60)]
    result = make_group_split(groups, ratios=RATIOS, seed=2026)
    assert result == make_group_split(groups, ratios=RATIOS, seed=2026)
    assert {s for s in result} == {"train", "validation", "test"}
    audit = audit_assignments(groups, groups, result)
    assert audit["canonical_groups_crossing_partitions"] == 0
    assert audit["scaffolds_crossing_partitions"] == 0
    assert sum(audit["rows"].values()) == 60


def test_scaffold_split_keeps_entire_scaffolds_together():
    canonical = [f"mol-{i}" for i in range(90)]
    scaffolds = [f"scaffold-{i//3}" for i in range(90)]
    assigned = make_group_split(scaffolds, ratios=RATIOS, seed=7)
    report = audit_assignments(canonical, scaffolds, assigned)
    assert report["scaffolds_crossing_partitions"] == 0
    assert report["canonical_groups_crossing_partitions"] == 0
    assert abs(report["rows"]["train"] / 90 - 0.8) < 0.10


def test_splits_independent_of_reordered_target_labels():
    group_ids = [f"scaffold-{i//2}" for i in range(40)]
    targets_a = list(range(40))
    targets_b = targets_a[::-1]
    # No target input to this API by design.
    assert targets_a != targets_b
    assert make_group_split(group_ids, ratios=RATIOS, seed=2026) == make_group_split(group_ids, ratios=RATIOS, seed=2026)


def test_leakage_is_detected():
    with pytest.raises(SplitError, match="Canonical molecule leakage"):
        audit_assignments(["A", "A", "B", "C"], ["S1", "S1", "S2", "S3"],
                          ["train", "test", "validation", "train"])
    good = audit_assignments(["A", "B", "C"], ["S1", "S1", "S2"],
                             ["train", "validation", "test"])
    assert good["scaffolds_crossing_partitions"] == 1


def test_too_few_groups_rejected():
    with pytest.raises(SplitError, match="three distinct"):
        make_group_split(["A", "A", "B"], ratios=RATIOS, seed=1)
    with pytest.raises(SplitError, match="sum to one"):
        make_group_split(["A", "B", "C"], ratios={"train": 0.9, "validation": 0.1, "test": 0.1}, seed=1)
