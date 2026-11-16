"""Deterministic molecule-group splits and leakage checks (labels are not used)."""
from __future__ import annotations

from collections import Counter, defaultdict
from random import Random
from typing import Any

SPLITS = ("train", "validation", "test")


class SplitError(ValueError):
    """Insufficient groups or leakage detected during data partitioning."""


def make_group_split(groups: list[str], *, ratios: dict[str, float], seed: int) -> list[str]:
    """Assign intact groups to approximately 80/10/10 row fractions.

    Randomized order within each group size is seeded. The assignment uses
    only group membership and size, not labels. Exact row fractions cannot
    always be attained when a scaffold group is larger than its target bin.
    """
    if not groups or any(not isinstance(g, str) or not g for g in groups):
        raise SplitError("Groups must be non-empty strings")
    if set(ratios) != set(SPLITS) or any(
        not isinstance(ratios[k], (int, float)) or not 0 < ratios[k] < 1 for k in SPLITS
    ) or abs(sum(ratios.values()) - 1.0) > 1e-8:
        raise SplitError("Split ratios must be positive and sum to one")
    members: dict[str, list[int]] = defaultdict(list)
    for i, key in enumerate(groups):
        members[key].append(i)
    if len(members) < 3:
        raise SplitError("At least three distinct groups are required for three partitions")
    rng = Random(seed)
    keys = sorted(members)
    rng.shuffle(keys)
    # Largest groups first: group-size balancing without using targets.
    keys.sort(key=lambda key: -len(members[key]))
    counts = {part: 0 for part in SPLITS}
    desired = {part: ratios[part] * len(groups) for part in SPLITS}
    allocation: dict[str, str] = {}
    for index, key in enumerate(keys):
        remaining_groups = len(keys) - index
        empty = [part for part in SPLITS if counts[part] == 0]
        options = empty if remaining_groups <= len(empty) else list(SPLITS)
        # Choose the smallest prospective occupancy relative to quota.
        # This yields near-target sizes while avoiding overly small partitions.
        target = min(options, key=lambda part: ((counts[part] + len(members[key])) / desired[part], SPLITS.index(part)))
        allocation[key] = target
        counts[target] += len(members[key])
    result = [allocation[group] for group in groups]
    if any(counts[p] == 0 for p in SPLITS):
        raise SplitError(f"Could not make three non-empty splits: {counts}")
    return result


def audit_assignments(canonical: list[str], scaffolds: list[str], assignments: list[str]) -> dict[str, Any]:
    if not (len(canonical) == len(scaffolds) == len(assignments)) or not canonical:
        raise SplitError("Split arrays are missing or have different lengths")
    if any(s not in SPLITS for s in assignments):
        raise SplitError("Unknown split label")
    grouped: dict[str, set[str]] = defaultdict(set)
    scaffold_grouped: dict[str, set[str]] = defaultdict(set)
    for smi, scaffold, part in zip(canonical, scaffolds, assignments):
        grouped[smi].add(part)
        scaffold_grouped[scaffold].add(part)
    duplicate_overlap = sum(len(x) > 1 for x in grouped.values())
    scaffold_overlap = sum(len(x) > 1 for x in scaffold_grouped.values())
    if duplicate_overlap:
        raise SplitError(f"Canonical molecule leakage detected in {duplicate_overlap} groups")
    counts = Counter(assignments)
    if any(counts.get(split, 0) == 0 for split in SPLITS):
        raise SplitError("A split is empty")
    return {
        "rows": {split: counts[split] for split in SPLITS},
        "fraction": {split: round(counts[split] / len(assignments), 6) for split in SPLITS},
        "distinct_canonical": len(grouped),
        "distinct_scaffolds": len(scaffold_grouped),
        "canonical_groups_crossing_partitions": duplicate_overlap,
        "scaffolds_crossing_partitions": scaffold_overlap,
        "scaffold_groups_per_partition": {
            part: len({scaffolds[i] for i, s in enumerate(assignments) if s == part})
            for part in SPLITS
        },
    }
