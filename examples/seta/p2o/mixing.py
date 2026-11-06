"""Finite, nonblocking quota adjustment over already available task batches."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class MixConfig:
    target: float = 0.25
    maximum: float = 0.5

    def __post_init__(self):
        if not 0 < self.target <= self.maximum < 1:
            raise ValueError("mix ratios require 0 < target <= maximum < 1")

    def bounds(self, size):
        if type(size) is not int or size < 2:
            raise ValueError("mixing requires an integer task batch size >= 2")
        lower, upper = math.ceil(size * self.target), math.floor(size * self.maximum)
        if lower > upper:
            raise ValueError("batch size cannot represent the configured mix interval")
        return lower, upper


@dataclass(frozen=True)
class Selection:
    indices: tuple[int, ...]
    initial_templated: int
    final_templated: int
    replacements: int
    fallback: str | None


def newest_first(indices, *, priorities, rng):
    indices = list(indices)
    rng.shuffle(indices)
    return sorted(indices, key=lambda i: priorities[i], reverse=True)


def select_training_batch(*, templated, size, config):
    """Choose the earliest feasible complete batches in buffer insertion order.

    Keep the FIFO prefix when it meets the quota or the pool cannot meet it.
    Otherwise defer its latest excess-category entries and take the earliest
    missing-category entries. Training consumption does not draw RNG or rank
    by GEPA cycle; submission-time injection still owns that priority policy.
    """
    if len(templated) < size:
        raise ValueError("selection requires a full available batch")
    lower, upper = config.bounds(size)
    initial = list(range(size))
    count = sum(templated[i] for i in initial)
    available_hard = sum(templated)
    available_plain = len(templated) - available_hard
    reason = None
    if available_hard < lower:
        reason = "insufficient_templated"
    elif available_plain < size - upper:
        reason = "insufficient_plain"
    if reason or lower <= count <= upper:
        return Selection(tuple(initial), count, count, 0, reason)
    selected = list(initial)
    outside = range(size, len(templated))
    if count < lower:
        incoming = [i for i in outside if templated[i]][: lower - count]
        outgoing = [i for i in selected if not templated[i]][-len(incoming) :]
    else:
        outgoing = [i for i in selected if templated[i]][upper:]
        incoming = [i for i in outside if not templated[i]][: len(outgoing)]
    for old, new in zip(outgoing, incoming, strict=True):
        selected[selected.index(old)] = new
    return Selection(tuple(sorted(selected)), count, sum(templated[i] for i in selected), len(incoming), None)
