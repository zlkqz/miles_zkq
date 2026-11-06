import random
from itertools import combinations, product

import pytest

from examples.seta.p2o.mixing import MixConfig, select_training_batch


@pytest.mark.parametrize("hard", range(17))
@pytest.mark.parametrize("seed", range(8))
def test_finite_pool_selection_quota_or_exact_fifo_fallback(hard, seed):
    flags = [True] * hard + [False] * (16 - hard)
    random.Random(seed).shuffle(flags)  # Simulate different asynchronous completion orders.
    initial = tuple(range(8))
    selected = select_training_batch(templated=flags, size=8, config=MixConfig())
    assert len(set(selected.indices)) == 8
    assert selected.indices == tuple(sorted(selected.indices))
    assert selected.final_templated == sum(flags[i] for i in selected.indices)
    if hard < 2 or 16 - hard < 4:
        assert selected.indices == initial and selected.fallback
    else:
        assert 2 <= selected.final_templated <= 4 and selected.fallback is None
        if 2 <= sum(flags[i] for i in initial) <= 4:
            assert selected.indices == initial


def test_supplement_earliest_templates_and_defer_latest_plain_entries():
    flags = [False] * 20 + [True] * 10
    selected = select_training_batch(templated=flags, size=8, config=MixConfig())
    assert selected.indices == (0, 1, 2, 3, 4, 5, 20, 21)
    assert selected.replacements == 2


def test_keep_earliest_four_templates_and_supplement_earliest_plain_entries():
    flags = [True] * 12 + [False] * 6
    selected = select_training_batch(templated=flags, size=8, config=MixConfig())
    assert selected.indices == (0, 1, 2, 3, 12, 13, 14, 15)
    assert selected.replacements == 4


def test_exhaustive_small_pool_returns_lexicographically_earliest_feasible_batch():
    # Independent combinatorial oracle: earliest feasible subset, or exact FIFO.
    for flags in product((False, True), repeat=8):
        feasible = [indices for indices in combinations(range(8), 4) if 1 <= sum(flags[i] for i in indices) <= 2]
        expected = min(feasible) if feasible else (0, 1, 2, 3)
        selected = select_training_batch(templated=flags, size=4, config=MixConfig())
        assert selected.indices == expected


@pytest.mark.parametrize("size", [0, 1])
def test_impossible_batch_rejected(size):
    with pytest.raises(ValueError):
        MixConfig().bounds(size)
