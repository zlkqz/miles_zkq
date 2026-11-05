import random

import pytest

from examples.seta.p2o.mixing import MixConfig, select_training_batch


@pytest.mark.parametrize("hard", range(17))
@pytest.mark.parametrize("seed", range(8))
def test_finite_pool_selection_quota_or_exact_random_fallback(hard, seed):
    flags = [True] * hard + [False] * (16 - hard)
    initial = random.Random(seed).sample(range(16), 8)
    selected = select_training_batch(
        templated=flags, priorities=list(range(16)), size=8, config=MixConfig(), rng=random.Random(seed)
    )
    assert len(set(selected.indices)) == 8
    assert selected.final_templated == sum(flags[i] for i in selected.indices)
    if hard < 2 or 16 - hard < 4:
        assert selected.indices == tuple(initial) and selected.fallback
    else:
        assert 2 <= selected.final_templated <= 4 and selected.fallback is None
        if 2 <= sum(flags[i] for i in initial) <= 4:
            assert selected.indices == tuple(initial)


def test_newest_available_template_tasks_are_supplemented_first():
    flags = [False] * 20 + [True] * 10
    seed = next(s for s in range(100) if not any(flags[i] for i in random.Random(s).sample(range(30), 8)))
    selected = select_training_batch(
        templated=flags, priorities=list(range(30)), size=8, config=MixConfig(), rng=random.Random(seed)
    )
    assert {i for i in selected.indices if flags[i]} == {28, 29}


def test_oldest_excess_hits_are_returned_to_available_pool():
    flags = [True] * 12 + [False] * 4
    seed = next(s for s in range(100) if sum(flags[i] for i in random.Random(s).sample(range(16), 8)) > 4)
    initial = random.Random(seed).sample(range(16), 8)
    selected = select_training_batch(
        templated=flags, priorities=list(range(16)), size=8, config=MixConfig(), rng=random.Random(seed)
    )
    assert {i for i in selected.indices if flags[i]} == set(sorted(i for i in initial if flags[i])[-4:])


@pytest.mark.parametrize("size", [0, 1])
def test_impossible_batch_rejected(size):
    with pytest.raises(ValueError):
        MixConfig().bounds(size)
