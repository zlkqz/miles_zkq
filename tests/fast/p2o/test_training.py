import copy
from argparse import Namespace

import pytest
import torch
from examples.seta.p2o.training import CRMConfig, _align_response, attach_training_context, crm_weights, score_contexts
from tests.fast.backends.training_utils.loss.loss_test_utils import make_parallel_state
from tests.fast.backends.training_utils.test_true_on_policy_loss_metrics import _make_args

from miles.backends.training_utils.data.context_parallel import assemble_log_prob_from_cp, slice_log_prob_with_cp
from miles.backends.training_utils.loss.hub import losses
from miles.backends.training_utils.parallel import GroupInfo, get_parallel_state, set_parallel_state
from miles.utils.types import Sample


@pytest.fixture(autouse=True)
def single_rank():
    from miles.backends.training_utils import parallel

    previous = parallel._parallel_state
    make_parallel_state()
    yield
    parallel._parallel_state = previous


def args():
    result = _make_args(use_rollout_logprobs=False)
    result.loss_type = "policy_loss"
    result.advantage_estimator = "grpo"
    result.use_tis = True
    result.tis_clip_low = 0.0
    result.tis_clip = 2.0
    result.custom_tis_function_path = "miles.backends.training_utils.loss.hub.corrections.icepop_function"
    result.compute_advantages_and_returns = True
    result.eps_clip_high = 0.28
    result.p2o_crm_lower = 0.01
    result.p2o_crm_upper = 10.0
    return result


def sample(templated=True):
    result = Sample(
        tokens=[10, 11, 12, 20, 30, 21],
        response_length=3,
        loss_mask=[1, 0, 1],
        rollout_log_probs=[-0.2, 0.0, -0.3],
        status=Sample.Status.COMPLETED,
        metadata={"p2o": {"templated": templated, "template_id": "t" if templated else None}},
    )
    if templated:
        result.metadata["p2o_context"] = dict(version=1, augmented_prompt_length=3, plain_prompt_token_ids=[10])
    return result


def batch(templated=True):
    s = sample(templated)
    attach_training_context(s)
    return dict(
        tokens=[torch.tensor(s.tokens)],
        total_lengths=[len(s.tokens)],
        response_lengths=[s.response_length],
        loss_masks=[torch.tensor(s.loss_mask)],
        rollout_log_probs=[torch.tensor(s.rollout_log_probs)],
        rollout_mask_sums=torch.tensor([2.0]),
        rewards=[0.4],
        rollout_ids=[7],
        metadata=[s.train_metadata],
    )


@pytest.mark.parametrize("templated", [False, True])
def test_dual_scoring_keeps_actual_actions_behavior_reward_and_original_masks(templated):
    data = batch(templated)
    before = copy.deepcopy(data)
    calls = []

    def score(value, prefix):
        calls.append((prefix, value["tokens"][0].clone()))
        return {f"{prefix}log_probs": [torch.tensor([-0.4, -3.0, -1.0]) + (0 if prefix else 0.1)]}

    score_contexts(args(), data, score=score)
    assert [c[0] for c in calls] == ["p2o_aug_", ""]
    assert calls[0][1].tolist() == before["tokens"][0].tolist()
    assert data["tokens"][0].tolist() == ([10, 20, 30, 21] if templated else sample(False).tokens)
    assert data["tokens"][0][-3:].tolist() == before["tokens"][0][-3:].tolist()
    assert data["loss_masks"] is before["loss_masks"] or torch.equal(data["loss_masks"][0], before["loss_masks"][0])
    torch.testing.assert_close(data["rollout_log_probs"][0], before["rollout_log_probs"][0])
    torch.testing.assert_close(data["importance_log_probs"][0], torch.tensor([-0.4, -3.0, -1.0]))
    torch.testing.assert_close(data["log_probs"][0], torch.tensor([-0.3, -2.9, -0.9]))
    assert data["rollout_mask_sums"].tolist() == [2] and data["rollout_ids"] == [7] and data["rewards"] == [0.4]
    assert all(not x.requires_grad for x in data["pg_loss_weights"] + data["importance_log_probs"])


def test_crm_inclusive_log_bounds_detached_and_plain_always_one():
    ratios = torch.tensor([0.0099, 0.01, 0.0101, 1.0, 10.0, 10.01], dtype=torch.float64)
    plain = ratios.log().requires_grad_()
    augmented = torch.zeros_like(plain, requires_grad=True)
    weights = crm_weights(plain, augmented, templated=True, config=CRMConfig())
    assert weights.tolist() == [0, 1, 1, 1, 1, 0] and not weights.requires_grad
    assert crm_weights(plain, augmented, templated=False, config=CRMConfig()).tolist() == [1] * 6


def test_combined_native_loss_uses_three_distinct_ratios_and_original_denominator(monkeypatch):
    old = torch.tensor([0.12, 0.0005, 0.12, 0.1, 0.1, 0.1], dtype=torch.float64).log().requires_grad_()
    aug = torch.tensor([0.15, 0.15, 0.3, 0.15, 0.15, 0.15], dtype=torch.float64).log().requires_grad_()
    behavior = torch.full_like(old, 0.1).log().requires_grad_()
    current = torch.tensor([0.132, 0.0005, 0.12, 0.14, 0.07, 0.1], dtype=torch.float64).log().requires_grad_()
    weights = crm_weights(old, aug, templated=True, config=CRMConfig()).requires_grad_()
    mask = torch.tensor([1, 1, 1, 1, 1, 0], dtype=torch.float64)
    data = dict(
        advantages=[torch.tensor([1, 1, 1, 1, -1, 10.0], requires_grad=True)],
        log_probs=[old],
        importance_log_probs=[aug],
        rollout_log_probs=[behavior],
        pg_loss_weights=[weights],
        total_lengths=[7],
        response_lengths=[6],
        loss_masks=[mask],
        rollout_mask_sums=torch.tensor([5.0]),
        unconcat_tokens=[torch.arange(7)],
    )
    monkeypatch.setattr(losses, "get_log_probs_and_entropy", lambda *a, **k: {"log_probs": [current]})
    loss, metrics = losses.policy_loss_function(args(), data, torch.zeros(1), lambda value: (value * mask).sum() / 5)
    # token 0: 1.1 * 1.5; token 1: CRM rejected; token 2: IcePop rejected;
    # token 3: positive PPO capped at 1.28; token 4: negative capped at .8; tool excluded.
    assert loss.item() == pytest.approx((-1.65 - 1.92 + 1.2) / 5)
    loss.backward()
    torch.testing.assert_close(current.grad, torch.tensor([-0.33, 0, 0, 0, 0, 0], dtype=torch.float64))
    assert all(value.grad is None for value in [old, aug, behavior, weights, data["advantages"][0]])
    assert metrics["pg_loss_weight"].item() == pytest.approx(0.8)
    assert data["loss_masks"][0].tolist() == [1, 1, 1, 1, 1, 0]


@pytest.mark.parametrize("zero_kind", ["crm", "advantage", "original_mask"])
def test_zero_loss_is_finite_and_zero_gradient(monkeypatch, zero_kind):
    current = torch.tensor([-0.2, -0.3], requires_grad=True)
    mask = torch.zeros(2) if zero_kind == "original_mask" else torch.ones(2)
    data = dict(
        advantages=[torch.zeros(2) if zero_kind == "advantage" else torch.ones(2)],
        log_probs=[current.detach().clone()],
        importance_log_probs=[current.detach().clone()],
        rollout_log_probs=[current.detach().clone()],
        pg_loss_weights=[torch.zeros(2) if zero_kind == "crm" else torch.ones(2)],
        total_lengths=[3],
        response_lengths=[2],
        loss_masks=[mask],
        rollout_mask_sums=mask.sum().reshape(1),
        unconcat_tokens=[torch.arange(3)],
    )
    monkeypatch.setattr(losses, "get_log_probs_and_entropy", lambda *a, **k: {"log_probs": [current]})
    loss, _ = losses.policy_loss_function(
        args(), data, torch.zeros(1), lambda value: (value * mask).sum() / mask.sum().clamp_min(1)
    )
    assert torch.isfinite(loss) and loss.item() == 0
    loss.backward()
    assert current.grad.tolist() == [0, 0]


def test_untemplated_loss_and_gradients_are_exact_native_baseline(monkeypatch):
    def run(guided):
        current = torch.tensor([-0.2, -0.8], requires_grad=True)
        old = torch.tensor([-0.3, -0.7])
        behavior = torch.tensor([-0.4, -0.6])
        data = dict(
            advantages=[torch.tensor([1.0, -1.0])],
            log_probs=[old],
            rollout_log_probs=[behavior],
            total_lengths=[3],
            response_lengths=[2],
            loss_masks=[torch.ones(2)],
            rollout_mask_sums=torch.tensor([2.0]),
            unconcat_tokens=[torch.arange(3)],
        )
        if guided:
            data.update(importance_log_probs=[old.clone()], pg_loss_weights=[torch.ones(2)])
        monkeypatch.setattr(losses, "get_log_probs_and_entropy", lambda *a, **k: {"log_probs": [current]})
        loss, _ = losses.policy_loss_function(args(), data, torch.zeros(1), lambda value: value.mean())
        loss.backward()
        return loss.detach(), current.grad

    baseline, p2o = run(False), run(True)
    assert torch.equal(baseline[0], p2o[0]) and torch.equal(baseline[1], p2o[1])


@pytest.mark.parametrize("cp_size", [2, 4])
@pytest.mark.parametrize("qkv_format", ["thd", "bshd"])
def test_changed_prefix_repartitions_cp_scores_without_losing_any_response_token(monkeypatch, cp_size, qkv_format):
    state = get_parallel_state()
    full = torch.arange(11, dtype=torch.float32) / 10
    augmented = dict(total_lengths=[24], response_lengths=[11])
    plain = dict(total_lengths=[15], response_lengths=[11])
    if qkv_format == "bshd":
        augmented["max_seq_lens"] = plain["max_seq_lens"] = [32]
    outputs = {}
    for rank in range(cp_size):
        state.cp = GroupInfo(rank=rank, size=cp_size, group=None)
        local = slice_log_prob_with_cp(full, 24, 11, qkv_format, 32 if qkv_format == "bshd" else None)

        def gather(value, *a):
            torch.testing.assert_close(value, local)
            return full.clone()

        monkeypatch.setattr("examples.seta.p2o.training.all_gather_with_cp", gather)
        outputs[rank] = _align_response([local], augmented, plain, Namespace(qkv_format=qkv_format))[0]
    actual = assemble_log_prob_from_cp(outputs, 15, 11, cp_size, qkv_format, 32 if qkv_format == "bshd" else None)
    assert torch.equal(actual, full)


@pytest.mark.parametrize("balance", [False, True])
def test_native_dp_partition_keeps_context_with_its_episode(balance):
    from miles.ray.rollout.train_data_conversion import process_rollout_data_shard, split_train_data_by_dp_raw

    data = dict(
        tokens=[list(range(n)) for n in (8, 13, 5, 10)],
        metadata=[{"p2o": {"plain_prompt_token_ids": [100 + i], "episode": i}} for i in range(4)],
        sample_indices=list(range(4)),
    )
    shards = split_train_data_by_dp_raw(Namespace(balance_data=balance), data, dp_size=2)
    seen = []
    for shard in shards:
        shard = process_rollout_data_shard(None, shard)
        for index, meta, tokens in zip(shard["sample_indices"], shard["metadata"], shard["tokens"], strict=True):
            assert meta["p2o"] == {"plain_prompt_token_ids": [100 + index], "episode": index}
            assert tokens == data["tokens"][index]
            seen.append(index)
    assert sorted(seen) == list(range(4))
