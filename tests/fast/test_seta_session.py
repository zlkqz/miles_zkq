"""SETA validity must survive v2's server-side reward assignment and leaf merging."""

from copy import deepcopy

import pytest

from examples.seta.agent import postprocess_session_samples, reward_func
from miles.utils.types import Sample


def _leaf(node_id, path_node_ids):
    return Sample(
        group_index=3,
        index=7,
        rollout_id=7,
        status=Sample.Status.COMPLETED,
        tokens=[1, 10, 20, 30],
        response_length=3,
        loss_mask=[1, 0, 1],
        rollout_log_probs=[-0.1, 0.0, -0.2],
        metadata={"leaf": {"node_id": node_id, "path_node_ids": path_node_ids}},
    )


@pytest.mark.parametrize(
    ("agent_metadata", "reward", "status"),
    [
        ({"seta_valid_reward": True, "reward": 1.2}, 1.2, Sample.Status.COMPLETED),
        ({"seta_valid_reward": True, "reward": 0.0}, 0.0, Sample.Status.COMPLETED),
        ({"seta_valid_reward": True, "reward": 0.4}, 0.4, Sample.Status.COMPLETED),
        ({"seta_valid_reward": False, "reward": 0.0}, 0.0, Sample.Status.ABORTED),
        ({"seta_valid_reward": False, "reward": 1.2}, 0.0, Sample.Status.ABORTED),
        ({"reward": 0.0}, 0.0, Sample.Status.ABORTED),
        ({}, 0.0, Sample.Status.ABORTED),
    ],
)
async def test_v2_applies_the_same_validity_contract_as_the_reward_hook(agent_metadata, reward, status):
    original = _leaf(1, [0, 1])
    baseline = deepcopy(original)
    baseline.metadata.update(agent_metadata)
    metadata = {"agent": agent_metadata, "tree": {"nodes": [{"id": 0}, {"id": 1}]}}
    (result,) = postprocess_session_samples([original], metadata)
    assert result.reward == await reward_func(None, baseline) == reward
    assert result.status == baseline.status == status
    assert (result.group_index, result.rollout_id) == (3, 7)
    assert result.tokens == baseline.tokens
    assert result.loss_mask == baseline.loss_mask
    assert result.rollout_log_probs == baseline.rollout_log_probs


def test_v2_keeps_shared_actions_trainable_once_and_reward_at_episode_scope():
    first, second = _leaf(1, [0, 1]), _leaf(2, [0, 2])
    second.tokens[-1] = 40
    metadata = {
        "agent": {"seta_valid_reward": True, "reward": 1.2, "remote_request_id": "episode-7"},
        "tree": {"nodes": [{"id": 0, "completion_span": [1, 2]}, {"id": 1}, {"id": 2}]},
    }
    results = postprocess_session_samples([second, first], metadata)
    assert results == [second, first]
    assert first.loss_mask == [1, 0, 1]
    assert second.loss_mask == [0, 0, 1]
    assert [sample.reward for sample in results] == [1.2, 1.2]
    assert [sample.rollout_id for sample in results] == [7, 7]
    assert [sample.metadata["remote_request_id"] for sample in results] == ["episode-7", "episode-7"]
