import json

import pytest
from examples.seta.p2o.evidence import EpisodeEvidence, TaskKey
from examples.seta.p2o.scoring import CollectionConfig, DevScores, aggregate_dev, aggregate_dev_task, classify_hard
from pydantic import ValidationError


@pytest.mark.parametrize(("full_passes", "expected"), [(0, True), (1, True), (2, False)])
def test_hard_boundary_excludes_bonus(make_episode, make_observation, full_passes, expected):
    episodes = tuple(make_episode(i, passed=16 if i < full_passes else 0) for i in range(16))
    result = classify_hard(make_observation(episodes=episodes), CollectionConfig(n_samples_per_prompt=16))
    assert result.is_hard is expected
    assert result.progress_mean == full_passes / 16
    assert result.reward_mean == pytest.approx(full_passes * 1.2 / 16)


def test_continuous_progress_is_not_binary_success(make_observation):
    result = classify_hard(make_observation(passed=2), CollectionConfig(n_samples_per_prompt=16))
    assert result.progress_mean == 0.125 and result.is_hard is False
    assert classify_hard(make_observation(passed=1), CollectionConfig(n_samples_per_prompt=16)).is_hard


def test_threshold_follows_k_or_explicit_override(make_observation):
    config = CollectionConfig(n_samples_per_prompt=4)
    assert config.threshold == 0.25
    assert classify_hard(make_observation(count=4, passed=4), config).is_hard
    strict = CollectionConfig(n_samples_per_prompt=4, hard_reward_threshold=0)
    assert not classify_hard(make_observation(count=4, passed=1), strict).is_hard
    assert classify_hard(
        make_observation(passed=16), CollectionConfig(n_samples_per_prompt=16, hard_reward_threshold=1)
    ).is_hard


@pytest.mark.parametrize("threshold", [-0.01, 1.01, float("nan"), float("inf")])
def test_invalid_threshold_rejected(threshold):
    with pytest.raises(ValidationError):
        CollectionConfig(n_samples_per_prompt=16, hard_reward_threshold=threshold)


@pytest.mark.parametrize("count", [0, 15, 17])
def test_partial_or_oversized_training_batch_is_not_hard(make_observation, count):
    assert classify_hard(make_observation(count=count), CollectionConfig(n_samples_per_prompt=16)).is_hard is None


def test_one_invalid_episode_invalidates_hard_decision(make_episode, make_observation):
    episodes = tuple(make_episode(i, valid=i != 15) for i in range(16))
    result = classify_hard(make_observation(episodes=episodes), CollectionConfig(n_samples_per_prompt=16))
    assert result.is_hard is None and result.progress_mean is None


@pytest.mark.parametrize("valid_count", [0, 1, 2, 3, 4])
def test_dev_averages_only_valid_terminal_attempts(task, make_episode, valid_count):
    episodes = tuple(make_episode(i, passed=i + 1, total=4, valid=i < valid_count) for i in range(4))
    score = aggregate_dev_task(task, episodes, expected_repeats=4)
    values = [0.25, 0.5, 0.75, 1.2][:valid_count]
    assert score.reward_mean == pytest.approx(sum(values) / valid_count if valid_count else 0)
    assert score.attempted_count == 4 and score.valid_count == valid_count and score.invalid_count == 4 - valid_count
    assert score.all_invalid is (valid_count == 0)
    assert (score.reward_std is None) is (valid_count < 2)
    assert len([e.error for e in score.episodes if not e.valid]) == 4 - valid_count


def test_pending_dev_attempt_is_not_an_invalid_terminal_result(task, make_episode):
    with pytest.raises(ValueError, match="terminate"):
        aggregate_dev_task(task, [make_episode(i) for i in range(3)], expected_repeats=4)


def test_duplicate_environment_attempt_cannot_count_as_repeats(task, make_episode, make_observation):
    duplicate = make_episode(0)
    with pytest.raises(ValueError, match="duplicate attempt_id"):
        aggregate_dev_task(task, [duplicate] * 4, expected_repeats=4)
    episodes = tuple(make_episode(i, remote_request_id="same-env") for i in range(16))
    with pytest.raises(ValueError, match="duplicate remote_request_id"):
        make_observation(episodes=episodes)


def test_dev_vector_preserves_task_order_and_equal_task_weight(make_episode):
    tasks = tuple(TaskKey(dataset_id="train", instance_id=str(i)) for i in range(16))
    episodes = [
        make_episode(j, task_key=t, passed=16, valid=i == 0 and j == 0) for i, t in enumerate(tasks) for j in range(4)
    ]
    result = aggregate_dev(tasks, list(reversed(episodes)), expected_repeats=4)
    assert len(result.vector) == 16 and result.vector == (1.2,) + (0,) * 15
    assert result.mean == pytest.approx(1.2 / 16)  # Not mean of all valid episodes (1.2).
    restored = DevScores.model_validate(json.loads(result.model_dump_json()))
    assert restored == result and restored.vector == result.vector
    with pytest.raises(ValueError, match="outside"):
        aggregate_dev(tasks[1:], episodes, expected_repeats=4)


def test_invalid_evidence_cannot_pretend_to_be_numeric_zero(task):
    with pytest.raises(ValidationError, match="invalid evidence"):
        EpisodeEvidence(task=task, attempt_id="bad", valid=False, reward=0, error="timeout")
