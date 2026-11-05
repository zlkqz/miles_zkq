from copy import deepcopy

import pytest
from examples.seta.p2o.evidence import TaskKey, episode_from_samples
from pydantic import ValidationError

from miles.utils.types import Sample, SampleLineage, WeightVersionSpan, WeightVersionsPerCall


def _sample(index=7, **overrides):
    values = dict(
        index=index,
        group_index=3,
        status=Sample.Status.COMPLETED,
        tokens=[1, 2, 3],
        response_length=2,
        loss_mask=[0, 1],
        rollout_log_probs=[0.0, -0.2],
        reward=1.2,
        metadata={
            "instance_id": "SETA_Synth__task",
            "seta_valid_reward": True,
            "reward": 1.2,
            "eval_report": {"passed": 10, "total": 10},
            "remote_request_id": "request-1",
            "trial_dir": "/trials/1",
        },
        weight_versions=[WeightVersionsPerCall(spans=[WeightVersionSpan("0", 2, 3)])],
    )
    values.update(overrides)
    return Sample(**values)


def _read(samples):
    return episode_from_samples(samples, dataset_id="revision", attempt_id="attempt", template_id="template-a")


def test_single_leaf_uses_index_fallback_and_keeps_bonus_and_artifacts():
    sample = _sample()
    original = deepcopy(sample)
    evidence = _read([sample])
    assert evidence.valid and evidence.progress == 1 and evidence.reward == 1.2
    assert evidence.policy_versions == ("0",)
    assert evidence.trial_dir == "/trials/1" and evidence.remote_request_id == "request-1"
    assert evidence.template_id == "template-a"
    assert sample.to_dict() == original.to_dict()


def test_branched_episode_is_one_result_with_all_weight_versions():
    first = _sample(7, rollout_id=7, lineage=SampleLineage(7, 0, 2))
    second = _sample(
        8,
        rollout_id=7,
        lineage=SampleLineage(7, 1, 2),
        weight_versions=[WeightVersionsPerCall(spans=[WeightVersionSpan("1", 2, 3)])],
    )
    evidence = _read([second, first])
    assert evidence.valid and evidence.reward == 1.2 and evidence.policy_versions == ("0", "1")
    assert not _read([first]).valid
    assert not _read([first, first]).valid


@pytest.mark.parametrize(
    "mutation",
    [
        "zero_total",
        "too_many_passes",
        "missing_report",
        "nan_reward",
        "wrong_reward",
        "missing_validity",
        "aborted",
        "failed",
        "removed",
    ],
)
def test_unusable_results_have_no_numeric_score(mutation):
    sample = _sample()
    if mutation == "zero_total":
        sample.metadata["eval_report"]["total"] = 0
    elif mutation == "too_many_passes":
        sample.metadata["eval_report"]["passed"] = 11
    elif mutation == "missing_report":
        sample.metadata.pop("eval_report")
    elif mutation == "nan_reward":
        sample.reward = float("nan")
    elif mutation == "wrong_reward":
        sample.reward = 0.0
    elif mutation == "missing_validity":
        sample.metadata.pop("seta_valid_reward")
    elif mutation == "aborted":
        sample.status = Sample.Status.ABORTED
    elif mutation == "failed":
        sample.status = Sample.Status.FAILED
    else:
        sample.remove_sample = True
    evidence = _read([sample])
    assert not evidence.valid and evidence.progress is None and evidence.reward is None
    assert evidence.error and evidence.trial_dir == "/trials/1"


def test_valid_truncation_and_zero_reward_are_not_infrastructure_failures():
    sample = _sample(status=Sample.Status.TRUNCATED, reward=0.0)
    sample.metadata.update(reward=0.0, eval_report={"passed": 0, "total": 10})
    evidence = _read([sample])
    assert evidence.valid and evidence.reward == evidence.progress == 0


def test_disagreeing_siblings_are_invalid():
    first = _sample(7, rollout_id=7, lineage=SampleLineage(7, 0, 2))
    second = _sample(8, rollout_id=7, lineage=SampleLineage(7, 1, 2), reward=0.2)
    second.metadata.update(reward=0.2, eval_report={"passed": 2, "total": 10})
    assert "disagree" in _read([first, second]).error


def test_identity_errors_do_not_get_silently_collected():
    with pytest.raises(ValueError, match="different Miles"):
        _read([_sample(1), _sample(2)])
    with pytest.raises(ValidationError):
        _read([_sample(metadata={})])
    assert TaskKey(dataset_id="v1", instance_id="SETA_Synth__same") != TaskKey(
        dataset_id="v1", instance_id="SETA_Evolve__same"
    )
    assert TaskKey(dataset_id="v1", instance_id="same") != TaskKey(dataset_id="v2", instance_id="same")
