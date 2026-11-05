import pytest

from examples.seta.p2o.evidence import EpisodeEvidence, Origin, TaskKey, TaskObservation


@pytest.fixture
def task():
    return TaskKey(dataset_id="seta-revision-test", instance_id="SETA_Synth__task_0")


@pytest.fixture
def make_episode(task):
    def make(index=0, *, task_key=task, passed=0, total=16, reward=None, valid=True, **kwargs):
        if not valid:
            return EpisodeEvidence(
                task=task_key,
                attempt_id=f"{task_key.instance_id}:{index}",
                valid=False,
                error="verifier unavailable",
                **kwargs,
            )
        if reward is None:
            reward = passed / total + (0.2 if passed == total else 0)
        return EpisodeEvidence(
            task=task_key,
            attempt_id=f"{task_key.instance_id}:{index}",
            valid=True,
            passed=passed,
            total=total,
            reward=reward,
            **kwargs,
        )

    return make


@pytest.fixture
def make_observation(task, make_episode):
    def make(
        *, task_key=task, count=16, cycle=1, origin=Origin.NATURAL, episodes=None, observation_id="batch", **kwargs
    ):
        if episodes is None:
            episodes = tuple(make_episode(i, task_key=task_key, **kwargs) for i in range(count))
        return TaskObservation(
            cycle_id=cycle, observation_id=observation_id, task=task_key, origin=origin, episodes=episodes
        )

    return make
