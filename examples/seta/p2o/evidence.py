"""Immutable episode evidence, preserving Miles' episode/leaf distinction."""

from collections.abc import Sequence
from enum import Enum

from pydantic import Field, FiniteFloat, model_validator

from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.types import Sample


class TaskKey(FrozenStrictBaseModel):
    # The caller supplies a dataset revision/fingerprint, not a transient row index.
    dataset_id: str = Field(min_length=1)
    instance_id: str = Field(min_length=1)


class Origin(str, Enum):
    NATURAL = "natural"
    REPLAY = "replay"
    SEARCH = "search"
    REFLECTION = "reflection"
    EVALUATION = "evaluation"


class EpisodeEvidence(FrozenStrictBaseModel):
    task: TaskKey
    attempt_id: str = Field(min_length=1)
    valid: bool
    passed: int | None = Field(default=None, ge=0, strict=True)
    total: int | None = Field(default=None, gt=0, strict=True)
    reward: FiniteFloat | None = Field(default=None, ge=0)
    error: str | None = None
    remote_request_id: str | None = None
    remote_endpoint: str | None = None
    trial_dir: str | None = None
    policy_versions: tuple[str, ...] = ()
    template_id: str | None = None

    @model_validator(mode="after")
    def _check_score(self):
        if self.valid:
            if self.passed is None or self.total is None or self.reward is None:
                raise ValueError("valid evidence requires passed, total and reward")
            if self.passed > self.total:
                raise ValueError("passed exceeds total")
            if self.error is not None:
                raise ValueError("valid evidence cannot carry an error")
        elif self.reward is not None or self.passed is not None or self.total is not None:
            raise ValueError("invalid evidence has no usable score; retain its artifact references instead")
        elif not self.error:
            raise ValueError("invalid evidence requires a reason")
        return self

    @property
    def progress(self) -> float | None:
        return self.passed / self.total if self.valid else None


def check_distinct_attempts(episodes: Sequence[EpisodeEvidence]) -> None:
    if len({e.attempt_id for e in episodes}) != len(episodes):
        raise ValueError("duplicate attempt_id; leaves/retries are not independent attempts")
    remote_ids = [e.remote_request_id for e in episodes if e.remote_request_id is not None]
    if len(set(remote_ids)) != len(remote_ids):
        raise ValueError("duplicate remote_request_id; one environment cannot count twice")


class TaskObservation(FrozenStrictBaseModel):
    cycle_id: int = Field(ge=1, strict=True)
    observation_id: str = Field(min_length=1)
    task: TaskKey
    origin: Origin
    episodes: tuple[EpisodeEvidence, ...]

    @model_validator(mode="after")
    def _check_episodes(self):
        check_distinct_attempts(self.episodes)
        if any(e.task != self.task for e in self.episodes):
            raise ValueError("observation mixes different tasks")
        return self


def _episode_key(sample: Sample) -> int:
    # Same identity fallback as Miles' training reward normalization.
    key = sample.rollout_id if sample.rollout_id is not None else sample.index
    if key is None:
        raise ValueError("episode requires Miles rollout_id or index")
    return key


def episode_from_samples(
    samples: Sequence[Sample], *, dataset_id: str, attempt_id: str, template_id: str | None = None
) -> EpisodeEvidence:
    """Collapse all kept leaves of one finalized episode; never mutate Samples.

    Structural identity errors raise. Failed/malformed verifier outcomes become
    invalid evidence with artifact pointers, so reflection can still inspect them.
    """
    if not samples:
        raise ValueError("no samples; construct invalid evidence from the scheduled attempt identity")
    first = samples[0]
    task = TaskKey(dataset_id=dataset_id, instance_id=first.metadata.get("instance_id", ""))
    if any(s.metadata.get("instance_id") != task.instance_id for s in samples):
        raise ValueError("episode mixes different tasks")
    if len({_episode_key(s) for s in samples}) != 1 or len({s.group_index for s in samples}) != 1:
        raise ValueError("samples belong to different Miles episodes or task batches")
    versions = tuple(sorted({span.version for s in samples for turn in s.weight_versions for span in turn.spans}))
    refs = {key: first.metadata.get(key) for key in ("remote_request_id", "remote_endpoint", "trial_dir")}
    common = dict(task=task, attempt_id=attempt_id, template_id=template_id, policy_versions=versions, **refs)
    error = _leaf_error(samples)
    if error:
        return EpisodeEvidence(**common, valid=False, error=error)
    try:
        scores = []
        for sample in samples:
            report = sample.metadata["eval_report"]
            score = EpisodeEvidence(
                **common, valid=True, passed=report["passed"], total=report["total"], reward=sample.reward
            )
            if sample.reward != sample.metadata["reward"]:
                raise ValueError("Sample reward differs from verifier metadata")
            scores.append(score)
        if any(score != scores[0] for score in scores[1:]):
            raise ValueError("sibling leaves disagree on verifier score")
        return scores[0]
    except (KeyError, TypeError, ValueError) as error:
        return EpisodeEvidence(**common, valid=False, error=f"invalid_verifier_result: {error}")


def _leaf_error(samples: Sequence[Sample]) -> str | None:
    lineages = [s.lineage for s in samples]
    if any(lineage is not None for lineage in lineages):
        if any(lineage is None for lineage in lineages):
            return "incomplete_leaf_lineage"
        if len({(x.source_sample_index, x.output_count) for x in lineages}) != 1:
            return "inconsistent_leaf_lineage"
        if sorted(x.output_index for x in lineages) != list(range(lineages[0].output_count)):
            return "incomplete_or_duplicate_leaves"
    elif len(samples) > 1:
        return "missing_multi_leaf_lineage"
    if len({s.metadata.get("remote_request_id") for s in samples}) != 1:
        return "inconsistent_remote_request_id"
    for sample in samples:
        if sample.status not in (Sample.Status.COMPLETED, Sample.Status.TRUNCATED) or sample.remove_sample:
            return f"unusable_sample_status: {sample.status.value}"
        if sample.metadata.get("seta_valid_reward") is not True:
            return str(sample.metadata.get("seta_error") or "missing_valid_verifier_result")
    return None
