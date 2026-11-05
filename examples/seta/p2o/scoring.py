"""Two score contracts: bonus-free hard detection and unchanged GEPA rewards."""

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import fmean, stdev

from examples.seta.p2o.evidence import EpisodeEvidence, TaskKey, TaskObservation, check_distinct_attempts
from pydantic import Field, FiniteFloat, model_validator

from miles.utils.pydantic_utils import FrozenStrictBaseModel


class CollectionConfig(FrozenStrictBaseModel):
    # Read K from the Miles run when wiring the hook; do not maintain a second default.
    n_samples_per_prompt: int = Field(ge=1, strict=True)
    hard_reward_threshold: FiniteFloat | None = Field(default=None, ge=0, le=1)
    gepa_trigger_unique_hard_tasks: int = Field(default=64, ge=1, strict=True)
    gepa_dev_task_count: int = Field(default=16, ge=1, strict=True)
    gepa_dev_repeats: int = Field(default=4, ge=1, strict=True)

    @model_validator(mode="after")
    def _check_split(self):
        if self.gepa_dev_task_count >= self.gepa_trigger_unique_hard_tasks:
            raise ValueError("trigger must leave at least one search-train task after dev selection")
        return self

    @property
    def threshold(self) -> float:
        return 1 / self.n_samples_per_prompt if self.hard_reward_threshold is None else self.hard_reward_threshold


@dataclass(frozen=True)
class HardDecision:
    is_hard: bool | None
    progress_mean: float | None
    reward_mean: float | None
    threshold: float
    reason: str


def classify_hard(observation: TaskObservation, config: CollectionConfig) -> HardDecision:
    episodes = observation.episodes
    if len(episodes) != config.n_samples_per_prompt:
        return HardDecision(None, None, None, config.threshold, "incomplete_or_oversized_episode_batch")
    if not all(e.valid for e in episodes):
        return HardDecision(None, None, None, config.threshold, "invalid_episode")
    progress = fmean(e.progress for e in episodes)
    return HardDecision(
        progress <= config.threshold, progress, fmean(e.reward for e in episodes), config.threshold, "scored"
    )


class DevTaskScore(FrozenStrictBaseModel):
    task: TaskKey
    episodes: tuple[EpisodeEvidence, ...]

    @model_validator(mode="after")
    def _check_episodes(self):
        check_distinct_attempts(self.episodes)
        if not self.episodes or any(e.task != self.task for e in self.episodes):
            raise ValueError("dev score requires terminal attempts for exactly one task")
        return self

    @property
    def attempted_count(self) -> int:
        return len(self.episodes)

    @property
    def valid_count(self) -> int:
        return sum(e.valid for e in self.episodes)

    @property
    def invalid_count(self) -> int:
        return self.attempted_count - self.valid_count

    @property
    def all_invalid(self) -> bool:
        return self.valid_count == 0

    @property
    def reward_mean(self) -> float:
        return fmean(e.reward for e in self.episodes if e.valid) if self.valid_count else 0.0

    @property
    def reward_std(self) -> float | None:
        return stdev(e.reward for e in self.episodes if e.valid) if self.valid_count > 1 else None


def aggregate_dev_task(task: TaskKey, episodes: Sequence[EpisodeEvidence], *, expected_repeats: int) -> DevTaskScore:
    # Invalid terminal attempts count toward the planned repetitions; pending ones do not.
    if expected_repeats < 1 or len(episodes) != expected_repeats:
        raise ValueError("all planned dev attempts must terminate before aggregation")
    return DevTaskScore(task=task, episodes=tuple(episodes))


class DevScores(FrozenStrictBaseModel):
    tasks: tuple[DevTaskScore, ...]

    @model_validator(mode="after")
    def _check_tasks(self):
        if not self.tasks or len({s.task for s in self.tasks}) != len(self.tasks):
            raise ValueError("dev vector requires nonempty, distinct tasks")
        return self

    @property
    def vector(self) -> tuple[float, ...]:
        return tuple(s.reward_mean for s in self.tasks)

    @property
    def mean(self) -> float:
        return fmean(self.vector)


def aggregate_dev(
    task_order: Sequence[TaskKey], episodes: Sequence[EpisodeEvidence], *, expected_repeats: int
) -> DevScores:
    check_distinct_attempts(episodes)
    by_task = {task: [] for task in task_order}
    for episode in episodes:
        if episode.task not in by_task:
            raise ValueError("result is outside the fixed dev panel")
        by_task[episode.task].append(episode)
    return DevScores(
        tasks=tuple(aggregate_dev_task(t, by_task[t], expected_repeats=expected_repeats) for t in task_order)
    )
