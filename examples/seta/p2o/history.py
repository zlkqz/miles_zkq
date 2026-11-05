"""Completed-cycle template history and task-conditioned, cross-cycle sampling."""

import math
import random
from dataclasses import dataclass
from statistics import fmean, pstdev
from typing import Literal

from examples.seta.p2o.evidence import EpisodeEvidence, TaskKey
from examples.seta.p2o.scoring import DevScores
from examples.seta.p2o.templates import Template
from pydantic import Field, model_validator

from miles.utils.pydantic_utils import FrozenStrictBaseModel

DEFAULT_TEMPLATE_TEMPERATURE = 1.0


class TemplateEvaluation(FrozenStrictBaseModel):
    text: str  # Empty bootstrap is retained as evidence, never sampled for training.
    scores: DevScores
    policy_version: str = Field(min_length=1)
    search_episodes: tuple[EpisodeEvidence, ...] = ()

    @model_validator(mode="after")
    def _validate_text(self):
        if self.text:
            Template(self.text)
        return self

    @property
    def template_id(self):
        return Template(self.text).template_id if self.text else ""


class TemplateCycle(FrozenStrictBaseModel):
    cycle_id: int = Field(ge=1, strict=True)
    evaluations: tuple[TemplateEvaluation, ...]

    @model_validator(mode="after")
    def _validate_panel(self):
        ids = [e.template_id for e in self.evaluations]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate template in completed cycle")
        panels = {tuple(s.task for s in e.scores.tasks) for e in self.evaluations}
        if len(panels) > 1 or len({e.policy_version for e in self.evaluations}) > 1:
            raise ValueError("cycle evaluations must share a dev panel and policy")
        return self


class TaskAssignment(FrozenStrictBaseModel):
    task: TaskKey
    template_ids: tuple[str, ...]


class HistoryState(FrozenStrictBaseModel):
    schema_version: Literal[1] = 1
    cycles: tuple[TemplateCycle, ...]
    assignments: tuple[TaskAssignment, ...]


@dataclass(frozen=True)
class TemplateChoice:
    template: Template
    score_cycle: int
    normalized_score: float
    eligibility: str
    probability: float


class TemplateHistory:
    """Raw evaluations own scores; normalized entries are derived once per commit.

    RNG belongs to the sampling coordinator, not to the registry. Duplicate
    hashes retain every evaluation but only their latest completed-cycle weight.
    """

    def __init__(self, *, temperature: float = DEFAULT_TEMPLATE_TEMPERATURE):
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("template temperature must be finite and positive")
        self.temperature = temperature
        self._cycles: list[TemplateCycle] = []
        self._entries: dict[str, tuple[TemplateEvaluation, int, float]] = {}
        self._solvers: dict[TaskKey, set[str]] = {}
        self._assignments: dict[TaskKey, tuple[str, ...]] = {}

    def __len__(self):
        return len(self._entries)

    def commit(self, cycle: TemplateCycle) -> None:
        if self._cycles and cycle.cycle_id <= self._cycles[-1].cycle_id:
            if cycle == self._cycles[-1]:
                return
            raise ValueError("history cycles must be committed in order")
        entries = [e for e in cycle.evaluations if e.text]
        scores = [e.scores.mean for e in entries]
        mean, std = (fmean(scores), pstdev(scores)) if scores else (0.0, 0.0)
        for entry, score in zip(entries, scores, strict=True):
            q = (score - mean) / std if std > 1e-12 else 0.0
            self._entries[entry.template_id] = (entry, cycle.cycle_id, q)
            for task_score in entry.scores.tasks:
                if any(e.valid and e.passed == e.total for e in task_score.episodes):
                    self._solvers.setdefault(task_score.task, set()).add(entry.template_id)
            for episode in entry.search_episodes:
                if episode.valid and episode.passed == episode.total:
                    self._solvers.setdefault(episode.task, set()).add(entry.template_id)
        self._cycles.append(cycle)

    def probabilities(self, task: TaskKey):
        if ids := self._solvers.get(task):
            reason = "known_solver"
        elif ids := self._assignments.get(task):
            reason = "previous_assignment"
        else:
            ids, reason = self._entries, "global_history"
        keys = sorted(set(ids))
        if not keys:
            return (), (), "empty_history"
        values = [self._entries[key][2] for key in keys]
        # Subtract before division, so even tiny positive T cannot overflow.
        maximum = max(values)
        weights = [math.exp((q - maximum) / self.temperature) for q in values]
        total = sum(weights)
        return tuple(keys), tuple(w / total for w in weights), reason

    def choose(self, task: TaskKey, *, rng: random.Random) -> TemplateChoice | None:
        ids, probabilities, reason = self.probabilities(task)
        if not ids:
            return None
        index = rng.choices(range(len(ids)), weights=probabilities, k=1)[0]
        entry, cycle_id, q = self._entries[ids[index]]
        return TemplateChoice(Template(entry.text), cycle_id, q, reason, probabilities[index])

    def assign(self, tasks, *, repeats: int, rng: random.Random) -> None:
        if repeats < 1:
            raise ValueError("assignment repeats must be positive")
        for task in tasks:
            # Retain the old assignment when no new solver evidence exists.
            if task in self._assignments and task not in self._solvers:
                continue
            choices = [self.choose(task, rng=rng) for _ in range(repeats)]
            if choices[0] is not None:
                self._assignments[task] = tuple(choice.template.template_id for choice in choices)

    def snapshot(self) -> HistoryState:
        return HistoryState(
            cycles=tuple(self._cycles),
            assignments=tuple(TaskAssignment(task=t, template_ids=ids) for t, ids in self._assignments.items()),
        )

    @classmethod
    def restore(cls, state: HistoryState, *, temperature=DEFAULT_TEMPLATE_TEMPERATURE):
        history = cls(temperature=temperature)
        for cycle in state.cycles:
            history.commit(cycle)
        for entry in state.assignments:
            if entry.task in history._assignments or not entry.template_ids:
                raise ValueError("duplicate or empty task assignment")
            if any(key not in history._entries for key in entry.template_ids):
                raise ValueError("assignment references an unknown template")
            history._assignments[entry.task] = entry.template_ids
        return history
