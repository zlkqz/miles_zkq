"""Single-owner hard-task collection, fixed cycle panels and resumable RNG."""

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from examples.seta.p2o.evidence import Origin, TaskKey, TaskObservation
from examples.seta.p2o.scoring import CollectionConfig, HardDecision, classify_hard
from pydantic import Field, model_validator

from miles.utils.pydantic_utils import FrozenStrictBaseModel


class DevTaskPlan(FrozenStrictBaseModel):
    task: TaskKey
    seeds: tuple[int, ...]

    @model_validator(mode="after")
    def _check_seeds(self):
        if not self.seeds or len(set(self.seeds)) != len(self.seeds) or any(not 0 <= s < 2**32 for s in self.seeds):
            raise ValueError("dev repetitions require distinct uint32 seeds")
        return self


class CyclePanel(FrozenStrictBaseModel):
    cycle_id: int = Field(ge=1, strict=True)
    hard_tasks: tuple[TaskObservation, ...]
    dev: tuple[DevTaskPlan, ...]

    @model_validator(mode="after")
    def _check_partition(self):
        keys = {o.task for o in self.hard_tasks}
        dev_keys = {d.task for d in self.dev}
        if len(keys) != len(self.hard_tasks) or len(dev_keys) != len(self.dev):
            raise ValueError("duplicate task in cycle panel")
        if not dev_keys or not dev_keys < keys:
            raise ValueError("dev must be a proper nonempty subset of hard tasks")
        if any(o.cycle_id != self.cycle_id or o.origin != Origin.NATURAL for o in self.hard_tasks):
            raise ValueError("panel requires natural observations from this cycle")
        return self

    @property
    def search_train(self) -> tuple[TaskObservation, ...]:
        dev_keys = {d.task for d in self.dev}
        return tuple(o for o in self.hard_tasks if o.task not in dev_keys)


class Participation(FrozenStrictBaseModel):
    task: TaskKey
    last_participated_cycle: int = Field(ge=1, strict=True)


class CollectorState(FrozenStrictBaseModel):
    schema_version: Literal[1] = 1
    config: CollectionConfig
    training_tasks: tuple[TaskKey, ...]
    cycle_id: int = Field(ge=1, strict=True)
    pending: tuple[TaskObservation, ...]
    panel: CyclePanel | None
    history: tuple[Participation, ...]
    # Python Random state is JSON-safe through Pydantic, without pickle.
    rng_state: tuple[int, tuple[int, ...], float | None]


@dataclass(frozen=True)
class CollectionResult:
    added: bool
    trigger_pending: bool
    reason: str
    decision: HardDecision | None = None


class HardTaskCollector:
    """Called by one coordinator after finalized task batches, never by workers.

    freeze() is called only after the owner drains in-flight work. It freezes all
    H tasks, not just the first trigger-sized prefix. This module does not pause
    producers, launch GEPA, write checkpoints, or change training rewards.
    """

    def __init__(self, config: CollectionConfig, training_tasks: Sequence[TaskKey], *, seed: int):
        if not training_tasks or len(set(training_tasks)) != len(training_tasks):
            raise ValueError("training catalog must be nonempty and unique")
        self.config = config
        self.training_tasks = tuple(sorted(training_tasks, key=lambda t: (t.dataset_id, t.instance_id)))
        self._allowed = frozenset(training_tasks)
        self._rng = random.Random(seed)
        self._cycle_id = 1
        self._pending: dict[TaskKey, TaskObservation] = {}
        self._panel: CyclePanel | None = None
        self._history: dict[TaskKey, Participation] = {}

    @property
    def cycle_id(self) -> int:
        return self._cycle_id

    @property
    def hard_task_count(self) -> int:
        return len(self._pending)

    def last_participated_cycle(self, task: TaskKey) -> int:
        entry = self._history.get(task)
        return entry.last_participated_cycle if entry is not None else 0

    @property
    def participated_tasks(self) -> tuple[TaskKey, ...]:
        return tuple(self._history)

    @property
    def trigger_pending(self) -> bool:
        return self.hard_task_count >= self.config.gepa_trigger_unique_hard_tasks

    def observe(self, observation: TaskObservation) -> CollectionResult:
        if observation.origin != Origin.NATURAL:
            return CollectionResult(False, self.trigger_pending, "not_natural")
        if observation.cycle_id != self._cycle_id:
            raise ValueError("observation belongs to a different cycle")
        if self._panel is not None:
            raise ValueError("cycle is frozen; drain observations before freeze")
        if observation.task not in self._allowed:
            raise ValueError("natural task is outside the training catalog")
        decision = classify_hard(observation, self.config)
        if decision.is_hard is not True:
            return CollectionResult(
                False, self.trigger_pending, decision.reason if decision.is_hard is None else "not_hard", decision
            )
        if observation.task in self._pending:
            # Keep the first qualifying evidence; later easy results do not erase it.
            return CollectionResult(False, self.trigger_pending, "already_collected", decision)
        self._pending[observation.task] = observation
        return CollectionResult(True, self.trigger_pending, "added", decision)

    def freeze(self) -> CyclePanel:
        if self._panel is not None:
            return self._panel
        if not self.trigger_pending:
            raise ValueError("hard-task trigger not reached")
        observations = tuple(self._pending[key] for key in self.training_tasks if key in self._pending)
        dev_tasks = self._rng.sample([o.task for o in observations], self.config.gepa_dev_task_count)
        dev = tuple(
            DevTaskPlan(task=task, seeds=tuple(self._rng.sample(range(2**32), self.config.gepa_dev_repeats)))
            for task in dev_tasks
        )
        self._panel = CyclePanel(cycle_id=self._cycle_id, hard_tasks=observations, dev=dev)
        return self._panel

    def finish_cycle(self, cycle_id: int) -> None:
        if cycle_id >= 1 and cycle_id == self._cycle_id - 1:
            return  # Re-delivery of the last completed transition is harmless.
        if self._panel is None or cycle_id != self._cycle_id:
            raise ValueError("no matching frozen cycle to finish")
        # Participation survives even when the search found no useful template.
        for observation in self._panel.hard_tasks:
            self._history[observation.task] = Participation(task=observation.task, last_participated_cycle=cycle_id)
        self._pending.clear()
        self._panel = None
        self._cycle_id += 1

    def snapshot(self) -> CollectorState:
        return CollectorState(
            config=self.config,
            training_tasks=self.training_tasks,
            cycle_id=self._cycle_id,
            pending=tuple(self._pending.values()),
            panel=self._panel,
            history=tuple(self._history.values()),
            rng_state=self._rng.getstate(),
        )

    @classmethod
    def restore(
        cls, state: CollectorState, *, config: CollectionConfig, training_tasks: Sequence[TaskKey]
    ) -> "HardTaskCollector":
        collector = cls(config, training_tasks, seed=0)
        if state.config != config or state.training_tasks != collector.training_tasks:
            raise ValueError("checkpoint config or training catalog differs from this run")
        collector._cycle_id = state.cycle_id
        for observation in state.pending:
            if not collector.observe(observation).added:
                raise ValueError("checkpoint contains duplicate or ineligible hard evidence")
        for entry in state.history:
            if entry.task not in collector._allowed or entry.task in collector._history:
                raise ValueError("invalid checkpoint history task")
            if entry.last_participated_cycle >= state.cycle_id:
                raise ValueError("history refers to an unfinished cycle")
            collector._history[entry.task] = entry
        if state.panel is not None:
            collector._validate_restored_panel(state.panel)
        collector._panel = state.panel
        collector._rng.setstate(state.rng_state)
        return collector

    def _validate_restored_panel(self, panel: CyclePanel) -> None:
        if not self.trigger_pending or panel.cycle_id != self._cycle_id:
            raise ValueError("checkpoint panel is below trigger or belongs to another cycle")
        if {o.task: o for o in panel.hard_tasks} != self._pending:
            raise ValueError("checkpoint panel does not contain all pending hard tasks")
        if len(panel.dev) != self.config.gepa_dev_task_count:
            raise ValueError("checkpoint dev task count differs from config")
        if any(len(d.seeds) != self.config.gepa_dev_repeats for d in panel.dev):
            raise ValueError("checkpoint dev repeats differ from config")
