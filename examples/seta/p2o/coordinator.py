"""One rollout-side owner for collection, historical templates and sampling RNG."""

import hashlib
import json
import logging
import random
from collections import Counter

from examples.seta.p2o.evidence import Origin, TaskKey, TaskObservation, episode_from_samples
from examples.seta.p2o.hard_tasks import CollectorState, HardTaskCollector
from examples.seta.p2o.history import DEFAULT_TEMPLATE_TEMPERATURE, HistoryState, TemplateHistory
from examples.seta.p2o.scoring import CollectionConfig
from pydantic import Field

from miles.utils.pydantic_utils import FrozenStrictBaseModel

logger = logging.getLogger(__name__)


class CoordinatorState(FrozenStrictBaseModel):
    schema_version: int = Field(default=1, ge=1, le=1)
    collector: CollectorState
    history: HistoryState
    temperature: float
    rng_state: tuple[int, tuple[int, ...], float | None]


def catalog_identity(samples):
    """Hash the loaded immutable training rows, not their shuffled positions/path."""
    rows = [dict(instance_id=s.metadata["instance_id"], prompt=s.prompt, metadata=s.metadata) for s in samples]
    rows.sort(key=lambda row: row["instance_id"])
    if len({r["instance_id"] for r in rows}) != len(rows):
        raise ValueError("P2O requires unique task instances in the training catalog")
    return hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class SamplingCoordinator:
    def __init__(self, *, samples, config: CollectionConfig, seed, temperature=DEFAULT_TEMPLATE_TEMPERATURE):
        self.dataset_id = catalog_identity(samples)
        self.catalog = {self.task(sample): sample for sample in samples}
        self.collector = HardTaskCollector(config, tuple(self.catalog), seed=seed)
        self.history = TemplateHistory(temperature=temperature)
        self.rng = random.Random(seed)
        self.metrics = Counter()

    def task(self, sample):
        return TaskKey(dataset_id=self.dataset_id, instance_id=sample.metadata["instance_id"])

    def priority(self, task):
        return self.collector.last_participated_cycle(task)

    def eligible(self, task):
        return bool(self.priority(task) and len(self.history))

    def stamp(self, samples, *, origin, inject):
        task = self.task(samples[0])
        for sample in samples:
            metadata = dict(sample.metadata or {})
            metadata.pop("p2o_template", None)
            # A retried attempt is generated anew. Context alignment from the
            # previous attempt is never evidence for its replacement trajectory.
            metadata.pop("p2o_context", None)
            choice = self.history.choose(task, rng=self.rng) if inject else None
            details = dict(
                dataset_id=self.dataset_id,
                cycle_id=self.collector.cycle_id,
                origin=origin.value,
                last_participated_cycle=self.priority(task),
                templated=choice is not None,
            )
            if choice is not None:
                metadata["p2o_template"] = choice.template.to_dict()
                details.update(
                    template_id=choice.template.template_id,
                    score_cycle=choice.score_cycle,
                    normalized_score=choice.normalized_score,
                    eligibility=choice.eligibility,
                    probability=choice.probability,
                )
            metadata["p2o"] = details
            sample.metadata = metadata

    def observe(self, entry):
        scheduled = entry.prompt_group
        info = scheduled[0].metadata["p2o"]
        task = self.task(scheduled[0])
        if len(entry.group) != len(scheduled):
            raise ValueError("SETA result must preserve one output slot per scheduled episode")
        episodes = []
        for original, output in zip(scheduled, entry.group, strict=True):
            leaves = output if isinstance(output, list) else [output]
            # Output processing may replace Samples; the scheduled identity owns
            # origin, cycle and assignment, never an untrusted remote echo.
            for sample in leaves:
                sample.metadata = {**sample.metadata, "p2o": dict(original.metadata["p2o"])}
            episodes.append(
                episode_from_samples(
                    leaves,
                    dataset_id=self.dataset_id,
                    attempt_id=f"{original.group_index}:{original.index}",
                    template_id=original.metadata["p2o"].get("template_id"),
                )
            )
        observation = TaskObservation(
            cycle_id=info["cycle_id"],
            observation_id=str(scheduled[0].group_index),
            task=task,
            origin=Origin(info["origin"]),
            episodes=tuple(episodes),
        )
        result = self.collector.observe(observation)
        self.metrics[f"collection/{result.reason}"] += 1
        logger.info(
            "P2O collect task=%s origin=%s cycle=%s result=%s progress=%s reward=%s hard=%d trigger=%s",
            task.instance_id,
            info["origin"],
            info["cycle_id"],
            result.reason,
            result.decision.progress_mean if result.decision else None,
            result.decision.reward_mean if result.decision else None,
            self.collector.hard_task_count,
            result.trigger_pending,
        )
        return result

    def finish_cycle(self, cycle):
        panel = self.collector.freeze()
        if panel.cycle_id != cycle.cycle_id:
            raise ValueError("template cycle differs from frozen task panel")
        for entry in cycle.evaluations:
            if tuple(s.task for s in entry.scores.tasks) != tuple(d.task for d in panel.dev):
                raise ValueError("template evaluation differs from frozen dev panel")
            if any(s.attempted_count != self.collector.config.gepa_dev_repeats for s in entry.scores.tasks):
                raise ValueError("template evaluation has incomplete dev attempts")
        self.history.commit(cycle)
        self.history.assign(
            [o.task for o in panel.hard_tasks], repeats=self.collector.config.n_samples_per_prompt, rng=self.rng
        )
        self.collector.finish_cycle(cycle.cycle_id)

    def snapshot(self):
        return CoordinatorState(
            collector=self.collector.snapshot(),
            history=self.history.snapshot(),
            temperature=self.history.temperature,
            rng_state=self.rng.getstate(),
        )

    def load_state(self, state: CoordinatorState):
        if state.temperature != self.history.temperature:
            raise ValueError("checkpoint template temperature differs from this run")
        collector = HardTaskCollector.restore(
            state.collector, config=self.collector.config, training_tasks=tuple(self.catalog)
        )
        history = TemplateHistory.restore(state.history, temperature=self.history.temperature)
        if any(c.cycle_id >= collector.cycle_id for c in state.history.cycles):
            raise ValueError("template history contains an unfinished cycle")
        if any(a.task not in self.catalog for a in state.history.assignments):
            raise ValueError("template assignment outside the training catalog")
        self.collector, self.history = collector, history
        self.rng.setstate(state.rng_state)
