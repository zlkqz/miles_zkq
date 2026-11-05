"""Native sample identities, finite submission windows and random buffer selection."""

import logging
from collections import deque

from examples.seta.p2o.evidence import Origin
from examples.seta.p2o.mixing import newest_first, select_training_batch

from miles.rollout.base_types import RolloutPhaseBoundary
from miles.rollout.fully_async_data_buffer import DefaultDataBuffer, first_sample

logger = logging.getLogger(__name__)


class SubmissionWindows:
    def __init__(self, *, source, coordinator, size, config):
        config.bounds(size)
        self.source, self.owner, self.size, self.config = source, coordinator, size, config
        self.ready = deque()
        self.displaced = deque()
        self.window_id = 0

    def pop(self, *, retries, in_flight):
        if not self.ready:
            self._prepare(retries=retries, in_flight=in_flight)
        return self.ready.popleft()

    def _initial(self, retries):
        batches = []
        for queue in (retries, self.displaced):
            while queue and len(batches) < self.size:
                batches.append(queue.popleft())
        if len(batches) < self.size:
            batches.extend(self.source.get_samples(self.size - len(batches)))
        return batches

    def _prepare(self, *, retries, in_flight):
        owner = self.owner
        batches = self._initial(retries)
        tasks = [owner.task(batch[0]) for batch in batches]
        priorities = [owner.priority(task) for task in tasks]
        hits = [i for i, task in enumerate(tasks) if owner.eligible(task)]
        lower, upper = self.config.bounds(self.size)
        initial = len(hits)
        origin = [Origin(batch[0].metadata.get("p2o", {}).get("origin", Origin.NATURAL)) for batch in batches]
        candidates = [
            task
            for task in owner.collector.participated_tasks
            if task not in in_flight and task not in tasks and owner.eligible(task)
        ]
        fallback, replaced = None, 0
        if initial < lower:
            missing = lower - initial
            if len(candidates) < missing:
                fallback = "insufficient_available_hard_tasks"
            else:
                ordered = newest_first(
                    range(len(candidates)), priorities=[owner.priority(t) for t in candidates], rng=owner.rng
                )
                # Preserve the deferred FIFO prefix, replacing fresh suffix tasks first.
                destinations = [i for i in range(self.size) if i not in hits][-missing:]
                for index, destination in zip(ordered[:missing], destinations, strict=True):
                    self.displaced.append(batches[destination])
                    [replacement] = self.source.materialize_samples([owner.catalog[candidates[index]]])
                    batches[destination] = replacement
                    origin[destination] = Origin.REPLAY
                    hits.append(destination)
                    replaced += 1
        elif initial > upper:
            # These are still unsubmitted tasks: keep every natural task and
            # decide which ones receive a template before generating any tokens.
            hits = newest_first(hits, priorities=priorities, rng=owner.rng)[:upper]
        self.window_id += 1
        for index, batch in enumerate(batches):
            owner.stamp(batch, origin=origin[index], inject=index in hits)
            for sample in batch:
                sample.metadata["p2o"]["window_id"] = self.window_id
        self.ready.extend(batches)
        owner.metrics["submission/windows"] += 1
        owner.metrics["submission/initial_templated"] += initial
        owner.metrics["submission/final_templated"] += len(hits)
        owner.metrics["submission/tasks"] += len(batches)
        owner.metrics["submission/replacements"] += replaced
        if fallback:
            owner.metrics[f"submission/fallback/{fallback}"] += 1
        logger.info(
            "P2O submission window=%d initial=%d final=%d total=%d replay=%d available_replay=%d displaced=%d fallback=%s",
            self.window_id,
            initial,
            len(hits),
            len(batches),
            replaced,
            len(candidates),
            len(self.displaced),
            fallback,
        )

    def state_dict(self):
        return dict(
            ready=list(self.ready),
            displaced=list(self.displaced),
            window_id=self.window_id,
            size=self.size,
            ratios=(self.config.target, self.config.maximum),
        )

    def load_state_dict(self, state):
        if self.ready or self.displaced:
            raise ValueError("restore submission state before sampling")
        if state["size"] != self.size or state["ratios"] != (self.config.target, self.config.maximum):
            raise ValueError("checkpoint submission window configuration differs")
        self.ready = deque(state["ready"])
        self.displaced = deque(state["displaced"])
        self.window_id = state["window_id"]


class P2ODataBuffer(DefaultDataBuffer):
    """Preserve native validity/staleness/backpressure; change selection only."""

    def bind(self, coordinator, config):
        self.owner, self.config = coordinator, config
        self._normal_capacity = self._capacity
        self._phase_ready = False

    async def drain_capacity(self, in_flight):
        # Admission has stopped. Reserve bounded space for every submitted task
        # so producer flushing cannot deadlock behind a paused trainer.
        async with self._cond:
            self._capacity = max(self._capacity, len(self._buffer) + in_flight)
            self._cond.notify_all()

    async def mark_boundary(self):
        async with self._cond:
            self._phase_ready = True
            self._cond.notify_all()

    def resume(self):
        self._phase_ready = False
        self._capacity = self._normal_capacity

    def _check_wait_condition(self):
        if self._phase_ready:
            raise RolloutPhaseBoundary()

    def _select_available(self, num_groups):
        # Classification uses immutable generation-time assignments, never the
        # latest registry: old plain trajectories cannot be relabelled as guided.
        infos = [first_sample(entry.group).metadata["p2o"] for entry in self._buffer]
        selected = select_training_batch(
            templated=[info["templated"] for info in infos],
            priorities=[self.owner.priority(self.owner.task(entry.prompt_group[0])) for entry in self._buffer],
            size=num_groups,
            config=self.config,
            rng=self.owner.rng,
        )
        entries = [self._buffer[i] for i in selected.indices]
        indices = set(selected.indices)
        self._buffer = [entry for i, entry in enumerate(self._buffer) if i not in indices]
        for key in ("initial_templated", "final_templated", "replacements"):
            self.owner.metrics[f"consumption/{key}"] += getattr(selected, key)
        self.owner.metrics["consumption/tasks"] += num_groups
        if selected.fallback:
            self.owner.metrics[f"consumption/fallback/{selected.fallback}"] += 1
        logger.info("P2O consume available=%d total=%d selection=%s", len(infos), num_groups, selected)
        return entries
