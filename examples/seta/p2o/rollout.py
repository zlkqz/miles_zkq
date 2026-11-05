"""P2O's native fully-async rollout extension (one coordinator per executor)."""

import asyncio
import logging
from pathlib import Path

from examples.seta.p2o.coordinator import CoordinatorState, SamplingCoordinator
from examples.seta.p2o.history import DEFAULT_TEMPLATE_TEMPERATURE, TemplateCycle
from examples.seta.p2o.mixing import MixConfig
from examples.seta.p2o.sampling import P2ODataBuffer, SubmissionWindows
from examples.seta.p2o.scoring import CollectionConfig
from examples.seta.p2o.search import SearchConfig
from examples.seta.p2o.training import CRMConfig, attach_training_context
from miles.rollout.filter_hub.base_types import iter_samples
from miles.rollout.fully_async_rollout import FullyAsyncRolloutFn
from miles.utils.simple_checkpointer import load_simple_checkpoint, save_simple_checkpoint

logger = logging.getLogger(__name__)


class P2ORolloutFn(FullyAsyncRolloutFn):
    @staticmethod
    def add_arguments(parser):
        FullyAsyncRolloutFn.add_arguments(parser)
        defaults = CollectionConfig(n_samples_per_prompt=1)
        parser.add_argument("--hard-reward-threshold", type=float, default=None)
        parser.add_argument(
            "--gepa-trigger-unique-hard-tasks", type=int, default=defaults.gepa_trigger_unique_hard_tasks
        )
        parser.add_argument("--gepa-dev-task-count", type=int, default=defaults.gepa_dev_task_count)
        parser.add_argument("--gepa-dev-repeats", type=int, default=defaults.gepa_dev_repeats)
        mix = MixConfig()
        parser.add_argument("--p2o-task-mix-ratio", type=float, default=mix.target)
        parser.add_argument("--p2o-max-template-ratio", type=float, default=mix.maximum)
        parser.add_argument("--p2o-template-temperature", type=float, default=DEFAULT_TEMPLATE_TEMPERATURE)
        crm = CRMConfig()
        parser.add_argument("--p2o-crm-lower", type=float, default=crm.lower)
        parser.add_argument("--p2o-crm-upper", type=float, default=crm.upper)
        search = SearchConfig()
        for name, default in vars(search).items():
            parser.add_argument("--gepa-" + name.replace("_", "-"), type=int, default=default)
        parser.add_argument("--gepa-reflection-num-gpus", type=int, default=1)
        parser.add_argument("--gepa-reflection-tp", type=int, default=1)
        parser.add_argument("--gepa-policy-tp", type=int, default=1)

    def __init__(self, input):
        super().__init__(input)
        if not isinstance(self._output, P2ODataBuffer):
            raise ValueError("P2O requires --custom-async-data-buffer-path examples.seta.p2o.sampling.P2ODataBuffer")
        if self.args.dynamic_sampling_filter_path is not None:
            raise ValueError("P2O requires same-reward filtering disabled")
        if not self.args.rollout_shuffle or self.data_source.dataset is None:
            raise ValueError("P2O requires the native shuffled global training dataset")
        if len(self.data_source.dataset) < self.args.rollout_batch_size:
            raise ValueError("P2O training catalog must contain at least one full submission window")
        config = CollectionConfig(**{key: getattr(self.args, key) for key in CollectionConfig.model_fields})
        self.owner = SamplingCoordinator(
            samples=self.data_source.dataset.origin_samples,
            config=config,
            seed=self.args.rollout_seed,
            temperature=self.args.p2o_template_temperature,
        )
        mix = MixConfig(self.args.p2o_task_mix_ratio, self.args.p2o_max_template_ratio)
        self.windows = SubmissionWindows(
            source=self.data_source, coordinator=self.owner, size=self.args.rollout_batch_size, config=mix
        )
        self._output.bind(self.owner, mix)
        self._phase_resumed = asyncio.Event()
        self._phase_resumed.set()

    async def _worker_loop(self):
        active = set()
        while True:
            if self.owner.collector.trigger_pending:
                await self._output.drain_capacity(len(self._running_tasks))
                if not active:
                    self._phase_resumed.clear()
                    await self._output.mark_boundary()
                    logger.info(
                        "P2O boundary drained cycle=%d pending_train=%d hard=%d",
                        self.owner.collector.cycle_id,
                        len(self._output.state_dict()),
                        self.owner.collector.hard_task_count,
                    )
                    await self._phase_resumed.wait()
                    continue
            else:
                await self._producer_resumed.wait()
                while self._scheduler.has_capacity(
                    pending_groups=len(active), group_budget=self._max_in_flight_groups()
                ):
                    active.add(self._submit_one_group())
            done, active = await self._scheduler.wait_for_progress(active)
            for task in done:
                prompt = next(entry.prompt_group for entry in self._running_tasks if entry.task is task)
                entry = self._collect_group_result(task, prompt)
                if self.owner.collector.trigger_pending:
                    await self._output.drain_capacity(len(self._running_tasks))
                await self._output.put(entry)
                self._running_tasks = [entry for entry in self._running_tasks if entry.task is not task]

    async def phase(self, operation, payload):
        if operation == "status":
            return dict(
                ready=self._output._phase_ready,
                cycle_id=self.owner.collector.cycle_id,
                trigger_pending=self.owner.collector.trigger_pending,
            )
        if operation == "freeze":
            if not self._output._phase_ready or self._running_tasks:
                raise RuntimeError("Cannot freeze before every submitted episode terminates")
            return self.owner.collector.freeze().model_dump(mode="json")
        if operation == "commit":
            cycle = TemplateCycle.model_validate(payload)
            if cycle.cycle_id < self.owner.collector.cycle_id:
                if self.owner.history.snapshot().cycles[-1] != cycle:
                    raise ValueError("Conflicting repeated cycle commit")
                return dict(committed=True)
            if not self._output._phase_ready or self._running_tasks:
                raise RuntimeError("Cannot commit before producer is drained")
            self.owner.finish_cycle(cycle)
            # Persist fresh assignment eligibility before the post-GEPA save,
            # while the producer is still paused. Restart cannot replay stale
            # assignments from a prefetched-but-unsubmitted window.
            self.windows.displaced.extendleft(reversed(self.windows.ready))
            self.windows.ready.clear()
            return dict(committed=True)
        if operation == "resume":
            if self.owner.collector.trigger_pending:
                raise RuntimeError("Cannot resume before the completed cycle is committed")
            # A prefetched but unsubmitted window gets new assignments, without
            # advancing the natural cursor or changing already generated tails.
            self.windows.displaced.extendleft(reversed(self.windows.ready))
            self.windows.ready.clear()
            self._output.resume()
            self._phase_resumed.set()
            return dict(resumed=True)
        raise ValueError(f"Unknown P2O phase operation: {operation}")

    async def dispose(self):
        await super().dispose()
        tasks = [entry.task for entry in self._running_tasks]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def _next_prompt_groups(self):
        in_flight = {self.owner.task(entry.prompt_group[0]) for entry in self._running_tasks}
        samples = self.windows.pop(retries=self._retry_buffer, in_flight=in_flight)
        # Unsubmitted prefetch/retries may survive a search; the attempt belongs
        # to the cycle it is submitted in. Its actual chosen template stays fixed.
        for sample in samples:
            sample.metadata["p2o"]["cycle_id"] = self.owner.collector.cycle_id
        return [samples]

    def _collect_group_result(self, task, prompt_group):
        entry = super()._collect_group_result(task, prompt_group)
        self.owner.observe(entry)  # Observe before native buffer validity/filter decisions.
        for sample in iter_samples(entry.group):
            attach_training_context(sample)
        return entry

    async def _drain(self, input):
        result = await super()._drain(input)
        result.metrics.update({f"p2o/{key}": value for key, value in self.owner.metrics.items()})
        result.metrics.update(
            {
                "p2o/hard_tasks": self.owner.collector.hard_task_count,
                "p2o/trigger_pending": int(self.owner.collector.trigger_pending),
                "p2o/history_templates": len(self.owner.history),
                "p2o/cycle_id": self.owner.collector.cycle_id,
            }
        )
        self.owner.metrics.clear()
        return result

    def save(self, directory: Path):
        super().save(directory / "native")
        save_simple_checkpoint(
            directory=directory / "p2o",
            data={
                "owner": self.owner.snapshot().model_dump_json(),
                "windows": self.windows.state_dict(),
            },
        )

    def load(self, directory: Path):
        super().load(directory / "native")
        state = load_simple_checkpoint(directory=directory / "p2o")
        self.owner.load_state(CoordinatorState.model_validate_json(state["owner"]))
        self.windows.load_state_dict(state["windows"])
