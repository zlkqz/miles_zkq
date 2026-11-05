"""Persisted asynchronous GEPA; budget reservations and result commits have one owner."""

import asyncio
import json
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from examples.seta.p2o.checkpoint import atomic_json
from examples.seta.p2o.history import TemplateCycle, TemplateEvaluation
from examples.seta.p2o.templates import Template

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchConfig:
    concurrency: int = 16
    budget: int = 256
    patience: int = 32
    minibatch: int = 3
    reflection_max_turns: int = 20
    max_template_tokens: int = 1024

    def __post_init__(self):
        if any(type(value) is not int or value < 1 for value in asdict(self).values()):
            raise ValueError("GEPA search counts must be positive integers")
        if self.reflection_max_turns > 100:
            raise ValueError("Reflection max turns must be <=100")


def pareto_indices(candidates):
    vectors = [entry.scores.vector for entry in candidates]
    return [
        i
        for i, vector in enumerate(vectors)
        if not any(
            all(a >= b for a, b in zip(other, vector, strict=True))
            and any(a > b for a, b in zip(other, vector, strict=True))
            for j, other in enumerate(vectors)
            if i != j
        )
    ]


class Search:
    """Runtime owns episode caching; this object owns ordered, exactly-once commits.

    An interrupted reserved exploration keeps its ID, parent and minibatch. On
    restart its completed episode caches are reused; it does not consume a new
    budget slot. External requests interrupted before being durably recorded may
    execute again, but cannot double-count candidate admission or patience.
    """

    def __init__(self, *, panel, config, runtime, directory, policy_version, seed, emit=None):
        self.panel, self.config, self.runtime = panel, config, runtime
        self.directory = Path(directory)
        self.path = self.directory / "search_state.json"
        self.emit = emit or (lambda event: None)
        self.identity = dict(
            panel=panel.model_dump(mode="json"), config=asdict(config), policy_version=policy_version, seed=seed
        )
        if len(panel.search_train) < config.minibatch:
            raise ValueError("Frozen search-train panel is smaller than GEPA minibatch")
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
            if self.state["identity"] != self.identity:
                raise ValueError("GEPA journal differs from frozen panel/config/policy")
        else:
            self.state = dict(identity=self.identity, candidates=[], explorations=[], failures=0, stop=None, events=[])
            self._persist()

    @property
    def candidates(self):
        return [TemplateEvaluation.model_validate(entry) for entry in self.state["candidates"]]

    def _persist(self):
        atomic_json(self.path, self.state)

    def _event(self, kind, **values):
        candidates = self.candidates
        event = dict(
            event=len(self.state["events"]),
            kind=kind,
            cycle=self.panel.cycle_id,
            started=len(self.state["explorations"]),
            completed=sum(e["done"] for e in self.state["explorations"]),
            candidates=len(candidates),
            pareto=len(pareto_indices(candidates)),
            best_dev=max((c.scores.mean for c in candidates), default=0),
            patience=self.state["failures"],
            stop=self.state["stop"],
            pending_dev=sum(e.get("stage") == "dev" and not e["done"] for e in self.state["explorations"]),
            **getattr(self.runtime, "metrics", {}),
            **values,
        )
        self.state["events"].append(event)
        self._persist()  # All state changes and their metrics commit atomically.
        self.emit(event)
        logger.info("P2O GEPA %s", json.dumps(event, ensure_ascii=False))

    def _reserve(self, busy):
        candidates = self.candidates
        rng = random.Random(self.identity["seed"] + len(self.state["explorations"]))
        if len(candidates) <= self.config.concurrency:
            available = [i for i in range(len(candidates)) if i not in busy]
        else:
            available = pareto_indices(candidates)
        if not available:
            return None
        # Prefer frontier parents explored least often; duplication is allowed
        # only once candidate count exceeds the concurrency cap.
        counts = {i: sum(e["parent"] == i for e in self.state["explorations"]) for i in available}
        minimum = min(counts.values())
        parent = rng.choice([i for i in available if counts[i] == minimum])
        tasks = rng.sample(range(len(self.panel.search_train)), self.config.minibatch)
        entry = dict(
            id=len(self.state["explorations"]),
            parent=parent,
            tasks=tasks,
            seeds=[rng.randrange(2**32) for _ in tasks],
            done=False,
            stage="reserved",
            candidate_count=len(candidates),
            frontier=pareto_indices(candidates),
        )
        self.state["explorations"].append(entry)
        if len(self.state["explorations"]) == self.config.budget:
            self.state["stop"] = "budget"
        self._event("reserved", exploration=entry["id"], parent=parent)
        return entry

    async def _explore(self, entry):
        parent = self.candidates[entry["parent"]]
        tasks = [self.panel.search_train[i].task for i in entry["tasks"]]
        self._stage(entry, "before")
        before = await self.runtime.minibatch(parent.text, tasks, entry["seeds"], entry["id"], "before")
        entry["parent_evidence"] = [self.runtime.evidence(x).model_dump(mode="json") for x in before]
        self._stage(entry, "reflection")
        proposal = await self.runtime.reflect(parent.text, before, entry["id"])
        if not proposal:
            return dict(reason="no_template")
        try:
            template = Template(proposal)
            self.runtime.validate_template(template)
        except ValueError as error:
            return dict(reason="invalid_template", error=str(error))
        entry["proposal"] = template.to_dict()
        self._stage(entry, "after")
        after = await self.runtime.minibatch(template.text, tasks, entry["seeds"], entry["id"], "after")
        before_score, after_score = self.runtime.mean(before), self.runtime.mean(after)
        if after_score <= before_score:
            return dict(reason="minibatch_rejected", before=before_score, after=after_score)
        self._stage(entry, "dev")
        evaluation = await self.runtime.evaluate(template.text, f"exploration_{entry['id']}")
        # Search-train successes inform task-specific assignment as well as dev successes.
        evaluation = evaluation.model_copy(update={"search_episodes": tuple(self.runtime.evidence(x) for x in after)})
        return dict(
            reason="admitted", evaluation=evaluation.model_dump(mode="json"), before=before_score, after=after_score
        )

    def _stage(self, entry, stage):
        entry["stage"] = stage
        self._event("stage", exploration=entry["id"], stage=stage)

    def _commit(self, entry, result):
        if entry["done"]:
            return
        previous = self.candidates
        best = max(c.scores.mean for c in previous)
        if entry.get("parent_evidence"):
            self.state["candidates"][entry["parent"]]["search_episodes"].extend(entry["parent_evidence"])
        improved = False
        if result["reason"] == "admitted":
            candidate = TemplateEvaluation.model_validate(result["evaluation"])
            if candidate.template_id not in {c.template_id for c in previous}:
                self.state["candidates"].append(candidate.model_dump(mode="json"))
                improved = candidate.scores.mean > best
            else:
                index = next(i for i, c in enumerate(previous) if c.template_id == candidate.template_id)
                self.state["candidates"][index]["search_episodes"].extend(result["evaluation"]["search_episodes"])
                result = {**result, "reason": "duplicate_template"}
        entry.update(done=True, result=result)
        if len(self.state["candidates"]) > self.config.concurrency:
            self.state["failures"] = 0 if improved else self.state["failures"] + 1
            if self.state["failures"] >= self.config.patience and self.state["stop"] is None:
                self.state["stop"] = "patience"
        self._event(
            "completed",
            exploration=entry["id"],
            reason=result["reason"],
            improved=improved,
            before=result.get("before"),
            after=result.get("after"),
        )

    async def run(self):
        if not self.state["candidates"]:
            bootstrap = await self.runtime.evaluate("", "bootstrap")
            self.state["candidates"].append(bootstrap.model_dump(mode="json"))
            self._event("bootstrap")
        active = {}
        try:
            # Replay reserved-but-uncommitted work, even when budget/early-stop
            # already latched. Such work must drain, not disappear on restart.
            for entry in self.state["explorations"]:
                if not entry["done"]:
                    active[asyncio.create_task(self._explore(entry))] = entry
            while True:
                while self.state["stop"] is None and len(active) < min(len(self.candidates), self.config.concurrency):
                    entry = self._reserve({e["parent"] for e in active.values()})
                    if entry is None:
                        break
                    active[asyncio.create_task(self._explore(entry))] = entry
                if not active:
                    break
                done, _ = await asyncio.wait(active, return_when=asyncio.FIRST_COMPLETED)
                # Ready tasks' order is persisted at commit time. No assumption
                # that submission order equals completion order is made.
                for future in done:
                    entry = active.pop(future)
                    self._commit(entry, future.result())
            self._event("finished")
            return TemplateCycle(cycle_id=self.panel.cycle_id, evaluations=tuple(self.candidates))
        finally:
            for future in active:
                future.cancel()
            await asyncio.gather(*active, return_exceptions=True)
