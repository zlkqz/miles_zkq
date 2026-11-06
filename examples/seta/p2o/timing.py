"""Durable wall-clock intervals for asynchronous GEPA stages and phase attempts."""

import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path

from examples.seta.p2o.checkpoint import atomic_json

logger = logging.getLogger(__name__)


class GepaTimings:
    """One owner per cycle; concurrent stages may overlap and must not be added as wall time.

    Closed intervals survive restart. Open intervals from a dead process are
    marked interrupted, with unknown duration; downtime is never guessed.
    """

    def __init__(self, directory, *, emit=None):
        self.directory = Path(directory)
        self.path = self.directory / "timing.json"
        self.emit = emit or (lambda event: None)
        self.records = json.loads(self.path.read_text())["records"] if self.path.exists() else []
        for record in self.records:
            if record["status"] == "running":
                record["status"] = "interrupted"
        self._persist()

    @contextmanager
    def measure(self, stage, **identity):
        record = dict(
            interval=len(self.records),
            stage=stage,
            status="running",
            started_unix=time.time(),
            seconds=None,
            **identity,
        )
        self.records.append(record)
        self._persist()
        started = time.monotonic()
        status = "failed"
        try:
            yield
            status = "completed"
        finally:
            record.update(seconds=time.monotonic() - started, status=status, finished_unix=time.time())
            record["event"] = sum(r["status"] != "running" for r in self.records) - 1
            record["cumulative_seconds"] = self.summary()[stage]["seconds"]
            self._persist()
            event = dict(record)
            self.emit(event)
            logger.info("P2O GEPA timing %s", json.dumps(event, ensure_ascii=False))

    def summary(self):
        result = {}
        for record in self.records:
            value = result.setdefault(
                record["stage"], dict(seconds=0.0, completed=0, failed=0, interrupted=0, running=0)
            )
            value[record["status"]] += 1
            if record["seconds"] is not None:
                value["seconds"] += record["seconds"]
        return result

    def _persist(self):
        atomic_json(self.path, dict(schema_version=1, records=self.records, summary=self.summary()))
        # Derived JSONL mirrors all intervals, including interrupted attempts.
        (self.directory / "timing.jsonl").write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in self.records)
        )

    def plot(self):
        # Optional example dependency, preflighted by the driver.
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        summary = self.summary()
        figure = Figure(figsize=(12, 6), constrained_layout=True)
        FigureCanvasAgg(figure)
        axis = figure.subplots()
        axis.barh(list(summary), [value["seconds"] for value in summary.values()])
        axis.set_xlabel("Measured seconds (overlapping stages are not additive)")
        axis.set_title("GEPA stage wall durations across process attempts; downtime excluded")
        figure.savefig(self.directory / "timing.png", dpi=150)
        figure.savefig(self.directory / "timing.svg")
