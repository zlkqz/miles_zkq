"""One raw event stream per cycle, with native W&B axes and local export plots."""

import json
from pathlib import Path

from examples.seta.p2o.checkpoint import atomic_json
from miles.utils.tracking_utils import tracking


class SearchMetrics:
    def __init__(self, *, args, directory, cycle_id, training_step):
        self.args = args
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.prefix = f"gepa/cycle_{cycle_id}"
        self.step_key = self.prefix + "/event"
        self.training_step = training_step
        tracking.define_step_key_metric_group(self.prefix, self.step_key)
        self.timing_prefix = self.prefix + "/timing"
        self.timing_step_key = self.timing_prefix + "/event"
        tracking.define_step_key_metric_group(self.timing_prefix, self.timing_step_key)

    def __call__(self, event):
        # Search.state is the transactional source of truth. JSONL is a convenient
        # derived stream; final export rebuilds it to remove any interrupted write.
        with (self.directory / "metrics.jsonl").open("a") as file:
            file.write(json.dumps(event, ensure_ascii=False) + "\n")
        metrics = {
            f"{self.prefix}/{key}": value for key, value in event.items() if isinstance(value, (int, float, bool))
        }
        metrics[self.prefix + "/training_step"] = self.training_step
        tracking.log(self.args, metrics, step_key=self.step_key)

    def timing(self, event):
        """Separate axis: phase timers continue after the last search event."""
        stage = event["stage"]
        tracking.log(
            self.args,
            {
                self.timing_step_key: event["event"],
                self.timing_prefix + f"/{stage}_seconds": event["seconds"],
                self.timing_prefix + f"/{stage}_cumulative_seconds": event["cumulative_seconds"],
                self.timing_prefix + "/training_step": self.training_step,
                self.timing_prefix + "/failed": int(event["status"] != "completed"),
            },
            step_key=self.timing_step_key,
        )

    def finish(self, state):
        # Plotting is an optional dependency of this example, imported only when
        # a cycle finishes; it is preflighted before starting GPU work.
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        events = state["events"]
        atomic_json(self.directory / "metrics.json", events)
        (self.directory / "metrics.jsonl").write_text(
            "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in events)
        )
        figure = Figure(figsize=(12, 8), constrained_layout=True)
        FigureCanvasAgg(figure)
        axes = figure.subplots(2, 2)
        for axis, keys in zip(
            axes.flat, [("best_dev",), ("candidates", "pareto"), ("started", "completed"), ("patience",)], strict=True
        ):
            for key in keys:
                axis.plot([e["event"] for e in events], [e[key] for e in events], label=key)
            axis.set_xlabel("GEPA event")
            axis.legend()
            axis.grid(alpha=0.2)
        figure.savefig(self.directory / "metrics.png", dpi=150)
        figure.savefig(self.directory / "metrics.svg")
