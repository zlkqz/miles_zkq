import json
import runpy
import sys
from pathlib import Path

import pytest

from miles.utils.external_utils.command_utils.ray_backend.backend import RayCommandBackend


def test_workplace_launch_uses_the_configured_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """The workplace recipe submits through its backend instead of removed module exports."""
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(RayCommandBackend, "execute_train", lambda self, **kwargs: calls.append(kwargs))
    module = runpy.run_path(
        str(
            Path(__file__).resolve().parents[3]
            / "examples/experimental/nemo-gym-workspace-assistant/run_nemotron35_workplace.py"
        )
    )
    args = module["ScriptArgs"]()

    module["execute"](args)

    assert len(calls) == 1
    assert calls[0]["config"] is args
    assert calls[0]["train_script"] == "train_async.py"


def test_workplace_cli_loads_json_configuration(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(RayCommandBackend, "execute_train", lambda self, **kwargs: calls.append(kwargs))
    config = tmp_path / "launcher.json"
    config.write_text(json.dumps({"num_rollout": 2, "verifier_url": "http://verifier:8211"}))
    monkeypatch.setattr(sys, "argv", ["workplace", "--config", str(config)])

    runpy.run_path(
        str(
            Path(__file__).resolve().parents[3]
            / "examples/experimental/nemo-gym-workspace-assistant/run_nemotron35_workplace.py"
        ),
        run_name="__main__",
    )

    assert len(calls) == 1
    assert "--num-rollout 2" in calls[0]["train_args"]
    assert calls[0]["extra_env_vars"]["WORKPLACE_RESOURCE_URL"] == "http://verifier:8211"
