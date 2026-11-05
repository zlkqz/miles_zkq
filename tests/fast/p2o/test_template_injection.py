"""The actual remote agent receives the template; no shared task-file editing."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.seta.p2o.templates import Template


@pytest.fixture
def terminus(monkeypatch):
    class Base:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def run(self, instruction, environment, context):
            self.received = instruction
            context.metadata = {"native": True}

    monkeypatch.setitem(sys.modules, "harbor.agents.terminus_2.terminus_2", SimpleNamespace(Terminus2=Base))
    monkeypatch.setitem(
        sys.modules,
        "harbor.llms.base",
        SimpleNamespace(
            ContextLengthExceededError=RuntimeError,
            MaxSeqLenExceededError=RuntimeError,
            SingleTurnMaxSeqLenExceededError=RuntimeError,
        ),
    )
    path = Path(__file__).resolve().parents[3] / "examples/seta/terminus.py"
    spec = importlib.util.spec_from_file_location("seta_terminus_injection_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SetaTerminus2


async def test_remote_input_injection_is_per_instance(terminus):
    template = Template("Check files before editing.")
    augmented, plain = terminus(p2o_template=template.to_dict()), terminus()
    context = SimpleNamespace(metadata={})
    await augmented.run("original task", None, context)
    await plain.run("original task", None, context)
    assert augmented.received == "original task" + template.block
    assert plain.received == "original task"
    assert "p2o_template" not in augmented.kwargs
    assert context.metadata == {"native": True}


def test_corrupt_payload_rejected_before_remote_execution(terminus):
    payload = Template("Check files.").to_dict()
    payload["text"] = "Changed"
    with pytest.raises(ValueError, match="hash"):
        terminus(p2o_template=payload)
