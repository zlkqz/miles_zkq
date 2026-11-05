"""DSV4.1 recordings require the same explicit checkpoint as a real launch."""

import pytest

from tests.fast.launch_scripts.py_harness import (
    REPO_ROOT,
    call_entrypoint,
    freeze_environment,
    import_launch_script,
    install_command_recorder,
)


@pytest.mark.parametrize("entrypoint", ["prepare_spmd", "train"])
def test_checkpoint_is_required_and_forwarded(monkeypatch, tmp_path, entrypoint):
    freeze_environment(monkeypatch)
    recording = install_command_recorder(monkeypatch)
    module = import_launch_script(REPO_ROOT / "scripts/run_deepseek_v4_1.py")

    with pytest.raises(AssertionError):
        call_entrypoint(module, entrypoint, {}, sandbox=tmp_path)
    assert not recording.commands

    checkpoint = str(tmp_path / "models" / "explicit-checkpoint")
    call_entrypoint(module, entrypoint, {"hf_checkpoint": checkpoint}, sandbox=tmp_path)
    assert any(f"--hf-checkpoint {checkpoint}" in command for command in recording.commands)
