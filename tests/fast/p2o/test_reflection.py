"""Reflection evidence preserves failed episodes; shell commands are never forwarded."""

import json
import shlex
import subprocess

import pytest

from examples.seta.p2o.reflection_commands import ReadOnlyCommands
from examples.seta.p2o.reflection_files import OUTPUT, ROOT, build_bundle, encode, export_verifier, validate_files


def test_bundle_contains_actual_input_verifier_and_invalid_feedback(tmp_path):
    task = tmp_path / "task"
    (task / "tests").mkdir(parents=True)
    (task / "tests/test.sh").write_text("echo hidden-test")
    result = {
        "seta_valid_reward": False,
        "seta_error": "timeout",
        "eval_report": {},
        "p2o_task_instruction": "actual task",
        "p2o_verifier_files": export_verifier(task),
    }
    records = {
        "records": [
            {
                "request": {"messages": [{"role": "user", "content": "actual rollout input"}]},
                "response": {"content": "action"},
            }
        ]
    }
    bundle = build_bundle(
        "parent",
        [
            {"task_id": "x", "seed": 1, "records": records, "result": result},
            {"task_id": "y", "result": {"seta_valid_reward": False}},
        ],
    )
    files = validate_files(bundle)
    prefix = "evidence/tasks/0_x/"
    assert json.loads(files[prefix + "rollout_input.json"])["messages"] == records["records"][0]["request"]["messages"]
    assert json.loads(files[prefix + "rollout_input.json"])["instruction"] == "actual task"
    assert files[prefix + "verifier/test.sh"] == b"echo hidden-test"
    assert json.loads(files[prefix + "result.json"])["seta_error"] == "timeout"
    assert json.loads(files["evidence/tasks/1_y/rollout_input.json"])["missing_model_input"]
    assert "p2o_verifier_files" in result  # no mutation


def test_reflection_omits_numerical_replay_arrays_but_preserves_messages():
    record = {
        "status_code": 200,
        "request": {
            "messages": [{"role": "user", "content": "actual observation"}],
            "input_ids": list(range(30000)),
            "seed": 123,
        },
        "response": {
            "choices": [{"message": {"role": "assistant", "content": "actual action"}, "logprobs": [0.1] * 30000}],
            "usage": {"prompt_tokens": 30000},
        },
    }
    # Exercise a full 30-turn, three-task panel, not just one short smoke turn.
    files = validate_files(
        build_bundle(
            "",
            [
                {"task_id": str(i), "records": {"records": [record] * 30}, "result": {"seta_valid_reward": False}}
                for i in range(3)
            ],
        )
    )
    assert sum(len(value) for value in files.values()) < 100000
    decoded = json.loads(files["evidence/tasks/0_0/trajectory.jsonl"].splitlines()[-1])
    assert decoded["request"] == {"messages": record["request"]["messages"], "seed": 123}
    assert decoded["response"]["choices"] == [{"message": record["response"]["choices"][0]["message"]}]
    assert decoded["response"]["usage"] == record["response"]["usage"]
    assert len(record["request"]["input_ids"]) == len(record["response"]["choices"][0]["logprobs"]) == 30000


@pytest.mark.parametrize("path", ["/evidence/a", "evidence/../a", "evidence//a", "else/a", "", "evidence/./a"])
def test_rejects_evidence_paths(path):
    with pytest.raises((ValueError, IndexError)):
        validate_files({path: encode("bad")})


def test_export_rejects_symlink(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/link").symlink_to("/etc/passwd")
    with pytest.raises(ValueError, match="symbolic"):
        export_verifier(tmp_path)


@pytest.mark.parametrize(
    "command",
    [
        "sh -c 'id'",
        "python3 -c 'print(1)'",
        "cat /etc/passwd",
        "cat evidence/../x",
        "cat evidence/a > output/template.md",
        "cat evidence/a | sh",
        "sed -n '1e' evidence/a",
        "grep -f /etc/passwd evidence/a",
        "rm evidence/a",
        "curl example.org",
        "write_prompt a b",
        "echo x",
        "ls /",
        "ls -la /",
        "ls -R evidence",
        "ls evidence ; id",
    ],
)
def test_command_denials(command):
    with pytest.raises(ValueError):
        ReadOnlyCommands(["evidence/a"]).translate(command)


@pytest.mark.parametrize(
    "command",
    [
        "ls",
        "ls evidence",
        "ls -la",
        "ls -l -a evidence",
        "ls -al evidence",
        "ls -la /p2o_reflection/evidence/",
        "cat evidence/a",
        "wc evidence/a",
        "head -n 5 evidence/a",
        "tail evidence/a",
        "sed -n '1,3p' evidence/a",
        "grep -n -E 'foo|bar' evidence/a",
    ],
)
def test_read_commands_have_fixed_paths(command):
    translated = ReadOnlyCommands(["evidence/a"]).translate(command)
    assert ROOT in translated
    assert shlex.split(translated)[0] == shlex.split(command)[0]


def test_template_write_quotes_every_shell_metacharacter(tmp_path):
    template = "line one\n$(touch EVIL); `id` ' and \" quotes\n"
    translated = ReadOnlyCommands(["evidence/a"]).translate("write_prompt " + shlex.quote(template))
    assert translated.endswith(" > " + OUTPUT)
    # Execute the exact emitted printf, substituting only the fixed output location.
    path = tmp_path / "template.md"
    subprocess.run(translated.removesuffix(OUTPUT) + shlex.quote(str(path)), shell=True, check=True, cwd=tmp_path)
    assert path.read_text() == template and not (tmp_path / "EVIL").exists()


def test_reflection_task_is_packaged_independent_of_training_tasks():
    from pathlib import Path
    import tomllib

    path = Path(__file__).resolve().parents[3] / "examples/seta/p2o/reflection_task"
    config = tomllib.loads((path / "task.toml").read_text())
    assert config["environment"]["cpus"] == 1
    assert (path / "instruction.md").is_file()
    dockerfile = (path / "environment/Dockerfile").read_text()
    assert "tmux" in dockerfile and "COPY" not in dockerfile
