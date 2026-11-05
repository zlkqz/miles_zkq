"""Bounded agent-readable evidence; full native records stay in the episode cache."""

import base64
import hashlib
import json
import re
from pathlib import Path, PurePosixPath

ROOT = "/p2o_reflection"
OUTPUT = ROOT + "/output/template.md"
MAX_BUNDLE_BYTES = 32 * 1024 * 1024


def encode(content):
    return base64.b64encode(content if isinstance(content, bytes) else content.encode()).decode()


def validate_files(files):
    if not isinstance(files, dict) or not files or len(files) > 2048:
        raise ValueError("Reflection requires a bounded evidence file map")
    total, decoded = 0, {}
    for name, value in files.items():
        path = PurePosixPath(name)
        if (
            not path.parts
            or path.is_absolute()
            or ".." in path.parts
            or str(path) != name
            or path.parts[0] != "evidence"
        ):
            raise ValueError("Reflection evidence path must stay under evidence/")
        raw = base64.b64decode(value, validate=True)
        total += len(raw)
        if total > MAX_BUNDLE_BYTES:
            raise ValueError("Reflection evidence exceeds transport budget")
        decoded[name] = raw
    return decoded


def export_verifier(task):
    root = Path(task) / "tests"
    files = {}
    total = 0
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Verifier evidence must not follow symbolic links")
        if path.is_file():
            if path.stat().st_size + total > MAX_BUNDLE_BYTES:
                raise ValueError("Verifier evidence exceeds transport budget")
            content = path.read_bytes()
            total += len(content)
            files[str(path.relative_to(root))] = encode(content)
    return files


def _readable_record(record):
    # Native session records retain exact token IDs/log-probs for training and
    # audit. They are redundant for reflection and grow with every context turn.
    # Keep messages, actions, sampling settings, timing and feedback unchanged.
    request = {key: value for key, value in record.get("request", {}).items() if key != "input_ids"}
    response = dict(record.get("response", {}))
    if "choices" in response:
        response["choices"] = [
            {key: value for key, value in choice.items() if key != "logprobs"} for choice in response["choices"]
        ]
    return {**record, "request": request, "response": response}


def build_bundle(parent, episodes):
    files = {"evidence/parent_template.md": encode(parent)}
    for index, episode in enumerate(episodes):
        task_id = episode["task_id"]
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", task_id):
            raise ValueError("Invalid task identity in reflection evidence")
        prefix = f"evidence/tasks/{index}_{task_id}"
        records = episode.get("records", {}).get("records", [])
        result = dict(episode["result"])
        verifier = result.pop("p2o_verifier_files", {})
        for name, content in verifier.items():
            files[f"{prefix}/verifier/{name}"] = content
        initial = records[0]["request"].get("messages") if records else None
        files[f"{prefix}/rollout_input.json"] = encode(
            json.dumps(
                {
                    "messages": initial,
                    "task_id": task_id,
                    "seed": episode.get("seed"),
                    "missing_model_input": initial is None,
                    "instruction": result.pop("p2o_task_instruction", None),
                },
                ensure_ascii=False,
            )
        )
        files[f"{prefix}/trajectory.jsonl"] = encode(
            "\n".join(json.dumps(_readable_record(r), ensure_ascii=False) for r in records)
        )
        readable = []
        for turn, record in enumerate(records):
            messages = record.get("request", {}).get("messages", [])
            observation = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
            choices = record.get("response", {}).get("choices", [])
            action = choices[0].get("message", {}).get("content", "") if choices else ""
            readable.append(f"## Turn {turn + 1}\nObservation/input:\n{observation}\n\nAction:\n{action}\n")
        files[f"{prefix}/trajectory.md"] = encode("\n".join(readable))
        files[f"{prefix}/result.json"] = encode(json.dumps(result, ensure_ascii=False))
        files[f"{prefix}/feedback.json"] = encode(
            json.dumps(
                {
                    "valid": result.get("seta_valid_reward", False),
                    "error": result.get("seta_error"),
                    "exit_status": result.get("exit_status"),
                    "verifier_files": list(verifier),
                },
                ensure_ascii=False,
            )
        )
    decoded = validate_files(files)
    manifest = {
        name: {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
        for name, content in decoded.items()
    }
    files["evidence/manifest.json"] = encode(json.dumps(manifest, indent=2))
    validate_files(files)
    return files
