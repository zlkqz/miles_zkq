"""Atomically pair native model checkpoints with immutable rollout/phase state.

A pre/post pair shares the same frozen model/optimizer/RNG directory. Only its
rollout state differs. Native save directories remain under SAVE; the small
pair directories are valid native --load roots, selected by one atomic pointer.
"""

import json
import logging
import os
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}-{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w") as output:
            json.dump(value, output, ensure_ascii=False, indent=2, allow_nan=False)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        temporary.unlink(missing_ok=True)


class CheckpointLedger:
    def __init__(self, root, *, identity):
        self.root = Path(root).resolve()
        self.identity = identity
        self.pointer = self.root / "p2o_checkpoint.json"
        self._saved_models = set()
        if latest := self.latest():
            self._saved_models.add(latest["rollout_id"])

    def latest(self):
        if not self.pointer.exists():
            return None
        pointer = json.loads(self.pointer.read_text())
        directory = (self.root / pointer["directory"]).resolve()
        if not directory.is_relative_to(self.root):
            raise ValueError("checkpoint pointer escapes SAVE")
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest["schema_version"] != 1 or manifest["identity"] != self.identity:
            raise ValueError("P2O checkpoint model/tokenizer/data identity differs from this run")
        step = manifest["rollout_id"]
        if not (directory / f"iter_{step:07d}").is_dir() or not (directory / "rollout" / str(step)).is_dir():
            raise ValueError("Incomplete P2O checkpoint pair")
        return {**manifest, "directory": str(directory)}

    async def save(self, *, rollout_id, cycle_id, phase, save_model, save_rollout):
        if rollout_id < 0 or phase not in {"periodic", "pre_gepa", "post_gepa"}:
            raise ValueError("A phase checkpoint requires a completed training update and a known phase")
        if rollout_id not in self._saved_models:
            await save_model(rollout_id, force_sync=True)
        model = self.root / f"iter_{rollout_id:07d}"
        if not model.is_dir():
            raise ValueError(f"Native checkpoint did not publish {model}")
        self._saved_models.add(rollout_id)
        directory = (
            self.root / "p2o_checkpoints" / f"step_{rollout_id:07d}_cycle_{cycle_id}_{phase}_{uuid.uuid4().hex}"
        )
        directory.mkdir(parents=True)
        await save_rollout(rollout_id, directory=str(directory / "rollout" / str(rollout_id)))
        (directory / model.name).symlink_to(model, target_is_directory=True)
        (directory / "latest_checkpointed_iteration.txt").write_text(str(rollout_id))
        manifest = dict(
            schema_version=1, rollout_id=rollout_id, cycle_id=cycle_id, phase=phase, identity=self.identity
        )
        atomic_json(directory / "manifest.json", manifest)
        atomic_json(self.pointer, dict(directory=str(directory.relative_to(self.root))))
        logger.info("P2O checkpoint paired step=%d cycle=%d phase=%s path=%s", rollout_id, cycle_id, phase, directory)
        return {**manifest, "directory": str(directory)}
