"""将完整 SETA-Env 转换为 Miles JSONL 和扁平任务目录。

参数：
  --dataset-root：包含 SETA_Synth 和 SETA_Evolve 的目录。
  --output-dir：train.jsonl 和 tasks/ 的输出目录。
  --agent-name：可选；确定脚手架后，才写入 metadata.agent_name。

示例：
  python examples/seta/prepare_dataset.py \
      --dataset-root /ceph_home/zhangkaiqi2024/SETA-Env \
      --output-dir /ceph_home/zhangkaiqi2024/seta_miles_data
"""

import argparse
import json
import shutil
from pathlib import Path


_SUBSETS = ("SETA_Synth", "SETA_Evolve")


def _tasks(dataset_root: Path):
    for subset in _SUBSETS:
        subset_dir = dataset_root / subset
        if not subset_dir.is_dir():
            raise FileNotFoundError(subset_dir)
        for task_dir in sorted(subset_dir.iterdir()):
            if task_dir.is_dir() and (task_dir / "task.toml").is_file():
                yield subset, task_dir


def _copy_task(source: Path, target: Path) -> None:
    if target.exists():
        if not (target / "task.toml").is_file():
            raise ValueError(f"Incomplete target task: {target}")
        return
    temp = target.with_name(target.name + ".incomplete")
    if temp.exists():
        shutil.rmtree(temp)
    shutil.copytree(source, temp, symlinks=False)
    temp.rename(target)


def prepare(dataset_root: Path, output_dir: Path, *, agent_name: str | None = None, copy_tasks: bool = True) -> dict[str, int]:
    """写入全部任务，不按难度或奖励过滤；默认不指定 agent。"""
    dataset_root = dataset_root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks_dir = output_dir / "tasks"
    if copy_tasks:
        tasks_dir.mkdir(exist_ok=True)
    rows = []
    counts = {subset: 0 for subset in _SUBSETS}
    seen_ids = set()
    for subset, task_dir in _tasks(dataset_root):
        instruction_path = task_dir / "instruction.md"
        if not instruction_path.is_file():
            raise FileNotFoundError(instruction_path)
        instance_id = f"{subset}__{task_dir.name}"
        if instance_id in seen_ids:
            raise ValueError(f"Duplicate instance_id: {instance_id}")
        seen_ids.add(instance_id)
        instruction = instruction_path.read_text(encoding="utf-8").strip()
        if not instruction:
            raise ValueError(f"Empty instruction: {instruction_path}")
        if copy_tasks:
            _copy_task(task_dir, tasks_dir / instance_id)
        rows.append(
            {
                "prompt": instruction,
                "metadata": {
                    "instance_id": instance_id,
                    "task_id": task_dir.name,
                    "subset": subset,
                    **({"agent_name": agent_name} if agent_name else {}),
                },
            }
        )
        counts[subset] += 1
    output = output_dir / "train.jsonl"
    temp_output = output.with_suffix(".jsonl.incomplete")
    with temp_output.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp_output.replace(output)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--agent-name", help="确定脚手架后填写；默认不指定")
    parser.add_argument("--no-copy-tasks", action="store_true")
    args = parser.parse_args()
    counts = prepare(args.dataset_root, args.output_dir, agent_name=args.agent_name, copy_tasks=not args.no_copy_tasks)
    print(f"已写入 {args.output_dir / 'train.jsonl'}：{sum(counts.values())} 个任务，{counts}")


if __name__ == "__main__":
    main()
