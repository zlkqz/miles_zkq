"""用参考解或空动作检查单个 SETA 任务，不调用模型、不训练。"""

import argparse
import asyncio
import json
from pathlib import Path

from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig, TrialConfig
from harbor.trial.trial import Trial

from examples.seta.progress import progress_reward, read_test_results
from examples.seta.docker_adapter import configure_docker


async def _check(args) -> None:
    config = TrialConfig(
        task=TaskConfig(path=args.tasks_dir / args.task_id),
        agent=AgentConfig(name=args.agent),
        environment=EnvironmentConfig(type="docker", delete=True),
        trials_dir=args.trials_dir,
        environment_build_timeout_multiplier=args.build_timeout_multiplier,
    )
    trial = await Trial.create(config)
    configure_docker(trial)
    result = await trial.run()
    if result.verifier_result is None:
        raise RuntimeError(f"任务未完成验证：{result.exception_info}；日志：{trial.paths.trial_dir}")
    tests = read_test_results(trial.paths.verifier_dir)
    print(
        json.dumps(
            {
                "trial_dir": str(trial.paths.trial_dir),
                "passed": sum(tests.values()),
                "total": len(tests),
                "reward": progress_reward(tests, args.full_pass_bonus),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-dir", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--trials-dir", type=Path, required=True)
    parser.add_argument("--agent", choices=("oracle", "nop"), default="oracle")
    parser.add_argument("--full-pass-bonus", type=float, default=0.2)
    parser.add_argument("--build-timeout-multiplier", type=float, default=1.0)
    asyncio.run(_check(parser.parse_args()))
