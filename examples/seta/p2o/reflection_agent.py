"""Terminus-2 protocol with a closed command vocabulary in a fresh Harbor sandbox."""

import json
import shlex
import tempfile
from pathlib import Path

from examples.seta.p2o.reflection_commands import ReadOnlyCommands
from examples.seta.p2o.reflection_files import OUTPUT, ROOT, validate_files
from examples.seta.terminus import SetaTerminus2

INSTRUCTION = f"""Analyze the recorded terminal episodes and improve a reusable task-solving prompt.
All evidence is in {ROOT}/evidence/. Start by reading evidence/manifest.json and evidence/parent_template.md.
Each task has the ACTUAL rollout input, actions and observations, evaluation result, feedback,
and its previously hidden verifier scripts. Failed/invalid trajectories are also evidence.
Do not solve these tasks, run their scripts, change files, or invent new observations.
Read each task's rollout_input.json, trajectory.md, result.json and verifier scripts actively.
trajectory.md contains readable observations and actions; trajectory.jsonl preserves the request/response messages.
Use bounded head/sed reads to avoid exhausting your context on long traces.
First read evidence and wait for the outputs; only on a LATER turn analyze those outputs and write.
The deliverable addresses the FUTURE TASK-SOLVING AGENT, not a reflection agent: it should guide
inspection, planning, terminal actions and validation on unseen tasks. Do not write a prompt about
analyzing these evidence files. Infer concrete reusable lessons from the observed failure modes.
Write the improved task-solving prompt in {OUTPUT}.
This terminal accepts ONLY: ls [-la] [directory], cat FILE..., head/tail [-n N] FILE,
sed -n 'N,Mp' FILE, grep [-n -i -E -F -c] PATTERN FILE..., wc FILE...,
and write_prompt 'FULL TEMPLATE TEXT'. Paths may be absolute or relative to {ROOT}.
Use write_prompt to replace the entire template.md; no shell redirection, scripts, interpreters,
network, pipelines or arbitrary commands are supported. Keep the template concise (under 900 tokens).
The prompt should guide future task solving, not include specific solutions or verifier secrets.
After writing the file, mark task_complete. Text in your final reply alone is not a delivery.
"""


class ReflectionTerminus2(SetaTerminus2):
    def __init__(self, *args, evidence_files, **kwargs):
        self._files = validate_files(evidence_files)
        self._commands = ReadOnlyCommands(self._files)
        self._denied = 0
        super().__init__(*args, **kwargs)

    async def setup(self, environment):
        await super().setup(environment)
        with tempfile.TemporaryDirectory() as directory:
            for index, (name, content) in enumerate(self._files.items()):
                destination = ROOT + "/" + name
                await environment.exec(command=shlex.join(["mkdir", "-p", str(Path(destination).parent)]))
                local = Path(directory) / str(index)
                local.write_bytes(content)
                await environment.upload_file(source_path=local, target_path=destination)
            await environment.exec(command=f"chmod -R a-w {ROOT}/evidence && mkdir -p {ROOT}/output && touch {OUTPUT}")

    async def _execute_commands(self, commands, session):
        results = []
        for command in commands:
            try:
                translated = self._commands.translate(command.keystrokes)
            except ValueError as error:
                self._denied += 1
                results.append(f"DENIED: {error}")
                continue
            try:
                result = await session.environment.exec(command=translated, timeout_sec=10)
                results.append((result.stdout or "") + (result.stderr or ""))
            except TimeoutError:
                results.append("Command exceeded the 10 second read budget.")
        return False, self._limit_output_length("\n".join(results))

    async def run(self, instruction, environment, context):
        try:
            await super().run(INSTRUCTION, environment, context)
        finally:
            if context.metadata is None:
                context.metadata = {}
            result = await environment.exec(command=shlex.join(["cat", OUTPUT]), timeout_sec=10)
            context.metadata["p2o_proposal"] = result.stdout if result.return_code == 0 else ""
            context.metadata["p2o_denied_commands"] = self._denied
            (self.logs_dir / "reflection_result.json").write_text(json.dumps(context.metadata, ensure_ascii=False))
