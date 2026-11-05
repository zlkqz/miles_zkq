"""SETA 的有限长度 episode：预算耗尽结束 agent，然后仍验证已有任务状态。"""

from examples.seta.p2o.templates import Template, augment_instruction
from harbor.agents.terminus_2.terminus_2 import Terminus2
from harbor.llms.base import ContextLengthExceededError, MaxSeqLenExceededError, SingleTurnMaxSeqLenExceededError


class SetaTerminus2(Terminus2):
    def __init__(self, *args, p2o_template=None, **kwargs):
        self._p2o_template = Template.from_dict(p2o_template) if p2o_template is not None else None
        super().__init__(*args, **kwargs)

    async def _run_agent_loop(self, *args, **kwargs):
        try:
            await super()._run_agent_loop(*args, **kwargs)
        except (ContextLengthExceededError, MaxSeqLenExceededError, SingleTurnMaxSeqLenExceededError) as error:
            # 不拼接/恢复截断的 assistant 消息；TITO 只收集实际生成的 token。
            # 预算用尽是策略终止状态，不能使整组轨迹失去任务验证奖励。
            self._seta_stop_reason = type(error).__name__
            self.logger.info("SETA agent 预算用尽，转入验证：%s", self._seta_stop_reason)

    async def run(self, instruction, environment, context):
        self._seta_stop_reason = None
        if self._p2o_template is not None:
            instruction = augment_instruction(instruction, self._p2o_template)
        await super().run(instruction, environment, context)
        if self._seta_stop_reason:
            context.metadata["seta_stop_reason"] = self._seta_stop_reason
