"""在远端 Harbor Python 中运行：python -m unittest tests.manual.test_seta_harbor_budget。"""

import asyncio
import logging
import unittest
from unittest.mock import patch

from harbor.agents.terminus_2.terminus_2 import Terminus2
from harbor.llms.base import ContextLengthExceededError

from examples.seta.terminus import SetaTerminus2


class BudgetTest(unittest.IsolatedAsyncioTestCase):
    async def test_budget_exhaustion_reaches_normal_agent_completion(self):
        agent = object.__new__(SetaTerminus2)
        agent.logger = logging.getLogger("budget-test")
        with patch.object(Terminus2, "_run_agent_loop", side_effect=ContextLengthExceededError):
            await agent._run_agent_loop()
        self.assertEqual(agent._seta_stop_reason, "ContextLengthExceededError")

    async def test_infrastructure_and_cancellation_still_propagate(self):
        agent = object.__new__(SetaTerminus2)
        for error in (RuntimeError("sandbox lost"), asyncio.CancelledError()):
            with patch.object(Terminus2, "_run_agent_loop", side_effect=error):
                with self.assertRaises(type(error)):
                    await agent._run_agent_loop()
