"""Tests for the run instruction until done tool."""

import asyncio
import unittest
from unittest import mock

from absl.testing import absltest
from safari_sdk.agent.framework import config as framework_config
from safari_sdk.agent.framework.event_bus import event_bus
from safari_sdk.agent.framework.tools import run_instruction_for_duration
from safari_sdk.agent.framework.tools import tool


class RunInstructionForDurationTest(unittest.IsolatedAsyncioTestCase):

  def setUp(self):
    super().setUp()
    self.loop = asyncio.new_event_loop()
    asyncio.set_event_loop(self.loop)
    self.config = framework_config.AgentFrameworkConfig()
    self.bus = event_bus.EventBus(config=self.config)

  def tearDown(self):
    self.loop.close()
    super().tearDown()

  async def test_instantiation_test(self):
    run_instruction_for_duration.RunInstructionForDurationTool(
        bus=self.bus,
        embodiment_run_instruction_tool=mock.create_autospec(tool.Tool),
        embodiment_stop_tool=mock.create_autospec(tool.Tool),
        stop_on_success=True,
    )


if __name__ == "__main__":
  absltest.main()
