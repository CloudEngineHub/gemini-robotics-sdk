"""Tests for the fulfill order tool."""

import asyncio
import unittest
from unittest import mock

from google.genai import types as genai_types

from absl.testing import absltest
from safari_sdk.agent.framework import config as framework_config
from safari_sdk.agent.framework.event_bus import event_bus
from safari_sdk.agent.framework.tools import fulfill_order_a1
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
    mock_tool = mock.create_autospec(tool.Tool)
    mock_tool.declaration = genai_types.FunctionDeclaration(
        name="mock_tool",
        description="A mock tool.",
    )
    fulfill_order_a1.FulfillOrderTool(
        bus=self.bus,
        model_name="models/gemini-2.5-flash",
        api_key="test_api_key",
        toolset=[mock_tool],
    )


if __name__ == "__main__":
  absltest.main()
