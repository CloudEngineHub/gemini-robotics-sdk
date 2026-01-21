"""Tests for the tool base class."""

import asyncio
import unittest
from unittest import mock

from google.genai import types as genai_types

from absl.testing import absltest
from safari_sdk.agent.framework import config as framework_config
from safari_sdk.agent.framework import types as framework_types
from safari_sdk.agent.framework.event_bus import event_bus
from safari_sdk.agent.framework.tools import tool


class ToolTest(unittest.IsolatedAsyncioTestCase):

  def setUp(self):
    super().setUp()
    self.loop = asyncio.new_event_loop()
    asyncio.set_event_loop(self.loop)
    self.config = framework_config.AgentFrameworkConfig()
    self.bus = event_bus.EventBus(config=self.config)

  def tearDown(self):
    self.loop.close()
    super().tearDown()

  async def test_tool_cancellation(self):

    class CancellableTool(tool.Tool):
      cancelled = False

      def _handle_tool_call_cancellation_events(self, event: event_bus.Event):
        self.cancelled = True

    tool_inst = CancellableTool(
        bus=self.bus,
        fn=mock.create_autospec(
            framework_types.AsyncFunction,
            instance=True,
            spec_set=True,
        ),
        declaration=genai_types.FunctionDeclaration(
            name="test_tool",
            description="Test tool.",
            behavior=genai_types.Behavior.NON_BLOCKING,
        ),
    )
    self.bus.start()
    await self.bus.publish(
        event=event_bus.Event(
            type=event_bus.EventType.TOOL_CALL_CANCELLATION,
            source=event_bus.EventSource.MAIN_AGENT,
        )
    )
    # The cancellation event is handled asynchronously, so we need to wait a
    # bit for the cancellation to be propagated.
    while not self.bus._event_queue.empty():
      await asyncio.sleep(0.1)
    self.assertTrue(tool_inst.cancelled)


if __name__ == "__main__":
  absltest.main()
