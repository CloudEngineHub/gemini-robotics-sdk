"""Tests for tool call event handler."""

import asyncio
import unittest

from google.genai import types as genai_types

from absl.testing import absltest
from safari_sdk.agent.framework import config as framework_config
from safari_sdk.agent.framework import types as framework_types
from safari_sdk.agent.framework.event_bus import event_bus
from safari_sdk.agent.framework.event_bus import tool_call_event_handler


class DummyAsyncFunction(framework_types.AsyncFunction):

  def __init__(self):
    self.call_log = []

  async def __call__(self, call_id: str, z: int):
    self.call_log.append((call_id, z))
    yield genai_types.FunctionResponse(response={"output": z})
    yield genai_types.FunctionResponse(response={"output": z + 1})


class ToolCallEventHandlerTest(unittest.IsolatedAsyncioTestCase):

  async def asyncSetUp(self):
    await super().asyncSetUp()
    self.config = framework_config.AgentFrameworkConfig()
    self.bus = event_bus.EventBus(config=self.config)
    self.events = []
    self.bus.subscribe(
        event_types=[event_bus.EventType.TOOL_RESULT],
        handler=self.events.append,
    )
    self.bus.start()

  async def asyncTearDown(self):
    self.bus.shutdown()
    await super().asyncTearDown()

  async def test_async_generator_function(self):
    async_fn = DummyAsyncFunction()
    tool_dict = {"async_tool": async_fn}
    _ = tool_call_event_handler.ToolCallEventHandler(
        bus=self.bus,
        tool_dict=tool_dict,
    )

    await self.bus.publish(
        event=framework_types.Event(
            type=framework_types.EventType.TOOL_CALL,
            source=event_bus.EventSource.MAIN_AGENT,
            data=genai_types.LiveServerToolCall(
                function_calls=[
                    genai_types.FunctionCall(
                        name="async_tool", args={"z": 4}, id="call-123"
                    ),
                ]
            ),
        )
    )
    await asyncio.sleep(0.1)

    self.assertEqual(async_fn.call_log, [("call-123", 4)])
    self.assertEqual(len(self.events), 2)

    response_1 = self.events[0].data.function_responses[0]
    response_2 = self.events[1].data.function_responses[0]
    self.assertEqual(response_1.id, "call-123")
    self.assertEqual(response_1.name, "async_tool")
    self.assertEqual(response_1.response, {"output": 4})
    self.assertEqual(response_2.id, "call-123")
    self.assertEqual(response_2.name, "async_tool")
    self.assertEqual(response_2.response, {"output": 5})


if __name__ == "__main__":
  absltest.main()
