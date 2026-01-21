"""Test for the Agent base class."""

from collections.abc import Sequence
import unittest
from unittest import mock

from google.genai import types

from absl.testing import absltest
from safari_sdk.agent.framework import config as framework_config
from safari_sdk.agent.framework.agents import agent
from safari_sdk.agent.framework.embodiments import embodiment
from safari_sdk.agent.framework.event_bus import event_bus
from safari_sdk.agent.framework.tools import tool


class AgentTest(unittest.IsolatedAsyncioTestCase):

  def setUp(self):
    super().setUp()
    self.config = framework_config.AgentFrameworkConfig()
    self.bus = event_bus.EventBus(config=self.config)
    self.embodiment = mock.create_autospec(embodiment.Embodiment)
    self.embodiment.camera_stream_names = []

  @mock.patch.object(
      agent.tool_call_event_handler, "ToolCallEventHandler", autospec=True
  )
  @mock.patch.object(agent.live_handler, "GeminiLiveAPIHandler", autospec=True)
  async def test_tool_exposure(
      self, mock_live_handler_ctor, mock_tool_call_event_handler_ctor
  ):
    tool_1 = "test_tool_1"
    tool_2 = "test_tool_2"

    class TestAgent(agent.Agent):

      def _get_all_tools(
          self,
          embodiment_tools: Sequence[tool.Tool],
      ) -> Sequence[agent.ToolUseConfig]:
        return [
            agent.ToolUseConfig(
                tool=tool,
                exposed_to_agent=True
                if tool.declaration.name == tool_1
                else False,
            )
            for tool in embodiment_tools
        ]

    self.embodiment.tools = [
        tool.Tool(
            bus=self.bus,
            fn=lambda: None,
            declaration=types.FunctionDeclaration(name=tool_1),
        ),
        tool.Tool(
            bus=self.bus,
            fn=lambda: None,
            declaration=types.FunctionDeclaration(name=tool_2),
        ),
    ]

    test_config = framework_config.AgentFrameworkConfig(
        api_key="test api key",
        agent_model_name="test model",
    )
    _ = TestAgent(
        bus=self.bus,
        config=test_config,
        embodiment=self.embodiment,
        system_prompt="test system prompt",
    )

    mock_live_handler_ctor.assert_called_once()
    with self.subTest(name="only_pass_exposed_tools_to_live_api"):
      config_str = repr(mock_live_handler_ctor.call_args.kwargs["live_config"])
      self.assertIn(tool_1, config_str)
      self.assertNotIn(tool_2, config_str)

    mock_tool_call_event_handler_ctor.assert_called_once()
    with self.subTest(name="pass_all_tools_to_tool_call_handler"):
      tools = mock_tool_call_event_handler_ctor.call_args.kwargs["tool_dict"]
      self.assertEqual(
          list(tools.keys()),
          [tool_1, tool_2],
      )


if __name__ == "__main__":
  absltest.main()
