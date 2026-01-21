"""Tests for terminal_ui."""

import asyncio
import unittest
from unittest import mock

from google.genai import types as genai_types

from absl.testing import absltest
from safari_sdk.agent.framework import config as framework_config
from safari_sdk.agent.framework.event_bus import event_bus
from safari_sdk.agent.framework.ui import terminal_ui


class TerminalUITest(unittest.IsolatedAsyncioTestCase):

  async def asyncSetUp(self):
    await super().asyncSetUp()
    self.config = framework_config.AgentFrameworkConfig()
    self.event_bus = event_bus.EventBus(config=self.config)
    self.terminal_ui = terminal_ui.TerminalUI(
        bus=self.event_bus, config=self.config
    )
    self.event_bus.start()

  async def asyncTearDown(self):
    self.event_bus.shutdown()
    await super().asyncTearDown()

  @mock.patch('builtins.print')
  async def test_print_events(self, mock_print):
    await self.event_bus.publish(
        event_bus.Event(
            type=event_bus.EventType.MODEL_TURN,
            source=event_bus.EventSource.MAIN_AGENT,
            data=genai_types.Content(
                parts=[
                    genai_types.Part(text='Hello! How can I help you today?')
                ],
            ),
        )
    )
    await asyncio.sleep(0.1)
    color = terminal_ui.COLOR_MAP.get(event_bus.EventType.MODEL_TURN, '')
    reset_color = terminal_ui.TERMINAL_COLOR_RESET
    mock_print.assert_called_once_with(
        f'{color}[LIVE_AGENT, MODEL_TURN - text]: '
        f'Hello! How can I help you today?{reset_color}'
    )

  @mock.patch('builtins.input')
  async def test_text_input_loop(self, mock_input):
    mock_handler = mock.AsyncMock()
    self.event_bus.subscribe(
        [event_bus.EventType.MODEL_TEXT_INPUT], mock_handler
    )
    user_input = 'Pick up the gray tray.'
    mock_input.side_effect = [user_input, KeyboardInterrupt()]
    await self.terminal_ui.connect()

    # Allow the input loop to run once
    await asyncio.sleep(0.1)

    mock_handler.assert_called_once()
    args, _ = mock_handler.call_args
    event = args[0]
    self.assertEqual(event.type, event_bus.EventType.MODEL_TEXT_INPUT)
    self.assertEqual(event.source, event_bus.EventSource.USER)
    self.assertEqual(event.data, user_input)

    await self.terminal_ui.disconnect()


if __name__ == '__main__':
  absltest.main()
