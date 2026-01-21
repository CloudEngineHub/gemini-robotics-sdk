"""Tests for the Aloha embodiment."""

import asyncio
import unittest

from absl.testing import absltest
from safari_sdk.agent.framework import config as framework_config
from safari_sdk.agent.framework.embodiments import aloha
from safari_sdk.agent.framework.event_bus import event_bus


class AlohaTest(unittest.IsolatedAsyncioTestCase):

  def setUp(self):
    super().setUp()
    self.loop = asyncio.new_event_loop()
    asyncio.set_event_loop(self.loop)
    self.config = framework_config.AgentFrameworkConfig()
    self.bus = event_bus.EventBus(config=self.config)

  def tearDown(self):
    self.loop.close()
    super().tearDown()

  async def test_instantiate(self):
    embodiment = aloha.AlohaEmbodiment(bus=self.bus)
    self.assertIsNotNone(embodiment)

  async def test_create_event_streams(self):
    embodiment = aloha.AlohaEmbodiment(bus=self.bus)
    event_streams = embodiment.event_streams

    self.assertEqual(len(event_streams), 4)

    expected_endpoints = {
        aloha.OVERHEAD_ENDPOINT,
        aloha.LEFT_WRIST_ENDPOINT,
        aloha.RIGHT_WRIST_ENDPOINT,
        aloha.WORMS_EYE_ENDPOINT,
    }
    actual_endpoints = {stream.name for stream in event_streams}
    self.assertEqual(actual_endpoints, expected_endpoints)

    for stream_def in event_streams:
      self.assertTrue(stream_def.is_image_stream)
      self.assertIsNotNone(stream_def.stream)

  async def test_create_tools(self):
    embodiment = aloha.AlohaEmbodiment(bus=self.bus)
    tools = embodiment.tools

    self.assertEqual(len(tools), 5)

    expected_tool_names = {
        aloha.RUN_INSTRUCTION_TOOL_NAME,
        aloha.RESET_TOOL_NAME,
        aloha.STOP_TOOL_NAME,
        aloha.OPEN_GRIPPER_TOOL_NAME,
        aloha.SLEEP_TOOL_NAME,
    }
    actual_tool_names = {tool.declaration.name for tool in tools}
    self.assertEqual(actual_tool_names, expected_tool_names)

  async def test_camera_stream_names(self):
    embodiment = aloha.AlohaEmbodiment(bus=self.bus)
    camera_stream_names = embodiment.camera_stream_names

    self.assertEqual(len(camera_stream_names), 4)

    expected_names = [
        aloha.OVERHEAD_ENDPOINT,
        aloha.LEFT_WRIST_ENDPOINT,
        aloha.RIGHT_WRIST_ENDPOINT,
        aloha.WORMS_EYE_ENDPOINT,
    ]
    self.assertCountEqual(camera_stream_names, expected_names)

  async def test_custom_server_and_port(self):
    custom_server = "192.168.1.100"
    custom_port = 9000
    embodiment = aloha.AlohaEmbodiment(
        bus=self.bus, server=custom_server, port=custom_port
    )

    self.assertIsNotNone(embodiment)
    self.assertEqual(
        embodiment._robot_server, f"http://{custom_server}:{custom_port}"
    )


if __name__ == "__main__":
  absltest.main()
