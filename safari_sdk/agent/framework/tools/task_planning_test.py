"""Tests for the run task planning tool."""

import asyncio
import unittest

from absl.testing import absltest
from safari_sdk.agent.framework import config as framework_config
from safari_sdk.agent.framework.event_bus import event_bus
from safari_sdk.agent.framework.tools import task_planning


class TaskPlanningTest(unittest.IsolatedAsyncioTestCase):

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
    task_planning.VisionTaskPlanningTool(
        bus=self.bus,
        camera_stream_name="test_camera_stream_name",
        camera_fps=30.0,
        api_key="test_api_key",
        num_history_frames=31,
        planning_images_fps=1.0,
        one_time_planning=False,
    )


if __name__ == "__main__":
  absltest.main()
