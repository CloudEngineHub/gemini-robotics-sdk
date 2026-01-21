# Copyright 2025 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Tests for the run instruction until done tool."""

import asyncio
import unittest
from unittest import mock

from absl.testing import absltest
from safari_sdk.agent.framework import config as framework_config
from safari_sdk.agent.framework.event_bus import event_bus
from safari_sdk.agent.framework.tools import run_instruction_until_done
from safari_sdk.agent.framework.tools import success_detection
from safari_sdk.agent.framework.tools import tool


class RunInstructionUntilDoneTest(unittest.IsolatedAsyncioTestCase):

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
    run_instruction_until_done.RunInstructionUntilDoneTool(
        bus=self.bus,
        config=self.config,
        success_detector_tool=mock.create_autospec(
            success_detection.SubtaskSuccessDetectorV4
        ),
        run_instruction_tool=mock.create_autospec(tool.Tool),
        stop_tool=mock.create_autospec(tool.Tool),
        maximum_run_duration=10.0,
    )


if __name__ == "__main__":
  absltest.main()
