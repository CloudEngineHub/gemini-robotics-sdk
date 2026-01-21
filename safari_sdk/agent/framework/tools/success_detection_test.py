"""Tests for the success detection tools."""

import asyncio
import unittest
from unittest import mock

from absl.testing import flagsaver

from absl.testing import absltest
from safari_sdk.agent.framework import config as framework_config
from safari_sdk.agent.framework.event_bus import event_bus
from safari_sdk.agent.framework.tools import success_detection


class SuccessDetectionTest(unittest.IsolatedAsyncioTestCase):

  def setUp(self):
    super().setUp()
    self.loop = asyncio.new_event_loop()
    asyncio.set_event_loop(self.loop)
    self.config = framework_config.AgentFrameworkConfig()
    self.bus = event_bus.EventBus(config=self.config)

  def tearDown(self):
    self.loop.close()
    super().tearDown()

  @mock.patch.object(success_detection.genai, "Client", autospec=True)
  async def test_subtask_success_detector_instantiation(self, _):
    _ = success_detection.SubtaskSuccessDetectorV4(
        bus=self.bus,
        config=self.config,
        api_key="1234567",
        sd_camera_endpoint_names=["camera0", "camera1"],
    )

  @flagsaver.flagsaver(**{
      "sd.use_start_images": False,
      "sd.num_history_frames": 0,
      "sd.use_explicit_thinking": False,
  })
  @mock.patch.object(success_detection.genai, "Client", autospec=True)
  def test_build_prompt_current_images_only(self, _):
    config = framework_config.AgentFrameworkConfig.create()
    detector = success_detection.SubtaskSuccessDetectorV4(
        bus=self.bus,
        config=config,
        api_key="test_key",
        sd_camera_endpoint_names=["camera0"],
    )

    mock_image_event = mock.MagicMock(data=b"latest_image_data")
    detector._image_buffer.get_camera_endpoint_names = mock.MagicMock(
        return_value=["camera0"]
    )
    detector._image_buffer.get_start_images_map = mock.MagicMock(
        return_value={"camera0": mock_image_event}
    )
    detector._image_buffer.get_latest_images_map = mock.MagicMock(
        return_value={"camera0": [mock_image_event]}
    )

    prompt = detector._build_prompt(subtask="pick up cup")

    expected_prompt = [
        (
            "The images below show what the robot sees now (the robot may have"
            " more than one camera):"
        ),
        "camera0:",
        b"latest_image_data",
        "The robot's task is: pick up cup",
        (
            "Given the information above, think silently about the following"
            " questions first (only if you have thinking budget):"
        ),
        success_detection._GUIDED_THINKING_SD_QUESTIONS,
        "Please limit your response to 50 words.",
    ]
    self.assertEqual(prompt, expected_prompt)

  @flagsaver.flagsaver(**{
      "sd.use_start_images": True,
      "sd.num_history_frames": 0,
      "sd.use_explicit_thinking": False,
  })
  @mock.patch.object(success_detection.genai, "Client", autospec=True)
  def test_build_prompt_with_start_images(self, _):
    config = framework_config.AgentFrameworkConfig.create()
    detector = success_detection.SubtaskSuccessDetectorV4(
        bus=self.bus,
        config=config,
        api_key="test_key",
        sd_camera_endpoint_names=["camera0"],
    )

    mock_start_image = mock.MagicMock(data=b"start_image_data")
    mock_latest_image = mock.MagicMock(data=b"latest_image_data")
    detector._image_buffer.get_camera_endpoint_names = mock.MagicMock(
        return_value=["camera0"]
    )
    detector._image_buffer.get_start_images_map = mock.MagicMock(
        return_value={"camera0": mock_start_image}
    )
    detector._image_buffer.get_latest_images_map = mock.MagicMock(
        return_value={"camera0": [mock_latest_image]}
    )

    prompt = detector._build_prompt(subtask="place cup on table")

    expected_prompt = [
        (
            "The images below show what the robot saw at the start of the task"
            " (the robot may have more than one camera):"
        ),
        "camera0:",
        b"start_image_data",
        (
            "The images below show what the robot sees now (the robot may have"
            " more than one camera):"
        ),
        "camera0:",
        b"latest_image_data",
        "The robot's task is: place cup on table",
        (
            "Given the information above, think silently about the following"
            " questions first (only if you have thinking budget):"
        ),
        success_detection._GUIDED_THINKING_SD_QUESTIONS,
        "Please limit your response to 50 words.",
    ]
    self.assertEqual(prompt, expected_prompt)

  @flagsaver.flagsaver(**{
      "sd.use_start_images": False,
      "sd.num_history_frames": 0,
      "sd.use_explicit_thinking": True,
  })
  @mock.patch.object(success_detection.genai, "Client", autospec=True)
  def test_build_prompt_with_explicit_thinking(self, _):
    config = framework_config.AgentFrameworkConfig.create()
    detector = success_detection.SubtaskSuccessDetectorV4(
        bus=self.bus,
        config=config,
        api_key="test_key",
        sd_camera_endpoint_names=["camera0"],
    )

    mock_image = mock.MagicMock(data=b"image_data")
    detector._image_buffer.get_camera_endpoint_names = mock.MagicMock(
        return_value=["camera0"]
    )
    detector._image_buffer.get_start_images_map = mock.MagicMock(
        return_value={"camera0": mock_image}
    )
    detector._image_buffer.get_latest_images_map = mock.MagicMock(
        return_value={"camera0": [mock_image]}
    )

    prompt = detector._build_prompt(subtask="test task")

    expected_prompt = [
        (
            "The images below show what the robot sees now (the robot may have"
            " more than one camera):"
        ),
        "camera0:",
        b"image_data",
        "The robot's task is: test task",
        "Given the information above, answer the following questions first:",
        success_detection._GUIDED_THINKING_SD_QUESTIONS,
        "Please limit your response to 50 words.",
    ]
    self.assertEqual(prompt, expected_prompt)

  @flagsaver.flagsaver(**{
      "sd.use_start_images": False,
      "sd.num_history_frames": 0,
      "sd.use_explicit_thinking": False,
      "sd.guided_thinking_word_limit": 150,
  })
  @mock.patch.object(success_detection.genai, "Client", autospec=True)
  def test_build_prompt_with_word_limit(self, _):

    config = framework_config.AgentFrameworkConfig.create()
    detector = success_detection.SubtaskSuccessDetectorV4(
        bus=self.bus,
        config=config,
        api_key="test_key",
        sd_camera_endpoint_names=["camera0"],
    )

    mock_image = mock.MagicMock(data=b"image_data")
    detector._image_buffer.get_camera_endpoint_names = mock.MagicMock(
        return_value=["camera0"]
    )
    detector._image_buffer.get_start_images_map = mock.MagicMock(
        return_value={"camera0": mock_image}
    )
    detector._image_buffer.get_latest_images_map = mock.MagicMock(
        return_value={"camera0": [mock_image]}
    )

    prompt = detector._build_prompt(subtask="test task")

    expected_prompt = [
        (
            "The images below show what the robot sees now (the robot may have"
            " more than one camera):"
        ),
        "camera0:",
        b"image_data",
        "The robot's task is: test task",
        (
            "Given the information above, think silently about the following"
            " questions first (only if you have thinking budget):"
        ),
        success_detection._GUIDED_THINKING_SD_QUESTIONS,
        "Please limit your response to 150 words.",
    ]
    self.assertEqual(prompt, expected_prompt)

  @mock.patch.object(success_detection.genai, "Client", autospec=True)
  @flagsaver.flagsaver(**{"sd.dry_run": True})
  async def test_detect_success_timeout(self, _):
    detector = success_detection.SubtaskSuccessDetectorV4(
        bus=self.bus,
        config=self.config,
        api_key="test_key",
        sd_camera_endpoint_names=["camera0"],
    )
    detector.set_timeout_seconds(0.1)

    response = await detector.detect_success(
        subtask="test task", call_id="test_id"
    )

    self.assertFalse(response.response["task_success"])
    self.assertTrue(response.response["timeout"])
    self.assertFalse(response.will_continue)

  @mock.patch.object(success_detection.genai, "Client", autospec=True)
  @flagsaver.flagsaver(**{"sd.dry_run": True})
  async def test_detect_success_cancellation(self, _):
    detector = success_detection.SubtaskSuccessDetectorV4(
        bus=self.bus,
        config=self.config,
        api_key="test_key",
        sd_camera_endpoint_names=["camera0"],
    )

    async def cancel_after_delay():
      await asyncio.sleep(0.1)
      detector._tool_call_cancelled = True

    cancel_task = asyncio.create_task(cancel_after_delay())
    response = await detector.detect_success(
        subtask="test task", call_id="test_id"
    )
    await cancel_task

    self.assertFalse(response.response["task_success"])
    self.assertFalse(response.response["timeout"])
    self.assertFalse(response.will_continue)

  @mock.patch.object(success_detection.genai, "Client", autospec=True)
  @flagsaver.flagsaver(**{"sd.dry_run": True})
  async def test_detect_success_external_signal(self, _):
    detector = success_detection.SubtaskSuccessDetectorV4(
        bus=self.bus,
        config=self.config,
        api_key="test_key",
        sd_camera_endpoint_names=["camera0"],
    )

    async def send_external_signal():
      await asyncio.sleep(0.1)
      detector._external_success_signal = True

    signal_task = asyncio.create_task(send_external_signal())
    response = await detector.detect_success(
        subtask="test task", call_id="test_id"
    )
    await signal_task

    self.assertTrue(response.response["task_success"])
    self.assertFalse(response.response["timeout"])
    self.assertFalse(response.will_continue)

  @mock.patch.object(success_detection.genai, "Client", autospec=True)
  async def test_detect_success_internal_signal(self, _):
    detector = success_detection.SubtaskSuccessDetectorV4(
        bus=self.bus,
        config=self.config,
        api_key="test_key",
        sd_camera_endpoint_names=["camera0"],
    )

    async def set_success_signal():
      await asyncio.sleep(0.1)
      detector._success_signal = True

    signal_task = asyncio.create_task(set_success_signal())
    response = await detector.detect_success(
        subtask="test task", call_id="test_id"
    )
    await signal_task

    self.assertTrue(response.response["task_success"])
    self.assertFalse(response.response["timeout"])
    self.assertFalse(response.will_continue)

  @mock.patch.object(success_detection.genai, "Client", autospec=True)
  async def test_detect_success_resets_state(self, _):
    detector = success_detection.SubtaskSuccessDetectorV4(
        bus=self.bus,
        config=self.config,
        api_key="test_key",
        sd_camera_endpoint_names=["camera0"],
    )
    detector._success_signal = True
    detector._external_success_signal = True
    detector._task = "old task"

    async def set_success_signal():
      await asyncio.sleep(0.6)
      detector._success_signal = True

    signal_task = asyncio.create_task(set_success_signal())
    response = await detector.detect_success(
        subtask="new task", call_id="test_id"
    )
    await signal_task

    self.assertEqual(detector._task, "new task")
    self.assertTrue(response.response["task_success"])


if __name__ == "__main__":
  absltest.main()
