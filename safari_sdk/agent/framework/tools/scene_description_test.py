"""Tests for the scene description tool."""

import asyncio
import unittest
from unittest import mock

from absl.testing import absltest
from safari_sdk.agent.framework import config as framework_config
from safari_sdk.agent.framework.event_bus import event_bus
from safari_sdk.agent.framework.tools import scene_description


_CAMERA_ENDPOINT_0 = "camera0"
_CAMERA_ENDPOINT_1 = "camera1"


class SceneDescriptionToolTest(unittest.IsolatedAsyncioTestCase):

  def setUp(self):
    super().setUp()
    self.loop = asyncio.new_event_loop()
    asyncio.set_event_loop(self.loop)

  def tearDown(self):
    self.loop.close()
    super().tearDown()

  @mock.patch.object(scene_description.genai, "Client", autospec=True)
  def test_scene_description_tool_instantiation(self, _):
    config = framework_config.AgentFrameworkConfig()
    bus = event_bus.EventBus(config=config)
    tool = scene_description.SceneDescriptionTool(
        bus=bus,
        config=config,
        api_key="test_api_key",
        camera_endpoint_names=[_CAMERA_ENDPOINT_0, _CAMERA_ENDPOINT_1],
    )
    self.assertIsNotNone(tool)
    self.assertEqual(
        tool._image_buffer.get_camera_endpoint_names(),
        [_CAMERA_ENDPOINT_0, _CAMERA_ENDPOINT_1],
    )

  @mock.patch.object(scene_description.genai, "Client", autospec=True)
  def test_build_prompt_with_single_camera(self, _):
    config = framework_config.AgentFrameworkConfig(
        scene_description_num_output_words=100
    )
    bus = event_bus.EventBus(config=config)
    tool = scene_description.SceneDescriptionTool(
        bus=bus,
        config=config,
        api_key="test_api_key",
        camera_endpoint_names=[_CAMERA_ENDPOINT_0],
    )
    mock_image = mock.MagicMock(data=b"image_data_0")
    tool._image_buffer.get_latest_images_map = mock.MagicMock(
        return_value={_CAMERA_ENDPOINT_0: [mock_image]}
    )
    tool._image_buffer.get_camera_endpoint_names = mock.MagicMock(
        return_value=[_CAMERA_ENDPOINT_0]
    )

    prompt = tool._build_prompt()

    print("\n=== FULL PROMPT OUTPUT ===")
    for i, element in enumerate(prompt):
      if isinstance(element, bytes):
        print(f"prompt[{i}]: <bytes data, length={len(element)}>")
      else:
        print(f"prompt[{i}]: {element!r}")
    print("=== END PROMPT OUTPUT ===\n")

    self.assertEqual(
        prompt[0],
        (
            "The images below show what the robot sees now (the robot may have"
            " more than one camera):"
        ),
    )
    self.assertEqual(prompt[1], "camera0:")
    self.assertEqual(prompt[2], b"image_data_0")
    self.assertEqual(
        prompt[3],
        (
            "\nYou are an expert at describing scenes based on images from one"
            " or more cameras.\nFor each camera, detect objects in the scene"
            " and describe their properties such as color, shape, material, and"
            " size.\nMerge the object descriptions from all cameras into a"
            " single, coherent scene description.\nIf there are duplicate"
            " objects in the scene, use natural language spatial references to"
            " disambiguate between them, such as 'the red large bowl on the"
            " left' and 'the red large bowl on the right'.\nLimit your response"
            " to 100 words.\n"
        ),
    )

  @mock.patch.object(scene_description.genai, "Client", autospec=True)
  def test_build_prompt_with_multiple_cameras(self, _):
    config = framework_config.AgentFrameworkConfig(
        scene_description_num_output_words=50
    )
    bus = event_bus.EventBus(config=config)
    tool = scene_description.SceneDescriptionTool(
        bus=bus,
        config=config,
        api_key="test_api_key",
        camera_endpoint_names=[_CAMERA_ENDPOINT_0, _CAMERA_ENDPOINT_1],
    )
    mock_image_0 = mock.MagicMock(data=b"image_data_0")
    mock_image_1 = mock.MagicMock(data=b"image_data_1")
    tool._image_buffer.get_latest_images_map = mock.MagicMock(
        return_value={
            _CAMERA_ENDPOINT_0: [mock_image_0],
            _CAMERA_ENDPOINT_1: [mock_image_1],
        }
    )
    tool._image_buffer.get_camera_endpoint_names = mock.MagicMock(
        return_value=[_CAMERA_ENDPOINT_0, _CAMERA_ENDPOINT_1]
    )

    prompt = tool._build_prompt()

    self.assertEqual(
        prompt[0],
        (
            "The images below show what the robot sees now (the robot may have"
            " more than one camera):"
        ),
    )
    self.assertEqual(prompt[1], "camera0:")
    self.assertEqual(prompt[2], b"image_data_0")
    self.assertEqual(prompt[3], "camera1:")
    self.assertEqual(prompt[4], b"image_data_1")
    self.assertEqual(
        prompt[5],
        (
            "\nYou are an expert at describing scenes based on images from one"
            " or more cameras.\nFor each camera, detect objects in the scene"
            " and describe their properties such as color, shape, material, and"
            " size.\nMerge the object descriptions from all cameras into a"
            " single, coherent scene description.\nIf there are duplicate"
            " objects in the scene, use natural language spatial references to"
            " disambiguate between them, such as 'the red large bowl on the"
            " left' and 'the red large bowl on the right'.\nLimit your response"
            " to 50 words.\n"
        ),
    )

  @mock.patch.object(scene_description.genai, "Client", autospec=True)
  def test_build_prompt_skips_none_images(self, _):
    config = framework_config.AgentFrameworkConfig()
    bus = event_bus.EventBus(config=config)
    tool = scene_description.SceneDescriptionTool(
        bus=bus,
        config=config,
        api_key="test_api_key",
        camera_endpoint_names=[_CAMERA_ENDPOINT_0, _CAMERA_ENDPOINT_1],
    )
    mock_image_0 = mock.MagicMock(data=b"image_data_0")
    tool._image_buffer.get_latest_images_map = mock.MagicMock(
        return_value={
            _CAMERA_ENDPOINT_0: [mock_image_0],
            _CAMERA_ENDPOINT_1: None,
        }
    )
    tool._image_buffer.get_camera_endpoint_names = mock.MagicMock(
        return_value=[_CAMERA_ENDPOINT_0, _CAMERA_ENDPOINT_1]
    )

    prompt = tool._build_prompt()

    self.assertIn("camera0:", prompt)
    self.assertNotIn("camera1:", prompt)
    self.assertIn(b"image_data_0", prompt)

  @mock.patch.object(scene_description.genai, "Client", autospec=True)
  async def test_describe_scene_success(self, _):
    mock_response = mock.MagicMock()
    mock_response.text = "A red bowl on the left and a blue cup on the right."

    config = framework_config.AgentFrameworkConfig()
    bus = event_bus.EventBus(config=config)
    tool = scene_description.SceneDescriptionTool(
        bus=bus,
        config=config,
        api_key="test_api_key",
        camera_endpoint_names=[_CAMERA_ENDPOINT_0],
    )
    tool._gemini_wrapper.generate_content = mock.AsyncMock(
        return_value=mock_response
    )
    tool._image_buffer.get_latest_images_map = mock.MagicMock(
        return_value={_CAMERA_ENDPOINT_0: [mock.MagicMock(data=b"image")]}
    )
    tool._image_buffer.get_camera_endpoint_names = mock.MagicMock(
        return_value=[_CAMERA_ENDPOINT_0]
    )

    response = await tool.describe_scene(call_id="test_call_id")

    self.assertEqual(
        response.response["scene_description"],
        "A red bowl on the left and a blue cup on the right.",
    )
    self.assertTrue(response.response["Do not call this tool again."])
    self.assertFalse(response.will_continue)

  @mock.patch.object(scene_description.genai, "Client", autospec=True)
  async def test_describe_scene_exception_handling(self, _):
    config = framework_config.AgentFrameworkConfig()
    bus = event_bus.EventBus(config=config)
    tool = scene_description.SceneDescriptionTool(
        bus=bus,
        config=config,
        api_key="test_api_key",
        camera_endpoint_names=[_CAMERA_ENDPOINT_0],
    )
    tool._gemini_wrapper.generate_content = mock.AsyncMock(
        side_effect=Exception("API error")
    )
    tool._image_buffer.get_latest_images_map = mock.MagicMock(
        return_value={_CAMERA_ENDPOINT_0: [mock.MagicMock(data=b"image")]}
    )
    tool._image_buffer.get_camera_endpoint_names = mock.MagicMock(
        return_value=[_CAMERA_ENDPOINT_0]
    )

    response = await tool.describe_scene(call_id="test_call_id")

    self.assertTrue(response.response["Failed to describe scene"])
    self.assertFalse(response.will_continue)

  @mock.patch.object(scene_description.genai, "Client", autospec=True)
  async def test_describe_scene_sets_call_id(self, _):
    mock_response = mock.MagicMock()
    mock_response.text = "Scene description."

    config = framework_config.AgentFrameworkConfig()
    bus = event_bus.EventBus(config=config)
    tool = scene_description.SceneDescriptionTool(
        bus=bus,
        config=config,
        api_key="test_api_key",
        camera_endpoint_names=[_CAMERA_ENDPOINT_0],
    )
    tool._gemini_wrapper.generate_content = mock.AsyncMock(
        return_value=mock_response
    )
    tool._image_buffer.get_latest_images_map = mock.MagicMock(
        return_value={_CAMERA_ENDPOINT_0: [mock.MagicMock(data=b"image")]}
    )
    tool._image_buffer.get_camera_endpoint_names = mock.MagicMock(
        return_value=[_CAMERA_ENDPOINT_0]
    )

    self.assertIsNone(tool._call_id)
    await tool.describe_scene(call_id="test_call_123")
    self.assertEqual(tool._call_id, "test_call_123")


if __name__ == "__main__":
  absltest.main()
