"""Tests for the Live API handler."""

import asyncio
import dataclasses
import pathlib
import unittest
from unittest import mock

from google import genai
from google.genai import types


from absl.testing import absltest
from safari_sdk.agent.framework import config as framework_config
from safari_sdk.agent.framework import constants
from safari_sdk.agent.framework.event_bus import event_bus
from safari_sdk.agent.framework.live_api import live_handler


_PRIMARY_CAMERA_NAME = 'foo'
_SECONDARY_CAMERA_NAME = 'bar'
_TESTDATA_DIR = pathlib.Path(
    'safari_sdk/agent/framework/live_api/testdata'
)


class LiveHandlerTest(unittest.IsolatedAsyncioTestCase):

  def setUp(self):
    super().setUp()
    self.loop = asyncio.new_event_loop()
    asyncio.set_event_loop(self.loop)

    # Cache a test image event.
    with open(
        (_TESTDATA_DIR / 'input_img.png'), 'rb'
    ) as f:
      image_bytes = f.read()
    self.image_event = event_bus.Event(
        type=event_bus.EventType.MODEL_IMAGE_INPUT,
        source=event_bus.EventSource.USER,
        data=image_bytes,
        metadata={constants.STREAM_NAME_METADATA_KEY: _PRIMARY_CAMERA_NAME},
    )

  def tearDown(self):
    self.loop.close()
    super().tearDown()

  async def test_react_to_image(self):
    test_config = framework_config.AgentFrameworkConfig(
        api_key='1234567890',
        update_vision_after_fr=True,
    )
    bus = event_bus.EventBus(config=test_config)
    live_config = types.LiveConnectConfig()
    handler = live_handler.GeminiLiveAPIHandler(
        bus=bus,
        config=test_config,
        live_config=live_config,
        camera_names=(_PRIMARY_CAMERA_NAME, _SECONDARY_CAMERA_NAME),
    )
    # only test with TURN_INCLUDES_ALL_INPUT
    # otherwise inject image directly
    handler._turn_coverage = types.TurnCoverage.TURN_INCLUDES_ALL_INPUT
    handler.register_event_subscribers()

    # Do *not* call `connect` here, it would connect to the actual API. Instead
    # we pretend that it's running.
    # TODO: Should add proper test mocking of Live API.
    handler._is_active = True
    handler._session = mock.create_autospec(genai.live.AsyncSession)

    # TODO: Once camera stream switching is implemented, adjust
    # this test accordingly, both for initial choice and after switching.
    with self.subTest('active_camera_used'):
      self.assertTrue(handler._streaming_input_queue.empty())
      await handler._handle_image_in_event(self.image_event)
      # Note: since we do not call `connect`, the handler will never take the
      # image out of the queue, so we don't have to worry about timing.
      self.assertFalse(handler._streaming_input_queue.empty())
      await handler._streaming_input_queue.get()

    with self.subTest('inactive_camera_ignored'):
      self.assertTrue(handler._streaming_input_queue.empty())
      image_event = dataclasses.replace(
          self.image_event,
          metadata={constants.STREAM_NAME_METADATA_KEY: _SECONDARY_CAMERA_NAME},
      )
      await handler._handle_image_in_event(image_event)
      # Note: since we do not call `connect`, the handler will never take the
      # image out of the queue, so we don't have to worry about timing.
      self.assertTrue(handler._streaming_input_queue.empty())

    with self.subTest('unnamed_camera_ignored'):
      self.assertTrue(handler._streaming_input_queue.empty())
      image_event = dataclasses.replace(self.image_event, metadata={})
      await handler._handle_image_in_event(image_event)
      # Note: since we do not call `connect`, the handler will never take the
      # image out of the queue, so we don't have to worry about timing.
      self.assertTrue(handler._streaming_input_queue.empty())

  async def test_handle_text_input(self):
    test_config = framework_config.AgentFrameworkConfig(
        api_key='1234567890',
    )
    bus = event_bus.EventBus(config=test_config)
    live_config = types.LiveConnectConfig()
    handler = live_handler.GeminiLiveAPIHandler(
        bus=bus,
        config=test_config,
        live_config=live_config,
        camera_names=(_PRIMARY_CAMERA_NAME,),
    )
    handler._is_active = True
    handler._session = mock.create_autospec(genai.live.AsyncSession)

    text_event = event_bus.Event(
        type=event_bus.EventType.MODEL_TEXT_INPUT,
        source=event_bus.EventSource.USER,
        data='Hello, world!',
    )

    self.assertTrue(handler._non_streaming_inputs_queue.empty())
    await handler._handle_text_in_event(text_event)
    self.assertFalse(handler._non_streaming_inputs_queue.empty())
    queued_text = await handler._non_streaming_inputs_queue.get()
    self.assertEqual(queued_text, 'Hello, world!')

  async def test_handle_audio_input(self):
    test_config = framework_config.AgentFrameworkConfig(
        api_key='1234567890',
    )
    bus = event_bus.EventBus(config=test_config)
    live_config = types.LiveConnectConfig()
    handler = live_handler.GeminiLiveAPIHandler(
        bus=bus,
        config=test_config,
        live_config=live_config,
        camera_names=(_PRIMARY_CAMERA_NAME,),
    )
    handler._is_active = True
    handler._session = mock.create_autospec(genai.live.AsyncSession)

    audio_data = b'fake_audio_bytes'
    audio_event = event_bus.Event(
        type=event_bus.EventType.MODEL_AUDIO_INPUT,
        source=event_bus.EventSource.USER,
        data=audio_data,
    )

    self.assertTrue(handler._streaming_input_queue.empty())
    await handler._handle_audio_in_event(audio_event)
    self.assertFalse(handler._streaming_input_queue.empty())
    queued_blob = await handler._streaming_input_queue.get()
    self.assertIsInstance(queued_blob, types.Blob)
    self.assertEqual(queued_blob.data, audio_data)
    self.assertTrue(queued_blob.mime_type.startswith('audio/pcm'))

  async def test_handle_tool_result(self):
    test_config = framework_config.AgentFrameworkConfig(
        api_key='1234567890',
    )
    bus = event_bus.EventBus(config=test_config)
    live_config = types.LiveConnectConfig()
    handler = live_handler.GeminiLiveAPIHandler(
        bus=bus,
        config=test_config,
        live_config=live_config,
        camera_names=(_PRIMARY_CAMERA_NAME,),
    )
    handler._is_active = True
    handler._session = mock.create_autospec(genai.live.AsyncSession)

    tool_response = types.LiveClientToolResponse(
        function_responses=[
            types.FunctionResponse(
                name='test_function',
                id='test_id',
                response={'result': 'success'},
            )
        ]
    )
    tool_event = event_bus.Event(
        type=event_bus.EventType.TOOL_RESULT,
        source=event_bus.EventSource.USER,
        data=tool_response,
    )

    self.assertTrue(handler._non_streaming_inputs_queue.empty())
    await handler._handle_tool_result_event(tool_event)
    self.assertFalse(handler._non_streaming_inputs_queue.empty())
    queued_response = await handler._non_streaming_inputs_queue.get()
    self.assertEqual(queued_response, tool_response)

  async def test_inactive_session_ignores_events(self):
    test_config = framework_config.AgentFrameworkConfig(
        api_key='1234567890',
    )
    bus = event_bus.EventBus(config=test_config)
    live_config = types.LiveConnectConfig()
    handler = live_handler.GeminiLiveAPIHandler(
        bus=bus,
        config=test_config,
        live_config=live_config,
        camera_names=(_PRIMARY_CAMERA_NAME,),
    )

    text_event = event_bus.Event(
        type=event_bus.EventType.MODEL_TEXT_INPUT,
        source=event_bus.EventSource.USER,
        data='Hello',
    )

    await handler._handle_text_in_event(text_event)
    self.assertTrue(handler._non_streaming_inputs_queue.empty())

    await handler._handle_image_in_event(self.image_event)
    self.assertTrue(handler._streaming_input_queue.empty())

  def test_api_key_validation(self):
    with self.subTest('api_key_from_config'):
      test_config = framework_config.AgentFrameworkConfig(
          api_key='test_key_123',
      )
      bus = event_bus.EventBus(config=test_config)
      handler = live_handler.GeminiLiveAPIHandler(
          bus=bus,
          config=test_config,
          live_config=types.LiveConnectConfig(),
          camera_names=(_PRIMARY_CAMERA_NAME,),
      )
      self.assertIsNotNone(handler._client)

    with self.subTest('api_key_missing_raises_error'):
      test_config = framework_config.AgentFrameworkConfig(api_key=None)
      bus = event_bus.EventBus(config=test_config)
      with self.assertRaises(ValueError) as context:
        live_handler.GeminiLiveAPIHandler(
            bus=bus,
            config=test_config,
            live_config=types.LiveConnectConfig(),
            camera_names=(_PRIMARY_CAMERA_NAME,),
        )
      self.assertIn('No API key provided', str(context.exception))

  def test_turn_coverage_detection(self):
    with self.subTest('turn_coverage_from_config'):
      test_config = framework_config.AgentFrameworkConfig(
          api_key='1234567890',
      )
      bus = event_bus.EventBus(config=test_config)
      live_config = types.LiveConnectConfig(
          realtime_input_config=types.RealtimeInputConfig(
              turn_coverage=types.TurnCoverage.TURN_INCLUDES_ALL_INPUT,
          ),
      )
      handler = live_handler.GeminiLiveAPIHandler(
          bus=bus,
          config=test_config,
          live_config=live_config,
          camera_names=(_PRIMARY_CAMERA_NAME,),
      )
      self.assertEqual(
          handler._turn_coverage, types.TurnCoverage.TURN_INCLUDES_ALL_INPUT
      )

    with self.subTest('default_turn_coverage_when_missing'):
      test_config = framework_config.AgentFrameworkConfig(
          api_key='1234567890',
      )
      bus = event_bus.EventBus(config=test_config)
      handler = live_handler.GeminiLiveAPIHandler(
          bus=bus,
          config=test_config,
          live_config=types.LiveConnectConfig(),
          camera_names=(_PRIMARY_CAMERA_NAME,),
      )
      self.assertEqual(
          handler._turn_coverage,
          types.TurnCoverage.TURN_INCLUDES_ONLY_ACTIVITY,
      )

  def test_prepare_image_parts(self):
    img_dict = {
        'camera1': types.Blob(
            display_name='Camera 1',
            data=b'image_data_1',
            mime_type='image/jpeg',
        ),
        'camera2': types.Blob(
            display_name='Camera 2',
            data=b'image_data_2',
            mime_type='image/jpeg',
        ),
    }
    parts = live_handler._prepare_image_parts(img_dict)
    self.assertEqual(len(parts), 4)
    self.assertEqual(parts[0].text, 'Camera 1')
    self.assertEqual(parts[1].inline_data.data, b'image_data_1')
    self.assertEqual(parts[2].text, 'Camera 2')
    self.assertEqual(parts[3].inline_data.data, b'image_data_2')

  def test_invalid_stream_name_to_camera_name(self):
    test_config = framework_config.AgentFrameworkConfig(
        api_key='1234567890',
    )
    bus = event_bus.EventBus(config=test_config)
    with self.assertRaises(ValueError) as context:
      live_handler.GeminiLiveAPIHandler(
          bus=bus,
          config=test_config,
          live_config=types.LiveConnectConfig(),
          camera_names=(_PRIMARY_CAMERA_NAME,),
          stream_name_to_camera_name={'nonexistent_camera': 'description'},
      )
    self.assertIn('not found in available cameras', str(context.exception))

  def test_ignore_image_inputs_flag(self):
    test_config = framework_config.AgentFrameworkConfig(
        api_key='1234567890',
    )
    bus = event_bus.EventBus(config=test_config)
    handler = live_handler.GeminiLiveAPIHandler(
        bus=bus,
        config=test_config,
        live_config=types.LiveConnectConfig(),
        camera_names=(_PRIMARY_CAMERA_NAME,),
        ignore_image_inputs=True,
    )
    self.assertTrue(handler._ignore_image_inputs)


if __name__ == '__main__':
  absltest.main()
