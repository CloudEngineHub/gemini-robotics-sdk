import asyncio
import unittest
from unittest import mock

from google.genai import types as genai_types

from absl.testing import absltest
from safari_sdk.agent.framework import config as framework_config
from safari_sdk.agent.framework import types
from safari_sdk.agent.framework.event_bus import audio_event_handler
from safari_sdk.agent.framework.event_bus import event_bus


class AudioHandlerTest(unittest.IsolatedAsyncioTestCase):

  async def asyncSetUp(self):
    await super().asyncSetUp()
    self.config = framework_config.AgentFrameworkConfig()
    self.bus = event_bus.EventBus(config=self.config)
    self.handler = audio_event_handler.AudioHandler(
        bus=self.bus,
        enable_audio_input=True,
        enable_audio_output=True,
    )
    self.mock_process = mock.AsyncMock()
    self.mock_process.stdin = mock.AsyncMock()
    self.mock_process.stdin.drain.return_value = None
    self.mock_process.stdin.wait_closed.return_value = None
    self.mock_process.stdout = mock.AsyncMock()
    self.mock_process.stderr = mock.AsyncMock()
    self.mock_process.pid = 666
    # Simulate reading some data then EOF
    self.mock_process.stdout.read.side_effect = [b"audio data", b""]
    create_subprocess_exec_patch = mock.patch(
        "asyncio.create_subprocess_exec", new_callable=mock.AsyncMock
    )
    self.mock_create_subprocess_exec = create_subprocess_exec_patch.start()
    self.addCleanup(create_subprocess_exec_patch.stop)
    self.mock_create_subprocess_exec.return_value = self.mock_process
    self.bus.start()
    await self.handler.connect()

  async def asyncTearDown(self):
    await self.handler.disconnect()
    self.bus.shutdown()
    await super().asyncTearDown()

  async def test_handle_event(self):
    # Check starting the recording and playback processes.
    self.mock_create_subprocess_exec.assert_any_call(
        *audio_event_handler.APLAY_CMD, stdin=mock.ANY
    )
    self.mock_create_subprocess_exec.assert_any_call(
        *audio_event_handler.ARECORD_CMD, stdout=mock.ANY
    )
    self.assertFalse(self.handler._is_speaking)
    # Check that the audio data is queued for playback.
    event = types.Event(
        type=types.EventType.MODEL_TURN,
        source=types.EventSource.MAIN_AGENT,
        data=genai_types.Content(
            parts=[
                genai_types.Part(
                    inline_data=genai_types.Blob(
                        data=b"audio data", mime_type="audio/wav"
                    ),
                )
            ],
            role="model",
        ),
    )
    await self.bus.publish(event)
    await asyncio.sleep(1.0)  # Allow event to be processed
    self.assertTrue(self.handler._is_speaking)
    self.mock_process.stdin.write.assert_called_once_with(b"audio data")
    # Check that the recording is resumed after the model turn is complete.
    event = types.Event(
        type=types.EventType.MODEL_TURN_COMPLETE,
        source=types.EventSource.MAIN_AGENT,
        data=None,
    )
    await self.bus.publish(event)
    await asyncio.sleep(1.0)  # Allow event to be processed
    self.assertFalse(self.handler._is_speaking)
    # Check that the recording is resumed after the model turn is interrupted.
    event = types.Event(
        type=types.EventType.MODEL_TURN_INTERRUPTED,
        source=types.EventSource.MAIN_AGENT,
        data=None,
    )
    await self.bus.publish(event)
    await asyncio.sleep(1.0)  # Allow event to be processed
    self.assertFalse(self.handler._is_speaking)


if __name__ == "__main__":
  absltest.main()
