import datetime

from absl.testing import flagsaver
import numpy as np

from absl.testing import absltest
from safari_sdk.agent.framework import config as framework_config
from safari_sdk.agent.framework import constants
from safari_sdk.agent.framework import flags as agent_flags
from safari_sdk.agent.framework import types as framework_types
from safari_sdk.agent.framework.event_bus import event_bus
from safari_sdk.agent.framework.tools import image_buffer

_CAMERA_ENDPOINT_0 = "camera0"
_CAMERA_ENDPOINT_1 = "camera1"


class ImageBufferTest(absltest.TestCase):

  def _create_model_image_input_event(
      self,
      timestamp: datetime.datetime,
      stream_name: str,
  ) -> event_bus.Event:
    """Creates a model image input event."""
    return event_bus.Event(
        type=framework_types.EventType.MODEL_IMAGE_INPUT,
        timestamp=timestamp,
        data=np.array([0, 1, 2], dtype=np.uint8),
        metadata={constants.STREAM_NAME_METADATA_KEY: stream_name},
        source="test",
    )

  def test_get_latest_images_map_returns_empty_map_if_no_images(self):
    image_buffer_instance = image_buffer.ImageBuffer(
        [], framework_config.AgentFrameworkConfig()
    )
    self.assertEmpty(image_buffer_instance.get_latest_images_map())

  def test_get_latest_images_map_returns_images_map(self):
    image_buffer_instance = image_buffer.ImageBuffer(
        [_CAMERA_ENDPOINT_0], framework_config.AgentFrameworkConfig()
    )
    image_buffer_instance.handle_model_image_input_event(
        self._create_model_image_input_event(
            timestamp=datetime.datetime.fromisoformat(
                "2025-03-29T10:00:00.000"
            ),
            stream_name=_CAMERA_ENDPOINT_0,
        )
    )
    self.assertLen(image_buffer_instance.get_latest_images_map(), 1)
    self.assertLen(
        image_buffer_instance.get_latest_images_map()[_CAMERA_ENDPOINT_0], 1
    )

  def test_get_latest_images_map_skips_images_from_unknown_cameras(self):
    image_buffer_instance = image_buffer.ImageBuffer(
        [_CAMERA_ENDPOINT_0], framework_config.AgentFrameworkConfig()
    )
    image_buffer_instance.handle_model_image_input_event(
        self._create_model_image_input_event(
            timestamp=datetime.datetime.fromisoformat(
                "2025-03-29T10:00:00.000"
            ),
            stream_name=_CAMERA_ENDPOINT_1,
        )
    )
    self.assertEmpty(image_buffer_instance.get_latest_images_map())

  @flagsaver.flagsaver(
      (agent_flags.AGENTIC_SD_NUM_HISTORY_FRAMES, 1),
  )
  def test_get_latest_images_map_returns_images_map_with_history_frames(self):
    config = framework_config.AgentFrameworkConfig.create()
    image_buffer_instance = image_buffer.ImageBuffer(
        [_CAMERA_ENDPOINT_0], config
    )
    image_buffer_instance.handle_model_image_input_event(
        self._create_model_image_input_event(
            timestamp=datetime.datetime.fromisoformat(
                "2025-03-29T10:00:00.000"
            ),
            stream_name=_CAMERA_ENDPOINT_0,
        )
    )
    image_buffer_instance.handle_model_image_input_event(
        self._create_model_image_input_event(
            timestamp=datetime.datetime.fromisoformat(
                "2025-03-29T10:00:02.000"
            ),
            stream_name=_CAMERA_ENDPOINT_0,
        )
    )
    image_buffer_instance.handle_model_image_input_event(
        self._create_model_image_input_event(
            timestamp=datetime.datetime.fromisoformat(
                "2025-03-29T10:00:04.000"
            ),
            stream_name=_CAMERA_ENDPOINT_0,
        )
    )
    self.assertLen(image_buffer_instance.get_latest_images_map(), 1)
    self.assertLen(
        image_buffer_instance.get_latest_images_map()[_CAMERA_ENDPOINT_0], 2
    )

  def test_get_latest_images_map_skips_images_within_interval(self):
    image_buffer_instance = image_buffer.ImageBuffer(
        [_CAMERA_ENDPOINT_0], framework_config.AgentFrameworkConfig()
    )
    image_buffer_instance.handle_model_image_input_event(
        self._create_model_image_input_event(
            timestamp=datetime.datetime.fromisoformat(
                "2025-03-29T10:00:00.000"
            ),
            stream_name=_CAMERA_ENDPOINT_0,
        )
    )
    image_buffer_instance.handle_model_image_input_event(
        self._create_model_image_input_event(
            timestamp=datetime.datetime.fromisoformat(
                "2025-03-29T10:00:00.900"
            ),
            stream_name=_CAMERA_ENDPOINT_0,
        )
    )

    self.assertLen(image_buffer_instance.get_latest_images_map(), 1)
    self.assertLen(
        image_buffer_instance.get_latest_images_map()[_CAMERA_ENDPOINT_0], 1
    )

  def test_get_latest_images_map_with_multiple_cameras(self):
    image_buffer_instance = image_buffer.ImageBuffer(
        [_CAMERA_ENDPOINT_0, _CAMERA_ENDPOINT_1],
        framework_config.AgentFrameworkConfig(),
    )
    image_buffer_instance.handle_model_image_input_event(
        self._create_model_image_input_event(
            timestamp=datetime.datetime.fromisoformat(
                "2025-03-29T10:00:00.000"
            ),
            stream_name=_CAMERA_ENDPOINT_0,
        )
    )
    image_buffer_instance.handle_model_image_input_event(
        self._create_model_image_input_event(
            timestamp=datetime.datetime.fromisoformat(
                "2025-03-29T10:00:00.000"
            ),
            stream_name=_CAMERA_ENDPOINT_1,
        )
    )

    self.assertLen(image_buffer_instance.get_latest_images_map(), 2)
    self.assertLen(
        image_buffer_instance.get_latest_images_map()[_CAMERA_ENDPOINT_0], 1
    )
    self.assertLen(
        image_buffer_instance.get_latest_images_map()[_CAMERA_ENDPOINT_1], 1
    )

  def test_reset_latest_images_map_returns_empty_map(self):
    image_buffer_instance = image_buffer.ImageBuffer(
        [_CAMERA_ENDPOINT_0], framework_config.AgentFrameworkConfig()
    )
    image_buffer_instance.handle_model_image_input_event(
        self._create_model_image_input_event(
            timestamp=datetime.datetime.fromisoformat(
                "2025-03-29T10:00:00.000"
            ),
            stream_name=_CAMERA_ENDPOINT_0,
        )
    )
    image_buffer_instance.reset_latest_images_map()
    self.assertEmpty(image_buffer_instance.get_latest_images_map())

  def test_get_latest_image_timestamp_returns_none_if_no_images(self):
    image_buffer_instance = image_buffer.ImageBuffer(
        [], framework_config.AgentFrameworkConfig()
    )
    self.assertIsNone(image_buffer_instance.get_latest_image_timestamp())

  def test_get_latest_image_timestamp_returns_timestamp_of_latest_image(self):
    image_buffer_instance = image_buffer.ImageBuffer(
        [_CAMERA_ENDPOINT_0, _CAMERA_ENDPOINT_1],
        framework_config.AgentFrameworkConfig(),
    )
    image_buffer_instance.handle_model_image_input_event(
        self._create_model_image_input_event(
            timestamp=datetime.datetime.fromisoformat(
                "2025-03-29T10:00:00.000"
            ),
            stream_name=_CAMERA_ENDPOINT_0,
        )
    )
    image_buffer_instance.handle_model_image_input_event(
        self._create_model_image_input_event(
            timestamp=datetime.datetime.fromisoformat(
                "2025-03-29T10:00:01.000"
            ),
            stream_name=_CAMERA_ENDPOINT_1,
        )
    )
    self.assertEqual(
        image_buffer_instance.get_latest_image_timestamp(),
        datetime.datetime.fromisoformat("2025-03-29T10:00:01.000"),
    )

  def test_get_start_images_map_returns_empty_map_if_no_images(self):
    image_buffer_instance = image_buffer.ImageBuffer(
        [], framework_config.AgentFrameworkConfig()
    )
    self.assertEmpty(image_buffer_instance.get_start_images_map())

  def test_get_start_images_map_returns_images_map(self):
    image_buffer_instance = image_buffer.ImageBuffer(
        [_CAMERA_ENDPOINT_0, _CAMERA_ENDPOINT_1],
        framework_config.AgentFrameworkConfig(),
    )
    image_buffer_instance.handle_model_image_input_event(
        self._create_model_image_input_event(
            timestamp=datetime.datetime.fromisoformat(
                "2025-03-29T10:00:00.000"
            ),
            stream_name=_CAMERA_ENDPOINT_0,
        )
    )
    image_buffer_instance.handle_model_image_input_event(
        self._create_model_image_input_event(
            timestamp=datetime.datetime.fromisoformat(
                "2025-03-29T10:00:01.000"
            ),
            stream_name=_CAMERA_ENDPOINT_1,
        )
    )
    image_buffer_instance.handle_model_image_input_event(
        self._create_model_image_input_event(
            timestamp=datetime.datetime.fromisoformat(
                "2025-03-29T10:00:02.000"
            ),
            stream_name=_CAMERA_ENDPOINT_0,
        )
    )
    start_images_map = image_buffer_instance.get_start_images_map()
    self.assertLen(start_images_map, 2)
    self.assertEqual(
        start_images_map[_CAMERA_ENDPOINT_0].timestamp,
        datetime.datetime.fromisoformat("2025-03-29T10:00:00.000"),
    )
    self.assertEqual(
        start_images_map[_CAMERA_ENDPOINT_1].timestamp,
        datetime.datetime.fromisoformat("2025-03-29T10:00:01.000"),
    )

  def test_reset_start_images_map_returns_empty_map(self):
    image_buffer_instance = image_buffer.ImageBuffer(
        [_CAMERA_ENDPOINT_0, _CAMERA_ENDPOINT_1],
        framework_config.AgentFrameworkConfig(),
    )
    image_buffer_instance.handle_model_image_input_event(
        self._create_model_image_input_event(
            timestamp=datetime.datetime.fromisoformat(
                "2025-03-29T10:00:00.000"
            ),
            stream_name=_CAMERA_ENDPOINT_0,
        )
    )
    image_buffer_instance.reset_start_images_map()
    self.assertEmpty(image_buffer_instance.get_start_images_map())

  def test_get_camera_endpoint_names_returns_camera_endpoint_names(self):
    image_buffer_instance = image_buffer.ImageBuffer(
        [_CAMERA_ENDPOINT_0, _CAMERA_ENDPOINT_1],
        framework_config.AgentFrameworkConfig(),
    )
    self.assertEqual(
        image_buffer_instance.get_camera_endpoint_names(),
        [_CAMERA_ENDPOINT_0, _CAMERA_ENDPOINT_1],
    )


if __name__ == "__main__":
  absltest.main()
