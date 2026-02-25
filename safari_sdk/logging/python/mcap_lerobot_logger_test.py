# Copyright 2026 Google LLC
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

from unittest import mock
import numpy as np
from absl.testing import absltest
from safari_sdk.logging.python import episodic_logger
from safari_sdk.logging.python import mcap_lerobot_logger

_TEST_TASK_ID = "test_task"
_TEST_IMAGE_KEY = "image"
_TEST_PROPRIO_KEY = "proprio"
_NUM_EPISODES = 2
_NUM_STEPS_PER_EPISODE = 3


class FakeLeRobotDataset:
  """Fake LeRobotDataset for testing."""

  def __init__(self, num_episodes, steps_per_episode):
    self._num_episodes = num_episodes
    self._steps_per_episode = steps_per_episode
    self.features = {
        "action": {"shape": [7], "dtype": "float32"},
        f"observation.{_TEST_IMAGE_KEY}": {
            "shape": [3, 224, 224],
            "dtype": "video",
            "names": ["channels", "height", "width"],
        },
        f"observation.{_TEST_PROPRIO_KEY}": {"shape": [7], "dtype": "float32"},
        "task": {"shape": (), "dtype": "string"},
        "timestamp": {"shape": (), "dtype": "float32"},
    }

    class Meta:

      def __init__(self, num_episodes, steps_per_episode):
        self.episodes = {
            i: {
                "dataset_from_index": i * steps_per_episode,
                "dataset_to_index": (i + 1) * steps_per_episode,
            }
            for i in range(num_episodes)
        }
        self.camera_keys = [f"observation.{_TEST_IMAGE_KEY}"]

    self.meta = Meta(num_episodes, steps_per_episode)

  @property
  def num_episodes(self):
    return self._num_episodes

  def __getitem__(self, index):
    episode_index = index // self._steps_per_episode
    frame_index = index % self._steps_per_episode
    return {
        "action": np.random.rand(7).astype(np.float32),
        f"observation.{_TEST_IMAGE_KEY}": (
            np.random.rand(3, 224, 224).astype(np.float32)
        ),
        f"observation.{_TEST_PROPRIO_KEY}": (
            np.random.rand(7).astype(np.float32)
        ),
        "frame_index": frame_index,
        "next.done": frame_index == self._steps_per_episode - 1,
        "task": f"task_{episode_index}",
        "timestamp": float(frame_index * 0.1),
    }


class McapLerobotLoggerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.output_dir = self.create_tempdir().full_path

  @mock.patch.object(episodic_logger, "EpisodicLogger", autospec=True)
  def test_lerobot_episodic_logger_record_step(self, mock_episodic_logger):
    mock_logger_instance = mock_episodic_logger.create.return_value
    logger = mcap_lerobot_logger.LeRobotEpisodicLogger(
        task_id=_TEST_TASK_ID,
        output_directory=self.output_dir,
        image_observation_keys=[_TEST_IMAGE_KEY],
        proprioceptive_observation_keys=[_TEST_PROPRIO_KEY],
        features=FakeLeRobotDataset(1, 1).features,
        generate_episode_timestamps=True,
    )

    logger.start_episode(0)
    step_data = {
        "action": np.array([1.0] * 7),
        "observation.image": np.zeros((3, 224, 224)),
        "observation.proprio": np.array([2.0] * 7),
        "frame_index": 0,
        "next.done": False,
        "task": "test_task",
        "timestamp": 0.0,
    }
    logger.record_step(step_data, timestamp_ns=123)

    mock_logger_instance.reset.assert_called_once()
    args, _ = mock_logger_instance.reset.call_args
    timestep = args[0]
    self.assertEqual(timestep.observation["proprio"][0], 2.0)
    self.assertEqual(
        timestep.observation["image"].shape, (224, 224, 3)
    )

  def test_convert_lerobot_data_to_mcap(self):
    dataset = FakeLeRobotDataset(
        num_episodes=_NUM_EPISODES, steps_per_episode=_NUM_STEPS_PER_EPISODE
    )
    episode_start_timestamps_ns = {
        i: i * 1_000_000_000 for i in range(_NUM_EPISODES)
    }

    with mock.patch(
        "safari_sdk.logging.python.mcap_lerobot_logger.LeRobotEpisodicLogger"
    ) as mock_logger:
      mcap_lerobot_logger.convert_lerobot_data_to_mcap(
          dataset=dataset,
          task_id=_TEST_TASK_ID,
          output_directory=self.output_dir,
          proprioceptive_observation_keys=[_TEST_PROPRIO_KEY],
          episodes_limit=0,
          max_workers=1,
          episode_start_timestamps_ns=episode_start_timestamps_ns,
      )

      self.assertEqual(mock_logger.call_count, _NUM_EPISODES)
      self.assertEqual(
          mock_logger.return_value.start_episode.call_count, _NUM_EPISODES
      )
      self.assertEqual(
          mock_logger.return_value.record_step.call_count,
          _NUM_EPISODES * _NUM_STEPS_PER_EPISODE,
      )
      self.assertEqual(
          mock_logger.return_value.finish_episode.call_count, _NUM_EPISODES
      )


if __name__ == "__main__":
  absltest.main()
