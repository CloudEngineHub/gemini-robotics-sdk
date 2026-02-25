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

"""ModelInterface implementation for querying remote models."""

from collections.abc import Mapping, Sequence
import json

import numpy as np
from typing_extensions import override

from safari_sdk.model import constants
from safari_sdk.model import genai_robotics
from safari_sdk.model import model_interface
from safari_sdk.model import observation_to_model_query_contents


class RemoteModelInterface(model_interface.ModelInterface):
  """Model that queries a remote model."""

  def __init__(
      self,
      serve_id: str,
      robotics_api_connection: constants.RoboticsApiConnectionType,
      string_observations_keys: Sequence[str],
      task_instruction_key: str,
      proprioceptive_observation_keys: Sequence[str],
      image_observation_keys: Sequence[str],
      image_compression_jpeg_quality: int,
      num_of_retries: int = 1,
  ):
    self._serve_id = serve_id
    self._robotics_api_connection = robotics_api_connection
    self._image_observation_keys = image_observation_keys
    self._string_observations_keys = string_observations_keys
    self._task_instruction_key = task_instruction_key
    self._proprioceptive_observation_keys = proprioceptive_observation_keys
    self._image_compression_jpeg_quality = image_compression_jpeg_quality

    self._client = genai_robotics.Client(
        robotics_api_connection=robotics_api_connection,
        num_retries=num_of_retries,
    )

  @override
  def query_model(
      self,
      observation: Mapping[str, np.ndarray],
  ) -> np.ndarray:
    """Queries the model with the given observation."""

    # Serialize the observation to the format expected by the transport.
    serialized_contents = observation_to_model_query_contents.observation_to_model_query_contents(
        observation=observation,
        string_observations_keys=self._string_observations_keys,
        task_instruction_key=self._task_instruction_key,
        proprioceptive_observation_keys=self._proprioceptive_observation_keys,
        image_observation_keys=self._image_observation_keys,
    )
    if (
        self._robotics_api_connection
        == constants.RoboticsApiConnectionType.CLOUD_GENAI
    ):
      serialized_contents = genai_robotics.update_robotics_content_to_genai_format(
          serialized_contents,
          image_compression_jpeg_quality=self._image_compression_jpeg_quality,
      )

    response = self._client.models.generate_content(
        model=self._serve_id,
        contents=serialized_contents,
    )

    # Parse the response text (assuming its JSON containing the action)
    if response.text:
      response_data = json.loads(response.text)
    elif response.candidates:
      response_data = json.loads(
          response.candidates[0].content.parts[0].inline_data.data
      )
    else:
      raise ValueError("Response does not contain text or candidates.")

    if not isinstance(response_data, dict):
      raise ValueError(
          "Response data does not have a single object as root object."
      )

    # Assuming the structure is {'action_chunk': [...]}
    action_chunk = response_data.get(constants.ACTION_CHUNK_RESPONSE_KEY)
    if action_chunk is None:
      raise ValueError(
          "Response JSON does not contain"
          f" '{constants.ACTION_CHUNK_RESPONSE_KEY}'"
      )
    action_chunk = np.array(action_chunk)
    if action_chunk.ndim != 2:
      raise ValueError(
          "Action chunk has more than 2 dimensions:"
          f" {action_chunk.shape}. Please make sure the model is configured to"
          " output a 2D array."
      )

    return action_chunk
