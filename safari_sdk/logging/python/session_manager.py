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

"""SessionManager class for managing the lifecycle of a session."""

from collections.abc import Callable, Sequence, Set
from safari_sdk.logging.python import metadata_utils
from safari_sdk.protos import label_pb2
from safari_sdk.protos.logging import metadata_pb2
from safari_sdk.protos.logging import orchestrator_info_pb2
from safari_sdk.protos.logging import policy_type_pb2


class SessionManager:
  """Class for managing the lifecycle of a session."""

  def __init__(
      self,
      topics: Set[str],
      required_topics: Set[str],
      policy_environment_metadata_params: metadata_utils.PolicyEnvironmentMetadataParams,
      fixed_tags: Sequence[str] = (),
      dynamic_episode_taggers: Sequence[Callable[[], Sequence[str]]] = (),
      orchestrator_info_provider: (
          Callable[[], orchestrator_info_pb2.OrchestratorInfo] | None
      ) = None,
  ):
    """Initializes the SessionManager.

    Args:
      topics: The topics to be logged.
      required_topics: The topics that are required to be logged.
      policy_environment_metadata_params: parameters for setting the policy
        environment metadata. Note that the dictionaries passed must be
        flattened.
      fixed_tags: Fixed tags to be added to all sessions when they are stopped.
      dynamic_episode_taggers: A Sequence of Callables each returning a Sequence
        of strings. This is used to generate tags that are added to the session
        when it is stopped. The tags generated may change between episodes /
        sessions.
      orchestrator_info_provider: An optional callable that returns the current
        OrchestratorInfo. If provided, it is called when `stop_session` is
        called and the result is set in the session.
    """

    self._session_started: bool = False
    self._session: metadata_pb2.Session | None = None

    self._topics: Set[str] = topics
    self._required_topics: Set[str] = required_topics

    self._policy_environment_metadata_params = (
        policy_environment_metadata_params
    )
    self._fixed_tags: Sequence[str] = fixed_tags
    self._dynamic_episode_taggers: Sequence[Callable[[], Sequence[str]]] = (
        dynamic_episode_taggers
    )
    self._orchestrator_info_provider = orchestrator_info_provider

    self._validate_topics()

  def _validate_topics(self) -> None:
    """Validates that required topics are a subset of all topics.

    Raises:
        ValueError: If required_topics is not a subset of topics or if a
        reserved topic is present.
    """
    # Checks that required topics are a subset of all topics.
    if not self._required_topics <= self._topics:
      missing_topics = self._required_topics - self._topics
      raise ValueError(
          'required_topics must be a subset of topics. '
          f'Missing topics: {missing_topics}'
      )

  @property
  def session_started(self) -> bool:
    return self._session_started

  def _get_policy_type(self) -> policy_type_pb2.PolicyType:
    """Returns the policy type."""
    policy_type = self._policy_environment_metadata_params.policy_type
    if callable(policy_type):
      return policy_type()

    return policy_type

  def _set_policy_environment_metadata(self) -> None:
    """Sets the policy environment metadata in the session.

    Raises:
      ValueError: If the session has not been started.
    """
    if self._session is None:
      raise ValueError(
          'Session is None. Cannot set policy environment metadata.'
      )
    # Set the feature specs.
    feature_specs = metadata_utils.create_feature_specs_proto(
        self._policy_environment_metadata_params
    )
    self._session.policy_environment_metadata.feature_specs.CopyFrom(
        feature_specs
    )
    # Set the policy type.
    self._session.policy_environment_metadata.policy_type = (
        self._get_policy_type()
    )
    # Set the control timestep.
    self._session.policy_environment_metadata.control_timestep = (
        self._policy_environment_metadata_params.control_timestep
    )
    # Set the embodiment version.
    self._session.policy_environment_metadata.embodiment_version = (
        self._policy_environment_metadata_params.embodiment_version
    )

  def start_session(self, *, start_timestamp_nsec: int, task_id: str) -> None:
    """Starts a new session for logging.

    Args:
      start_timestamp_nsec: The start timestamp of the session.
      task_id: The task ID of the session.
    """
    if self._session_started:
      raise ValueError('Session has already been started.')

    self._session = metadata_pb2.Session(
        interval=label_pb2.IntervalValue(
            start_nsec=start_timestamp_nsec,
        ),
        task_id=task_id,
    )
    # Set the policy environment metadata once the session is created.
    self._set_policy_environment_metadata()

    for topic in self._topics:
      self._session.streams.append(
          metadata_pb2.Session.StreamMetadata(
              key_range=metadata_pb2.KeyRange(
                  topic=topic,
                  interval=label_pb2.IntervalValue(
                      start_nsec=start_timestamp_nsec,
                  ),
              ),
              is_required=topic in self._required_topics,
          )
      )
    self._session_started = True

  def add_session_label(self, label: label_pb2.LabelMessage) -> None:
    """Adds a label to the session.

    Args:
      label: The label to be added to the session. This will be added to the
        session aspects.

    Raises:
      ValueError: If the session has not been started.
    """
    if not self._session_started or self._session is None:
      raise ValueError(
          'add_session_label is called before session has been started.'
      )
    self._session.labels.append(label)

  def _set_tags(self) -> None:
    """Sets the tags in the session."""

    if not self._session_started or self._session is None:
      raise ValueError('Session is not started. Cannot set tags.')

    self._session.tags.extend(self._fixed_tags)

    # Set dynamic tags.
    for tagger in self._dynamic_episode_taggers:
      tags = tagger()
      self._session.tags.extend(tags)

  def stop_session(self, stop_timestamp_nsec: int) -> metadata_pb2.Session:
    """Stops the current session and updates the session metadata.

    Updates the session interval and stream key ranges with the stop timestamp.
    Additionally, sets the fixed and dynamic tags and invokes the any providers
    passed to the constructor which will update the session in place.

    Args:
      stop_timestamp_nsec: The stop timestamp of the session.

    Returns:
      The Session object.

    Raises:
      ValueError: If the session has not been started.
    """
    if not self._session_started or self._session is None:
      raise ValueError('Session is not started. Cannot stop session.')

    self._session.interval.stop_nsec = stop_timestamp_nsec

    for stream in self._session.streams:
      stream.key_range.interval.stop_nsec = stop_timestamp_nsec

    self._set_tags()

    # Set orchestrator info from provider if available.
    if self._orchestrator_info_provider is not None:
      self._session.orchestrator_info.CopyFrom(
          self._orchestrator_info_provider()
      )

    # Update the policy type in case it is a callable.
    self._session.policy_environment_metadata.policy_type = (
        self._get_policy_type()
    )

    self._session_started = False

    return self._session
