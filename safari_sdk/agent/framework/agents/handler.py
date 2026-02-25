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

"""Protocol definition for API handlers."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class Handler(Protocol):
  """Protocol defining the interface for API handlers.

  This protocol defines the common interface that both GeminiLiveAPIHandler
  (streaming) and NonStreamingGenAIHandler (non-streaming) implement. It
  allows agents to work with either handler type interchangeably.
  """

  async def connect(self) -> None:
    """Establishes connection or activates the handler."""
    ...

  async def disconnect(self) -> None:
    """Disconnects or deactivates the handler."""
    ...

  def register_event_subscribers(self) -> None:
    """Registers subscribers for events from the event bus."""
    ...
