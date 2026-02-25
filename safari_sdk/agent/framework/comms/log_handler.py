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

"""Custom logging handler that publishes log messages to the event bus."""

import asyncio
# We need stdlib logging for the Handler base class and LogRecord type
# annotation. absl.logging is built on top of Python's logging module and
# doesn't re-export these.
import logging as stdlib_logging

from absl import logging

from safari_sdk.agent.framework import types


class EventBusLogHandler(stdlib_logging.Handler):
  """A logging handler that publishes log messages to the event bus."""

  def __init__(self, event_bus, min_level: int = logging.INFO):
    """Initializes the handler.

    Args:
      event_bus: The event bus to publish log messages to.
      min_level: Minimum logging level to publish (default: INFO).
    """
    super().__init__(level=min_level)
    self._event_bus = event_bus
    self._loop: asyncio.AbstractEventLoop | None = None

  def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
    """Set the asyncio event loop for async publishing.

    Args:
      loop: The asyncio event loop to use for publishing events.
    """
    self._loop = loop

  def emit(self, record: stdlib_logging.LogRecord) -> None:
    """Emit a log record as an event to the event bus.

    Args:
      record: The log record to emit.
    """
    try:
      event = types.Event(
          type=types.EventType.SYSTEM_LOG,
          source=types.EventSource.MAIN_AGENT,
          data={
              "level": record.levelname,
              "message": record.getMessage(),
              "module": record.module,
              "funcName": record.funcName,
              "lineno": record.lineno,
          },
      )
      if self._loop and self._loop.is_running():
        self._loop.call_soon_threadsafe(
            lambda e=event: asyncio.create_task(self._event_bus.publish(e))
        )
    except Exception:  # pylint: disable=broad-exception-caught
      self.handleError(record)
