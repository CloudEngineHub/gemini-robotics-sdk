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

"""Operator text UI handler for data collection scenarios."""

import time

from absl import logging

from safari_sdk.agent.framework.event_bus import event_bus
from safari_sdk.ui import client


_LOG_PREFIX_TEXT = "<color=#777777>[GEMINI] "  # Gray for model text.
_LOG_PREFIX_STEP = "<color=#AAAAAA>[GEMINI] "  # Brighter gray for info.
_LOG_SUFFIX = "</color>"
_framework: client.Framework | None = None

_message_buffer: str = ""
_last_time_buffer_updated: float = 0.0
_BUFFER_FLUSH_TIMEOUT_SECONDS: float = 2.0


def _send_to_robotics_ui(message: str) -> None:
  global _framework
  try:
    if _framework is None:
      logging.info(
          "Logging to Robotics UI before it is ready: trying to connect..."
      )
      _framework = client.Framework()
      _framework.connect(block_until_connected=True)
    _framework.add_chat_line("omega_star:chatbox", message)
  except Exception as e:  # pylint: disable=broad-except
    logging.warning("Failed to log to Robotics UI: %s", e)


def _should_skip_text(text: str) -> bool:
  return text.startswith((
      "code_output",
      "FunctionResponse",
      "function_response",
  ))


def _should_flush_buffer(text: str) -> bool:
  return text.rstrip().endswith((
      ".",
      "?",
      "!",
      "...",
  ))


def handle_event(event: event_bus.Event) -> None:
  """Handles events for operator data collection UI.

  Prints to both terminal and robotics UI with simplified formatting.

  Args:
    event: The event to handle.

  Behavior:
    - MODEL_TURN: Print part.text in gray. Track unfinished sentences (not
      ending with period) and append subsequent text. Format: [GEMINI]
    - TOOL_CALL: Print in red with format [GEMINI] <instruction>
    - TOOL_CALL_CANCELLATION: Print in red [GEMINI] CANCELLED.
    - TOOL_RESULT: Print [GEMINI] SUCCESS or FAILED based on
      subtask_success field.
  """
  global _message_buffer, _last_time_buffer_updated

  match event.type:
    case event_bus.EventType.MODEL_TURN:
      for part in event.data.parts:
        if part.text:
          text = part.text
          current_time = time.time()
          # TODO: This may print messages with huge delay since
          #   it only prints when the next event arrives.
          # If the buffer has been there for >2s, just print it.
          if (
              _message_buffer
              and current_time - _last_time_buffer_updated
              > _BUFFER_FLUSH_TIMEOUT_SECONDS
          ):
            _send_to_robotics_ui(
                f"{_LOG_PREFIX_TEXT}{_message_buffer}{_LOG_SUFFIX}"
            )
            _message_buffer = ""

          # If the new text starts with code_output or FR, skip it.
          # Special treatment: model code output and the acknowledgement FR
          # is classified as model turn text. Skip them.
          if _should_skip_text(text):
            continue

          # If there is unfinished buffer, append the current text to it.
          if _message_buffer:
            _message_buffer = " ".join([_message_buffer, text])
          else:
            _message_buffer = text

          # Print and flush the buffer or wait.
          if _should_flush_buffer(_message_buffer):
            _send_to_robotics_ui(
                f"{_LOG_PREFIX_TEXT}{_message_buffer}{_LOG_SUFFIX}"
            )
            _message_buffer = ""
          else:
            _last_time_buffer_updated = current_time

    case event_bus.EventType.TOOL_CALL:
      _message_buffer = ""
      instruction = event.data.function_calls[0].args.get("instruction", None)
      if instruction:
        _send_to_robotics_ui(
            f"{_LOG_PREFIX_STEP}INSTRUCTION: {instruction}{_LOG_SUFFIX}"
        )

    case event_bus.EventType.TOOL_CALL_CANCELLATION:
      _message_buffer = ""
      _send_to_robotics_ui(
          f"{_LOG_PREFIX_STEP}INSTRUCTION CANCELLED.{_LOG_SUFFIX}"
      )

    case event_bus.EventType.TOOL_RESULT:
      _message_buffer = ""
      response = event.data.function_responses[0].response
      if "subtask_success" not in response:
        logging.error(
            (
                "TOOL_RESULT event missing 'subtask_success' field in "
                "response: %s\n This event will be omitted from the UI."
            ),
            response,
        )
        return
      status = response.get("subtask_success", "unknown")
      _send_to_robotics_ui(
          f"{_LOG_PREFIX_STEP}SUCCESS: {status}{_LOG_SUFFIX}"
      )

    case _:
      pass
