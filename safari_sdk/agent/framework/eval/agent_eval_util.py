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

"""Agent evaluation utility library."""

import abc
import base64
import enum
import os
import subprocess
import threading
import time
from typing import Any, Callable

from absl import logging


_STATUS_PRINT_INTERVAL_SECONDS = 5.0


# ===========================================================================
# Configuration / Policy parsing helpers
# ===========================================================================


def get_runner_path_from_policy_details(policy_details, config_key: str) -> str:
  """Extracts a runner path from policy details."""
  runner_path = policy_details.get_parameter_value(config_key, None)
  if runner_path is None:
    raise ValueError(f"Runner path {config_key} not found in policy details.")
  return runner_path


def serialize_policy_details(policy_details) -> str:
  """Serializes policy details to base64-encoded JSON string."""
  json_str = policy_details.to_json()
  return base64.b64encode(json_str.encode("utf-8")).decode("utf-8")


# ===========================================================================
# Subprocess management
# ===========================================================================


def stream_output(
    pipe,
    prefix: str,
    log_file_path: str | None = None,
    show_logs: bool = False,
) -> None:
  """Reads and optionally prints output from a pipe."""
  log_file = None
  if log_file_path:
    try:
      log_file = open(log_file_path, "w")
    except IOError:
      logging.warning("Failed to open log file: %s", log_file_path)

  for line in iter(pipe.readline, b""):
    decoded_line = line.decode("utf-8").strip()
    if show_logs:
      logging.info("[%s] %s\n", prefix, decoded_line)
    if log_file:
      log_file.write(decoded_line + "\n")
      log_file.flush()

  pipe.close()
  if log_file:
    log_file.close()


def start_backend_subprocess(
    runner_path: str,
    policy_details_json_b64: str,
    log_prefix: str,
    save_logs: bool = True,
    log_dir: str | None = None,
    show_logs: bool = False,
) -> subprocess.Popen[Any]:
  """Starts a backend subprocess with serialized policy details."""
  cmd = [
      runner_path,
      f"--policy_details_json={policy_details_json_b64}",
  ]
  cmd_str = " ".join(cmd)
  logging.info("Starting %s with command: %s", log_prefix, cmd_str)
  process = subprocess.Popen(
      cmd_str,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      shell=True,
  )

  stdout_log_path = None
  stderr_log_path = None
  if save_logs and log_dir:
    clean_prefix = log_prefix.replace(" ", "_")
    stdout_log_path = os.path.join(log_dir, f"{clean_prefix}_stdout.log")
    stderr_log_path = os.path.join(log_dir, f"{clean_prefix}_stderr.log")

  stdout_thread = threading.Thread(
      target=stream_output,
      args=(process.stdout, f"{log_prefix}_STDOUT", stdout_log_path, show_logs),
  )
  stderr_thread = threading.Thread(
      target=stream_output,
      args=(process.stderr, f"{log_prefix}_STDERR", stderr_log_path, show_logs),
  )
  stdout_thread.daemon = True
  stderr_thread.daemon = True
  stdout_thread.start()
  stderr_thread.start()
  return process


def terminate_process(
    process: subprocess.Popen[Any] | None,
    name: str,
    timeout: int = 2,
) -> None:
  """Terminates a subprocess, force killing if necessary."""
  if process is None:
    return
  logging.info("Terminating %s process...", name)
  process.terminate()
  try:
    process.wait(timeout=timeout)
    logging.info("%s process terminated.", name)
  except subprocess.TimeoutExpired:
    logging.warning(
        "%s process did not terminate within %d seconds, forcing kill.",
        name,
        timeout,
    )
    process.kill()
    logging.info("%s process killed.", name)


# ===========================================================================
# Utility functions
# ===========================================================================


def print_every_x_seconds(
    message: Any,
    last_print_time: float,
    print_fn: Callable[[str], None],
    interval_seconds: float = _STATUS_PRINT_INTERVAL_SECONDS,
) -> float:
  """Prints a message if enough time has passed since the last print.

  Args:
    message: The message to print.
    last_print_time: The timestamp of the last print.
    print_fn: The function to use for printing.
    interval_seconds: Minimum seconds between prints.

  Returns:
    The updated last print time.
  """
  current_time = time.time()
  if current_time - last_print_time >= interval_seconds:
    print_fn(message)
    return current_time
  return last_print_time


class TerminationReason(enum.Enum):
  """The reason the episode terminated."""

  # The agent decided that the it has finished the episode.
  AGENT_TERMINATION_SIGNAL = "agent_termination_signal"
  # The operator pressed the abort signal via e.g., keyboard or pedal.
  OPERATOR_ABORT = "operator_abort"
  # The time limit for the episode was reached.
  TIME_LIMIT_REACHED = "time_limit_reached"
  # The backend server is unhealthy.
  BACKEND_UNHEALTHY = "backend_unhealthy"


# Utility functions.
class CustomUserIoConnection(abc.ABC):
  """An abstract base class for user IO functions."""

  @abc.abstractmethod
  def input(
      self,
      prompt: str,
      choices: list[str] | None = None,
      is_cancelable: bool = False,
      use_checklist_format: bool = False,
  ) -> str | list[str] | None:
    """Prompts the user and returns the user's input.

    Args:
      prompt: A message to prompt the user's input.
      choices: A list of choices to present to the user. If provided, the user
        will be prompted to select one of the choices.
      is_cancelable: Whether the user can cancel the input.
      use_checklist_format: Whether to use a checklist format for the user
        input.

    Returns:
      The user's input.
    """

  @abc.abstractmethod
  def print(self, message: str) -> None:
    """Prints a message for the user.

    Args:
      message: The message to print.
    """


class SetUserIoConnectionAsConsole(CustomUserIoConnection):
  """Set the user IO connection as standard python console."""

  def _input_with_choices(
      self, prompt: str, choices: list[str], is_cancelable: bool
  ) -> str | None:
    """Prompts the user and returns the user's input with choices."""
    choices_msg = "\n --- Choices: ---\n"
    for i, choice in enumerate(choices):
      choices_msg += f"  [{i+1}] {choice}\n"

    highest_choice_index = len(choices)
    lowest_choice_index = 1
    if is_cancelable:
      choices_msg += "\n  [0] Cancel\n"
      lowest_choice_index = 0

    display_prompt = prompt + choices_msg + " Your selection: "
    while True:
      user_input = input(display_prompt)

      try:
        user_input = int(user_input.strip())
      except Exception:  # pylint: disable=broad-except
        self.print(f"\n --- Invalid input: {user_input} ---\n")
        continue

      if is_cancelable and user_input == 0:
        return None

      if user_input < lowest_choice_index or user_input > highest_choice_index:
        self.print(f"\n --- Invalid input: {user_input} ---\n")
        continue

      return choices[user_input - 1]

  def _input_without_choices(self, prompt: str) -> str:
    """Prompts the user and returns the user's input with choices."""
    return input(prompt)

  def input(
      self,
      prompt: str,
      choices: list[str] | None = None,
      is_cancelable: bool = False,
      use_checklist_format: bool = False,
  ) -> str | list[str] | None:
    if choices:
      return self._input_with_choices(
          prompt=prompt, choices=choices, is_cancelable=is_cancelable
      )
    else:
      return self._input_without_choices(prompt=prompt)

  def print(self, message: str) -> None:
    print(message)
