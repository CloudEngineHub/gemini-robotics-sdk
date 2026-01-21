"""A collection of Success detection tools."""

import asyncio
import dataclasses
import time
from typing import Optional, Sequence

from absl import logging
from google.genai import types
import pytz

from safari_sdk.agent.framework import config as framework_config
from safari_sdk.agent.framework.event_bus import event_bus
from safari_sdk.agent.framework.tools import genai_client
from safari_sdk.agent.framework.tools import success_detection


_TASK_SUCCESS_KEY = "task_success"
_OVERALL_TASK_SUCCESS_KEY = "overall_task_success"
_TIMEOUT_KEY = "timeout"
_REASON_KEY = "reason"
_LOG_DIR = "/tmp/sd_logs"
_LOS_ANGELES_TIMEZONE = pytz.timezone("America/Los_Angeles")
_COMPACT_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
_COMPACT_TIMESTAMP_FORMAT_WITH_MILLIS = "%Y%m%d_%H%M%S.%f"


@dataclasses.dataclass
class SuccessResponse:
  success: bool = False
  ts: float = -1.0
  lock: asyncio.Lock = asyncio.Lock()


class EnsembleSubtaskSuccessDetectorV2(
    success_detection.VisionSuccessDetectionTool
):
  """Success detector for one-shot tasks based on state verification."""

  def __init__(
      self,
      bus: event_bus.EventBus,
      config: framework_config.AgentFrameworkConfig,
      api_key: str,
      model_name: str,
      sd_camera_endpoint_names: Optional[Sequence[str]],
      ensemble_size: int,
      ensemble_threshold: int,
      temperature: float,
      sleep_interval_s: float = 0.2,
      thinking_budget: int = 0,
      print_raw_sd_response: bool = False,
  ):
    super().__init__(
        bus=bus,
        fn=self.detect_success,
        declaration=self._get_declaration(),
        config=config,
        api_key=api_key,
        camera_endpoint_names=sd_camera_endpoint_names,
    )
    self._camera_names_for_prompt = sd_camera_endpoint_names
    self._num_image_streams = len(sd_camera_endpoint_names)
    # The name of the task.
    self._task = None
    # Whether the task is successful.
    self._success_signal = False
    # The timestamp of the images used to determine the success signal.
    self._success_signal_images_time = None
    # Create Gemini client wrapper for success detection model.
    thinking_config = types.ThinkingConfig(
        include_thoughts=False,
        # CAUTION: Instantly overloads the server if using thinking!
        thinking_budget=thinking_budget,
    )
    generate_content_config = types.GenerateContentConfig(
        temperature=temperature,
        thinking_config=thinking_config,
        # Suppress function calling message spam. Doesn't actually affect
        # function calling.
        automatic_function_calling=types.AutomaticFunctionCallingConfig(
            disable=True,
        ),
    )
    self._gemini_wrapper = genai_client.GeminiClientWrapper(
        bus=bus,
        client=self._client,
        model_name=model_name,
        config=generate_content_config,
        print_raw_response=print_raw_sd_response,
    )
    self._ensemble_size = ensemble_size
    self._ensemble_threshold = ensemble_threshold
    self._sleep_interval_s = sleep_interval_s

  def _get_declaration(self):
    return types.FunctionDeclaration(
        name="detect_success",
        description=(
            "Detects subtask success as frequent as possible. This is a"
            " non-blocking tool. It will detect success and return"
            ' {"task_success": "maybe true"} when the subtask is likely'
            ' successful or {"task_success": "maybe false"} when the subtask is'
            " likely not successful. Although not part of the parameters, this"
            " function requires at least one video stream from a camera"
            " endpoint. But will likely work better with more cameras to"
            " provide different angles."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "subtask": types.Schema(
                    type=types.Type.STRING,
                    description=(
                        "A short, clear subtask for the robot (e.g. 'pick up"
                        " the cup', 'place on the table', 'move left')"
                    ),
                ),
            },
            required=["subtask"],
        ),
        behavior=types.Behavior.NON_BLOCKING,
    )

  async def detect_success(
      self, subtask: str, call_id: str
  ) -> types.FunctionResponse:
    """Returns the success detection result for a given subtask.

    Args:
      subtask: The subtask to be evaluated for success.
      call_id: The id of the tool call.
    """
    # Record the id of the tool call and mark it as running.
    self._call_id = call_id
    self._tool_call_cancelled = False

    # Reset the task success properties.
    self._task = subtask
    self._success_signal = False
    self._success_signal_images_time = None
    self._external_success_signal = False

    # Set a timeout for the task to be successful.
    timeout_seconds = self.get_timeout_seconds()
    end_time = time.time() + timeout_seconds

    # Check if the task is successful.
    background_tasks = set()
    response = types.FunctionResponse(
        response={_TASK_SUCCESS_KEY: "maybe true", _TIMEOUT_KEY: False},
        will_continue=False,
    )
    ensemble_responses = [SuccessResponse() for _ in range(self._ensemble_size)]
    start_images = [
        self._image_buffer.get_start_images_map()[camera_endpoint_name].data
        for camera_endpoint_name in self._image_buffer.get_camera_endpoint_names()
    ]
    while not self._success_signal:
      # Image buffer refreshes at 5Hz.
      await asyncio.sleep(self._sleep_interval_s)
      # Check if the tool call is cancelled.
      if self._tool_call_cancelled:
        response = types.FunctionResponse(
            response={_TASK_SUCCESS_KEY: "maybe false", _TIMEOUT_KEY: False},
            will_continue=False,
        )
        break
      # Check if we received an external success signal.
      if self._external_success_signal:
        self._success_signal = self._external_success_signal
        self._success_signal_images_time = time.time()
        response = types.FunctionResponse(
            response={
                _TASK_SUCCESS_KEY: "maybe true",
                _TIMEOUT_KEY: False,
            },
            will_continue=False,
        )
        break
      # Check if the time limit is reached.
      if time.time() > end_time:
        logging.info("SD: Timed out after %s seconds.", timeout_seconds)
        response = types.FunctionResponse(
            response={_TASK_SUCCESS_KEY: False, _TIMEOUT_KEY: True},
            will_continue=False,
        )
        break

      await self._check_current_task_success(
          start_images, subtask, ensemble_responses, background_tasks
      )
    for background_task in background_tasks:
      background_task.cancel()
    return response

  async def _check_current_task_success(
      self,
      start_images: Sequence[bytes],
      subtask: str,
      ensemble_responses: Sequence[SuccessResponse],
      background_tasks: set[asyncio.Task],
  ) -> None:
    current_images = [
        self._image_buffer.get_latest_images_map()[camera_endpoint_name][
            -1
        ].data
        for camera_endpoint_name in self._image_buffer.get_camera_endpoint_names()
    ]
    current_images_timestamp = self._image_buffer.get_latest_image_timestamp()
    if not current_images or not current_images_timestamp:
      logging.warning("[SD]: No images available.")
      return
    sd_signal = await self._get_sd_signal(
        start_images=start_images,
        current_images=current_images,
        subtask=subtask,
        ensemble_responses=ensemble_responses,
        background_tasks=background_tasks,
    )

    # TODO: Get rid of these class variables, doesn't mesh well
    # with the possibility of the agent issuing multiple concurrent tasks.

    # The subtask is no longer the current task.
    if self._task != subtask:
      return

    # The first time we receive a success signal for the current task.
    if not self._success_signal_images_time:
      self._success_signal_images_time = current_images_timestamp
      self._success_signal = sd_signal
      return
    self._success_signal_images_time = current_images_timestamp
    self._success_signal = sd_signal

  async def _get_sd_signal(
      self,
      start_images: Sequence[bytes],
      current_images: Sequence[bytes],
      subtask: str,
      ensemble_responses: Sequence[SuccessResponse],
      background_tasks: set[asyncio.Task],
  ) -> bool:
    """Query the specialist model for the success detection signal."""
    # pylint: disable=line-too-long
    sd_question = (
        f"The task is: {subtask}.Q1 What state should the objects be in if the"
        " robot successfully completed the task? Answer this by combining all"
        " views to form a complete picture.Q2 Are the objects currently in the"
        " state consistent with the state where the robot successfully"
        " completed at the current time? What are the differences? Tell me"
        " what views and times do you base your answer on first.Remember that"
        " the objects may look differently from diffferent sides, so if the"
        " object is missing from its origianl location and a new object has"
        " appeared in the new location, most likely the target object has been"
        " moved.If you think the robot finished the task, answer: 'FINAL"
        " ANSWER: yes'If you think the roboy did not finish the task, answer:"
        " 'FINAL ANSWER: no'.Do not mention the word 'FINAL ANSWER' anywhere"
        " else."
    )
    # pylint: enable=line-too-long
    image_prompt_start = ["Start of the episode:"]
    image_prompt_current = ["End of the episode:"]
    for camera_name, image in zip(self._camera_names_for_prompt, start_images):
      image_prompt_start.append(f"{camera_name}:")
      image_prompt_start.append(image)
    for camera_name, image in zip(
        self._camera_names_for_prompt, current_images
    ):
      image_prompt_current.append(f"{camera_name}:")
      image_prompt_current.append(image)
    prompt = [*image_prompt_start, *image_prompt_current, sd_question]

    # Send ensemble queries.
    for i in range(self._ensemble_size):
      task = asyncio.create_task(
          self._send_ensemble_query(i, prompt, ensemble_responses)
      )
      background_tasks.add(task)
      task.add_done_callback(background_tasks.discard)

    # DON'T wait for ensemble queries to complete. Just check whatever is ready.
    success_count = 0
    now = time.time()
    for response in ensemble_responses:
      latency = now - response.ts
      logging.log_every_n_seconds(logging.INFO, "SD: latency: %s", 10, latency)
      if response.success:
        success_count += 1
    logging.info("SD: success_count: %s", success_count)
    return success_count >= self._ensemble_threshold

  async def _send_ensemble_query(
      self,
      worker: int,
      prompt: Sequence[bytes],
      ensemble_responses: Sequence[SuccessResponse],
  ):
    ts = time.time()
    try:
      response = await self._gemini_wrapper.generate_content(prompt)
      if response.text is None:
        logging.warning("SD query response text is None.")
        return
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.exception("SD model failed to respond:\n%s", e)
      return
    async with ensemble_responses[worker].lock:
      # Ignore responses if a later thread was faster.
      if ensemble_responses[worker].ts > ts:
        pass
      else:
        ensemble_responses[worker].ts = ts
        ensemble_responses[worker].success = "yes" in response.text.lower()
