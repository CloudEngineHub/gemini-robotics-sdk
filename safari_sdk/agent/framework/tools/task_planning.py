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

"""A collection of task planning tools."""

import asyncio
import copy
import functools
import time
from typing import Any, AsyncIterator, Sequence, cast

from absl import logging
from google import genai
from google.genai import types

from safari_sdk.agent.framework import constants
from safari_sdk.agent.framework import flags as agentic_flags
from safari_sdk.agent.framework.event_bus import event_bus
from safari_sdk.agent.framework.tools import genai_client
from safari_sdk.agent.framework.tools import tool
from safari_sdk.agent.framework.utils import image_processing


_SUBTASK_GUIDELINE = """
1) The subtask should be atomic, i.e. involves a single, low-level action (put, place, move, push/pull) onto an object.
2) Natively Supported Actions (Treat as Atomic): Some actions can be treated as single steps even though it sounds complex:
  - Wiping table with an object
  - Stacking two objectss
  - Zipping and unzipping an object with a zipper
  - Folding paper / napkin / cloth
  - Opening and closing something
3) Subtasks should not be 'pick up/grab/hold object X' nor 'put down object X'. Instead, it should be 'put object X in/on Y.'
4) The robot can only grab one object at a time and many tasks requires both arms.
5) You should the mention the number of objects to interact with.
   For example, you should say "put one bread from the plate to the wooden plate"
   instead of "put breads from the plate to the wooden plate" if you only want
   to interact with one bread.
7) You cannot hold one object and interact with another object.
8) Do not create subtasks like "grab the red block" nor "pick up the red block". Instead, they should be like "put the red block in the green tray".
9) Do not include the spatial coordiantes of the object in the subtask. For example, you should say "put the red block on the left in the container" put the red block at (1, 2, 2) in the container".
10 Do use specific spatial references in subtasks, but do not include the amount of objects in the subtask.
   For example, you should say "put the red block on the left in the middle drawer" instead of "put one red block in the drawer".
11) Do not specify which arm for the robot to use in the subtask. For example, you should say "put the red block in the bin" instead of "put ther red block in the bin with your left arm".
12) Do not include the number of objects in the subtask. For example, you should never say "put one red block in the middle drawer".
13) Spatial references should be relative to the camera view. For example, you should say "put the red block on the most left in the middle drawer" instead of "put the red block closest to me in the middle drawer".
"""

"""
Wait until the run_instruction_for_duration function returns. The function response just confirms that the robot executed the instruction for a duration of time. It does NOT confirm that the instruction was fulfilled.
If you see that the robot is still working on the instruction, you should continue to call the same function with the instruction "open the middle drawer" and maintain the instruction consistency (preferably by using the exact same instruction).
Then, when you see that the middle drawer is open, you should call a function with the instruction "put the red block on the left in the middle drawer".
Then, when you see that the red block is in the middle drawer, you should call a function with the instruction "put the red block closest to the green block in the middle drawer".
Next, when you see that there is only one red block left on the table (since 2 blocks are already in the middle drawer), you should call a function with the instruction "put the red block on the table in the middle drawer".
Finally, when you see that there are no more red blocks on the table and there are 3 red blocks in the middle drawer, you should call a function with the instruction "close the middle drawer".
Lastly, before you announce to the user that you have fulfilled their request, you should describe in detail everything you see in the current image and then explain to the user why you fulfilled their request.
"""

_MULTI_CAMERA_PLANNING_PROMPT = """
Based on the information above.
Generate a task plan if the previous task plan is empty.
Otherwise, update the task plan if needed.
The task plan should be a ' -> ' separated list of subtasks, e.g. 'put the red square block in the basket -> open the middle drawer -> ...'.

You should think about the what how the robot should achieve the overall goal by breaking it down into subtasks using the camera information.

Make sure the task plan considers the physical feasibility of the subtasks. For example, a big object may not fit into a small container.

If you think the robot is still working on the first subtask of the previous task plan and there's no need to update the task plan, then output the previous task plan exactly.

If the overall goal is achieved, then simply output exactly 'overall goal achieved'.

Before you output the task plan, to the following first:
1) Tell me what objects do you see in each camera and which objects shows up in multiple cameras.
2) Tell me the reasoning behind your task plan
Only then, output the task plan after the text: "TASK PLAN:"

Be careful about the 'previous task plan'. It is given as the best effort by a ML model,
so please double check it visually and trust your vision more than the 'previous task plan'.
However, it can be very useful if you cannot determine the current task plan purely from the start and current images.
For example, if the overall goal is 'find the red block in the drawer. You can only open one drawer at a time.'
and the previous task plan is 'open the left drawer (success) -> 'pick up the red block (failed, no red block found)' -> close the left drawer (success) then you can use this information to determine the current task plan is 'open the right drawer' since the you know that the red block is not in the left drawer.

You should strictly follow the guidelines below when generating the task plan:
"""

_PREVIOUS_AND_CURRENT_TASK_PLAN_DESCRIPTION = """
The previous and current task plan as a string of ' -> ' separated subtasks.
The task planner will use the previous and current task plan and return an
updated plan (maybe the same).
If empty, the tool will create a brand new task plan. All previously executed
subtasks should be included here with indications of it being successful or not.
Example input for an overall goal of 'find the red block in the drawer':

'open the left drawer (success) -> 'pick up the red block (failed,
no red block found)' -> 'open the right drawer (current)'.
"""

_MULTI_CAMERA_PLANNING_PROMPT_V2 = """
Based on the information above.
Generate a task plan for the robot to achieve the overall goal.
Note that if the previous subtasks and remaining subtasks are both empty, it means that there is no existing task plan.
You need to generate a task plan from scratch based on the information above.

Before you generate the task plan, answer the following questions first:
Q1: Tell me about the state of task relevant objects by combining all camera views to form a complete picture.
Q2: What are the relationships between the relevant objects?
Q3: What would the state of relevant objects be if the overall goal is achieved?
Q4: Is the overall goal achievable?
Q5: Is the overall goal already achieved?
Limit this part of your response to {num_words} words.

General guidelines:
* After the answer to the above questions, output the task plan after the text: "TASK PLAN:"
* The task plan should be a ' -> ' separated list of subtasks.
* The first subtask must be EXACTLY the currently executing subtask decorated by the text "(CURRENT)".
* Limit the number of subtasks to {num_planned_subtasks} steps into the future.
  You do not have to finish the overall task within the steps.
* Make sure the task plan considers the physical feasibility of the subtasks.
  For example, a big object may not fit into a small container.
* If the overall goal can be achieved within {num_planned_subtasks} steps, the last subtask should always be: "DONE" to indicate that the overall goal will be achieved then.
* It is very important to utilize the camera information to determine the remaining subtasks as the remaining subtasks come from previous task planning result and may be stale or inaccurate due to changes in the scene.
* Sometimes you may find that the robot actually did not finish all the previous subtasks despite them being marked as 'executed'.
  In that case, you should think about how to include it in the new task plan so that the overall goal can be achieved.
* If you noticed that the same subtask had been executed more than {num_subtask_retry} times, it probably means that the robot can't execute that subtask you should probably find a way around it.
* If a subtask is in the previous subtasks (executed), and you can visually verify its completion, e.g., the drawer is visibly open, you should not include that subtask in the new task plan.
  If you cannot visually verify a previous subtask's completion, e.g., "put the red square block in the left drawer (executed)" and see that the left drawer if closed, you should assume that the subtask is completed and do NOT include it in the new task plan.
  However, if you can visually see that a previous subtask is not completed, e.g., "open the left drawer (executed)" and see that the left drawer is still closed, and importantly, nothing in the previousl subtasks or current subtask can potentially cause the the left drawer to be closed, you should include it in the new task plan.
* If the overall goal is not achievable, you should output "overall goal not achievable" followed by the reasons.
* Never output remaining subtasks that are essentially the same as the current subtask. For example, if the current subtask is "open the left drawer", you should NOT have remaining subtasks like "open the drawer on the left" or "open the left drawer".
  However, if the current subtask is "open the left drawer" and the remaining subtasks are "put the red square block in the left drawer -> close the left drawer -> DONE", you should include the "open the left drawer" in the new task plan since it is not essentially the same as the current subtask.
* Try to group similar subtasks together. For example, prefer: "put the red cube block in the red tray -> put the red cylinder block in the red tray -> put the blue sphere block in the blue tray" over "put the red cube block in the red tray -> put the blue sphere block in the blue tray -> put the red cylinder block in the red tray".
  over "put the red cube block in the red tray -> close the left drawer -> put the red cylinder block in the red tray -> put the blue sphere block in the blue tray -> put the blue cylinder block in the blue tray -> close the left drawer"

Example task plan output given the following information:
overall goal: "put one object in each drawer"
previous subtasks: "open the left drawer (executed) -> put the red square block in the left drawer (executed) -> close the left drawer (executed)"
current subtask: "open the middle drawer"
remaining subtasks: "put the blue wedged block in middle drawer -> close the middle drawer -> open the right drawer -> put the green block in the right drawer -> close the right drawer -> DONE"
example task plan: TASK PLAN: "open the middle drawer (CURRENT) -> close the left drawer -> put the blue wedged block in middle drawer -> close the middle drawer -> DONE"
In this example, the previous task plan is stale since the left drawer was not closed and the green block to be put in the right drawer is no longer visible in any camera view (perhaps the user took it).

Subtask granularity guidelines:
1) The subtask should be atomic, i.e. involves a single, low-level action (put, place, move, push/pull) onto an object.
2) Natively Supported Actions (Treat as Atomic): Some actions can be treated as single steps even though it sounds complex:
  - Wiping table with an object
  - Stacking two objects
  - Zipping and unzipping an object with a zipper
  - Folding paper / napkin / cloth
  - Opening and closing something
3) Subtasks should not be 'pick up/grab/hold object X' nor 'put down object X'. Instead, it should be 'put object X in/on Y.'
4) The robot can only grab one object at a time and many tasks requires both arms.
5) You cannot hold one object and interact with another object.
6) Do not create subtasks like "grab the red block" nor "pick up the red block". Instead, they should be like "put the red block in the green tray".
7) Do not include the spatial coordinates of the object in the subtask. For example, you should say "put the red block on the left in the container" put the red block at (1, 2, 2) in the container".
8) Do use specific spatial references in subtasks, but do not include the amount of objects in the subtask.
   For example, you should say "put the red block on the left in the middle drawer on the table" instead of "put one red block in the drawer".
9) Do not specify which arm for the robot to use in the subtask. For example, you should say "put the red block in the bin" instead of "put ther red block in the bin with your left arm".
10) Do not include the number of objects in the subtask. For example, you should never say "put one red block in the middle drawer".
11) Do not issue subtasks like "put the other red block in the tray", instead you should say "put the red block on the left on the table in the tray". Describe the subject using spatial references instead of temporal ones like "the other one".
12) Spatial references should be relative to the camera view. For example, you should say "put the red block on the most left in the middle drawer" instead of "put the red block closest to me in the middle drawer".
13) If the user's request already fits the conditions above, you should not modify it. For example, if the user says "put the bread from the plate to the wooden plate", you should just have a subtask of "put the bread from the plate to the wooden plate".
"""


class ImageBuffer:
  """Keeps track of the latest images from the event bus. (V1 task planning)."""

  def __init__(self, camera_endpoint_names: Sequence[str]):
    self._latest_images = {}
    self._camera_endpoint_names = camera_endpoint_names

  def update_from_event(self, event: event_bus.Event):
    """Handle the subscribed events."""
    # Store the latest image for each camera stream.
    self._latest_images[event.metadata[constants.STREAM_NAME_METADATA_KEY]] = (
        event.data
    )

  def get_latest_images(self) -> dict[str, bytes]:
    """Get the latest images from the event bus."""
    return self._latest_images

  def get_interleaved_images(self, imgs_dict: dict[str, bytes]) -> list[Any]:
    """Converts an imgs_dict of images to a string of interleaved images."""
    interleaved = []
    for stream_name in self._camera_endpoint_names:
      interleaved.append(stream_name.replace("/", "").strip() + ":")
      interleaved.append(imgs_dict[stream_name])
      interleaved.append("\n")
    return interleaved


class ImageBufferWithHistory:
  """Keeps track of the latest images from the event bus. (V2 task planning)."""

  def __init__(
      self,
      camera_endpoint_names: Sequence[str],
      num_history_frames: int = 2,
      history_interval_s: float = 1.0,
  ):
    self._last_image_update_time = {}
    self._history_images = []
    self._latest_images = {}
    self._latest_image_timestamp = 0.0
    self._camera_endpoint_names = camera_endpoint_names
    self._num_history_images = num_history_frames
    self._history_interval_s = history_interval_s

    for stream_name in self._camera_endpoint_names:
      self._last_image_update_time[stream_name] = 0.0

  def get_latest_image_timestamp(self) -> float:
    """Returns the timestamp of the latest images."""
    return self._latest_image_timestamp

  def get_latest_image_prompt_with_names(self) -> Sequence[Any]:
    prompt = []
    for stream_name in self._camera_endpoint_names:
      prompt.append(stream_name.replace("/", "").strip() + ":")
      prompt.append(self._latest_images[stream_name])
    return prompt

  def get_history_image_prompt_with_names(self) -> Sequence[Any]:
    prompt = []
    for history_image_dict in self._history_images:
      elapsed_time = time.time() - history_image_dict["timestamp"]
      endpoint_name = history_image_dict["endpoint_name"]
      prompt.append(
          endpoint_name.replace("/", "").strip()
          + f" from {elapsed_time:.1f} seconds ago:"
      )
      prompt.append(history_image_dict["image"])
    return prompt

  def update_from_event(self, event: event_bus.Event):
    """Handle the subscribed events."""
    # Store the latest image for each camera stream.
    camera_endpoint_name = event.metadata[constants.STREAM_NAME_METADATA_KEY]
    if camera_endpoint_name not in self._camera_endpoint_names:
      return
    self._latest_images[camera_endpoint_name] = copy.deepcopy(event.data)
    self._latest_image_timestamp = event.timestamp
    if (
        time.time() - self._last_image_update_time[camera_endpoint_name]
        > self._history_interval_s
    ):
      self._history_images.append({
          "endpoint_name": camera_endpoint_name,
          "image": copy.deepcopy(event.data),
          "timestamp": time.time(),
      })
      self._last_image_update_time[camera_endpoint_name] = time.time()
      # Pop the oldest image if we have too many.
      if len(self._history_images) > self._num_history_images * len(
          self._camera_endpoint_names
      ):
        self._history_images.pop(0)


class MultiCameraVisionTaskPlanningTool(tool.Tool):
  """A stateful tool that does task planning.

  This tool can create a new task plan or update an existing one based on the
  vision inputs and the overall goal. It uses Gemini API to generate the task
  plan. This tool outputs a task plan that is a string of ' -> ' separated
  subtasks and will return overall goal achieved if the overall goal is
  achieved. This is meant to be used with the run_until_done agent, i.e., use
  this tool after a subtask is completed.
  """

  def __init__(
      self,
      bus: event_bus.EventBus,
      camera_endpoint_names: Sequence[str],
      api_key: str,
      # Set below for V2 mode only.
      v2_mode: bool = False,
      num_history_frames: int = 0,
      history_interval_s: float = 1.0,
      print_raw_task_planning_response: bool = False,
      planning_model_name: str = "models/gemini-2.5-flash",
      planning_thinking_budget: int = 200,
      planning_num_words: int = 200,
      num_planned_subtasks: int = 5,
      num_planned_subtasks_retry: int = 3,
  ):
    # One time planning.
    self._v2_mode = v2_mode
    self.get_task_plan = (
        self.get_task_plan_v2 if v2_mode else self.get_task_plan_v1
    )
    declaration = self._get_decalration(v2_mode)
    python_function = self.get_task_plan
    super().__init__(fn=python_function, declaration=declaration, bus=bus)
    self._call_id = None
    self._current_task_plan = "None"
    self._start_marked = False
    self._start_imgs = None
    self._current_overall_goal = ""
    self._print_raw_task_planning_response = print_raw_task_planning_response
    self._planning_num_words = planning_num_words
    self._num_planned_subtasks = num_planned_subtasks
    self._num_planned_subtasks_retry = num_planned_subtasks_retry
    ###### START: v2 mode only
    self._start_images_prompt = []
    self._num_history_frames = num_history_frames
    self._history_interval_s = history_interval_s
    ###### END: v2 mode only

    # Initialize the LLM with the Gemini API.
    client = genai.Client(api_key=api_key)
    # Controls the thinking budget.
    logging.info(
        "task planning thinking budget: %s",
        planning_thinking_budget,
    )
    thinking_config = types.ThinkingConfig(
        thinking_budget=planning_thinking_budget
    )
    config = types.GenerateContentConfig(
        temperature=0.0, thinking_config=thinking_config
    )
    self._gemini_client = genai_client.GeminiClientWrapper(
        bus=bus,
        client=client,
        model_name=planning_model_name,
        config=config,
        print_raw_response=self._print_raw_task_planning_response,
    )
    # Stores the last num_history_frames frames.
    self.history_images = []

    # The buffer to store the latest images from the event bus.
    if v2_mode:
      self._image_buffer = ImageBufferWithHistory(
          camera_endpoint_names,
          num_history_frames=self._num_history_frames,
          history_interval_s=self._history_interval_s,
      )
    else:
      self._image_buffer = ImageBuffer(camera_endpoint_names)
    bus.subscribe(
        event_types=[event_bus.EventType.MODEL_IMAGE_INPUT],
        handler=self._image_buffer.update_from_event,
    )

  def _get_decalration(self, v2_mode: bool):
    if not v2_mode:
      return types.FunctionDeclaration(
          name="get_task_plan",
          description=(
              "Get a task plan to achieve the overall goal based on the current"
              " task plan, vision inputs and additional context. This tool uses"
              " a large vision language model to generate the task plan."
              " This function requires marking at the start of the overall goal"
              " to work. When an overall goal starts, one must call this"
              " function with the mode argument set to 'mark_start'. Only then,"
              " can this function be called with the mode argument set to"
              " 'check_success' to check if the overall goal is successful."
              " Failure to call this function with mode set to 'mark_start'"
              " prior to calling with mode set to 'check_success' will result"
              " in an Error. If mode='mark_start' it will return {'mark_start':"
              " True}. If the overall goal is achieved, this function will"
              " return {'task_plan': overall goal achieved'}."
          ),
          parameters=types.Schema(
              type=types.Type.OBJECT,
              properties={
                  "mode": types.Schema(
                      type=types.Type.STRING,
                      description=(
                          "The mode this function operates. Use 'mark_start' at"
                          " the beginning of the overall task. Use"
                          " 'get_task_plan' to get the task plan. Note that"
                          " failure to call this function with mode set to"
                          " 'mark_start' prior to calling with mode set to"
                          " 'get_task_plan' will result in an Error."
                      ),
                  ),
                  "overall_goal": types.Schema(
                      type=types.Type.STRING,
                      description=(
                          "The overall goal of the robot to be broken"
                          " down into a task plan consisting of a"
                          " sequence of subtasks."
                      ),
                  ),
                  "previous_and_current_task_plan": types.Schema(
                      type=types.Type.STRING,
                      description=_PREVIOUS_AND_CURRENT_TASK_PLAN_DESCRIPTION,
                  ),
                  "additional_context": types.Schema(
                      type=types.Type.STRING,
                      description=(
                          "Additional context to help the robot achieve the"
                          " overall goal. You can use this field to convey"
                          " things like: 'The robot had been trying the same"
                          " task plan for X times and it did not make any"
                          " progress. Please update accordingly.' (replace X)"
                          " or 'Here are the robot's capabilities and preferred"
                          " task plan granularity: ...'."
                      ),
                  ),
              },
              required=[
                  "mode",
                  "overall_goal",
                  "previous_and_current_task_plan",
                  "additional_context",
              ],
          ),
          behavior=types.Behavior.BLOCKING,
      )
    else:
      return types.FunctionDeclaration(
          name="get_task_plan",
          description=(
              "Get a task plan to achieve the overall goal based on the vision"
              " inputs, previous task plan, executing subtask and remaining"
              " subtasks. This tool uses a large vision language model to"
              " generate the task plan. This function requires marking at the"
              " start of the overall goal to work. When an overall goal starts,"
              " one must call this function with the mode argument set to"
              " 'mark_start'. Only then, can this function be called with the"
              " mode argument set to 'check_success' to check if the overall"
              " goal is successful. Failure to call this function with mode set"
              " to 'mark_start' prior to calling with mode set to"
              " 'check_success' will result in an Error. If mode='mark_start'"
              " it will return {'mark_start': True}."
          ),
          parameters=types.Schema(
              type=types.Type.OBJECT,
              properties={
                  "mode": types.Schema(
                      type=types.Type.STRING,
                      description=(
                          "The mode this function operates. Use 'mark_start' at"
                          " the beginning of the overall task. Use"
                          " 'get_task_plan' to get the task plan. Note that"
                          " failure to call this function with mode set to"
                          " 'mark_start' prior to calling with mode set to"
                          " 'get_task_plan' will result in an Error."
                      ),
                  ),
                  "overall_goal": types.Schema(
                      type=types.Type.STRING,
                      description=(
                          "The overall goal of the robot to be broken"
                          " down into a task plan consisting of a"
                          " sequence of subtasks."
                      ),
                  ),
                  "previous_subtasks": types.Schema(
                      type=types.Type.STRING,
                      description=(
                          "previously executed subtasks in a '->' separated"
                          " list with execution status comments. E.g. 'put the"
                          " red block in the red tray (executed) -> put the"
                          " blue block in the blue tray (timeout)'. This can be"
                          " en empty string if it is the first time calling"
                          " this function."
                      ),
                  ),
                  "executing_subtask": types.Schema(
                      type=types.Type.STRING,
                      description=(
                          "The currently executing subtask, e.g., 'put the"
                          " green block in the green tray'. This can be"
                          " 'nothing' if it is the first time calling this"
                          " function."
                      ),
                  ),
                  "remaining_subtasks": types.Schema(
                      type=types.Type.STRING,
                      description=(
                          "The remaining subtasks that have not been executed"
                          " yet, as a '->' separated list. This should come"
                          " from the previous task plan. If the overall goal"
                          " will be achieved at the end of the plan, the last"
                          " element should be 'DONE'. This can be an empty"
                          " string if it is the first time calling this"
                          " function. Example input: 'put the yellow block in"
                          " the yellow tray -> put the cyan block in the cyan"
                          " tray -> DONE' "
                      ),
                  ),
              },
              required=[
                  "mode",
                  "overall_goal",
                  "previous_subtasks",
                  "executing_subtask",
                  "remaining_subtasks",
              ],
          ),
          behavior=types.Behavior.NON_BLOCKING,
      )

  def set_gemini_client_config(self, config: types.GenerateContentConfigOrDict):
    self._gemini_client.set_config(config)

  def get_gemini_client_config(self) -> types.GenerateContentConfigOrDict:
    return self._gemini_client.get_config()

  async def _get_task_plan_v1(
      self,
      overall_goal: str,
      previous_and_current_task_plan: str,
      additional_context: str,
  ) -> types.FunctionResponse:
    if self._v2_mode:
      raise ValueError("POSSIBLE BUG: _get_task_plan_v1 is called in V2 mode.")
    self._image_buffer = cast(ImageBuffer, self._image_buffer)
    if self._start_imgs is None:
      raise ValueError(
          "No start images recorded. You must have called this function with"
          " mode 'mark_start' before calling it with mode 'check_success'."
      )

    prompt = [
        f"overall task: {overall_goal}",
        f"previous task plan: {previous_and_current_task_plan}",
        f"additional context: {additional_context}",
        (
            "This is what the robot sees when the overall task started. All"
            " images below correspond to the same time, just from different"
            " cameras."
        ),
        *self._image_buffer.get_interleaved_images(self._start_imgs),
        (
            "This is what the robot sees now. All images below correspond to"
            " the same timestamp, just from different cameras."
        ),
        *self._image_buffer.get_interleaved_images(
            self._image_buffer.get_latest_images()
        ),
        _MULTI_CAMERA_PLANNING_PROMPT,
        _SUBTASK_GUIDELINE,
        (
            "You should limit the number of subtasks to"
            f" {self._num_planned_subtasks}"
            " steps into the future."
        ),
        " You do not have to finish the overall task within the steps.",
    ]
    prompt = image_processing.convert_bytes_to_image(prompt)

    try:
      response = await self._gemini_client.generate_content(contents=prompt)

      # Directly return the raw response as the task plan.
      return types.FunctionResponse(
          response={
              "task_plan": response.text,
              "only_use_first_subtask": True,  # Prevents multiple rapid fire FC
          },
          will_continue=False,
      )

    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.exception(
          "Multi-camera task planning model failed to respond:\n%s", e
      )
      await asyncio.sleep(0.1)  # This is so that it returns a coroutine.
      return types.FunctionResponse(
          response={
              "task_plan": "null",
              "reason": (
                  "Multi-camera task planning model failed to respond, try"
                  " again."
              ),
          },
          will_continue=False,
      )

  async def get_task_plan_v1(
      self,
      mode: str,
      overall_goal: str,
      previous_and_current_task_plan: str = "",
      additional_context: str = "",
      call_id: str = "",
  ) -> types.FunctionResponse:
    """Exposed stateless function that gets the task plan."""
    if self._v2_mode:
      raise ValueError("POSSIBLE BUG: get_task_plan_v1 is called in V2 mode.")
    self._image_buffer = cast(ImageBuffer, self._image_buffer)
    self._call_id = call_id
    if mode == "mark_start":
      self._start_imgs = copy.deepcopy(self._image_buffer.get_latest_images())
      self._start_marked = True
      self._current_overall_goal = overall_goal
      return types.FunctionResponse(
          response={"marked_start": True},
          will_continue=False,
      )
    elif mode == "get_task_plan":
      if not self._start_marked:
        return types.FunctionResponse(
            response={
                "start_not_marked, please call with mode='mark_start' first": (
                    True
                )
            },
            will_continue=False,
        )
      if overall_goal != self._current_overall_goal:
        self._start_marked = False
        return types.FunctionResponse(
            response={
                "overall_goal_changed, please call with mode='mark_start' first": (
                    True
                )
            },
            will_continue=False,
        )
      return await self._get_task_plan_v1(
          overall_goal=overall_goal,
          previous_and_current_task_plan=previous_and_current_task_plan,
          additional_context=additional_context,
      )
    else:
      logging.warning("Mode %s is not supported.", mode)
      return types.FunctionResponse(
          response={"unsupported_mode": True},
          will_continue=False,
      )

  async def _get_task_plan_v2(
      self,
      overall_goal: str,
      previous_subtasks: str,
      executing_subtask: str,
      remaining_subtasks: str,
  ) -> types.FunctionResponse:
    if not self._v2_mode:
      raise ValueError("POSSIBLE BUG: _get_task_plan_v2 is called in V1 mode.")
    self._image_buffer = cast(ImageBufferWithHistory, self._image_buffer)
    if not self._start_images_prompt:
      raise ValueError(
          "No start images recorded. You must have called this function with"
          " mode 'mark_start' before calling it with mode 'check_success'."
      )

    # Make prompt.
    prompt = []
    # Start.
    prompt.append(
        "The images below show what the robot sees at the start of the"
        " robot's overall task (the robot may have more than one camera):"
    )
    prompt.extend(self._start_images_prompt)
    # History.
    if self._num_history_frames > 0:
      prompt.append("Here are what the robot sees recently:")
      prompt.extend(self._image_buffer.get_history_image_prompt_with_names())
    # Current.
    prompt.append("Here are what the robot sees now:")
    prompt.extend(self._image_buffer.get_latest_image_prompt_with_names())
    # Overall goal.
    prompt.append(f"The robot's overall goal is: {overall_goal}")
    # Previous task plan.
    prompt.append(
        f"The robot's previously executed subtasks are: {previous_subtasks}"
    )
    # Executing subtask.
    prompt.append(
        f"The robot is currently executing subtask: {executing_subtask}"
    )
    # Remaining subtasks.
    prompt.append(f"The robot's remaining subtasks are: {remaining_subtasks}")
    # Main prompt.
    main_prompt = _MULTI_CAMERA_PLANNING_PROMPT_V2.format(
        num_words=self._planning_num_words,
        num_planned_subtasks=self._num_planned_subtasks,
        num_subtask_retry=self._num_planned_subtasks_retry,
    )
    prompt.append(main_prompt)

    try:
      response = await self._gemini_client.generate_content(contents=prompt)
      # Directly return the raw response as the task plan.
      return types.FunctionResponse(
          response={
              "task_plan": response.text,
          },
          will_continue=False,
      )
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.exception(
          "Multi-camera task planning model failed to respond:\n%s", e
      )
      await asyncio.sleep(0.1)  # This is so that it returns a coroutine.
      return types.FunctionResponse(
          response={
              "task_plan": "null",
              "reason": (
                  "Multi-camera task planning V2 model failed to respond, try"
                  " again."
              ),
          },
          will_continue=False,
      )

  async def get_task_plan_v2(
      self,
      mode: str,
      overall_goal: str,
      call_id: str = "",
      previous_subtasks: str = "",
      executing_subtask: str = "",
      remaining_subtasks: str = "",
  ) -> types.FunctionResponse:
    """Exposed stateless function that gets the task plan."""
    if not self._v2_mode:
      raise ValueError("POSSIBLE BUG: _get_task_plan_v2 is called in V1 mode.")
    self._image_buffer = cast(ImageBufferWithHistory, self._image_buffer)
    self._call_id = call_id
    if mode == "mark_start":
      self._start_images_prompt = (
          self._image_buffer.get_latest_image_prompt_with_names()
      )
      self._start_marked = True
      self._current_overall_goal = overall_goal
      return types.FunctionResponse(
          response={"marked_start": True},
          will_continue=False,
      )
    elif mode == "get_task_plan":
      if not self._start_marked:
        return types.FunctionResponse(
            response={
                "start_not_marked, please call with mode='mark_start' first": (
                    True
                )
            },
            will_continue=False,
        )
      if overall_goal != self._current_overall_goal:
        self._start_marked = False
        return types.FunctionResponse(
            response={
                "overall_goal_changed, please call with mode='mark_start' first": (
                    True
                )
            },
            will_continue=False,
        )
      return await self._get_task_plan_v2(
          overall_goal=overall_goal,
          previous_subtasks=previous_subtasks,
          executing_subtask=executing_subtask,
          remaining_subtasks=remaining_subtasks,
      )
    else:
      logging.warning("Mode %s is not supported.", mode)
      return types.FunctionResponse(
          response={"unsupported_mode": True},
          will_continue=False,
      )


class VisionTaskPlanningTool(tool.Tool):
  """A stateful tool that does task planning.

  This tool can create a new task plan or update an existing one based on the
  vision inputs and the overall goal. It uses a fixed amount of history frames
  to create the planning images. Then, it uses a specialist model to update the
  task plan or create a new one.
  """

  def __init__(
      self,
      bus: event_bus.EventBus,
      camera_stream_name: str,
      camera_fps: float,
      api_key: str,
      num_history_frames: int = 31,
      planning_images_fps: float = 1.0,
      one_time_planning: bool = False,
  ):
    if one_time_planning:
      # One time planning.
      declaration = types.FunctionDeclaration(
          name="get_or_update_task_plan",
          description=(
              "Get a task plan for the overall goal. If the current task"
              " plan is empty. Otherwise, update the task plan if needed"
              " based on the new vision inputs."
          ),
          parameters=types.Schema(
              type=types.Type.OBJECT,
              properties={
                  "overall_goal": types.Schema(
                      type=types.Type.STRING,
                      description=(
                          "The overall goal of the robot to be broken"
                          " down into a task plan consisting of a"
                          " sequence of subtasks."
                      ),
                  ),
                  "current_task_plan": types.Schema(
                      type=types.Type.STRING,
                      description=(
                          "The current task plan as a string of comma"
                          " separated subtasks. If empty, the tool will"
                          " create a new task plan. Otherwise, the tool"
                          " will update the current task plan if needed"
                          " based on the new vision inputs."
                      ),
                  ),
              },
              required=["overall_goal"],
          ),
          behavior=types.Behavior.NON_BLOCKING,
      )
      python_function = self.get_or_update_task_plan
    else:
      # Continuous planning.
      # One time planning.
      declaration = types.FunctionDeclaration(
          name="stream_task_plan",
          description=(
              "Stream the task plan for the overall goal based on the vision"
              " inputs. A new task plan will be yielded when the current task"
              " plan is different from the previous one."
          ),
          parameters=types.Schema(
              type=types.Type.OBJECT,
              properties={
                  "overall_goal": types.Schema(
                      type=types.Type.STRING,
                      description=(
                          "The overall goal of the robot to be broken"
                          " down into a task plan consisting of a"
                          " sequence of subtasks."
                      ),
                  ),
              },
              required=["overall_goal"],
          ),
          behavior=types.Behavior.NON_BLOCKING,
      )
      python_function = self.stream_task_plan
    super().__init__(fn=python_function, declaration=declaration, bus=bus)
    self._call_id = None
    self.camera_stream_name = camera_stream_name
    self.num_history_frames = num_history_frames
    self.camera_fps = camera_fps
    self.planning_images_fps = planning_images_fps
    self._current_task_plan = "None"

    self.planning_downsample_factor = int(self.camera_fps / planning_images_fps)

    # Initialize the LLM with the Gemini API.
    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(temperature=0.0)
    # TODO: Change to byom and configurable.
    self.vlm = functools.partial(
        client.models.generate_content,
        model="gemini-2.5-flash",
        config=config,
    )
    # Stores the last num_history_frames frames.
    self.history_images = []

    bus.subscribe(
        [event_bus.EventType.MODEL_IMAGE_INPUT], self._handle_image_input
    )

  def _get_planning_images(self):
    # Downsamples the history images to the planning images.
    return self.history_images[:: self.planning_downsample_factor]

  def _handle_image_input(self, event: event_bus.Event):
    """Handle the image input event."""
    if (
        event.metadata[constants.STREAM_NAME_METADATA_KEY]
        != self.camera_stream_name
    ):
      return
    if len(self.history_images) >= self.num_history_frames:
      self.history_images.pop(0)
    self.history_images.append(event.data)

  def _handle_tool_call_cancellation_events(self, event: event_bus.Event):
    """Handle the subscribed events."""
    if self._call_id in event.data.ids:
      self._tool_call_cancelled = True
      logging.info("Tool call cancelled for id: %s.", self._call_id)

  async def get_or_update_task_plan(
      self, overall_goal: str, call_id: str, current_task_plan: str = ""
  ):
    """Exposed stateless function that updates the task plan."""
    # Record the id of the tool call.
    self._call_id = call_id
    # Reset the tool call cancellation flag for a new subtask.
    self._tool_call_cancelled = False
    planning_imgs = self._get_planning_images()
    # TODO: Make the prompt better.
    prompt = [
        (
            "given the past frames, overall goal and current plan, give me an"
            " updated comma separated plan (can be the same)."
        ),
        *planning_imgs,
        f"overall goal: {overall_goal}",
        f"current task plan: {current_task_plan}",
    ]
    prompt = image_processing.convert_bytes_to_image(prompt)
    response = await asyncio.to_thread(self.vlm, contents=prompt)
    subtasks = response.text
    did_task_plan_change = subtasks != current_task_plan
    return {
        "new_task_plan": subtasks,
        "did_task_plan_change": did_task_plan_change,
    }, False

  async def stream_task_plan(
      self, overall_goal: str, call_id: str
  ) -> AsyncIterator[types.FunctionResponse]:
    """Exposed stateless function that streams the task plan."""
    # Record the id of the tool call.
    self._call_id = call_id
    # Reset the tool call cancellation flag for a new subtask.
    self._tool_call_cancelled = False

    overall_goal_completed = False

    while not self._tool_call_cancelled and not overall_goal_completed:
      planning_imgs = self._get_planning_images()
      # TODO: Make the prompt better.
      prompt = [
          *planning_imgs,
          (
              "Given the past frames above and the overall goal, tell me what"
              " subtasks should the robot do next. Here is the guideline for"
              " the subtasks: "
          ),
          _SUBTASK_GUIDELINE,
          (
              "Limit to"
              f" {agentic_flags.AGENTIC_TASK_PLANNING_NUM_PLANNED_SUBTASKS.value} subtask"
              " please. You should also consider the previously task plan."
          ),
          f"previous task plan: {self._current_task_plan}",
          f"overall goal: {overall_goal}",
          (
              "If you think the new task plan is the same as or very similar to"
              " the previous task plan, just return exactly the previous task"
              " plan (do not rephrase). Also, if you think that the"
              " overall_goal is completed, return exactly"
              " 'overall_goal_completed'."
          ),
      ]
      prompt = image_processing.convert_bytes_to_image(prompt)
      response = await asyncio.to_thread(self.vlm, contents=prompt)
      try:
        subtasks = response.text

        logging.info("Streaming task plan: %s", subtasks)
        logging.info("Current task plan: %s", self._current_task_plan)
        if subtasks != self._current_task_plan:
          logging.info("DIFFERENT!!! New run_instruction: %s", subtasks)
        else:
          logging.info("SAME!!!")

        # Only yield a new task plan if it is different from the current task
        # plan or if the overall goal is completed.
        if subtasks == "overall_goal_completed":
          self._current_task_plan = "None"
          overall_goal_completed = True  # this will exit the while loop.
          yield types.FunctionResponse(
              response={"overall_goal_completed": True},
              will_continue=False,
          )
        elif subtasks != self._current_task_plan:
          self._current_task_plan = subtasks
          yield types.FunctionResponse(
              response={"new_task_plan": subtasks}, will_continue=True
          )
      except Exception as e:  # pylint: disable=broad-exception-caught
        logging.exception("Task planning model failed to respond:\n%s", e)
        logging.warning("Task planning model failed to respond. Retrying.")

    # Function call cancelled.
    yield types.FunctionResponse(
        response={"new_task_plan": "Cancelled."},
        will_continue=False,
    )
