"""A collection of task planning tools."""

import asyncio
import enum
import re
import time
from typing import AsyncIterator, Awaitable, Callable, List, Sequence, cast

from absl import logging
from google import genai
from google.genai import types

from safari_sdk.agent.framework import flags as agentic_flags
from safari_sdk.agent.framework.event_bus import event_bus
from safari_sdk.agent.framework.tools import genai_client
from safari_sdk.agent.framework.tools import run_instruction_until_done
from safari_sdk.agent.framework.tools import success_detection
from safari_sdk.agent.framework.tools import task_planning
from safari_sdk.agent.framework.tools import tool


@enum.unique
class UserRequestStatus(enum.Enum):
  """Class for tracking the type of the external controller endpoint."""

  # Last request is cancelled or no request sent yet.
  NO_REQUEST = "NO_REQUEST"
  # Last request is completed.
  COMPLETED = "COMPLETED"
  # Request is in progress.
  IN_PROGRESS = "IN_PROGRESS"
  # Request failed due to a problem.
  FAILED = "FAILED"


_USER_REQUEST_EXECUTOR_TOOL_MODE = """
The mode this function operates.
# Mode 1: new_user_request
Use 'new_user_request' if you want to send a new user request to the executor.
If there is a preexisting request in progress, the function will automatically cancel the old request and replace it with a new request.
# Mode 2: check_status
Use 'check_status' to get the current status of the executor.
The status will include the running user request and the details of the execution (e.g., what step the robot is on, what the scene looks like, history of events during the execution, etc.)
NOTE: the report outputted from this function only includes information since the start of the most recent user request.
# Mode 3: cancel_request
Use 'cancel_request' to cancel the current user request that is in progress.
The executor will stop executing the request and return to its idle state.
Use this mode if you want to stop the robot.
"""

_USER_REQUEST_EXECUTOR_TOOL_USER_REQUEST = """
The user request for the robot to carry out. This argument is required when
 'mode' is 'new_user_request', and ignored otherwise. You do not need to copy
 the user's request verbatim. When needed, you are encouraged to expand the
 request to make sure it is unambiguous. For instance, the executor is reset
 every time a new request is sent, so expand any reference to past context
 appropriately, e.g. 'same as before', 'one more time', etc. user_request can be
 at most 5 sentences. This tool is stateless, so when you are making a new
 request after another, the tool cannot reference anything from the
 previous request(s). This means that each request should be self-explanatory.
"""

_TASK_PLANNING_ADDITIONAL_CONTEXT = """
When reviewing previous steps, YOU WILL carefully inspect if the current step is
 completed. If not, you will prefer to retry the same step unless there is a
 strong reason to divert to a different step.
"""

_TASK_PLAN_MERGING_PROMPT = """
Your are a robot task planner. Your job is to merge the following two task plans into one.

A. New task plan:
{new_task_plan}
This is the new task plan just generated about 15s ago. It contains the latest information about the scene and what the robot should do next.

B. Current status of the old task plan:
{current_task_plan}
This is the task plan that was generated before and the robot is currently using.
As the robot finishes each subtask, it simply executes the next subtask in the list, i.e., move the (CURRENT) decorator to the next subtask and mark the previous subtask as (executed).

Your job is to update the old task plan using the newly generated task plan.
By default, the new plan should override the old plan. That is, you should just output the new plan in its entirety. Only remove the (CURRENT) decorator.

However, the following complexities apply.
The new plan might become slightly outdated. During the last 15s while the new plan was generated, the robot may have finished a subtask in the old plan already, and moved to the next subtask in the old plan.
To tell if it happened, check if the step marked as (CURRENT) in the old plan is different from the first step (marked as (CURRENT))in the new plan.

If so, then your output should start with the current step in the old plan, plus the

If the first 1-2 steps in the new task plan is the same as the steps in the old plan marked as (CURRENT) or immediately precede it (executed), please remove them from the output.

Important note on formatting: output only the remaining subtasks as a list of ' -> ' separated subtasks after the text "REMAINING SUBTASKS:".

Example 1:
current task plan: "open the left drawer (executed) -> put the red block in the left drawer (executed) -> close the left drawer (CURRNET) -> open the right drawer -> put the green block in the right drawer -> close the right drawer -> DONE"
new task plan: "put the red block in the left drawer (CURRENT) -> close the left drawer -> open the right drawer -> put the yellow block in the right drawer -> close the right drawer -> DONE"
desired output (REAMINING SUBTASKS): "open the right drawer -> put the yellow block in the right drawer -> close the right drawer -> DONE"
In this example, the robot already executed "open the left drawer" and "put the red block in the left drawer" during the 15s, so the new task plan's CURRENT is outdated and needs to move right to the "close the left drawer".
In addition, the new task plan changed the object to be put in the right drawer from green to yellow, due to changes in the scene (e.g. the green block was taken by the user).
If this style of merging does not make sense, perhaps due to the current subtask in the new task plan cannot be found (please be linient about matching the current subtask, do not do exact matching) in the current task plan, you should simply output the new task plan's elements after (CURRENT) as the remaining subtasks.

Example 2:
current task plan: "put the red L-shaped block in the red tray (executed) -> put the blue cylindrical block in the blue tray (executed) -> put the green rectangular block in the green tray (CURRENT) -> DONE"
new task plan: "put the blue cylindrical block in the blue tray (CURRENT) -> put the green rectangular block in the green tray -> DONE"
desired output (REAMINING SUBTASKS): "DONE"
In this example, the robot already executed "put the blue cylindrical block in the blue tray" and is working on "put the green rectangular block in the green tray", so the new task plan's CURRENT is outdated and needs to move right.
As a result, the remaining subtasks is just "DONE".
"""


class ToolVersion(enum.Enum):
  V1 = "v1"
  V2 = "v2"


class UserRequestExecutorTool(tool.Tool):
  """A stateful tool that runs both long-horizon task planning & execution."""

  def __init__(
      self,
      api_key: str,
      config,
      stop_tool: tool.Tool,
      run_instruction_tool: tool.Tool,
      task_planning_tool: task_planning.MultiCameraVisionTaskPlanningTool,
      bus: event_bus.EventBus,
      camera_endpoint_names: Sequence[str],
      success_detector_tool: success_detection.VisionSuccessDetectionTool,
      num_planned_subtasks: int = 1,
      minimum_step_duration: float = 5.0,
      maximum_step_duration: float = 30.0,
      summarize_history_to_planner: bool = False,
      initial_task_planning_thinking_budget=100,
      initial_task_planning_num_words=100,
      use_gemini_task_merger: bool = False,
  ):
    # 1. Set up tool.
    declaration = types.FunctionDeclaration(
        name="user_request_executor",
        description=(
            "Executes a user request until it is completed or timed-out. "
        ),
        behavior=types.Behavior.NON_BLOCKING,  # Streaming tool
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "mode": types.Schema(
                    type=types.Type.STRING,
                    description=_USER_REQUEST_EXECUTOR_TOOL_MODE,
                ),
                "user_request": types.Schema(
                    type=types.Type.STRING,
                    description=_USER_REQUEST_EXECUTOR_TOOL_USER_REQUEST,
                ),
            },
            required=[
                "mode",
            ],
        ),
    )
    match agentic_flags.AGENTIC_USER_REQUEST_EXECUTOR_VERSION.value:
      case agentic_flags.UserRequestExecutorVersion.V1:
        python_function = self.user_request_executor
      case agentic_flags.UserRequestExecutorVersion.V2:
        python_function = self.user_request_executor_v2
      case _:
        raise ValueError(
            "Unsupported tool version: "
            f"{agentic_flags.AGENTIC_USER_REQUEST_EXECUTOR_VERSION.value}"
        )
    super().__init__(fn=python_function, declaration=declaration, bus=bus)  # pytype: disable=wrong-arg-types
    self._lock = asyncio.Lock()

    # 2. Set class states
    self._call_id = ""
    self._preexisting_user_request = None
    self._request_status = UserRequestStatus.NO_REQUEST
    self._user_request_timer = 0.0
    # narrated_history maintains the list of string items. The buffer is flushed
    # when a new user request is sent or when check_status is called.
    self._history_buffer = []
    self._summarize_history_to_planner = summarize_history_to_planner
    self._initial_task_planning_thinking_budget = (
        initial_task_planning_thinking_budget
    )
    self._initial_task_planning_num_words = initial_task_planning_num_words
    self._preexisting_user_request = None
    self._previous_subtasks = []  # updated when switching subtasks.
    self._current_subtask = ""  # updated when switching subtasks.
    self._remaining_subtasks = []  # updated when task plan is updated.
    self._lock = asyncio.Lock()
    self._background_planning_task = None
    self._use_gemini_task_merger = use_gemini_task_merger

    if self._use_gemini_task_merger:
      self._merger = self._merge_new_and_current_task_plan_gemini
    else:
      self._merger = self._merge_new_and_current_task_plan

    # 3. Set up sub-tools
    self._api_key = api_key
    self._config = config
    self._task_planning_tool = task_planning_tool
    self._task_planning_fn = cast(
        Callable[..., Awaitable[types.FunctionResponse]],
        self._task_planning_tool.fn,
    )  # To avoid pytype error of expecting to output AsyncIterator.
    # Record the default config for the task planning tool. This is used to
    # switch between fast and normal mode. This is the normal mode config.
    self._task_planning_default_config = (
        self._task_planning_tool.get_gemini_client_config()
    )
    self._run_instruction_until_done_tool = (
        run_instruction_until_done.RunInstructionUntilDoneTool(
            self._bus,
            config=self._config,
            success_detector_tool=success_detector_tool,
            run_instruction_tool=run_instruction_tool,
            stop_tool=stop_tool,
            minimum_run_duration=minimum_step_duration,
            maximum_run_duration=maximum_step_duration,
            make_success_known_to_agent=True,
        )
    )
    self._run_instruction_until_done_fn = cast(
        Callable[..., Awaitable[types.FunctionResponse]],
        self._run_instruction_until_done_tool.fn,
    )  # To avoid pytype error of expecting to output AsyncIterator.
    self._stop_tool = stop_tool

    # 4. Set up fast task plan merger Gemini client.
    if self._use_gemini_task_merger:
      client = genai.Client(api_key=api_key)
      # Controls the thinking budget.
      thinking_config = types.ThinkingConfig(thinking_budget=0)
      config = types.GenerateContentConfig(
          temperature=0.0, thinking_config=thinking_config
      )
      self._fast_merge_gemini_client = genai_client.GeminiClientWrapper(
          bus=bus,
          client=client,
          model_name="models/gemini-2.5-flash-lite",
          config=config,
          print_raw_response=False,
      )

  def _handle_tool_call_cancellation_events(self, event: event_bus.Event):
    """Handle the subscribed events."""
    if self._call_id in event.data.ids:
      self._tool_call_cancelled = True
      logging.info("Tool call cancelled for id: %s.", self._call_id)

  def _extract_outermost_quoted(self, in_string):
    """Extract the substring that's inside the outermost quotation marks."""
    double_quote_match = re.search(r'"(.*)"', in_string, re.DOTALL)
    if double_quote_match:
      # The group (1) contains the content *between* the quotes
      return double_quote_match.group(1)

    # Check for single quotes
    single_quote_match = re.search(r"'(.*)'", in_string, re.DOTALL)
    if single_quote_match:
      return single_quote_match.group(1)

    return in_string  # Return the same string if no quotes are found

  def _convert_response_to_task_plan(
      self, response: str
  ) -> tuple[List[str], str]:
    """Convert the response to remaining subtasks and current subtask."""
    task_plan = response.response["task_plan"]  # pytype: disable=attribute-error

    # Remove possible quotation marks
    current_task_plan = self._extract_outermost_quoted(
        task_plan.split("TASK PLAN:")[1]
    )

    logging.info("******************************************************")
    logging.info(
        "current_task_plan (in _convert_response_to_task_plan):\n%s",
        current_task_plan,
    )
    logging.info("******************************************************")

    current_task_plan = current_task_plan.split("->")
    current_task_plan = [elem.strip() for elem in current_task_plan]
    current_subtask = current_task_plan[0].replace("(CURRENT)", "")
    if len(current_task_plan) > 1:
      remaining_subtasks = current_task_plan[1:]
    else:
      remaining_subtasks = ["DONE"]
    return remaining_subtasks, current_subtask

  async def _merge_new_and_current_task_plan_gemini(
      self,
      new_current_subtask: str,
      new_remaining_subtasks: List[str],
      previous_subtasks: List[str],
      current_subtask: str,
      current_remaining_subtasks: List[str],
  ) -> List[str]:
    """Merge the new and current task plan."""

    # Make the current and previous task plan string.
    current_task_plan = []
    for st in previous_subtasks:
      current_task_plan.append(f"{st} (executed)")
    current_task_plan.append(f"{current_subtask} (CURRENT)")
    current_task_plan.extend(current_remaining_subtasks)
    current_task_plan = " -> ".join(current_task_plan)

    # Make the new task plan string.
    new_task_plan = [f"{new_current_subtask} (CURRENT)"]
    new_task_plan.extend(new_remaining_subtasks)
    new_task_plan = " -> ".join(new_task_plan)

    prompt = _TASK_PLAN_MERGING_PROMPT.format(
        current_task_plan=current_task_plan, new_task_plan=new_task_plan
    )

    logging.info("****************** IN MERGE *************************")
    logging.info("current_task_plan:\n%s", current_task_plan)
    logging.info("new_task_plan:\n%s", new_task_plan)

    response = await self._fast_merge_gemini_client.generate_content(prompt)
    merged_remaining_subtasks = response.text.split("REMAINING SUBTASKS:")[1]  # pytype: disable=attribute-error

    logging.info("***********")
    logging.info(
        "merged_remaining_subtasks response:\n%s", merged_remaining_subtasks
    )
    logging.info("CURRENT SUBTASK: %s", self._current_subtask)
    logging.info("***********")

    merged_remaining_subtasks = merged_remaining_subtasks.split("->")
    merged_remaining_subtasks = [
        elem.strip() for elem in merged_remaining_subtasks
    ]

    if "DONE" not in merged_remaining_subtasks[-1].strip():
      merged_remaining_subtasks.append("DONE")

    logging.info("*****************END MERGE**************************")

    return merged_remaining_subtasks

  async def _merge_new_and_current_task_plan(
      self,
      new_current_subtask: str,
      new_remaining_subtasks: List[str],
      previous_subtasks: List[str],
      current_subtask: str,
      current_remaining_subtasks: List[str],
  ) -> List[str]:
    """Merge the new and current task plan."""
    # Sanitise strings.
    new_current_subtask = new_current_subtask.strip()
    new_remaining_subtasks = [elem.strip() for elem in new_remaining_subtasks]
    previous_subtasks = [elem.strip() for elem in previous_subtasks]
    current_subtask = current_subtask.strip()
    current_remaining_subtasks = [
        elem.strip() for elem in current_remaining_subtasks
    ]

    # *************************** LOGGING ***************************
    # Make the new task plan string.
    old_task_plan = [f"{current_subtask} (CURRENT)"]
    new_task_plan = [f"{new_current_subtask} (CURRENT)"]
    old_task_plan.extend(current_remaining_subtasks)
    new_task_plan.extend(new_remaining_subtasks)
    old_task_plan = " -> ".join(old_task_plan)
    new_task_plan = " -> ".join(new_task_plan)
    logging.info("********************** PRE-MERGE ***************************")
    logging.info("current_task_plan:\n%s", old_task_plan)
    logging.info("new_task_plan:\n%s", new_task_plan)

    if current_subtask == new_current_subtask:
      # New plan is not stale.
      # -> Override the old plan with the new plan.
      merged_remaining_subtasks = new_remaining_subtasks
    elif previous_subtasks[-1] == new_current_subtask:
      # New plan is stale (stepped forward by 1)
      if current_subtask == new_remaining_subtasks[0]:
        # Current subtask is the first remaining step in the new plan.
        # -> Just step the new plan forward.
        merged_remaining_subtasks = (
            new_remaining_subtasks[1:]
            if len(new_remaining_subtasks) > 1
            else ["DONE"]
        )
      else:
        # Current subtask is not the first remaining step in the new plan.
        # -> Do not step the new plan forward. If this results in redundancy,
        # It will be removed in the next planning iteration.
        merged_remaining_subtasks = new_remaining_subtasks
    else:
      # Something else happened.
      # -> Override the old plan with the new plan. If this results in
      # redundancy, it will be removed in the next planning iteration.
      logging.info("PLAN MERGING: THIS BETTER NOT HAPPENED")
      merged_remaining_subtasks = [new_current_subtask] + new_remaining_subtasks
    logging.info("********************** POST-MERGE **************************")
    logging.info("merged_remaining_subtasks:\n%s", merged_remaining_subtasks)
    logging.info("CURRENT SUBTASK: %s", self._current_subtask)
    logging.info("********************************************************")
    return merged_remaining_subtasks

  async def _launch_background_task_planning(self):
    """Callback for the task planning task."""
    logging.info("Background task planning started")

    try:
      while self._request_status == UserRequestStatus.IN_PROGRESS:
        response = await self._task_planning_fn(  # pytype: disable=wrong-arg-types
            mode="get_task_plan",
            overall_goal=self._preexisting_user_request,
            previous_subtasks=" -> ".join(self._previous_subtasks),
            executing_subtask=self._current_subtask,
            remaining_subtasks=" -> ".join(self._remaining_subtasks),
            call_id=self._call_id,
        )
        try:
          new_remaining_subtasks, new_current_subtask = (
              self._convert_response_to_task_plan(response)
          )
        except Exception as e:  # pylint: disable=broad-except
          logging.exception(
              "Failure in _convert_response_to_task_plan() (WILL RETRY): %s", e
          )
          continue
        # Merge with current task plan with fast Gemini client.
        # Use lock to avoid race conditions when merging.
        async with self._lock:
          try:
            self._remaining_subtasks = await self._merger(
                new_current_subtask=new_current_subtask,
                new_remaining_subtasks=new_remaining_subtasks,
                previous_subtasks=self._previous_subtasks,
                current_subtask=self._current_subtask,
                current_remaining_subtasks=self._remaining_subtasks,
            )
          except Exception as e:  # pylint: disable=broad-except
            logging.exception(
                "Failure in _merge_new_and_current_task_plan(): %s", e
            )
            continue
    except asyncio.CancelledError:
      logging.info("Task planning cancelled. Stopping task planning.")
      return
    except Exception as e:  # pylint: disable=broad-except
      logging.exception("Task planning failed: %s", e)

  async def _run_user_request_v2(
      self, user_request: str, call_id: str
  ) -> types.FunctionResponse:
    """Runs the user request."""
    # Sanity: status must be NO_REQUEST or COMPLETED or FAILED.
    assert self._request_status != UserRequestStatus.IN_PROGRESS

    # State transition (which can be changed internally or externally)
    self._request_status = UserRequestStatus.IN_PROGRESS
    self._preexisting_user_request = user_request
    self._previous_subtasks = []  # updated when switching subtasks.
    self._current_subtask = ""  # updated when switching subtasks.
    self._remaining_subtasks = []  # updated when task plan is updated.
    self._user_request_timer = time.time()

    # Mark the start for the tasking planning tool.
    logging.info("Task planner marked start.")
    _ = await self._task_planning_fn(
        mode="mark_start",
        overall_goal=user_request,
        previous_subtasks="",
        executing_subtask="",
        remaining_subtasks="",
        call_id=call_id,
    )

    # default final response should be external cancellation.
    final_response = types.FunctionResponse(
        response={
            "current_request_status": self._request_status.value,
            "current_user_request": self._preexisting_user_request,
            "result": "CANCELLED.",
        },
        will_continue=False,
    )

    # First task planning needs to be fast as everything has to wait for it.
    # Set the task planning config to the initial (fast) config.
    logging.info("First task planning")
    thinking_config = types.ThinkingConfig(
        thinking_budget=self._initial_task_planning_thinking_budget
    )
    config = types.GenerateContentConfig(
        temperature=0.0, thinking_config=thinking_config
    )
    self._task_planning_tool.set_gemini_client_config(config)
    # TODO: Add error handling.
    response = await self._task_planning_fn(
        mode="get_task_plan",
        overall_goal=user_request,
        previous_subtasks="",
        executing_subtask="",
        remaining_subtasks="",
        call_id=call_id,
    )

    # Goal not achievable.
    if "overall goal not achievable" in response.response["task_plan"]:
      final_response = types.FunctionResponse(
          response={
              "current_request_status": UserRequestStatus.FAILED.value,
              "current_user_request": self._preexisting_user_request,
              "reason": response.response["task_plan"],
          },
          will_continue=False,
      )
      self._request_status = UserRequestStatus.FAILED
      return final_response

    self._remaining_subtasks, self._current_subtask = (
        self._convert_response_to_task_plan(response)
    )
    # Restore the default config for the task planning tool in follow-up calls.
    # So the task planning tool will be accurate.
    self._task_planning_tool.set_gemini_client_config(
        self._task_planning_default_config
    )

    # Create recurring task planning task. The callback will create another
    # task planning task so this is recurring.
    self._background_planning_task = asyncio.create_task(
        self._launch_background_task_planning()
    )

    # Main loop.
    while self._request_status == UserRequestStatus.IN_PROGRESS:
      # Execute the current subtask.
      logging.info("============= EXECUTING SUBTASK ======================")
      logging.info("*** PREVIOUS SUBTASKS ***: %s", self._previous_subtasks)
      logging.info("*** CURRENT SUBTASK ***: %s", self._current_subtask)
      logging.info("*** REMAINING SUBTASKS ***: %s", self._remaining_subtasks)
      logging.info("============= END EXECUTING OF SUBTASK =================")
      # logging.info("previous subtasks: %s", self._previous_subtasks)
      _ = await self._run_instruction_until_done_fn(
          self._current_subtask, call_id
      )

      # Progress to the next subtask.
      logging.info("Finished executing subtask: %s", self._current_subtask)
      self._previous_subtasks.append(f"{self._current_subtask}")
      async with self._lock:
        self._current_subtask = self._remaining_subtasks.pop(0)

      # Check if the overall goal is achieved (current subtask is DONE).
      if self._current_subtask.strip() == "DONE":
        logging.info("Overall goal is achieved.")

        self._request_status = UserRequestStatus.COMPLETED
        await self._stop_tool.fn(call_id=call_id)
        final_response = types.FunctionResponse(
            response={
                "current_request_status": self._request_status.value,
                "current_user_request": self._preexisting_user_request,
                "result": "SUCCESSFULLY COMPLETED",
            },
            will_continue=False,
        )
        break

    # Cancel the task planning task.
    self._background_planning_task.cancel()

    return final_response

  async def user_request_executor_v2(
      self,
      mode: str,
      user_request: str = "",
      call_id: str = "",
  ) -> AsyncIterator[types.FunctionResponse]:
    self._call_id = call_id
    self._tool_call_cancelled = False
    match mode:
      case "new_user_request":
        if user_request:
          result = (
              f"New `user_request` starting: {user_request}. "
              "WARNING: it DOES NOT mean that you finished the request. "
              "For that, you will receive another response that signals "
              "completion or failure. You may check the progress by calling "
              "this function with 'check_status' mode. "
              "Report to the user that you are getting started on the request."
          )
          if self._request_status == UserRequestStatus.IN_PROGRESS:
            # Cancel the preexisting in-progress request first.
            cancelled_request = await self._cancel_user_request()
            result += (
                " Also cancelled the preexisting in-progress request: "
                f"{cancelled_request}."
            )
          # Start a new user_request and yield the response when completed or
          # failed.
          yield types.FunctionResponse(
              response={
                  "result": result,
              },
              will_continue=True,
          )
          # Await the long-running user request and yield its final response.
          # This is non-blocking for the event loop, allowing other tasks
          # (including other calls to this function) to run.
          final_response = await self._run_user_request_v2(
              user_request, call_id
          )
          yield final_response
        else:
          # No user request provided. Throw an error.
          yield types.FunctionResponse(
              response={
                  "debug_note": (
                      "No `user_request` provided. This call is ignored."
                  ),
              },
              will_continue=False,
          )
      case "check_status":
        response = {}
        if user_request:
          response["debug_note"] = (
              "You passed 'check_status' for `mode` AND a new `user_request`:"
              f" '{user_request}'. This arg is ignored."
          )
        report = await self._pop_history_v2()
        response["current_request_status"] = self._request_status.value
        response["current_user_request"] = self._preexisting_user_request
        response["report"] = report
        response["will_provide update later, please wait"] = True
        yield types.FunctionResponse(
            response=response,
            will_continue=False,
        )
      case "cancel_request":
        if self._request_status == UserRequestStatus.IN_PROGRESS:
          response = {}
          if user_request:
            response["debug_note"] = (
                "You passed 'cancel_request' for `mode` AND a new "
                f"`user_request`: '{user_request}'. This arg is ignored."
            )
          cancelled_request = await self._cancel_user_request()
          response["result"] = f"Successfully cancelled: {cancelled_request}."
          yield types.FunctionResponse(
              response=response,
              will_continue=False,
          )
        else:
          yield types.FunctionResponse(
              response={
                  "debug_note": (
                      "No in-progress request to cancel. This call is ignored."
                  ),
              },
              will_continue=False,
          )
      case _:
        yield types.FunctionResponse(
            response={
                "debug_note": (
                    f"Unsupported mode: {mode}. This call is ignored."
                ),
            },
            will_continue=False,
        )

  async def user_request_executor(
      self,
      mode: str,
      user_request: str = "",
      call_id: str = "",
  ) -> AsyncIterator[types.FunctionResponse]:
    self._call_id = call_id
    self._tool_call_cancelled = False
    match mode:
      case "new_user_request":
        if user_request:
          result = (
              f"New `user_request` starting: {user_request}. "
              "WARNING: it DOES NOT mean that you finished the request. "
              "For that, you will receive another response that signals "
              "completion or failure. You may check the progress by calling "
              "this function with 'check_status' mode. "
              "Report to the user that you are getting started on the request."
          )
          if self._request_status == UserRequestStatus.IN_PROGRESS:
            # Cancel the preexisting in-progress request first.
            cancelled_request = await self._cancel_user_request()
            result += (
                " Also cancelled the preexisting in-progress request: "
                f"{cancelled_request}."
            )
          # Start a new user_request and yield the response when completed or
          # failed.
          yield types.FunctionResponse(
              response={
                  "result": result,
              },
              will_continue=True,
          )
          # Await the long-running user request and yield its final response.
          # This is non-blocking for the event loop, allowing other tasks
          # (including other calls to this function) to run.
          final_response = await self._run_user_request(user_request, call_id)
          yield final_response
        else:
          # No user request provided. Throw an error.
          yield types.FunctionResponse(
              response={
                  "debug_note": (
                      "No `user_request` provided. This call is ignored."
                  ),
              },
              will_continue=False,
          )
      case "check_status":
        response = {}
        if user_request:
          response["debug_note"] = (
              "You passed 'check_status' for `mode` AND a new `user_request`:"
              f" '{user_request}'. This arg is ignored."
          )
        report = await self._pop_history()
        response["current_request_status"] = self._request_status.value
        response["current_user_request"] = self._preexisting_user_request
        response["report"] = report
        yield types.FunctionResponse(
            response=response,
            will_continue=False,
        )
      case "cancel_request":
        if self._request_status == UserRequestStatus.IN_PROGRESS:
          response = {}
          if user_request:
            response["debug_note"] = (
                "You passed 'cancel_request' for `mode` AND a new "
                f"`user_request`: '{user_request}'. This arg is ignored."
            )
          cancelled_request = await self._cancel_user_request()
          response["result"] = f"Successfully cancelled: {cancelled_request}."
          yield types.FunctionResponse(
              response=response,
              will_continue=False,
          )
        else:
          yield types.FunctionResponse(
              response={
                  "debug_note": (
                      "No in-progress request to cancel. This call is ignored."
                  ),
              },
              will_continue=False,
          )
      case _:
        yield types.FunctionResponse(
            response={
                "debug_note": (
                    f"Unsupported mode: {mode}. This call is ignored."
                ),
            },
            will_continue=False,
        )

  async def _cancel_user_request(self) -> str | None:
    # This function is aassumed to only be called when the request is in
    # progress.
    async with self._lock:
      assert self._request_status == UserRequestStatus.IN_PROGRESS
      request_to_cancel = self._preexisting_user_request
      # Stop the robot.
      await self._stop_tool.fn(call_id=self._call_id)
      # Cancel the task planning task.
      if self._background_planning_task is not None:
        self._background_planning_task.cancel()
      logging.info("User request cancelled. Stopping robot and task planning.")
      # State Transition
      self._request_status = UserRequestStatus.NO_REQUEST
      self._preexisting_user_request = None
      self._history_buffer.append(
          f"Request cancelled at {time.time() - self._user_request_timer} "
          "seconds after start."
      )
    return request_to_cancel

  async def _pop_history(self) -> str:
    async with self._lock:
      history_to_return = "\n".join(self._history_buffer)
      self._history_buffer = []
    return history_to_return

  async def _pop_history_v2(self) -> str:
    # This version does not use self._history_buffer
    history_str = "Previously executed subtasks:"
    for i, subtask in enumerate(self._previous_subtasks):
      history_str += f"\n{i}. {subtask}"
    history_str += f"\nCurrent subtask: {self._current_subtask}"
    history_str += "\nRemaining subtasks:"
    for i, subtask in enumerate(self._remaining_subtasks):
      history_str += f"\n{i}. {subtask}"
    return history_str

  async def _get_planner_context_from_history(self) -> str:
    """Returns a string of the history to be used as additional context."""
    instruction = """
    The following is a narrated history of events that happened since the start
    of the current user request. You may reference this to assist in planning
    the next step. It is possible that there might be conflicting information
    between statements recorded at different times. In many cases, this happens
    when the robot's vision system misunderstands the scene. You should use your
    best judgement and logic to decide which history is more accurate. You
    should record such resolutions in your output. (history may also include
    such past judgements, which you can follow or override based on your
    current reasoning and observations.)
    """
    history_to_return = "\n".join([instruction] + self._history_buffer)
    self._history_buffer = []
    return history_to_return

  async def _run_user_request(
      self, user_request: str, call_id: str
  ) -> types.FunctionResponse:
    # Sanity: status must be NO_REQUEST or COMPLETED or FAILED.
    assert self._request_status != UserRequestStatus.IN_PROGRESS

    # State transition (which can be changed internally or externally)
    self._request_status = UserRequestStatus.IN_PROGRESS
    self._preexisting_user_request = user_request
    self._history_buffer = []
    self._user_request_timer = time.time()

    # First time, set the task planning config to the initial (fast) config.
    thinking_config = types.ThinkingConfig(
        thinking_budget=self._initial_task_planning_thinking_budget
    )
    config = types.GenerateContentConfig(
        temperature=0.0, thinking_config=thinking_config
    )
    self._task_planning_tool.set_gemini_client_config(config)

    # Mark the start of the user request.
    _ = await self._task_planning_fn(
        mode="mark_start",
        overall_goal=user_request,
        previous_and_current_task_plan="",
        additional_context=_TASK_PLANNING_ADDITIONAL_CONTEXT,
        call_id=call_id,
    )
    previous_and_current_task_plan = []

    # Restore the task planning config to the default config (slow).
    self._task_planning_tool.set_gemini_client_config(
        self._task_planning_default_config
    )

    # default final response should be external cancellation.
    final_response = types.FunctionResponse(
        response={
            "current_request_status": self._request_status.value,
            "current_user_request": self._preexisting_user_request,
            "result": "CANCELLED.",
        },
        will_continue=False,
    )
    while self._request_status == UserRequestStatus.IN_PROGRESS:
      # First time, set the task planning config to the initial (fast) config.
      if not previous_and_current_task_plan:
        thinking_config = types.ThinkingConfig(
            thinking_budget=self._initial_task_planning_thinking_budget
        )
        config = types.GenerateContentConfig(
            temperature=0.0, thinking_config=thinking_config
        )
        self._task_planning_tool.set_gemini_client_config(config)
      else:
        # Restore the task planning config to the default config (slow).
        self._task_planning_tool.set_gemini_client_config(
            self._task_planning_default_config
        )

      # Constantly reassess the scene and obtain a plan.
      response = await self._task_planning_fn(
          mode="get_task_plan",
          overall_goal=user_request,
          previous_and_current_task_plan=(
              " -> ".join(previous_and_current_task_plan)
              if previous_and_current_task_plan
              else ""
          ),
          additional_context=(
              self._get_planner_context_from_history()
              if self._summarize_history_to_planner
              else ""
          ),
          call_id=call_id,
      )
      split_plan = response.response["task_plan"].split("TASK PLAN:")
      reasoning = split_plan[0]
      actual_plan = split_plan[1]
      time_since_start = int((time.time() - self._user_request_timer) * 10) / 10
      self._history_buffer.append(
          f"<<Narration at {time.time() - time_since_start} "
          f"seconds since user request start>>:\n\n{reasoning}"
      )

      # Detect overall goal completion.
      if (
          "goal achieved" in actual_plan
          or "goal completed" in actual_plan
          or "goal finished" in actual_plan
          or "goal fulfilled" in actual_plan
      ):
        # Announce completion
        self._request_status = UserRequestStatus.COMPLETED
        await self._stop_tool.fn(call_id=call_id)
        final_response = types.FunctionResponse(
            response={
                "current_request_status": self._request_status.value,
                "current_user_request": self._preexisting_user_request,
                "result": "SUCCESSFULLY COMPLETED (but double check)",
            },
            will_continue=False,
        )
        break

      # NOTE: catch failure
      if time_since_start > 200:
        self._request_status = UserRequestStatus.FAILED
        final_response = types.FunctionResponse(
            response={
                "current_request_status": self._request_status.value,
                "current_user_request": self._preexisting_user_request,
                "result": "WENT ON FOR TOO LONG, ABORTING.",
            },
            will_continue=False,
        )
        break

      # Execute the subtask from the task plan.
      steps = actual_plan.strip().split("->")
      current_step = steps[0].strip()
      previous_and_current_task_plan.append(current_step + " (success unknown)")
      response = await self._run_instruction_until_done_fn(
          current_step, call_id
      )
      if not response.response["subtask_success"]:
        await self._stop_tool.fn(call_id=call_id)
        logging.info(
            "Subtask failed or timed out. Stopping robot and replanning"
        )

    return final_response
