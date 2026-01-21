"""A collection of Success detection tools."""

import asyncio
from typing import Optional

from absl import logging
from google.genai import types

from safari_sdk.agent.framework.event_bus import event_bus
from safari_sdk.agent.framework.tools import tool

_TASK_SUCCESS_KEY = "task_success"
_TIMEOUT_KEY = "timeout"

_DEFAULT_FUNCTION_DECLARATION = types.FunctionDeclaration(
    name="run_instruction_for_duration",
    description="""
  Run a natural language instruction on the robot. This will cause the agent on the robot to execute low-level actions (in a run high frequency run loop) needed to complete the language instruction.
  This function will return after the specified duration.
  Generally the duration be as short as possible, since the robot might complete the instruction and then undo it.
  Around 8 seconds is a good number, but it is your call.
  The language instruction needs to be specific, unambiguous, atomic (involving single action with one object) instructions, e.g.:
    * "bring the cup to the table"
    * "put the book on the table"
    * "put the red dice in the green tray"
    * "push the bowl to the left"
    """,
    behavior=types.Behavior.NON_BLOCKING,
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "instruction": types.Schema(
                type=types.Type.STRING,
                description="""
  A specific, unambiguous, atomic (involving single action) natural language instruction for the robot (e.g. 'bring the cup to the table', 'put the dice in the green tray').
  This instruction should be specific enough to describe the exact object on the table, e.g. 'the white cup on the center of the table' or 'the green triangluar block on the most left'.
  It probably should not contain the word 'and'.
  """,
            ),
            "duration": types.Schema(
                type=types.Type.INTEGER,
                description="The duration in seconds to run the instruction.",
            ),
        },
        required=["instruction", "duration"],
    ),
)


class RunInstructionForDurationTool(tool.Tool):
  """An augmented version of robot's run_instruction tool.

  Will return a task success signal to the main agent after the robot finishes
  the instruction.
  """

  def __init__(
      self,
      bus: event_bus.EventBus,
      embodiment_run_instruction_tool: tool.Tool,
      embodiment_stop_tool: Optional[tool.Tool] = None,
      stop_on_success: bool = True,
      default_duration: float = 5.0,
      function_declaration: types.FunctionDeclaration = _DEFAULT_FUNCTION_DECLARATION,
  ):
    """Initializes the tool.

    Args:
      bus: The event bus to subscribe to.
      embodiment_run_instruction_tool: The tool to use for running instructions.
      embodiment_stop_tool: If provided, this tool will be called when
        stop_on_success is True and the instruction completes. If None and
        stop_on_success is True, raises ValueError.
      stop_on_success: If True, the robot will be stopped after the instruction
        completes successfully.
      default_duration: The default duration in seconds to run instructions.
      function_declaration: The function declaration to use for the tool.
    """
    declaration = function_declaration
    super().__init__(
        fn=self.run_instruction_for_duration,
        declaration=declaration,
        bus=bus,
    )
    self._call_id = None
    self._latest_images = {}
    self._start_imgs = []
    self._default_duration = default_duration
    self._stop_on_success = stop_on_success

    self._embodiment_run_instruction_tool = embodiment_run_instruction_tool

    self._embodiment_stop_tool = embodiment_stop_tool
    if stop_on_success and (embodiment_stop_tool is None):
      raise ValueError(
          "No stop tool from the embodiment found. You must have a stop tool"
          " to use the run_instruction_until_done tool when stop_on_success"
          " is set to True."
      )

  def _handle_tool_call_cancellation_events(self, event: event_bus.Event):
    """Handle the subscribed events."""
    if self._call_id in event.data.ids:
      self._tool_call_cancelled = True
      logging.info(
          "Tool call cancelled for run_instruction_until_done. id: %s.",
          self._call_id,
      )

  async def run_instruction_for_duration(
      self, instruction: str, call_id: str, duration: int | None = None
  ) -> types.FunctionResponse:
    """Runs the robot with an instruction, stops the robot after given duration."""
    if duration is None:
      duration = self._default_duration
    self._call_id = call_id

    # Send run instruction to the robot using the embodiment's tool.
    await self._embodiment_run_instruction_tool.fn(
        instruction, call_id
    )  # pytype: disable=bad-return-type

    task_success_dict, _ = await _run_instruction_timer(float(duration))

    # Stop the robot if needed.
    if self._stop_on_success and task_success_dict["task_success"]:
      assert self._embodiment_stop_tool is not None, (
          "No stop tool from the embodiment found. You must have a stop tool"
          " to use the run_instruction_until_done tool when stop_on_success"
          " is set to True."
      )
      await self._embodiment_stop_tool.fn(call_id)  # pytype: disable=bad-return-type
    return types.FunctionResponse(
        response={
            # TODO: This does not adhere to GenAI standards which
            # expect an "output" field.
            "subtask": instruction,
            "subtask_success": task_success_dict["task_success"],
            "should_change_subtask": task_success_dict["timeout"],
            "is_robot_stopped": self._stop_on_success,
        },
        will_continue=False,
    )


async def _run_instruction_timer(time_limit: float):
  await asyncio.sleep(time_limit)
  return {_TASK_SUCCESS_KEY: True, _TIMEOUT_KEY: True}, False
