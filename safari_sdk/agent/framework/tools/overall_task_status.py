"""A collection of Success detection tools."""

from google.genai import types

from safari_sdk.agent.framework.event_bus import event_bus
from safari_sdk.agent.framework.tools import tool


_STARTED_TOOL_DESCRIPTION = """
  The agent should call this tool to signal that the overall task is starting.
  This is important as there are other entities that want to know that the agent
  is starting to execute an overall task.
  For example, an experiment controller may send an overall task to the agent to
  execute and wants confirmation that the agent indeed started working on the
  task.
  """

_DONE_TOOL_DESCRIPTION = """
  The agent should call this tool to signal that the overall task is done.
  This is important as there are other entities that are waiting for the
  overall task to be done. For example, an experiment controller may be waiting
  for the agent to signal that the overall task is done before running the next
  experiment with a new overall (often long-horizon) task.
  """


class OverallTaskStartedTool(tool.Tool):
  """A tool for the agent to signal that the overall task is starting.

  Will change the framework status to RUNNING by publishing a FRAMEWORK_STATUS
  event.
  """

  def __init__(
      self,
      bus: event_bus.EventBus,
  ):
    # Define the tools exposed to the main agent here.
    # Run instruction until done tool
    declaration = types.FunctionDeclaration(
        name="overall_task_started",
        description=_STARTED_TOOL_DESCRIPTION,
        behavior=types.Behavior.NON_BLOCKING,
    )
    super().__init__(
        fn=self.overall_task_started,
        declaration=declaration,
        bus=bus,
    )
    self._call_id = None
    self._bus = bus

  async def overall_task_started(
      self, call_id: str
  ) -> types.FunctionResponse:
    """Sends a FRAMEWORK_STATUS event to signal that the overall task is started."""
    del call_id  # Unused.

    self._bus.publish(
        event=event_bus.Event(
            type=event_bus.EventType.FRAMEWORK_STATUS,
            source=event_bus.EventSource.ROBOT,
            data="RUNNING",
        )
    )
    return types.FunctionResponse(
        response={"overall_task_started signal sent": True},
        will_continue=False,
    )


class OverallTaskDoneTool(tool.Tool):
  """A tool for the agent to signal that the overall task is done.

  Will change the framework status to FINISHED by publishing a FRAMEWORK_STATUS
  event.
  """

  def __init__(
      self,
      bus: event_bus.EventBus,
  ):
    # Define the tools exposed to the main agent here.
    # Run instruction until done tool
    declaration = types.FunctionDeclaration(
        name="overall_task_done",
        description=_DONE_TOOL_DESCRIPTION,
        behavior=types.Behavior.NON_BLOCKING,
    )
    super().__init__(
        fn=self.overall_task_done,
        declaration=declaration,
        bus=bus,
    )
    self._call_id = None
    self._bus = bus

  async def overall_task_done(
      self, call_id: str
  ) -> types.FunctionResponse:
    """Sends a FRAMEWORK_STATUS event to signal that the overall task is done."""
    del call_id  # Unused.

    self._bus.publish(
        event=event_bus.Event(
            type=event_bus.EventType.FRAMEWORK_STATUS,
            source=event_bus.EventSource.ROBOT,
            data="FINISHED",
        )
    )
    return types.FunctionResponse(
        response={"overall_task_done signal sent": True},
        will_continue=False,
    )
