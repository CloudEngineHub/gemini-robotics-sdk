"""Robot functions for an OmegaStar robot.

Should be usable for OmegaStar as well, to allow running both against the
same API instance.
"""

from typing import Sequence

from google.genai import types

from safari_sdk.agent.framework.embodiments import embodiment
from safari_sdk.agent.framework.embodiments import omega_star_fast_api_endpoints
from safari_sdk.agent.framework.event_bus import event_bus
from safari_sdk.agent.framework.tools import fast_api_tools
from safari_sdk.agent.framework.tools import tool


_Endpoints = omega_star_fast_api_endpoints.FastApiEndpoints

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 8888


# Constants for the endpoint paths.
CAGE_CENTER_ENDPOINT = _Endpoints.CAGE_CENTER_CAMERA_STREAM.value.path
CAGE_LEFT_ENDPOINT = _Endpoints.CAGE_LEFT_CAMERA_STREAM.value.path
CAGE_RIGHT_ENDPOINT = _Endpoints.CAGE_RIGHT_CAMERA_STREAM.value.path
WRIST_LEFT_ENDPOINT = _Endpoints.WRIST_LEFT_CAMERA_STREAM.value.path
WRIST_RIGHT_ENDPOINT = _Endpoints.WRIST_RIGHT_CAMERA_STREAM.value.path

# Names for the tools this embodiment provides. Available so that agents can
# search for them in the embodiment tools collection.
RUN_INSTRUCTION_TOOL_NAME = "robot_run_instruction"
RESET_TOOL_NAME = "robot_reset"
STOP_TOOL_NAME = "robot_stop"
OPEN_GRIPPER_TOOL_NAME = "robot_open_gripper"
SLEEP_TOOL_NAME = "robot_sleep"


_RUN_INSTRUCTION_TOOL_DESCRIPTION = """
  Run a natural language instruction on the robot. This will cause the agent on the robot to execute low-level actions (in a run high frequency run loop) needed to complete the language instruction.
  This function returns right away (before the language instruction is completed) and the agent on the robot will not stop the robot even if the language instruction is completed.
  The language instruction needs to be specific, unambiguous, atomic (involving single action with one object) instructions, e.g.:
    * "bring the cup to the table"
    * "put the book on the table"
    * "put the red dice in the green tray"
    * "push the bowl to the left"
"""


class OmegaStarEmbodiment(embodiment.Embodiment):
  """Robot handler connecting FastAPI robots to the event bus."""

  def __init__(
      self,
      bus: event_bus.EventBus,
      server: str = DEFAULT_HOST,
      port: int = DEFAULT_PORT,
  ):
    # Must come before super init!
    self._robot_server = f"http://{server}:{port}"
    super().__init__(bus=bus)

  def _create_video_stream(
      self, endpoint: str
  ) -> embodiment.EventStreamDefinition:
    return embodiment.EventStreamDefinition(
        # Name must match the `stream_name` of the stream instance. `name` will
        # be passed to the live API handler to identify individual streams,
        # while `stream_name` is inserted into the event metadata.
        name=endpoint,
        stream=fast_api_tools.FastApiVideoStream(
            server=self._robot_server,
            endpoint=endpoint,
            stream_name=endpoint,
        ).stream(),
        is_image_stream=True,
    )

  def _create_event_streams(self) -> Sequence[embodiment.EventStreamDefinition]:
    return [
        self._create_video_stream(endpoint)
        for endpoint in (
            CAGE_CENTER_ENDPOINT,
            CAGE_LEFT_ENDPOINT,
            CAGE_RIGHT_ENDPOINT,
            WRIST_LEFT_ENDPOINT,
            WRIST_RIGHT_ENDPOINT,
        )
    ]

  def _create_tools(self, bus: event_bus.EventBus) -> Sequence[tool.Tool]:
    """Instantiates a list of tools specific to the OmegaStar robot."""
    return [
        tool.Tool(
            fn=fast_api_tools.FastApiGet(
                self._robot_server,
                _Endpoints.RUN_INSTRUCTION.value,
                ["instruction"],
            ),
            declaration=types.FunctionDeclaration(
                name=RUN_INSTRUCTION_TOOL_NAME,
                description=_RUN_INSTRUCTION_TOOL_DESCRIPTION,
                behavior=types.Behavior.NON_BLOCKING,
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "instruction": types.Schema(
                            type=types.Type.STRING,
                            description=(
                                "A short, clear instruction for the robot (e.g."
                                " 'pick up cup', 'place on table', 'move left')"
                            ),
                        ),
                    },
                    required=["instruction"],
                ),
            ),
            bus=bus,
        ),
        tool.Tool(
            fn=fast_api_tools.FastApiGet(
                self._robot_server, _Endpoints.RESET.value, []
            ),
            declaration=types.FunctionDeclaration(
                name=RESET_TOOL_NAME,
                description="Reset the robot to its default starting position.",
                behavior=types.Behavior.NON_BLOCKING,
            ),
            bus=bus,
        ),
        tool.Tool(
            fn=fast_api_tools.FastApiGet(
                self._robot_server, _Endpoints.STOP_INSTRUCTION.value, []
            ),
            declaration=types.FunctionDeclaration(
                name=STOP_TOOL_NAME,
                description="Stop the robot in its current position.",
                behavior=types.Behavior.NON_BLOCKING,
            ),
            bus=bus,
        ),
        tool.Tool(
            fn=fast_api_tools.FastApiGet(
                self._robot_server,
                _Endpoints.OPEN_GRIPPERS.value,
                [],
            ),
            declaration=types.FunctionDeclaration(
                name=OPEN_GRIPPER_TOOL_NAME,
                description="Open the grippers of the robot.",
                behavior=types.Behavior.BLOCKING,
            ),
            bus=bus,
        ),
        tool.Tool(
            fn=fast_api_tools.FastApiGet(
                self._robot_server, _Endpoints.SLEEP.value, []
            ),
            declaration=types.FunctionDeclaration(
                name=SLEEP_TOOL_NAME,
                description=(
                    "Put the robot in a safe resting position for shutdown."
                ),
                behavior=types.Behavior.NON_BLOCKING,
            ),
            bus=bus,
        ),
    ]
