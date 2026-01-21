"""Fast API endpoints for the OmegaStar robot."""

import enum

import fastapi

from safari_sdk.agent.framework.embodiments import fast_api_endpoint

# For brevity.
FastApiEndpoint = fast_api_endpoint.FastApiEndpoint


DEFAULT_PORT = 8888

# Special token the robot will stream when the episode is over.
END_OF_EPISODE_TOKEN = "END_EPISODE"


class FastApiEndpoints(enum.Enum):
  """Canonical Fast API endpoints for the OmegaStar robot."""

 ##############################
  ### INSTRUCTION EXECUTION ###
  #############################

  # Runs an instruction with the server's policy.
  RUN_INSTRUCTION = FastApiEndpoint(path="/run/")

  # Continues the last instruction.
  CONTINUE_INSTRUCTION = FastApiEndpoint(path="/continue/")

  # Stop the current instruction.
  STOP_INSTRUCTION = FastApiEndpoint(path="/stop/")

  # Reset the current instruction.
  RESET_INSTRUCTION = FastApiEndpoint(path="/reset-instruction/")

  # Open all of the robot's grippers.
  OPEN_GRIPPERS = FastApiEndpoint(path="/open_grippers/")

  # Open all of the robot's grippers.
  CLOSE_GRIPPERS = FastApiEndpoint(path="/close_grippers/")

  # Reset the robot to its starting pose.
  RESET = FastApiEndpoint(path="/reset/")

  # Place the robot in a safe resting position for shutdown.
  SLEEP = FastApiEndpoint(path="/sleep/")

  ###########################
  ### STREAMING ENDPOINTS ###
  ###########################

  # TODO: Replace names with the ones from omega_star_constants.py
  # once checked in.
  # Various camera streams.
  CAGE_CENTER_CAMERA_STREAM = FastApiEndpoint(
      path="/cage_back_rgb_img/",
      response_class=fastapi.responses.StreamingResponse,
  )

  CAGE_LEFT_CAMERA_STREAM = FastApiEndpoint(
      path="/cage_left_rgb_img/",
      response_class=fastapi.responses.StreamingResponse,
  )

  CAGE_RIGHT_CAMERA_STREAM = FastApiEndpoint(
      path="/cage_right_rgb_img/",
      response_class=fastapi.responses.StreamingResponse,
  )

  WRIST_LEFT_CAMERA_STREAM = FastApiEndpoint(
      path="/wrist_back_left_rgb_img/",
      response_class=fastapi.responses.StreamingResponse,
  )

  WRIST_RIGHT_CAMERA_STREAM = FastApiEndpoint(
      path="/wrist_back_right_rgb_img/",
      response_class=fastapi.responses.StreamingResponse,
  )

  # Streams the latest instructions that could be set by the environment or
  # a system like Orca for benchmark task definitions.
  INSTRUCTION_STREAM = FastApiEndpoint(
      path="/instruction-stream/",
      response_class=fastapi.responses.StreamingResponse,
  )

  # (Simulation only) Streams the camera video.
  VIDEO_STREAM = FastApiEndpoint(
      path="/video-stream/", response_class=fastapi.responses.StreamingResponse
  )

  # (Simulation only) Gets an HTML with the live video feed.
  VIDEO_FEED_HTML = FastApiEndpoint(
      path="/video/", response_class=fastapi.responses.HTMLResponse
  )

# TODO: Replace names with the ones from omega_star_constants.py
# once checked in.
CAMERA_ID_TO_ENDPOINT = {
    "cage_back_rgb_img": FastApiEndpoints.CAGE_CENTER_CAMERA_STREAM,
    "cage_left_rgb_img": FastApiEndpoints.CAGE_LEFT_CAMERA_STREAM,
    "cage_right_rgb_img": FastApiEndpoints.CAGE_RIGHT_CAMERA_STREAM,
    "wrist_back_left_rgb_img": (
        FastApiEndpoints.WRIST_LEFT_CAMERA_STREAM
    ),
    "wrist_back_right_rgb_img": (
        FastApiEndpoints.WRIST_RIGHT_CAMERA_STREAM
    ),
}
