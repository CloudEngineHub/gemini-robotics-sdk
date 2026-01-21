"""Fast API endpoints for the OmegaStar robot."""

import enum

import fastapi

from google3.third_party.safari.partners.fr3_duo.reaf import constants as fr3_duo_constants
from safari_sdk.agent.framework.embodiments import fast_api_endpoint


# For brevity.
FastApiEndpoint = fast_api_endpoint.FastApiEndpoint


DEFAULT_PORT = 8888

# Special token the robot will stream when the episode is over.
END_OF_EPISODE_TOKEN = "END_EPISODE"

# Keys for the FR3 Duo cameras.
_HEAD_LEFT_CAMERA = fr3_duo_constants.HEAD_CAMERA_LEFT_RGB_IMAGE_KEY
_HEAD_RIGHT_CAMERA = fr3_duo_constants.HEAD_CAMERA_RIGHT_RGB_IMAGE_KEY
_WRIST_LEFT_CAMERA = fr3_duo_constants.WRIST_CAMERA_LEFT_RGB_IMAGE_KEY
_WRIST_RIGHT_CAMERA = fr3_duo_constants.WRIST_CAMERA_RIGHT_RGB_IMAGE_KEY
_CAGE_MULTICAMERA = "multicamera_rgb"


class FastApiEndpoints(enum.Enum):
  """Canonical Fast API endpoints for the OmegaStar robot."""

  # TODO: These should mostly be shared with simulation. Right now
  #   we manually maintain their correspondence; this should either be done via
  #   a common source of truth, or covered by tests.

  ##############################
  ### INSTRUCTION EXECUTION ###
  #############################

  # Get the current high-level instruction from the environment.
  # TODO: This is a temporary endpoint. Currently LH tasks are
  #   provided by the env, which has access to Orca and RobotUI. In the future,
  #   this may be moved into the session manager.
  GET_INSTRUCTION = FastApiEndpoint(path="/get_env_instruction/")

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

  # Reset the robot to its starting pose.
  RESET = FastApiEndpoint(path="/reset/")

  # Place the robot in a safe resting position for shutdown.
  SLEEP = FastApiEndpoint(path="/sleep/")

  #######################
  ### BACKEND CONTROL ###
  #######################

  # Trigger the environment to start a new episode.
  RESET_EPISODE = FastApiEndpoint(path="/reset_episode/")

  # Get the status of the backend, whether it's ready to reveive commands.
  BACKEND_STATUS = FastApiEndpoint(path="/get_backend_status/")

  ###########################
  ### STREAMING ENDPOINTS ###
  ###########################

  # Various camera streams.
  HEAD_LEFT_CAMERA_STREAM = FastApiEndpoint(
      path=f"/{_HEAD_LEFT_CAMERA}/",
      response_class=fastapi.responses.StreamingResponse,
  )

  HEAD_RIGHT_CAMERA_STREAM = FastApiEndpoint(
      path=f"/{_HEAD_RIGHT_CAMERA}/",
      response_class=fastapi.responses.StreamingResponse,
  )

  WRIST_LEFT_CAMERA_STREAM = FastApiEndpoint(
      path=f"/{_WRIST_LEFT_CAMERA}/",
      response_class=fastapi.responses.StreamingResponse,
  )

  WRIST_RIGHT_CAMERA_STREAM = FastApiEndpoint(
      path=f"/{_WRIST_RIGHT_CAMERA}/",
      response_class=fastapi.responses.StreamingResponse,
  )

  # (Simulation only) Streams the multicamera video.
  CAGE_MULTICAMERA_STREAM = FastApiEndpoint(
      path=f"/{_CAGE_MULTICAMERA}/",
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

CAMERA_ID_TO_ENDPOINT = {
    _HEAD_LEFT_CAMERA: FastApiEndpoints.HEAD_LEFT_CAMERA_STREAM,
    _HEAD_RIGHT_CAMERA: FastApiEndpoints.HEAD_RIGHT_CAMERA_STREAM,
    _WRIST_LEFT_CAMERA: FastApiEndpoints.WRIST_LEFT_CAMERA_STREAM,
    _WRIST_RIGHT_CAMERA: FastApiEndpoints.WRIST_RIGHT_CAMERA_STREAM,
}
