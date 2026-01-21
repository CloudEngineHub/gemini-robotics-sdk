"""VQA tool."""

import asyncio

from absl import logging
from google import genai
from google.genai import types

from safari_sdk.agent.framework import config as framework_config
from safari_sdk.agent.framework.event_bus import event_bus
from safari_sdk.agent.framework.tools import genai_client
from safari_sdk.agent.framework.tools import task_planning
from safari_sdk.agent.framework.tools import tool
from safari_sdk.agent.framework.utils import image_processing


class VQATool(tool.Tool):
  """Tools used for visual question answering."""

  def __init__(
      self,
      bus: event_bus.EventBus,
      config: framework_config.AgentFrameworkConfig,
      api_key: str,
      camera_endpoint_names: list[str],
  ):
    # One time planning.
    declaration = types.FunctionDeclaration(
        name="ask_visual_question",
        description=(
            "Ask a question that can be answered by looking at the current"
            " camera view. For example: 'Is the cup on the table?', 'How many"
            " objects are in the scene?', 'What color is the apple?'"
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "question": types.Schema(
                    type=types.Type.STRING,
                    description=(
                        "Question about the current camera view or physical"
                        " scene."
                    ),
                ),
            },
            required=[
                "question",
            ],
        ),
        behavior=types.Behavior.BLOCKING,
    )
    python_function = self.ask_vision_question
    super().__init__(fn=python_function, declaration=declaration, bus=bus)
    self._call_id = None
    self._config = config

    # Initialize the LLM with the Gemini API.
    client = genai.Client(api_key=api_key)
    # Controls the thinking budget.
    logging.info(
        "VQA thinking budget: %s",
        config.vqa_thinking_budget,
    )
    thinking_config = types.ThinkingConfig(
        thinking_budget=config.vqa_thinking_budget
    )
    generation_config = types.GenerateContentConfig(
        temperature=0.0, thinking_config=thinking_config
    )
    self._vlm = genai_client.GeminiClientWrapper(
        bus=bus,
        client=client,
        model_name=config.vqa_model_name,
        config=generation_config,
    ).generate_content
    # Stores the last num_history_frames frames.
    self.history_images = []

    # The buffer to store the latest images from the event bus.
    self._image_buffer = task_planning.ImageBuffer(camera_endpoint_names)
    bus.subscribe(
        event_types=[event_bus.EventType.MODEL_IMAGE_INPUT],
        handler=self._image_buffer.update_from_event,
    )

  async def ask_vision_question(
      self,
      question: str,
      call_id: str = "",
  ) -> types.FunctionResponse:
    del call_id
    prompt = [
        (
            "You are an advanced, highly accurate vision AI, responsible for"
            " analysing the scene where robot arms interact with various"
            " objects. Your primary duty is to answer the question from the"
            " user about the scene. You have access to one or more cameras"
            "installed on the robot."
        ),
        (
            "This is what the current scene looks like now, "
            "taken from all cameras:"
        ),
        *self._image_buffer.get_interleaved_images(
            self._image_buffer.get_latest_images()
        ),
        f"Question:\n{question}",
        (
            "Please deliberate on the question then answer the question."
            " You need to combine all camera views when answering the question."
            " Please limit your response to"
            f" {self._config.vqa_num_output_words} words."
        ),
    ]
    prompt = image_processing.convert_bytes_to_image(prompt)

    try:
      response = await self._vlm(contents=prompt)
      logging.info("Multi-camera VQA model raw response: %s", response.text)

      # Directly return the raw response as the task plan.
      return types.FunctionResponse(
          response={
              "answer": response.text,
          },
          will_continue=False,
      )

    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.exception("Multi-camera VQA model failed to respond:\n%s", e)
      await asyncio.sleep(0.1)  # This is so that it returns a coroutine.
      return types.FunctionResponse(
          response={
              "answer": "The model failed to respond. Please try again.",
          },
          will_continue=False,
      )
