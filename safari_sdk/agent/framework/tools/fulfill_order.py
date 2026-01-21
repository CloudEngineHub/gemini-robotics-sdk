"""A tool for fulfilling an order ."""

import asyncio
import functools
from typing import AsyncIterator, Dict, Sequence, cast

from absl import logging
from google import genai
from google.genai import types

from safari_sdk.agent.framework import types as framework_types
from safari_sdk.agent.framework.event_bus import event_bus
from safari_sdk.agent.framework.tools import tool
from safari_sdk.agent.framework.utils import image_processing


_TOOL_NAME = "fulfill_order"
_DONE_SIGNAL = "Done"
_RESPONSE_OUTPUT_KEY = "output"
_RESPONSE_ERROR_KEY = "error"
_RESPONSE_STATUS_KEY = "status"
_USER_ROLE = "user"

_TOOL_DESCRIPTION = """
  Fulfills a shopping order by planning an optimal route and using navigation
  and manipulation tools. This tool manages the entire process from grabbing a
  cart to picking and placing items. The order should be a comma-separated list
  of items, e.g., 'apples, detergent, coke'.
  """

_SYSTEM_INSTRUCTION = """
  You are a helpful and intelligent humanoid robot with two arms and two legs.
  You can look at your surroundings through the camera image in your context.

  You are a shopping assistant in a grocery store. You will be asked to fulfill
  orders by customers in a grocery/retail store. You are provided with tools to
  execute physical actions to complete tasks.

  Here's how you will fulfill an order given as a list of items. You are
  standing in front of a shopping cart. Follow the steps below to fill an order:

  1: *Planning*  This is the most important step. Use the "get_scene_layout"
  tool to determine where the items are located.  Do not guess or invent
  locations. Make a list of all locations to go to from the response of the
  "get_scene_layout" tool. For example, if the items are in multiple locations,
  you will need to go to each location. If more that one items are in the same
  location, you only need to go there once. You can pick up multiple items from
  the same location.
  2: After planning, prepare to fulfill the order. First, "go_to" the cart.
  3. Then grab the shopping cart.
  3. For each location in your list, walk there with the "go_to" tool.
  4. Release the cart when you reach the location.
  5. Turn towards the location to face it with the "turn_towards" tool.
  6. Use the "get_scene_layout" tool to get the description of the items in the
  current location.
  7. For each order item to be picked up from the location, Construct an
  instruction to pack the item. Instruction should include the description of
  the item. For example, if the item is "cinnamon cereal squares", the
  instruction should be "pack the cinnamon cereal squares". Use the
  "run_instruction" tool to execute the instruction.
  8. Turn back towards the cart with the "turn_towards" tool.
  9. Repeat the above steps till the order is fulfilled.
  """

_STEP_PROMPT = """
  Instruction: If the order is fulfilled, output 'Done'. Otherwise, call the
  next relevant tool to continue fulfilling the order based on your plan.
  """


class FulfillOrderTool(tool.Tool):
  """Fulfills an order using navigation and manipulation tools on the robot."""
  _MAX_NO_TOOL_CALL_RETRIES = 3
  _MAX_TOOL_EXECUTION_RETRIES = 30

  def __init__(
      self,
      bus: event_bus.EventBus,
      api_key: str,
      model_name: str,
      toolset: Sequence[tool.Tool],
  ):
    """Initializes the tool, VLM client, and event bus subscriptions."""
    declaration = types.FunctionDeclaration(
        name=_TOOL_NAME,
        description=_TOOL_DESCRIPTION,
        behavior=types.Behavior.NON_BLOCKING,
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "order": types.Schema(
                    type=types.Type.STRING,
                    description=(
                        "A comma-separated list of items in the order, e.g., "
                        "'apples, detergent, coke'"
                    ),
                ),
            },
            required=["order"],
        ),
    )
    super().__init__(
        fn=cast(framework_types.AsyncFunction, self.fulfill_order),
        declaration=declaration,
        bus=bus,
    )

    # State management
    self._tool_call_cancelled = False
    self._order_in_progress = False
    self._call_id: str | None = None
    self.latest_image: bytes | None = None

    # Tool and VLM setup
    self._tooldict: Dict[str, tool.Tool] = {
        t.declaration.name: t for t in toolset
    }
    self._setup_vlm_client(api_key, model_name, toolset)

    # Event bus subscriptions
    bus.subscribe(
        event_types=[event_bus.EventType.MODEL_IMAGE_INPUT],
        handler=self._handle_image_input_events,
    )

  def _setup_vlm_client(
      self, api_key: str, model_name: str, toolset: Sequence[tool.Tool]
  ) -> None:
    """Configures and initializes the Generative Model client."""
    function_declarations = [t.declaration for t in toolset]
    for decl in function_declarations:
      decl.behavior = None  # Ensure sub-tools are blocking within this context

    logging.info("Using %s model for fulfilling order.", model_name)
    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        temperature=0.0,
        system_instruction=_SYSTEM_INSTRUCTION,
        tools=[types.Tool(function_declarations=function_declarations)],
        )
    self.vlm = functools.partial(
        client.models.generate_content,
        model=model_name,
        config=config,
    )

  def _handle_image_input_events(self, event: event_bus.Event) -> None:
    """Stores the latest image received from the event bus."""
    self.latest_image = event.data

  def _handle_tool_call_cancellation_events(self, event: event_bus.Event):
    """Handles tool cancellation events."""
    if self._call_id and self._call_id in event.data.ids:
      self._tool_call_cancelled = True
      logging.info(
          "Cancellation request received for fulfill_order. id: %s.",
          self._call_id,
      )

  async def _execute_tool_call(
      self, call_id: str, part: types.Part
  ) -> types.Part:
    """Executes a tool call and returns a structured response part."""
    assert part.function_call is not None, "Function call is None."
    fn_name = part.function_call.name
    fn_args = part.function_call.args

    for attempt in range(self._MAX_TOOL_EXECUTION_RETRIES):
      logging.info("Attempt %d to execute tool: %s", attempt + 1, fn_name)
      fn_response_obj = await self._tooldict[fn_name].fn(
          call_id=call_id, **fn_args
      )
      fn_response_dict = fn_response_obj.response  # pytype: disable=attribute-error
      if _RESPONSE_ERROR_KEY in fn_response_dict:
        e = str(fn_response_dict[_RESPONSE_ERROR_KEY])
        if "409" in e:
          logging.warning(
              "Tool '%s' failed with 409 Conflict. Robot is busy. Retrying"
              " after delay...",
              fn_name,
          )
          # Increasing backoff: 1s, 2s, 3s...
          await asyncio.sleep(attempt + 1)
          continue  # Go to the next attempt
        else:
          logging.error("Sub-tool '%s' failed with HTTP error: %s", fn_name, e)
          # For unexpected errors, fail immediately
          break
      return types.Part.from_function_response(
          name=fn_name, response=fn_response_dict
      )

  async def fulfill_order(
      self, order: str, call_id: str
  ) -> AsyncIterator[types.FunctionResponse]:
    """Fulfills a shopping order by orchestrating sub-tools through a VLM.

    This function manages a continuous loop of:
    1. Sending the current state (including camera image) to the LLM.
    2. Receiving a tool call or a "Done" signal from the LLM.
    3. Executing the tool call.
    4. Feeding the result (success or error) back to the VLM for the next step.

    Args:
      order: A comma-separated string of items to fulfill.
      call_id: The unique identifier for this tool call.

    Yields:
      types.FunctionResponse: Responses indicating the status of the
      order fulfillment, including tool execution results, cancellation,
      completion, or errors.
    """
    self._call_id = call_id
    if self._order_in_progress:
      yield types.FunctionResponse(
          response={
              _RESPONSE_OUTPUT_KEY: (
                  "Order already in progress. Please cancel the previous "
                  "order first or wait for it to finish."
              ),
          },
          will_continue=False,
      )
      return

    self._order_in_progress = True
    self._tool_call_cancelled = False
    await self._tooldict["stop_eye_contact"].fn(call_id=call_id)

    initial_prompt = (
        f"Order received: {order}. My first step is to understand the store"
        " layout to plan my route. I must use the `get_scene_layout` tool now."
    )
    prompt_history = [initial_prompt]
    no_tool_call_retries = 0

    try:
      while True:
        if self._tool_call_cancelled:
          logging.info("Order fulfillment cancelled by user.")
          yield types.FunctionResponse(
              response={_RESPONSE_OUTPUT_KEY: "Order cancelled."},
              will_continue=False,
          )
          break

        # Append current visual context and prompt for the next action
        current_prompt = list(prompt_history)
        current_prompt.append("Current camera image: ")
        if self.latest_image:
          current_prompt.extend(
              image_processing.convert_bytes_to_image([self.latest_image])
          )
        current_prompt.append(_STEP_PROMPT)

        # Call the VLM for the next step
        response = await asyncio.to_thread(self.vlm, contents=current_prompt)
        model_response_content = response.candidates[0].content
        prompt_history.append(model_response_content)

        logging.info("Fulfill order: Model Response Text: %s", response.text)
        logging.info(
            "Fulfill order: Model Tool Call: %s", model_response_content
        )

        if response.text and _DONE_SIGNAL in response.text:
          yield types.FunctionResponse(
              response={_RESPONSE_OUTPUT_KEY: "Order fulfilled successfully."},
              will_continue=False,
          )
          break

        # Process and execute tool calls
        tool_call_executed = False
        for part in model_response_content.parts:
          if part.function_call:
            # If we get a valid tool call, reset the retry counter
            no_tool_call_retries = 0
            fn_response_part = await self._execute_tool_call(call_id, part)
            # Send tool response back to the model
            prompt_history.append(
                types.Content(role=_USER_ROLE, parts=[fn_response_part])
            )
            # Yield intermediate status back to the main agent
            fn_response_dict = fn_response_part.function_response.response  # pytype: disable=attribute-error
            # Check if the sub-tool execution resulted in an error
            if _RESPONSE_ERROR_KEY in fn_response_dict:
              logging.error(
                  "Fulfill order is stopping due to a sub-tool failure.")
              fn_response_dict[_RESPONSE_OUTPUT_KEY] = (
                  "Fulfill order failed due to an internal error.")
              yield types.FunctionResponse(
                  response=fn_response_dict,
                  will_continue=False,
              )
              return  # Stop the entire process

            fn_response_dict[_RESPONSE_STATUS_KEY] = "Order in progress."
            yield types.FunctionResponse(
                response=fn_response_dict,
                will_continue=True,
                scheduling=types.FunctionResponseScheduling.SILENT,
            )
            tool_call_executed = True
            break  # Process one tool call per loop iteration

        if not tool_call_executed:
          no_tool_call_retries += 1
          logging.warning(
              "Model did not return a tool call or 'Done'. Retry %d/%d.",
              no_tool_call_retries,
              self._MAX_NO_TOOL_CALL_RETRIES,
          )
          if no_tool_call_retries > self._MAX_NO_TOOL_CALL_RETRIES:
            raise RuntimeError(
                "Model failed to call a tool or signal completion after"
                f" {self._MAX_NO_TOOL_CALL_RETRIES} retries."
            )
          # Add a corrective prompt and continue the loop to try again
          corrective_prompt = types.Part(
              text=(
                  "Your response was not valid. You must either call a tool to"
                  " continue the task or output 'Done' if task is complete."
              )
          )
          prompt_history.append(
              types.Content(role=_USER_ROLE, parts=[corrective_prompt])
          )
          continue

    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.error("Fulfill order failed with an unhandled exception: %s", e)
      yield types.FunctionResponse(
          response={
              _RESPONSE_OUTPUT_KEY: "Fulfill order failed.",
              _RESPONSE_ERROR_KEY: str(e),
          },
          will_continue=False,
      )
    finally:
      self._order_in_progress = False
      self._call_id = None
      await self._tooldict["make_eye_contact"].fn(call_id=call_id)
