"""Simple FastAPI client that subscribes to external instructions."""

from absl import logging

from safari_sdk.agent.framework import types
from safari_sdk.agent.framework.embodiments import fast_api_endpoint
from safari_sdk.agent.framework.event_bus import event_bus
from safari_sdk.agent.framework.event_bus import event_stream_handler
from safari_sdk.agent.framework.tools import fast_api_tools


class PassiveControllerClient:
  """The Fast API server for the passive controller of EAR framework."""

  def __init__(
      self,
      bus: event_bus.EventBus,
      host: str,
      port: int,
      endpoint: fast_api_endpoint.FastApiEndpoint,
      reset_command: str,
  ):
    self._reset_command = reset_command
    self._last_data = reset_command
    instruction_stream = fast_api_tools.FastApiServerSentEventsStream(
        server=f"http://{host}:{port}",
        endpoint=endpoint,
        server_sent_event_data_to_event_formatter=self._parse_stream_data,
    )
    self._event_handler = event_stream_handler.EventStreamHandler(
        bus=bus,
        streams={"instruction_stream": instruction_stream.stream()},
    )

  def _parse_stream_data(self, data: str) -> types.Event | None:
    """Parses the stream data into a framework event."""
    if data == self._last_data:
      return None
    self._last_data = data
    if data == self._reset_command:
      logging.info("Received RESET event from the passive controller.")
      return types.Event(
          type=types.EventType.RESET,
          source=types.EventSource.USER,
          data=data,
      )
    else:
      return types.Event(
          type=types.EventType.MODEL_TEXT_INPUT,
          source=types.EventSource.USER,
          data=data,
      )

  async def connect(self):
    """Connects to the handler and starts streaming events to the event bus."""
    logging.info("Connecting to the passive controller.")
    await self._event_handler.connect()

  async def disconnect(self):
    """Disconnects to the handler and stops streaming events to the event bus."""
    await self._event_handler.disconnect()
