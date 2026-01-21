"""Tests for the external controller."""

import asyncio
import logging
import unittest

import httpx

from absl.testing import absltest
from safari_sdk.agent.framework import config as framework_config
from safari_sdk.agent.framework.comms import external_controller
from safari_sdk.agent.framework.event_bus import event_bus


CANONICAL_GET_ENDPOINTS = {
    "/execute_lh_task/?lh_task=clean%20the%20table%20and%20you%20decide%20how",
    "/stop/",
    "/reset/",
    "/get_framework_status/",
}


class ExternalControllerTest(unittest.IsolatedAsyncioTestCase):

  def setUp(self):
    super().setUp()
    self.config = framework_config.AgentFrameworkConfig()
    self.bus = event_bus.EventBus(config=self.config)
    self.controller = external_controller.ExternalControllerFastAPIServer(
        bus=self.bus, host="0.0.0.0", port=8887
    )
    self.client = httpx.AsyncClient()
    logging.getLogger("httpx").setLevel(logging.DEBUG)

  def tearDown(self):
    asyncio.run(self.client.aclose())
    try:
      self.controller.disconnect()
    except Exception:  # pylint: disable=broad-exception-caught
      pass
    super().tearDown()

  async def _wait_for_server_ready(
      self, timeout: float = 10.0, retry_interval: float = 1.0
  ):
    """Wait for the server to be ready."""
    start_time = asyncio.get_running_loop().time()
    while asyncio.get_running_loop().time() - start_time < timeout:
      try:
        response = await self.client.get(
            "http://0.0.0.0:8887/get_framework_status/", timeout=2.0
        )
        response.raise_for_status()
        logging.info("Server is ready.")
        return
      except httpx.RequestError:
        logging.debug(
            "Server not ready yet, retrying in %s seconds...", retry_interval
        )
        await asyncio.sleep(retry_interval)
    self.fail("Server did not become ready within timeout.")

  async def _hit_endpoint(self, endpoint: str) -> httpx.Response:
    """Hits the endpoint with a GET request and returns the response."""
    try:
      logging.info("Hitting endpoint: %s", endpoint)
      url = f"http://0.0.0.0:8887{endpoint}"
      logging.debug("URL: %s", url)
      response = await self.client.get(url, timeout=30.0)
      response.raise_for_status()
      return response
    except httpx.RequestError as e:
      logging.exception("HTTP Error: %s", e)
      self.fail(f"Endpoint: {endpoint} is not reachable.")

  async def test_canonical_endpoints_recheable(self):
    await self.controller.connect()
    await self._wait_for_server_ready()
    for endpoint in CANONICAL_GET_ENDPOINTS:
      await self._hit_endpoint(endpoint)

  async def test_publish_event_endpoint(self):
    await self.controller.connect()
    await self._wait_for_server_ready()
    response = await self.client.post(
        "http://0.0.0.0:8887/publish_event/",
        json={
            "type": "DEBUG",
            "source": "USER",
            "data": "hello",
            "metadata": {},
        },
    )
    response.raise_for_status()
    self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
  absltest.main()
