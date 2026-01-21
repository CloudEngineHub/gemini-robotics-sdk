"""Tests for FastAPI helpers.

These are currently very basic because we can't reasonably test for any response
other than an HTTP error.
"""

import asyncio
import unittest
from unittest import mock

from absl import app  # for fastapi need pylint: disable=unused-import

from absl.testing import absltest
from safari_sdk.agent.framework.embodiments import fast_api_endpoint
from safari_sdk.agent.framework.tools import fast_api_tools


_TEST_SERVER = "http://testserver"

_FAKE_ENDPOINT = fast_api_endpoint.FastApiEndpoint(path="/foo/")


class EventBusTest(unittest.IsolatedAsyncioTestCase):

  def setUp(self):
    super().setUp()
    self.loop = asyncio.new_event_loop()
    asyncio.set_event_loop(self.loop)

  def tearDown(self):
    self.loop.close()
    super().tearDown()

  async def test_post_without_arg(self):
    stop = fast_api_tools.FastApiGet(
        server=_TEST_SERVER,
        endpoint=_FAKE_ENDPOINT,
        param_names=[],
    )
    with mock.patch.object(fast_api_tools.httpx, "get", autospec=True):
      await stop("unused_call_id")

  async def test_post_with_arg(self):
    _ = fast_api_tools.FastApiGet(
        server=_TEST_SERVER,
        endpoint=_FAKE_ENDPOINT,
        param_names=["instruction"],
    )

  async def test_explicit_arg_before_implicit(self):
    run = fast_api_tools.FastApiGet(
        server=_TEST_SERVER,
        endpoint=_FAKE_ENDPOINT,
        param_names=["instruction"],
    )
    # The call_id argument is *appended* to the explicitly specified args,
    # so having one unnamed arg ("instruction") followed by a kwarg
    # "instruction" should be rejected.
    with mock.patch.object(fast_api_tools.httpx, "get", autospec=True):
      with self.assertRaises(TypeError):
        await run("unused_call_id", instruction="move")

  async def test_video_stream(self):
    _ = fast_api_tools.FastApiVideoStream(
        server=_TEST_SERVER,
        endpoint=_FAKE_ENDPOINT,
        stream_name="overhead_camera",
    )
    # Can't await since nothing will stream anything.


if __name__ == "__main__":
  absltest.main()
