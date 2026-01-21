"""Tests for base types."""

import asyncio
from typing import AsyncIterator
import unittest

from google.genai import types as genai_types

from absl.testing import absltest
from safari_sdk.agent.framework import types


class TypesTest(unittest.IsolatedAsyncioTestCase):

  def setUp(self):
    super().setUp()
    self.loop = asyncio.new_event_loop()
    asyncio.set_event_loop(self.loop)

  def tearDown(self):
    self.loop.close()
    super().tearDown()

  async def test_define_async_function(self):

    class AsyncTestFunction(types.AsyncFunction):

      async def __call__(self, call_id: str, *args, **kwargs):

        async def inner() -> AsyncIterator[genai_types.FunctionResponse]:
          yield genai_types.FunctionResponse(
              response={"output": "hello world"},
          )
          yield genai_types.FunctionResponse(
              response={"output": "goodbye world"},
          )

        return inner()

    fn = AsyncTestFunction()
    responses = []
    async for response in await fn(call_id="123"):
      responses.append(response)
    self.assertEqual(len(responses), 2)
    self.assertEqual(responses[0].response["output"], "hello world")

  async def test_function_response_value(self):
    response = genai_types.FunctionResponse(response={"output": "hello world"})
    self.assertEqual(response.response["output"], "hello world")

  async def test_function_response_flat_value(self):
    response = genai_types.FunctionResponse(response={"foo": "bar"})
    self.assertEqual(response.response, {"foo": "bar"})

  async def test_function_response_error(self):
    response = genai_types.FunctionResponse(response={"error": "error message"})
    self.assertIn("error", response.response)
    self.assertEqual(response.response["error"], "error message")


if __name__ == "__main__":
  absltest.main()
