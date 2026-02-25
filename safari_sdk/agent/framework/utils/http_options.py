# Copyright 2026 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Utility functions for HTTP options."""


from absl import logging

from safari_sdk.agent.framework import config as framework_config


def get_http_options(
    config: framework_config.AgentFrameworkConfig,
) -> dict[str, dict[str, str] | str]:
  """Returns the HTTP options for the agent.

  Returns the HTTP options to specify api version and Sherlog headers.
  This can be used by live API and other tools that uses Gemini API.
  Note that this utility function consumes the agentic flags directly.

  Args:
    config: The agent framework config.

  Returns:
    A dictionary of HTTP options.
  """
  http_options = {"base_url": config.base_url}

  return http_options
