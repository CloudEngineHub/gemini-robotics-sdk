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
