"""Tests for the AgentFrameworkConfig."""

import unittest

from absl.testing import flagsaver

from absl.testing import absltest
from safari_sdk.agent.framework import config as framework_config
from safari_sdk.agent.framework import flags as agentic_flags
from safari_sdk.agent.framework import types


class AgentFrameworkConfigTest(unittest.TestCase):

  def test_create_with_no_overrides(self):
    """Test that create() works without any overrides."""
    with flagsaver.flagsaver(
        (agentic_flags.AGENTIC_API_KEY, 'test_api_key'),
        (agentic_flags.AGENTIC_AGENT_NAME, 'test_agent'),
    ):
      config = framework_config.AgentFrameworkConfig.create()
      self.assertEqual(config.api_key, 'test_api_key')
      self.assertEqual(config.agent_name, 'test_agent')

  def test_create_with_overrides(self):
    """Test that create() works with overrides."""
    with flagsaver.flagsaver(
        (agentic_flags.AGENTIC_API_KEY, 'flag_api_key'),
        (agentic_flags.AGENTIC_AGENT_NAME, 'flag_agent'),
        (agentic_flags.AGENTIC_MEOW_MODE, False),
    ):
      config = framework_config.AgentFrameworkConfig.create(
          api_key='override_api_key',
          meow_mode=True,
      )
      # Overridden values
      self.assertEqual(config.api_key, 'override_api_key')
      self.assertEqual(config.meow_mode, True)
      # Non-overridden value from flags
      self.assertEqual(config.agent_name, 'flag_agent')

  def test_create_with_none_override_doesnt_override(self):
    """Test that passing None doesn't override the flag value."""
    with flagsaver.flagsaver(
        (agentic_flags.AGENTIC_API_KEY, 'flag_value'),
    ):
      config = framework_config.AgentFrameworkConfig.create(
          api_key=None,
      )
      # None should not override, so we should get the flag value
      self.assertEqual(config.api_key, 'flag_value')

  def test_create_with_bool_overrides(self):
    """Test that boolean overrides work correctly."""
    with flagsaver.flagsaver(
        (agentic_flags.AGENTIC_ENABLE_AUDIO_INPUT, False),
        (agentic_flags.AGENTIC_ENABLE_AUDIO_OUTPUT, False),
    ):
      config = framework_config.AgentFrameworkConfig.create(
          enable_audio_input=True,
          enable_audio_output=False,
      )
      self.assertEqual(config.enable_audio_input, True)
      self.assertEqual(config.enable_audio_output, False)

  def test_create_with_enum_overrides(self):
    """Test that enum overrides work correctly."""
    with flagsaver.flagsaver(
        (
            agentic_flags.AGENTIC_SD_TOOL_NAME,
            agentic_flags.SDToolName.SUBTASK_SUCCESS_DETECTOR_V4,
        ),
    ):
      config = framework_config.AgentFrameworkConfig.create(
          sd_tool_name=agentic_flags.SDToolName.SUBTASK_SUCCESS_DETECTOR_V2,
      )
      self.assertEqual(
          config.sd_tool_name,
          agentic_flags.SDToolName.SUBTASK_SUCCESS_DETECTOR_V2,
      )

  def test_create_with_numeric_overrides(self):
    """Test that numeric overrides work correctly."""
    with flagsaver.flagsaver(
        (agentic_flags.AGENTIC_EXTERNAL_CONTROLLER_PORT, 8888),
        (agentic_flags.AGENTIC_SD_TIMEOUT_SECONDS, 60.0),
    ):
      config = framework_config.AgentFrameworkConfig.create(
          external_controller_port=9999,
          sd_timeout_seconds=120.5,
      )
      self.assertEqual(config.external_controller_port, 9999)
      self.assertEqual(config.sd_timeout_seconds, 120.5)

  def test_create_with_list_overrides(self):
    """Test that list overrides work correctly."""
    with flagsaver.flagsaver(
        (agentic_flags.AGENTIC_REMINDER_TEXT_LIST, []),
    ):
      config = framework_config.AgentFrameworkConfig.create(
          reminder_text_list=['reminder1', 'reminder2'],
      )
      self.assertEqual(config.reminder_text_list, ['reminder1', 'reminder2'])

  def test_dataclass_is_frozen(self):
    """Test that AgentFrameworkConfig is frozen."""
    config = framework_config.AgentFrameworkConfig()
    with self.assertRaises(AttributeError):
      config.api_key = 'new_value'

  def test_direct_instantiation_with_defaults(self):
    """Test that AgentFrameworkConfig can be instantiated directly."""
    config = framework_config.AgentFrameworkConfig()
    # Check that default values are set
    self.assertIsNone(config.api_key)
    self.assertEqual(config.agent_name, 'simple_agent')
    self.assertEqual(config.control_mode, types.ControlMode.TERMINAL_ONLY)

  def test_direct_instantiation_with_values(self):
    """Test that AgentFrameworkConfig can be instantiated with values."""
    config = framework_config.AgentFrameworkConfig(
        api_key='custom_key',
        agent_name='custom_agent',
        meow_mode=True,
    )
    self.assertEqual(config.api_key, 'custom_key')
    self.assertEqual(config.agent_name, 'custom_agent')
    self.assertEqual(config.meow_mode, True)


if __name__ == '__main__':
  absltest.main()
