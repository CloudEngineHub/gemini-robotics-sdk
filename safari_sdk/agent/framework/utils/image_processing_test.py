from unittest import mock

from google.genai import types

from absl.testing import absltest
from safari_sdk.agent.framework.utils import image_processing


class ConvertBytesToImageTest(absltest.TestCase):

  def test_convert_single_bytes_to_image(self):
    prompt_elems = [b"image_data"]

    result = image_processing.convert_bytes_to_image(prompt_elems)

    self.assertLen(result, 1)
    self.assertIsInstance(result[0], types.Part)

  def test_convert_multiple_bytes_to_image(self):
    prompt_elems = [b"image_data_1", b"image_data_2", b"image_data_3"]

    result = image_processing.convert_bytes_to_image(prompt_elems)

    self.assertLen(result, 3)
    for elem in result:
      self.assertIsInstance(elem, types.Part)

  def test_convert_mixed_elements(self):
    prompt_elems = ["text_prompt", b"image_data", "more_text", b"another_image"]

    result = image_processing.convert_bytes_to_image(prompt_elems)

    self.assertLen(result, 4)
    self.assertEqual(result[0], "text_prompt")
    self.assertIsInstance(result[1], types.Part)
    self.assertEqual(result[2], "more_text")
    self.assertIsInstance(result[3], types.Part)

  def test_convert_non_bytes_elements_unchanged(self):
    prompt_elems = ["text1", "text2", 123, {"key": "value"}]

    result = image_processing.convert_bytes_to_image(prompt_elems)

    self.assertLen(result, 4)
    self.assertEqual(result[0], "text1")
    self.assertEqual(result[1], "text2")
    self.assertEqual(result[2], 123)
    self.assertEqual(result[3], {"key": "value"})

  def test_convert_empty_sequence(self):
    prompt_elems = []

    result = image_processing.convert_bytes_to_image(prompt_elems)

    self.assertEmpty(result)

  def test_convert_only_text_elements(self):
    prompt_elems = ["text1", "text2", "text3"]

    result = image_processing.convert_bytes_to_image(prompt_elems)

    self.assertLen(result, 3)
    self.assertEqual(result, ["text1", "text2", "text3"])

  @mock.patch.object(types.Part, "from_bytes", autospec=True)
  def test_convert_calls_from_bytes_with_correct_mime_type(
      self, mock_from_bytes
  ):
    mock_from_bytes.return_value = mock.MagicMock()
    prompt_elems = [b"image_data"]

    image_processing.convert_bytes_to_image(prompt_elems)

    mock_from_bytes.assert_called_once_with(
        data=b"image_data", mime_type="image/jpeg"
    )

  @mock.patch.object(types.Part, "from_bytes", autospec=True)
  def test_convert_preserves_order(self, mock_from_bytes):
    mock_image_1 = mock.MagicMock()
    mock_image_2 = mock.MagicMock()
    mock_from_bytes.side_effect = [mock_image_1, mock_image_2]
    prompt_elems = ["start", b"image1", "middle", b"image2", "end"]

    result = image_processing.convert_bytes_to_image(prompt_elems)

    self.assertLen(result, 5)
    self.assertEqual(result[0], "start")
    self.assertEqual(result[1], mock_image_1)
    self.assertEqual(result[2], "middle")
    self.assertEqual(result[3], mock_image_2)
    self.assertEqual(result[4], "end")

  def test_convert_with_empty_bytes(self):
    prompt_elems = [b""]

    result = image_processing.convert_bytes_to_image(prompt_elems)

    self.assertLen(result, 1)
    self.assertIsInstance(result[0], types.Part)


if __name__ == "__main__":
  absltest.main()
