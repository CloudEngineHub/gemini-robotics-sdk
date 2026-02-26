// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "safari_sdk/logging/cc/log_data_serializer_utils.h"

#include <cstdint>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/imgcodecs.hpp>
#include <absl/container/fixed_array.h>
#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <absl/log/check.h>
#include <absl/status/status.h>
#include "safari_sdk/logging/cc/episode_data.h"

namespace safari::logging {

std::unique_ptr<EpisodeData> CreateEpisodeDataForTesting(int num_timesteps) {
  auto payload = std::make_unique<EpisodeDataPayloadInterface>();
  std::vector<int64_t> timestamps(num_timesteps);
  std::iota(timestamps.begin(), timestamps.end(), 0);
  auto episode_data = EpisodeData::Create(std::move(payload), timestamps);
  CHECK_OK(episode_data);
  return std::move(*episode_data);
}

namespace {

using ::testing::EqualsProto;
using ::testing::status::StatusIs;

// Helper to create a C-contiguous EpisodeFeatureBuffer for testing
template <typename T>
EpisodeFeatureBuffer<T> CreateTestBuffer(
    T* ptr, const std::vector<ssize_t>& shape_vec) {
  absl::FixedArray<ssize_t> shape(shape_vec.begin(), shape_vec.end());
  int ndim = shape.size();
  if (ndim == 0) {
    return EpisodeFeatureBuffer<T>(ptr, shape, ndim,
                                   absl::FixedArray<ssize_t>(0));
  }

  absl::FixedArray<ssize_t> strides(ndim);
  strides[ndim - 1] = 1;
  for (int i = ndim - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }
  return EpisodeFeatureBuffer<T>(ptr, shape, ndim, strides);
}

TEST(EncodeImageForTimestepTest, EncodeGrayScaleImageSuccess) {
  const int num_timesteps = 1;
  const int height = 4, width = 5, channels = 1;
  std::vector<uint8_t> data(num_timesteps * height * width * channels);
  std::iota(data.begin(), data.end(), 0);

  auto buffer =
      CreateTestBuffer(data.data(), {num_timesteps, height, width, channels});
  std::vector<uchar> encoded_image;

  ASSERT_OK(EncodeImageForTimestep(buffer, /*timestep=*/0, &encoded_image));
  ASSERT_FALSE(encoded_image.empty());

  cv::Mat decoded_image = cv::imdecode(encoded_image, cv::IMREAD_GRAYSCALE);
  ASSERT_FALSE(decoded_image.empty());

  EXPECT_EQ(decoded_image.rows, height);
  EXPECT_EQ(decoded_image.cols, width);
  EXPECT_EQ(decoded_image.channels(), 1);

  const int tolerance = 15;
  // Check pixel values with tolerance for JPEG compression
  for (int r = 0; r < height; ++r) {
    for (int c = 0; c < width; ++c) {
      uint8_t original = data[r * width + c];
      uint8_t decoded = decoded_image.at<uint8_t>(r, c);
      EXPECT_NEAR(decoded, original, tolerance);
    }
  }
}

TEST(EncodeImageForTimestepTest, EncodeRGBImageSuccess) {
  const int num_timesteps = 1;
  const int height = 8, width = 8, channels = 3;
  // Image data in R, G, B order
  std::vector<uint8_t> data(num_timesteps * height * width * channels);
  for (int i = 0; i < height * width; ++i) {
    data[i * 3 + 0] = 255;  // R
    data[i * 3 + 1] = 0;    // G
    data[i * 3 + 2] = 0;    // B
  }
  auto buffer =
      CreateTestBuffer(data.data(), {num_timesteps, height, width, channels});
  std::vector<uchar> encoded_image;

  ASSERT_OK(EncodeImageForTimestep(buffer, /*timestep=*/0, &encoded_image));

  ASSERT_FALSE(encoded_image.empty());

  cv::Mat decoded_image_bgr = cv::imdecode(encoded_image, cv::IMREAD_COLOR);
  ASSERT_FALSE(decoded_image_bgr.empty());

  // Corrected code converted RGB {255,0,0} to BGR {0,0,255} before encoding.
  // So, decoded BGR should have B~0, G~0, R~255.
  const int tolerance = 15;
  for (int r = 0; r < decoded_image_bgr.rows; ++r) {
    for (int c = 0; c < decoded_image_bgr.cols; ++c) {
      const auto& pixel = decoded_image_bgr.at<cv::Vec3b>(r, c);
      EXPECT_NEAR(pixel[0], 0, tolerance);    // B
      EXPECT_NEAR(pixel[1], 0, tolerance);    // G
      EXPECT_NEAR(pixel[2], 255, tolerance);  // R
    }
  }
}

TEST(EncodeImageForTimestepTest, EncodeInvalidImageDimensions) {
  const int num_timesteps = 1;
  const int height = 4, width = 5;
  std::vector<uint8_t> data(num_timesteps * height * width);
  auto buffer = CreateTestBuffer(data.data(), {num_timesteps, height, width});
  std::vector<uchar> encoded_image;
  EXPECT_THAT(EncodeImageForTimestep(buffer, /*timestep=*/0, &encoded_image),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "Cannot process image feature. Expecting dimensions 4, "
                       "but got, 3"));
}

TEST(EncodeImageForTimestepTest, EncodeInvalidNumberOfChannels) {
  const int height = 4, width = 5, channels = 4;
  std::vector<uint8_t> data(1 * height * width * channels);
  auto buffer = CreateTestBuffer(data.data(), {1, height, width, channels});
  std::vector<uchar> encoded_image;
  EXPECT_THAT(EncodeImageForTimestep(buffer, /*timestep=*/0, &encoded_image),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "Unsupported number of channels: 4"));
}

TEST(EncodeImageForTimestepTest, EncodeImageAtSpecificTimestepOffset) {
  const int height = 1, width = 2;
  // Timestep 0: {10, 20}
  // Timestep 1: {80, 90}
  std::vector<uint8_t> data = {10, 20, 80, 90};
  auto buffer = CreateTestBuffer(data.data(), {2, height, width, 1});
  std::vector<uchar> encoded_image;

  // Encode Timestep 1
  ASSERT_OK(EncodeImageForTimestep(buffer, /*timestep=*/1, &encoded_image));
  ASSERT_FALSE(encoded_image.empty());

  cv::Mat decoded_image = cv::imdecode(encoded_image, cv::IMREAD_GRAYSCALE);
  ASSERT_FALSE(decoded_image.empty());

  EXPECT_EQ(decoded_image.rows, height);
  EXPECT_EQ(decoded_image.cols, width);

  // Check values from Timestep 1, allowing for JPEG compression
  EXPECT_NEAR(decoded_image.at<uint8_t>(0, 0), 80, 15);
  EXPECT_NEAR(decoded_image.at<uint8_t>(0, 1), 90, 15);
}

TEST(SerializeImageForTimestepTest, SerializeImageSuccess) {
  const int num_timesteps = 1;
  const int height = 4, width = 5, channels = 1;
  std::vector<uint8_t> data(num_timesteps * height * width * channels);
  std::iota(data.begin(), data.end(), 0);

  auto buffer =
      CreateTestBuffer(data.data(), {num_timesteps, height, width, channels});
  tensorflow::Features features;

  ASSERT_OK(SerializeImageForTimestep(/*name=*/"image_data", /*timestep=*/0,
                                      buffer, features));
  ASSERT_TRUE(features.feature().contains("image_data"));
  ASSERT_FALSE(
      features.feature().at("image_data").bytes_list().value().empty());
}

TEST(FillFeatureMapForTimestepTest, CallSerializeImageForTimestep) {
  const int num_timesteps = 1;
  // Create an EpisodeData object with a single timestep.
  std::unique_ptr<EpisodeData> episode_data =
      safari::logging::CreateEpisodeDataForTesting(num_timesteps);

  // Create a test image buffer and add it to the episode data.
  absl::flat_hash_map<std::string, std::variant<EpisodeFeatureBuffer<uint8_t>>>
      data_map;
  const int height = 4, width = 5, channels = 1;
  std::vector<uint8_t> data(num_timesteps * height * width * channels);
  std::iota(data.begin(), data.end(), 0);
  auto buffer =
      CreateTestBuffer(data.data(), {num_timesteps, height, width, channels});
  ASSERT_OK(episode_data->InsertBuffer("test_image", buffer));

  // Call FillFeatureMapForTimestep with the test image key.
  int timestep = 0;
  tensorflow::Features features;
  absl::flat_hash_set<std::string> image_keys = {"test_image"};
  ASSERT_OK(FillFeatureMapForTimestep(timestep, episode_data.get(), image_keys,
                                      features));

  // Serializing an image should add a single bytes_list value to the
  // feature_map
  ASSERT_TRUE(features.feature().contains("test_image"));
  ASSERT_FALSE(
      features.feature().at("test_image").bytes_list().value().empty());
  ASSERT_EQ(features.feature().at("test_image").bytes_list().value().size(), 1);
}

TEST(FillFeatureMapForTimestepTest, CallSerializeNumericDataForTimestep) {
  const int num_timesteps = 1;
  std::unique_ptr<EpisodeData> episode_data =
      safari::logging::CreateEpisodeDataForTesting(num_timesteps);

  // Create a test buffer and add it to the episode data.
  absl::flat_hash_map<std::string, std::variant<EpisodeFeatureBuffer<uint8_t>>>
      data_map;
  const int length = 10;
  std::vector<uint8_t> data(num_timesteps * length);
  std::iota(data.begin(), data.end(), 0);
  auto buffer = CreateTestBuffer(data.data(), {num_timesteps, length});
  ASSERT_OK(episode_data->InsertBuffer("test_uint8_data", buffer));

  tensorflow::Features features;
  ASSERT_OK(FillFeatureMapForTimestep(/*timestep=*/0, episode_data.get(),
                                      /*image_keys=*/{}, features));

  // When serializing numeric data, we would expect the byte list length to be
  // equal to the number of elements in the buffer for a single timestep.
  ASSERT_TRUE(features.feature().contains("test_uint8_data"));
  ASSERT_FALSE(
      features.feature().at("test_uint8_data").bytes_list().value().empty());
  ASSERT_EQ(
      features.feature().at("test_uint8_data").bytes_list().value().size(),
      length);
}

TEST(FillFeatureMapForTimestepTest, CallSerializeStringDataForTimestep) {
  std::unique_ptr<EpisodeData> episode_data =
      safari::logging::CreateEpisodeDataForTesting(/*num_timesteps=*/2);
  // Create a test image buffer and add it to the episode data.
  absl::flat_hash_map<std::string, std::variant<EpisodeFeatureBuffer<uint8_t>>>
      data_map;

  std::vector<std::string> string_data = {"Hello", "World"};
  ASSERT_OK(episode_data->InsertBuffer("test_string", string_data));

  tensorflow::Features features_1;
  ASSERT_OK(FillFeatureMapForTimestep(/*timestep=*/0, episode_data.get(),
                                      /*image_keys=*/{}, features_1));
  EXPECT_THAT(features_1.feature().at("test_string"), EqualsProto(R"pb(
                bytes_list { value: [ "Hello" ] }
              )pb"));

  tensorflow::Features features_2;
  ASSERT_OK(FillFeatureMapForTimestep(/*timestep=*/1, episode_data.get(),
                                      /*image_keys=*/{}, features_2));
  EXPECT_THAT(features_2.feature().at("test_string"), EqualsProto(R"pb(
                bytes_list { value: [ "World" ] }
              )pb"));
}

TEST(FillFeatureMapForTimestepTest,
     CallSerializeStringDataForTimestepEmptyStringVector) {
  std::unique_ptr<EpisodeData> episode_data =
      safari::logging::CreateEpisodeDataForTesting(/*num_timesteps=*/2);
  // Create a test image buffer and add it to the episode data.
  absl::flat_hash_map<std::string, std::variant<EpisodeFeatureBuffer<uint8_t>>>
      data_map;

  std::vector<std::string> string_data = {};
  ASSERT_OK(episode_data->InsertBuffer("test_string", string_data));

  tensorflow::Features features;
  EXPECT_THAT(
      FillFeatureMapForTimestep(/*timestep=*/0, episode_data.get(),
                                /*image_keys=*/{}, features),
      StatusIs(absl::StatusCode::kInvalidArgument,
               "Invalid timestep: 0 when serializing string data. String data "
               "vector size is: 0."));
}

TEST(FillFeatureMapForTimestepTest,
     CallSerializeNumericDataForTimestepUInt16Buffer) {
  const int num_timesteps = 1;
  std::unique_ptr<EpisodeData> episode_data =
      safari::logging::CreateEpisodeDataForTesting(num_timesteps);

  // Create a test buffer and add it to the episode data.
  absl::flat_hash_map<std::string, std::variant<EpisodeFeatureBuffer<uint16_t>>>
      data_map;
  const int length = 10;

  std::vector<uint16_t> data(num_timesteps * length);
  std::iota(data.begin(), data.end(), 0);
  auto buffer = CreateTestBuffer(data.data(), {num_timesteps, length});
  ASSERT_OK(episode_data->InsertBuffer("test_uint16_data", buffer));

  tensorflow::Features features;
  ASSERT_OK(FillFeatureMapForTimestep(/*timestep=*/0, episode_data.get(),
                                      /*image_keys=*/{}, features));
  EXPECT_THAT(features.feature().at("test_uint16_data"), EqualsProto(R"pb(
                int64_list { value: [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ] }
              )pb"));
}

TEST(FillFeatureMapForTimestepTest,
     CallSerializeNumericDataForTimestepUInt32Buffer) {
  const int num_timesteps = 1;
  std::unique_ptr<EpisodeData> episode_data =
      safari::logging::CreateEpisodeDataForTesting(num_timesteps);

  // Create a test buffer and add it to the episode data.
  absl::flat_hash_map<std::string, std::variant<EpisodeFeatureBuffer<uint32_t>>>
      data_map;
  const int length = 10;

  std::vector<uint32_t> data(num_timesteps * length);
  std::iota(data.begin(), data.end(), 0);
  auto buffer = CreateTestBuffer(data.data(), {num_timesteps, length});
  ASSERT_OK(episode_data->InsertBuffer("test_uint32_data", buffer));

  tensorflow::Features features;
  ASSERT_OK(FillFeatureMapForTimestep(/*timestep=*/0, episode_data.get(),
                                      /*image_keys=*/{}, features));
  EXPECT_THAT(features.feature().at("test_uint32_data"), EqualsProto(R"pb(
                int64_list { value: [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ] }
              )pb"));
}

TEST(FillFeatureMapForTimestepTest,
     CCallSerializeNumericDataForTimestepInt32Buffer) {
  const int num_timesteps = 1;
  std::unique_ptr<EpisodeData> episode_data =
      safari::logging::CreateEpisodeDataForTesting(num_timesteps);

  // Create a test buffer and add it to the episode data.
  absl::flat_hash_map<std::string, std::variant<EpisodeFeatureBuffer<int32_t>>>
      data_map;
  const int length = 10;

  std::vector<int32_t> data(num_timesteps * length);
  std::iota(data.begin(), data.end(), 0);
  auto buffer = CreateTestBuffer(data.data(), {num_timesteps, length});
  ASSERT_OK(episode_data->InsertBuffer("test_int32_data", buffer));

  tensorflow::Features features;
  ASSERT_OK(FillFeatureMapForTimestep(/*timestep=*/0, episode_data.get(),
                                      /*image_keys=*/{}, features));
  EXPECT_THAT(features.feature().at("test_int32_data"), EqualsProto(R"pb(
                int64_list { value: [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ] }
              )pb"));
}

TEST(FillFeatureMapForTimestepTest,
     CallSerializeNumericDataForTimestepInt64Buffer) {
  const int num_timesteps = 1;
  std::unique_ptr<EpisodeData> episode_data =
      safari::logging::CreateEpisodeDataForTesting(num_timesteps);

  // Create a test buffer and add it to the episode data.
  absl::flat_hash_map<std::string, std::variant<EpisodeFeatureBuffer<int64_t>>>
      data_map;
  const int length = 10;

  std::vector<int64_t> data(num_timesteps * length);
  std::iota(data.begin(), data.end(), 0);
  auto buffer = CreateTestBuffer(data.data(), {num_timesteps, length});
  ASSERT_OK(episode_data->InsertBuffer("test_int64_data", buffer));

  tensorflow::Features features;
  ASSERT_OK(FillFeatureMapForTimestep(/*timestep=*/0, episode_data.get(),
                                      /*image_keys=*/{}, features));
  EXPECT_THAT(features.feature().at("test_int64_data"), EqualsProto(R"pb(
                int64_list { value: [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ] }
              )pb"));
}

TEST(FillFeatureMapForTimestepTest,
     CallSerializeNumericDataForTimestepFloatBuffer) {
  const int num_timesteps = 1;
  std::unique_ptr<EpisodeData> episode_data =
      safari::logging::CreateEpisodeDataForTesting(num_timesteps);

  // Create a test buffer and add it to the episode data.
  absl::flat_hash_map<std::string, std::variant<EpisodeFeatureBuffer<float>>>
      data_map;
  const int length = 10;

  std::vector<float> data(num_timesteps * length);
  std::iota(data.begin(), data.end(), 0);
  auto buffer = CreateTestBuffer(data.data(), {num_timesteps, length});
  ASSERT_OK(episode_data->InsertBuffer("test_float_data", buffer));

  tensorflow::Features features;
  ASSERT_OK(FillFeatureMapForTimestep(/*timestep=*/0, episode_data.get(),
                                      /*image_keys=*/{}, features));
  EXPECT_THAT(features.feature().at("test_float_data"), EqualsProto(R"pb(
                float_list { value: [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ] }
              )pb"));
}

TEST(FillFeatureMapForTimestepTest,
     CallSerializeNumericDataForTimestepDoubleBuffer) {
  const int num_timesteps = 1;
  std::unique_ptr<EpisodeData> episode_data =
      safari::logging::CreateEpisodeDataForTesting(num_timesteps);

  // Create a test buffer and add it to the episode data.
  absl::flat_hash_map<std::string, std::variant<EpisodeFeatureBuffer<double>>>
      data_map;
  const int length = 10;

  std::vector<double> data(num_timesteps * length);
  std::iota(data.begin(), data.end(), 0);
  auto buffer = CreateTestBuffer(data.data(), {num_timesteps, length});
  ASSERT_OK(episode_data->InsertBuffer("test_float_data", buffer));

  tensorflow::Features features;
  ASSERT_OK(FillFeatureMapForTimestep(/*timestep=*/0, episode_data.get(),
                                      /*image_keys=*/{}, features));
  EXPECT_THAT(features.feature().at("test_float_data"), EqualsProto(R"pb(
                float_list { value: [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ] }
              )pb"));
}

TEST(FillFeatureMapForTimestepTest, NullEpisodeDataReturnsError) {
  tensorflow::Features features;
  EXPECT_THAT(
      FillFeatureMapForTimestep(/*timestep=*/0, /*episode_data=*/nullptr,
                                /*image_keys=*/{}, features),
      StatusIs(absl::StatusCode::kInvalidArgument,
               "Episode data cannot be nullptr"));
}

}  // namespace
}  // namespace safari::logging
