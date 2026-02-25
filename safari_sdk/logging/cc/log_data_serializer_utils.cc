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
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <absl/container/flat_hash_set.h>
#include <absl/log/log.h>
#include <absl/status/status.h>
#include <absl/strings/str_cat.h>
#include <absl/strings/string_view.h>
#include <absl/types/span.h>
#include "safari_sdk/logging/cc/episode_data.h"
#include <tensorflow/core/example/feature.pb.h>

namespace safari::logging {

absl::Status EncodeImageForTimestep(EpisodeFeatureBuffer<uint8_t> buffer,
                                    int timestep,
                                    std::vector<uchar>* encoded_image) {
  if (buffer.ndim() != kImageFeatureDimensions) {
    return absl::InvalidArgumentError(
        absl::StrCat("Cannot process image feature. Expecting dimensions ",
                     kImageFeatureDimensions, ", but got, ", buffer.ndim()));
  }

  int64_t rows = buffer.shape()[1];
  int64_t cols = buffer.shape()[2];
  int64_t channels = buffer.shape()[3];

  // Determine the correct OpenCV type based on the number of channels
  int cv_type;
  if (channels == kGrayscaleImageChannels) {  // 1 channel
    cv_type = CV_8UC1;
  } else if (channels == kRGBImageChannels) {  // 3 channels
    cv_type = CV_8UC3;
  } else {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported number of channels: ", channels));
  }

  // The memory offset for the image data at a given timestep.
  int64_t offset = timestep * buffer.strides()[0];

  // Initializes the image matrix with the underlying raw data buffer without
  // a copy. Using buffer.strides[1] (row stride) is generally recommended when
  // constructing cv::Mat from external data to ensure correct row-major access.
  cv::Mat image(rows, cols, cv_type,
                const_cast<uint8_t*>(buffer.ptr() + offset),
                buffer.strides()[1]);

  const std::vector<int> encode_options = {cv::IMWRITE_JPEG_QUALITY,
                                           kJpegQualityDefault};
  if (channels == kRGBImageChannels) {
    // OpenCV assumes images are BGR, so we need to convert from RGB.
    // This creates a copy of the image, which is then encoded. This is done to
    // avoid modifying the original image data.
    cv::Mat image_bgr;
    cv::cvtColor(image, image_bgr, cv::COLOR_RGB2BGR);
    cv::imencode(kJpegExtensionDefault, image_bgr, *encoded_image,
                 encode_options);
  } else {
    cv::imencode(kJpegExtensionDefault, image, *encoded_image, encode_options);
  }

  return absl::OkStatus();
}

// Serializes an image for a single timestep to a tensorflow::Feature proto and
// stores it in the feature_map under the given name.
absl::Status SerializeImageForTimestep(absl::string_view name, int timestep,
                                       EpisodeFeatureBuffer<uint8_t> buffer,
                                       tensorflow::Features& features) {
  std::vector<uchar> encoded_image;
  absl::Status status =
      EncodeImageForTimestep(buffer, timestep, &encoded_image);
  if (!status.ok()) {
    return status;
  }

  tensorflow::Feature feature;
  tensorflow::BytesList* bytes_list = feature.mutable_bytes_list();
  std::string encoded_image_string(encoded_image.begin(), encoded_image.end());
  bytes_list->add_value(std::move(encoded_image_string));

  (*features.mutable_feature())[name] = feature;
  return absl::OkStatus();
}

template <typename T>
absl::Status SerializeNumericDataForTimestep(absl::string_view name,
                                             int timestep,
                                             EpisodeFeatureBuffer<T> buffer,
                                             tensorflow::Features& features) {
  tensorflow::Feature feature;

  int64_t length = buffer.num_elements_in_timestep();
  // We use Byteslist for uint8_t - storing a single char for enum types.
  if constexpr (std::is_same_v<T, uint8_t>) {
    tensorflow::BytesList* bytes_list = feature.mutable_bytes_list();
    bytes_list->mutable_value()->Reserve(length);

    for (int64_t i = 0; i < length; ++i) {
      char value = static_cast<char>(buffer.ptr()[timestep * length + i]);
      bytes_list->add_value(std::string(1, value));
    }
  } else if constexpr (std::is_integral_v<T>) {
    tensorflow::Int64List* int64_list = feature.mutable_int64_list();
    int64_list->mutable_value()->Reserve(length);

    for (int64_t i = 0; i < length; ++i) {
      int64_t value = static_cast<int64_t>(buffer.ptr()[timestep * length + i]);
      int64_list->add_value(value);
    }
  } else if constexpr (std::is_floating_point_v<T>) {
    tensorflow::FloatList* float_list = feature.mutable_float_list();
    float_list->mutable_value()->Reserve(length);

    for (int64_t i = 0; i < length; ++i) {
      float value = static_cast<float>(buffer.ptr()[timestep * length + i]);
      float_list->add_value(value);
    }
  } else {
    return absl::InvalidArgumentError("Unsupported type");
  }

  (*features.mutable_feature())[name] = feature;
  return absl::OkStatus();
}

absl::Status SerializeStringDataForTimestep(
    absl::string_view name, int timestep,
    absl::Span<const std::string> string_data, tensorflow::Features& features) {
  tensorflow::Feature feature;
  tensorflow::BytesList* bytes_list = feature.mutable_bytes_list();

  if (timestep >= string_data.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Invalid timestep: ", timestep,
        " when serializing string data. String data vector size is: ",
        string_data.size(), "."));
  }

  bytes_list->add_value(string_data.at(timestep));

  (*features.mutable_feature())[name] = feature;
  return absl::OkStatus();
}

absl::Status FillFeatureMapForTimestep(
    int timestep, const EpisodeData* episode_data,
    const absl::flat_hash_set<std::string>& image_keys,
    tensorflow::Features& features) {
  if (episode_data == nullptr) {
    return absl::InvalidArgumentError("Episode data cannot be nullptr");
  }

  // Iterate over each key and value in the map.
  for (const auto& [key, value] : episode_data->data_map()) {
    if (std::holds_alternative<EpisodeFeatureBuffer<uint8_t>>(value)) {
      EpisodeFeatureBuffer<uint8_t> buffer =
          std::get<EpisodeFeatureBuffer<uint8_t>>(value);
      // Differentiate between image and non-image data.
      if (image_keys.contains(key)) {
        absl::Status status =
            SerializeImageForTimestep(key, timestep, buffer, features);
        if (!status.ok()) {
          return status;
        }
      } else {
        absl::Status status =
            SerializeNumericDataForTimestep(key, timestep, buffer, features);
        if (!status.ok()) {
          return status;
        }
      }
    }
    if (std::holds_alternative<std::vector<std::string>>(value)) {
      const std::vector<std::string>& string_data =
          std::get<std::vector<std::string>>(value);

      absl::Status status =
          SerializeStringDataForTimestep(key, timestep, string_data, features);
      if (!status.ok()) {
        return status;
      }
    }
    if (std::holds_alternative<EpisodeFeatureBuffer<uint16_t>>(value)) {
      EpisodeFeatureBuffer<uint16_t> buffer =
          std::get<EpisodeFeatureBuffer<uint16_t>>(value);
      absl::Status status =
          SerializeNumericDataForTimestep(key, timestep, buffer, features);
      if (!status.ok()) {
        return status;
      }
    }
    if (std::holds_alternative<EpisodeFeatureBuffer<uint32_t>>(value)) {
      EpisodeFeatureBuffer<uint32_t> buffer =
          std::get<EpisodeFeatureBuffer<uint32_t>>(value);
      absl::Status status =
          SerializeNumericDataForTimestep(key, timestep, buffer, features);
      if (!status.ok()) {
        return status;
      }
    }
    if (std::holds_alternative<EpisodeFeatureBuffer<int32_t>>(value)) {
      EpisodeFeatureBuffer<int32_t> buffer =
          std::get<EpisodeFeatureBuffer<int32_t>>(value);
      absl::Status status =
          SerializeNumericDataForTimestep(key, timestep, buffer, features);
      if (!status.ok()) {
        return status;
      }
    }
    if (std::holds_alternative<EpisodeFeatureBuffer<int64_t>>(value)) {
      EpisodeFeatureBuffer<int64_t> buffer =
          std::get<EpisodeFeatureBuffer<int64_t>>(value);
      absl::Status status =
          SerializeNumericDataForTimestep(key, timestep, buffer, features);
      if (!status.ok()) {
        return status;
      }
    }
    if (std::holds_alternative<EpisodeFeatureBuffer<float>>(value)) {
      EpisodeFeatureBuffer<float> buffer =
          std::get<EpisodeFeatureBuffer<float>>(value);
      absl::Status status =
          SerializeNumericDataForTimestep(key, timestep, buffer, features);
      if (!status.ok()) {
        return status;
      }
    }
    if (std::holds_alternative<EpisodeFeatureBuffer<double>>(value)) {
      EpisodeFeatureBuffer<double> buffer =
          std::get<EpisodeFeatureBuffer<double>>(value);
      absl::Status status =
          SerializeNumericDataForTimestep(key, timestep, buffer, features);
      if (!status.ok()) {
        return status;
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace safari::logging
