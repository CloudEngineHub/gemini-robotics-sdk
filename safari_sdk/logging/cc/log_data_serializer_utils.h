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

#ifndef SAFARI_LOGGING_CC_LOG_DATA_SERIALIZER_UTILS_H_
#define SAFARI_LOGGING_CC_LOG_DATA_SERIALIZER_UTILS_H_

#include <cstdint>
#include <string>
#include <vector>

#include <opencv2/core/hal/interface.h>
#include <absl/container/flat_hash_set.h>
#include <absl/log/log.h>
#include <absl/strings/string_view.h>
#include <absl/types/span.h>
#include "safari_sdk/logging/cc/episode_data.h"
#include <tensorflow/core/example/feature.pb.h>

namespace safari::logging {

// Default JPEG encoding quality.
constexpr int kJpegQualityDefault = 80;
// Default JPEG encoding extension.
constexpr char kJpegExtensionDefault[] = ".jpg";

// The expected dimensions of an image feature.
constexpr int kImageFeatureDimensions = 4;
// The expected number of channels for a grayscale image.
constexpr int kGrayscaleImageChannels = 1;
// The expected number of channels for a RGB image.
constexpr int kRGBImageChannels = 3;

// Compresses a given image for a single timestep and stores it in the memory
// buffer. The data is expected to be in the format of HWC (height, width,
// channels) in RGB format. Uses OpenCV imencode to perform the encoding. NOTE:
// As the number of timesteps in an episode get larger, we become more at risk
// of overflow and incorrectly accessing memory.
absl::Status EncodeImageForTimestep(EpisodeFeatureBuffer<uint8_t> buffer,
                                    int timestep,
                                    std::vector<uchar>* encoded_image);

// Serializes an image for a single timestep to a tensorflow::Feature proto and
// stores it in the feature_map under the given name.
absl::Status SerializeImageForTimestep(absl::string_view name, int timestep,
                                       EpisodeFeatureBuffer<uint8_t> buffer,
                                       tensorflow::Features& features);

// Serializes numeric data for a single timestep to a tensorflow::Feature proto
// and stores it in the feature_map under the given name.
template <typename T>
absl::Status SerializeNumericDataForTimestep(absl::string_view name,
                                             int timestep,
                                             EpisodeFeatureBuffer<T> buffer,
                                             tensorflow::Features& features);

// Serializes string data for a single timestep to a tensorflow::Feature proto
// and stores it in the feature_map under the given name.
absl::Status SerializeStringDataForTimestep(
    absl::string_view name, int timestep,
    absl::Span<const std::string> string_data, tensorflow::Features& features);

// Adds data to the feature_map under the given name.
// Calls the appropriate serialization method based on the data type.
absl::Status FillFeatureMapForTimestep(
    int timestep, const EpisodeData* episode_data,
    const absl::flat_hash_set<std::string>& image_keys,
    tensorflow::Features& features);

}  // namespace safari::logging
#endif  // SAFARI_LOGGING_CC_LOG_DATA_SERIALIZER_UTILS_H_
