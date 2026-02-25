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

#ifndef SAFARI_LOGGING_CC_EPISODE_DATA_H_
#define SAFARI_LOGGING_CC_EPISODE_DATA_H_

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <absl/container/fixed_array.h>
#include <absl/container/flat_hash_map.h>
#include <absl/status/status.h>
#include <absl/status/statusor.h>
#include <absl/strings/string_view.h>
#include <absl/types/span.h>

namespace safari::logging {

// A generic buffer struct to represent a view into contiguous data in memory.
//
// This struct does not own the underlying data; it only provides a reference
// to existing memory. The data it points to must outlive the
// `EpisodeFeatureBuffer` instance. This is used as a wrapper for the raw
// internal data for a single key (aka feature) of a given episode, across all
// timesteps of the episode.
//
// Primarily used for Numpy arrays, but can also represent other C++ types.
// It should be noted that the timestep is assumed to be the first dimension
// of the underlying data.
template <typename T>
class EpisodeFeatureBuffer {
 public:
  EpisodeFeatureBuffer(T* ptr, absl::FixedArray<ssize_t> shape, int ndim,
                       absl::FixedArray<ssize_t> strides)
      : ptr_(ptr),
        shape_(std::move(shape)),
        ndim_(ndim),
        strides_(std::move(strides)) {
    num_elements_in_timestep_ = NumElementsInTimestep();
  }

  EpisodeFeatureBuffer(const EpisodeFeatureBuffer&) = default;
  EpisodeFeatureBuffer(EpisodeFeatureBuffer&&) = default;

  // Assignments are deleted because we are using absl::FixedArray that
  // does not support assignments.
  EpisodeFeatureBuffer& operator=(const EpisodeFeatureBuffer&) = delete;
  EpisodeFeatureBuffer& operator=(EpisodeFeatureBuffer&&) = delete;

  T* ptr() const { return ptr_; }
  absl::FixedArray<ssize_t> shape() const { return shape_; }
  int ndim() const { return ndim_; }
  absl::FixedArray<ssize_t> strides() const { return strides_; }
  int64_t num_elements_in_timestep() const { return num_elements_in_timestep_; }

 private:
  // Calculates the number of elements in a timestep.
  // This is the product of all dimensions of the array except the first
  // dimension (which is the timestep dimension).
  // Iterating over this number of elements for a contiguous array will be
  // equivalent to iterating over a flattened slice of the original array at
  // each timestep.
  int64_t NumElementsInTimestep() {
    int64_t num_elements_in_timestep = 1;
    // Start from 1, to skip the timestamp dimension.
    // Note that ssize_t is system dependent. So we specifically cast to
    // int64_t.
    for (ssize_t i = 1; i < shape_.size(); ++i) {
      num_elements_in_timestep *= static_cast<int64_t>(shape_[i]);
    }
    return num_elements_in_timestep;
  }
  // Pointer to the underlying data.
  // The EpisodeFeatureBuffer does not own this pointer, and is not responsible
  // for managing its memory. The lifetime of the data pointed to by `ptr`
  // must outlive this EpisodeFeatureBuffer instance.
  T* ptr_;
  // Shape of the data.
  absl::FixedArray<ssize_t> shape_;
  // Number of dimensions.
  int ndim_;
  // Strides for each dimension.
  absl::FixedArray<ssize_t> strides_;
  // Number of elements in a timestep.
  int64_t num_elements_in_timestep_;
};

// Unsigned 64-bit is not supported in EpisodeFeatureBufferVariant because
// tensorflow.Int64List.value, used for serializing integer data, is a
// signed int64 field (see cs/symbol:tensorflow.Int64List.value). This
// limits the integer types that can be used in EpisodeFeatureBuffer to
// signed integers to prevent issues interpreting the original data when
// serializing/deserializing.
using EpisodeFeatureBufferVariant =
    std::variant<EpisodeFeatureBuffer<uint8_t>, EpisodeFeatureBuffer<uint16_t>,
                 EpisodeFeatureBuffer<uint32_t>, EpisodeFeatureBuffer<int32_t>,
                 EpisodeFeatureBuffer<int64_t>, EpisodeFeatureBuffer<float>,
                 EpisodeFeatureBuffer<double>, std::vector<std::string>>;

// Virtual class that is used to pass the original underlying episode data
// to the EpisodeData wrapper class.
// Primarily used to pass Python generated data to C++.
class EpisodeDataPayloadInterface {
 public:
  virtual ~EpisodeDataPayloadInterface() = default;
};

// A class that wraps the original episode data and provides a convenient
// interface to access the data.
// This class encapsulates the data of an episode across timesteps.
class EpisodeData {
 public:
  // Copy constructor and copy assignment are deleted to enforce unique
  // ownership of the payload_. Move semantics are supported.
  EpisodeData(const EpisodeData&) = delete;
  EpisodeData& operator=(const EpisodeData&) = delete;
  EpisodeData(EpisodeData&&) = default;
  EpisodeData& operator=(EpisodeData&&) = default;

  static absl::StatusOr<std::unique_ptr<EpisodeData>> Create(
      std::unique_ptr<EpisodeDataPayloadInterface> payload,
      std::vector<int64_t> timestamps);

  // Add key-value pair to the internal map.
  absl::Status InsertBuffer(absl::string_view key,
                            EpisodeFeatureBufferVariant buffer);

  const absl::flat_hash_map<std::string, EpisodeFeatureBufferVariant>&
  data_map() const {
    return data_map_;
  }

  absl::Span<const int64_t> timestamps() const { return timestamps_; }

 private:
  EpisodeData(std::unique_ptr<EpisodeDataPayloadInterface> payload,
              std::vector<int64_t> timestamps)
      : payload_(std::move(payload)), timestamps_(std::move(timestamps)) {}

  // Wrapper around the original episode data owned by Python.
  std::unique_ptr<EpisodeDataPayloadInterface> payload_;
  // A map containing the data buffers of the original data.
  absl::flat_hash_map<std::string, EpisodeFeatureBufferVariant> data_map_;

  // The Unix epoch timestamps in nanoseconds for each timestep in the episode.
  std::vector<int64_t> timestamps_;
};

}  // namespace safari::logging
#endif  // SAFARI_LOGGING_CC_EPISODE_DATA_H_
