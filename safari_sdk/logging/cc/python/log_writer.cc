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

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <absl/log/check.h>
#include <absl/log/log.h>
#include <absl/status/status.h>
#include <absl/status/statusor.h>
#include <absl/strings/str_cat.h>
#include <absl/strings/string_view.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11_abseil/absl_casters.h>
#include <pybind11_abseil/status_casters.h>
#include <pybind11_protobuf/native_proto_caster.h>
#include "safari_sdk/logging/cc/episode_data.h"
#include "safari_sdk/logging/cc/thread_pool_log_writer.h"
#include "safari_sdk/protos/logging/metadata.pb.h"

namespace safari::logging {

constexpr int kTimestampsNumpyArrayDimensions = 1;

// All data should at least have the time dimension.
constexpr int kMinNumpyArrayDimensions = 1;

// Wrapper class that stores the pybind11 reference to the underlying Python
// dictionary. This keeps the object alive while we need to process it.
class PythonEpisodeDataPayload : public EpisodeDataPayloadInterface {
 public:
  explicit PythonEpisodeDataPayload(pybind11::dict payload)
      : payload_(payload) {}

  ~PythonEpisodeDataPayload() override {
    pybind11::gil_scoped_acquire gil_scoped_acquire;

    CHECK(PyGILState_Check())
        << "PythonEpisodeDataPayload destructor needs the "
           "GIL to destroy pybind11::dict";

    // Manually decrease the ref count of the dictionary while the destructor
    // has the GIL.
    payload_.dec_ref();
    // Sets the raw ptr to nullptr in C++.
    payload_.release();
  }

  pybind11::dict payload() { return payload_; }

 private:
  // Lightweight wrapper around the underlying episode data created in Python.
  pybind11::dict payload_;
};

// Helper class for the C++ ThreadPoolLogWriter class, that is exposed to Python
// through pybind11.
class LogWriterHelper {
 public:
  static std::unique_ptr<ThreadPoolLogWriter> CreateLogWriter(
      ThreadPoolLogWriterConfig config) {
    absl::StatusOr<std::unique_ptr<ThreadPoolLogWriter>> log_writer =
        ThreadPoolLogWriter::Create(std::move(config));

    if (!log_writer.ok()) {
      LOG(ERROR) << "Failed to create ThreadPoolLogWriter: "
                 << log_writer.status();
      return nullptr;
    }
    // Starts background threads.
    (*log_writer)->Start();
    return std::move(*log_writer);
  }

  static absl::Status FinalizeEpisode(ThreadPoolLogWriter* this_,
                                      absl::string_view episode_uuid) {
    return this_->FinalizeEpisode(episode_uuid);
  }

  // Enqueues episode data to the ThreadPoolLogWriter.
  static absl::Status EnqueueEpisodeData(ThreadPoolLogWriter* this_,
                                         pybind11::dict payload,
                                         pybind11::array_t<int64_t> timestamps,
                                         EnqueueMcapFileOptions options) {
    auto payload_interface =
        std::make_unique<PythonEpisodeDataPayload>(payload);

    auto episode_data =
        CreateEpisodeData(std::move(payload_interface), timestamps);

    if (!episode_data.ok()) {
      return episode_data.status();
    }
    this_->EnqueueEpisodeData(std::move(*episode_data), std::move(options));
    return absl::OkStatus();
  }

  // Enqueues Session proto to the ThreadPoolLogWriter.
  static absl::Status EnqueueSessionData(
      ThreadPoolLogWriter* this_, safari_sdk::protos::logging::Session session,
      EnqueueMcapFileOptions options) {
    this_->EnqueueSessionData(std::move(session), std::move(options));
    return absl::OkStatus();
  }

  // Starts the ThreadPoolLogWriter.
  // Not thread safe.
  static void Start(ThreadPoolLogWriter* this_) { this_->Start(); }

  // Stops the ThreadPoolLogWriter.
  // Not thread safe.
  static void Stop(ThreadPoolLogWriter* this_) { this_->Stop(); }

 private:
  static absl::StatusOr<std::unique_ptr<EpisodeData>> CreateEpisodeData(
      std::unique_ptr<PythonEpisodeDataPayload> payload,
      pybind11::array_t<int64_t> timestamps) {
    if (payload == nullptr) {
      return absl::InvalidArgumentError(
          "PythonEpisodeDataPayload cannot be nullptr when creating "
          "EpisodeData. PythonEpisodeDataPayload should wrap the Python "
          "dictionary holding the original episode data.");
    }

    const pybind11::dict payload_dict = payload->payload();

    absl::StatusOr<std::vector<int64_t>> timestamps_vector =
        TimestampsToVector(timestamps);
    if (!timestamps_vector.ok()) {
      return timestamps_vector.status();
    }
    auto episode_data =
        EpisodeData::Create(std::move(payload), std::move(*timestamps_vector));
    if (!episode_data.ok()) {
      return episode_data.status();
    }

    // Fill the EpisodeData with the data from the Python dictionary.
    // The Python dictionary must be a dictionary of Numpy arrays.
    // We copy the key but we will create a Buffer for each numpy array
    for (const auto& [key, value] : payload_dict) {
      std::string key_str = key.cast<std::string>();
      if (value.is_none() || !pybind11::isinstance<pybind11::array>(value)) {
        return absl::InvalidArgumentError(
            absl::StrCat("Each value in the payload dictionary must be a Numpy "
                         "array. Value is not a Numpy array at key: ",
                         key_str));
      }

      pybind11::array array = value.cast<pybind11::array>();

      absl::StatusOr<EpisodeFeatureBufferVariant> buffer_variant =
          CreateEpisodeFeatureBufferVariant(array);

      if (!buffer_variant.ok()) {
        return absl::Status(buffer_variant.status().code(),
                            absl::StrCat(buffer_variant.status().message(),
                                         "; Key: ", key_str));
      }

      absl::Status status =
          (*episode_data)->InsertBuffer(key_str, *buffer_variant);
      if (!status.ok()) {
        return status;
      }
    }

    return std::move(*episode_data);
  }

  // Helper function to convert a pybind11::array_t<int64_t> to a
  // std::vector<int64_t>. Performs a full copy of the data.
  static absl::StatusOr<std::vector<int64_t>> TimestampsToVector(
      pybind11::array_t<int64_t> timestamps) {
    pybind11::buffer_info buf = timestamps.request();
    if (buf.ndim != kTimestampsNumpyArrayDimensions) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Number of dimensions must be ", kTimestampsNumpyArrayDimensions));
    }
    int64_t* ptr = static_cast<int64_t*>(buf.ptr);
    size_t size = buf.shape[0];
    return std::vector<int64_t>(ptr, ptr + size);
  }

  static absl::StatusOr<EpisodeFeatureBufferVariant>
  CreateEpisodeFeatureBufferVariant(pybind11::array array) {
    CHECK(PyGILState_Check()) << "CreateEpisodeFeatureBufferVariant needs the "
                                 "GIL to process the Python dictionary.";
    if (array.ndim() < kMinNumpyArrayDimensions) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Array has ndim = ", array.ndim(),
          " which is less than the minimum number of dimensions expected = ",
          kMinNumpyArrayDimensions));
    }

    // Create a BufferVariant based on the dtype of the array.
    pybind11::dtype dtype = array.dtype();

    // Numpy arrays which contain variable length strings must be handled
    // differently to other data types. We must perform a full copy of the
    // data into a std::vector<std::string>
    bool is_string_dtype =
        dtype.kind() == 'U' || dtype.kind() == 'O' || dtype.kind() == 'S';
    if (is_string_dtype) {
      return CreateStringEpisodeFeatureBufferVariant(array);
    }

    // For numeric data, we directly use the buffer protocol and create a
    // Buffer<T> from the pybind11::buffer_info.
    return CreateNumericalEpisodeFeatureBufferVariant(array);
  }

  static absl::StatusOr<std::vector<std::string>>
  CreateStringEpisodeFeatureBufferVariant(pybind11::array array) {
    // String arrays should have shape (num_timestamps,)
    if (array.ndim() != kMinNumpyArrayDimensions) {
      return absl::InvalidArgumentError(absl::StrCat(
          "String array has ndim = ", array.ndim(),
          " which is not equal to the number of dimensions expected = ",
          kMinNumpyArrayDimensions));
    }

    // Cast to array_t to index into the array.
    pybind11::array_t<pybind11::object> object_array;
    try {
      object_array = array.cast<pybind11::array_t<pybind11::object>>();
    } catch (const pybind11::cast_error& e) {
      return absl::InvalidArgumentError(
          "Failed to cast Numpy array to array_t");
    }

    int num_timestamps = object_array.shape(0);
    std::vector<std::string> buffer(num_timestamps);

    for (ssize_t i = 0; i < num_timestamps; ++i) {
      // This gets the ith element of the array.
      // This will be a string numpy array of shape=()
      pybind11::object element;
      if (pybind11::isinstance<pybind11::array>(object_array.at(i))) {
        // If the element is an array, we need to use tuple indexing.
        element = object_array.at(i)[pybind11::tuple()];
      } else {
        element = object_array.at(i);
      }

      try {
        buffer[i] = element.cast<std::string>();
      } catch (const pybind11::cast_error& e) {
        return absl::InvalidArgumentError(
            "Failed to cast element to std::string");
      }
    }
    return buffer;
  }

  // Helper function to create a EpisodeFeatureBuffer<T> from a
  // pybind11::buffer_info object.
  template <typename T>
  static absl::StatusOr<EpisodeFeatureBufferVariant> CreateEpisodeFeatureBuffer(
      const pybind11::buffer_info& array_buffer) {
    absl::FixedArray<ssize_t> shape(array_buffer.shape.begin(),
                                    array_buffer.shape.end());

    absl::FixedArray<ssize_t> strides(array_buffer.strides.begin(),
                                      array_buffer.strides.end());

    return EpisodeFeatureBuffer<T>(
        static_cast<T*>(array_buffer.ptr), std::move(shape),
        static_cast<int>(array_buffer.ndim), std::move(strides));
  }

  static absl::StatusOr<EpisodeFeatureBufferVariant>
  CreateNumericalEpisodeFeatureBufferVariant(pybind11::array array) {
    pybind11::dtype dtype = array.dtype();
    pybind11::buffer_info array_buffer = array.request();

    bool is_c_contiguous = array.flags() & pybind11::array::c_style;
    if (!is_c_contiguous) {
      return absl::InvalidArgumentError(
          "Array is not C contiguous. Only C contiguous arrays are supported. "
          "Note that this is the default for Numpy arrays. However, passing an "
          "input that is a slice or a view of the original array will not be C "
          "contiguous.");
    }

    if (dtype.is(pybind11::dtype::of<uint8_t>())) {
      auto buffer = CreateEpisodeFeatureBuffer<uint8_t>(array_buffer);
      if (!buffer.ok()) {
        return std::move(buffer).status();
      } else {
        return *std::move(buffer);
      }
    } else if (dtype.is(pybind11::dtype::of<uint16_t>())) {
      auto buffer = CreateEpisodeFeatureBuffer<uint16_t>(array_buffer);
      if (!buffer.ok()) {
        return std::move(buffer).status();
      } else {
        return *std::move(buffer);
      }
    } else if (dtype.is(pybind11::dtype::of<uint32_t>())) {
      auto buffer = CreateEpisodeFeatureBuffer<uint32_t>(array_buffer);
      if (!buffer.ok()) {
        return std::move(buffer).status();
      } else {
        return *std::move(buffer);
      }
    } else if (dtype.is(pybind11::dtype::of<int32_t>())) {
      auto buffer = CreateEpisodeFeatureBuffer<int32_t>(array_buffer);
      if (!buffer.ok()) {
        return std::move(buffer).status();
      } else {
        return *std::move(buffer);
      }
    } else if (dtype.is(pybind11::dtype::of<int64_t>())) {
      auto buffer = CreateEpisodeFeatureBuffer<int64_t>(array_buffer);
      if (!buffer.ok()) {
        return std::move(buffer).status();
      } else {
        return *std::move(buffer);
      }
    } else if (dtype.is(pybind11::dtype::of<float>())) {
      auto buffer = CreateEpisodeFeatureBuffer<float>(array_buffer);
      if (!buffer.ok()) {
        return std::move(buffer).status();
      } else {
        return *std::move(buffer);
      }
    } else if (dtype.is(pybind11::dtype::of<double>())) {
      auto buffer = CreateEpisodeFeatureBuffer<double>(array_buffer);
      if (!buffer.ok()) {
        return std::move(buffer).status();
      } else {
        return *std::move(buffer);
      }
    }
    return absl::InvalidArgumentError(
        absl::StrCat("Input array has unsupported type: ",
                     std::to_string(array.dtype().kind())));
  }
};

namespace {
PYBIND11_MODULE(log_writer, m) {
  ::pybind11::google::ImportStatusModule();
  // Configuration for Mcap files.
  pybind11::class_<McapFileConfig>(m, "McapFileConfig")
      .def(pybind11::init<std::string, std::string, std::string, std::string,
                          int64_t>(),
           pybind11::arg("output_dir"), pybind11::arg("filename_prefix"),
           pybind11::arg("file_metadata_topic"), pybind11::arg("agent_id"),
           pybind11::arg("file_shard_size_limit_bytes"))
      .def_readwrite("output_dir", &McapFileConfig::output_dir)
      .def_readwrite("file_metadata_topic",
                     &McapFileConfig::file_metadata_topic)
      .def_readwrite("agent_id", &McapFileConfig::agent_id)
      .def_readwrite("file_shard_size_limit_bytes",
                     &McapFileConfig::file_shard_size_limit_bytes);
  // Configuration for the ThreadPoolLogWriter.
  pybind11::class_<ThreadPoolLogWriterConfig>(m, "ThreadPoolLogWriterConfig")
      .def(pybind11::init<int64_t, std::vector<std::string>, McapFileConfig>(),
           pybind11::arg("max_num_workers"),
           pybind11::arg("image_observation_keys"),
           pybind11::arg("mcap_file_config"))
      .def_readwrite("max_num_workers",
                     &ThreadPoolLogWriterConfig::max_num_workers)
      .def_readwrite("image_observation_keys",
                     &ThreadPoolLogWriterConfig::image_observation_keys)
      .def_readwrite("mcap_file_config",
                     &ThreadPoolLogWriterConfig::mcap_file_config);
  // Factory method for creating a LogWriter.
  m.def("create_log_writer", &LogWriterHelper::CreateLogWriter,
        pybind11::arg("config"));
  pybind11::class_<ThreadPoolLogWriter>(m, "LogWriter")
      // Once the python dictionary is passed to enqueue_episode_data(),
      // reassign it. Don't call clear() as this will have undefined behaviour.
      // The GIL must be acquired during this function call.
      .def("enqueue_episode_data", &LogWriterHelper::EnqueueEpisodeData,
           pybind11::arg("payload"), pybind11::arg("timestamps"),
           pybind11::arg("options"))
      .def("enqueue_session_data", &LogWriterHelper::EnqueueSessionData,
           pybind11::arg("session"), pybind11::arg("options"))
      // The GIL is released when stop is called so that the destructor of the
      // PythonEpisodeDataPayload can acquire the GIL and manage the
      // pybind11::dict reference count to the Python dictionary.
      .def("stop", &LogWriterHelper::Stop,
           pybind11::call_guard<pybind11::gil_scoped_release>())
      .def("start", &LogWriterHelper::Start)
      .def("finalize_episode", &LogWriterHelper::FinalizeEpisode,
           pybind11::arg("episode_uuid"));
  // Class for passing metadata to the EnqueueEpisodeData and EnqueueSessionData
  // methods.
  pybind11::class_<EnqueueMcapFileOptions>(m, "EnqueueMcapFileOptions")
      .def(pybind11::init<std::string, std::string, int64_t>(),
           pybind11::arg("episode_uuid"), pybind11::arg("topic"),
           pybind11::arg("timestamp_ns"))
      .def_readwrite("episode_uuid", &EnqueueMcapFileOptions::episode_uuid)
      .def_readwrite("topic", &EnqueueMcapFileOptions::topic)
      .def_readwrite("timestamp_ns", &EnqueueMcapFileOptions::timestamp_ns);
}
}  // namespace

}  // namespace safari::logging
