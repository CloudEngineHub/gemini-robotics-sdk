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

#ifndef SAFARI_LOGGING_CC_MCAP_FILE_HANDLE_H_
#define SAFARI_LOGGING_CC_MCAP_FILE_HANDLE_H_

#include <cstdint>
#include <limits>
#include <memory>
#include <string>

#include <absl/base/attributes.h>
#include <absl/container/flat_hash_map.h>
#include <absl/status/status.h>
#include <absl/status/statusor.h>
#include <absl/strings/string_view.h>
#include <mcap/types.hpp>
#include <mcap/writer.hpp>
#include <google/protobuf/descriptor.h>
#include "safari_sdk/logging/cc/mcap_write_op.h"

namespace safari::logging {

std::string GetFinalDirectory(absl::string_view output_dir);

// Base class for McapFileHandle.
// This class is used to create McapFileHandle objects and is used to allow
// for mocking of the McapFileHandle in tests.
class BaseMcapFileHandle {
 public:
  virtual absl::Status WriteMessage(const McapWriteOp& op) = 0;
  virtual ~BaseMcapFileHandle() = default;
  virtual int64_t total_messages_size_bytes() const = 0;
  virtual int64_t shard_index() const = 0;
  virtual int64_t last_publish_time_ns() const = 0;
};

// Class that is used to encapsulate the state of a single MCAP file.
// This class is thread-compatible. Calls to WriteMessage() and Create() are
// expected to be synchronized by the caller.
class McapFileHandle : public BaseMcapFileHandle {
 public:
  static absl::StatusOr<std::unique_ptr<McapFileHandle>> Create(
      absl::string_view episode_uuid, int64_t shard_index,
      int64_t first_publish_time_ns,
      const McapFileConfig* config ABSL_ATTRIBUTE_LIFETIME_BOUND);

  // McapFileHandle is not copyable or movable.
  // This is to manage the lifetime of the writer.
  McapFileHandle(const McapFileHandle&) = delete;
  McapFileHandle& operator=(const McapFileHandle&) = delete;

  ~McapFileHandle() override;

  // Writes a message to the MCAP file corresponding this file handle, at
  // filename_.
  // This method calls RegisterSchemaAndChannels() to register the schema
  // and channels for the topic and descriptor in the op if they have not
  // been registered yet.
  absl::Status WriteMessage(const McapWriteOp& op) override;

  int64_t total_messages_size_bytes() const override;

  int64_t shard_index() const override;

  int64_t last_publish_time_ns() const override;

 private:
  McapFileHandle(absl::string_view filename, absl::string_view tmp_file_path,
                 int64_t shard_index, int64_t first_publish_time_ns,
                 const McapFileConfig* config,
                 std::unique_ptr<mcap::McapWriter> writer);

  // Closes the MCAP writer, flushing any pending messages to the file.
  // Also moves the file to the final location and removes write permissions for
  // all users.
  // The final file path is constructed as:
  // {config_->output_dir}/{YYYY}/{MM}/{DD}/{filename_}
  void Close();

  // Registers a schema and channels for the given topic and descriptor.
  // Returns the channel id for the given topic.
  mcap::ChannelId RegisterSchemaAndChannels(
      absl::string_view topic, const google::protobuf::Descriptor* descriptor);

  // Creates a file metadata proto and writes it to the MCAP file.
  // This is called when the McapFileHandle is destroyed, as each file must have
  // a file metadata proto at the end of the file.
  void CreateFileMetadataProto();

  // The filename of the MCAP file, without the full path.
  // For example, if the full path is
  // /output_dir/2024/12/12/episode_1234567890_shard0.mcap, then the filename is
  // episode_1234567890_shard0.mcap.
  std::string filename_;

  // The temporary file path to write to before moving to the final location.
  // This is to prevent the file from being used when it is partially written.
  // The final file path is constructed as:
  // {output_dir}/tmp/{filename_}
  std::string tmp_file_path_;

  // Configuration for creating the MCAP file.
  // This is not owned by the McapFileHandle and must outlive it.
  const McapFileConfig* config_;

  // The MCAP writer for the MCAP file.
  std::unique_ptr<mcap::McapWriter> writer_;
  // A map of topic to channel id for channels that are registered to the
  // MCAP writer.
  absl::flat_hash_map<std::string, mcap::ChannelId> topic_to_channel_id_map_;

  // The first publish time of any message written to the MCAP file.
  // This is used to populate the file metadata proto.
  int64_t first_publish_time_ns_ = std::numeric_limits<int64_t>::max();
  // The last publish time of any message written to the MCAP file.
  // This is used to populate the file metadata proto.
  int64_t last_publish_time_ns_ = 0;

  // The total size of all messages written to the MCAP file in bytes.
  // This size is calculated before compression and does not include MCAP
  // metadata overhead.
  int64_t total_messages_size_bytes_ = 0;
  // The shard number of this file handle.
  // The shard number represents the number of times the episode that this file
  // handle is associated with has been sharded across files.
  int64_t shard_index_ = 0;

  // The next sequence number to be written to the MCAP file.
  int64_t next_sequence_number_ = 0;
};

}  // namespace safari::logging

#endif  // SAFARI_LOGGING_CC_MCAP_FILE_HANDLE_H_
