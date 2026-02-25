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

#include "safari_sdk/logging/cc/mcap_file_handle.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>  // NOLINT
#include <memory>
#include <queue>
#include <string>
#include <system_error>  // NOLINT
#include <utility>

#include <google/protobuf/descriptor.pb.h>
#include <absl/base/attributes.h>
#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <absl/log/log.h>
#include <absl/memory/memory.h>
#include <absl/status/status.h>
#include <absl/status/statusor.h>
#include <absl/strings/str_cat.h>
#include <absl/strings/string_view.h>
#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <mcap/errors.hpp>
#include <mcap/types.hpp>
#include <mcap/writer.hpp>
#include <google/protobuf/descriptor.h>
#include "safari_sdk/logging/cc/mcap_write_op.h"
#include "safari_sdk/protos/label.pb.h"
#include "safari_sdk/protos/logging/metadata.pb.h"
#include <tensorflow/core/example/example.pb.h>
#include <tensorflow/core/example/feature.pb.h>

namespace safari::logging {
namespace {
// Helper function to get the file descriptor set for a given message
// descriptor.
std::string GetFileDescriptorSetString(
    const google::protobuf::Descriptor* msg_descriptor) {
  google::protobuf::FileDescriptorSet descriptor_set;
  const google::protobuf::FileDescriptor* file_descriptor = msg_descriptor->file();

  std::queue<const google::protobuf::FileDescriptor*> files_to_add;
  files_to_add.push(file_descriptor);
  absl::flat_hash_set<const google::protobuf::FileDescriptor*> added_files;
  added_files.insert(file_descriptor);

  while (!files_to_add.empty()) {
    const google::protobuf::FileDescriptor* current_file = files_to_add.front();
    files_to_add.pop();
    current_file->CopyTo(descriptor_set.add_file());

    for (int i = 0; i < current_file->dependency_count(); ++i) {
      const google::protobuf::FileDescriptor* dep = current_file->dependency(i);
      if (added_files.find(dep) == added_files.end()) {
        files_to_add.push(dep);
        added_files.insert(dep);
      }
    }
  }
  return descriptor_set.SerializeAsString();
}

// Helper function to a create directory if it does not exist.
absl::Status CreateDirectoryIfNotExist(absl::string_view directory) {
  std::error_code error_code;
  if (!std::filesystem::exists(directory)) {
    VLOG(1) << "Creating directory: " << directory;
    std::filesystem::create_directories(directory, error_code);
  }
  if (error_code) {
    return absl::InternalError(absl::StrCat(
        "Failed to create directory '", directory, "': ", error_code.message(),
        " (error code ", error_code.value(), ")"));
  }
  return absl::OkStatus();
}
}  // namespace

std::string GetFinalDirectory(absl::string_view output_dir) {
  absl::Time now = absl::Now();
  absl::TimeZone tz = absl::LocalTimeZone();
  std::string year = absl::FormatTime("%Y", now, tz);
  std::string month = absl::FormatTime("%m", now, tz);
  std::string day = absl::FormatTime("%d", now, tz);

  // Create the final directory for the file.
  std::string final_dir =
      absl::StrCat(output_dir, "/", year, "/", month, "/", day);
  return final_dir;
}

absl::StatusOr<std::unique_ptr<McapFileHandle>> McapFileHandle::Create(
    absl::string_view episode_uuid, int64_t shard_index,
    int64_t first_publish_time_ns,
    const McapFileConfig* config ABSL_ATTRIBUTE_LIFETIME_BOUND) {
  // Create a writer for the MCAP file.
  auto writer = std::make_unique<mcap::McapWriter>();
  // Create a temporary file path to write to.
  std::string tmp_dir = absl::StrCat(config->output_dir, "/tmp/");
  absl::Status status = CreateDirectoryIfNotExist(tmp_dir);
  if (!status.ok()) {
    return absl::InternalError(
        absl::StrCat("Failed to create directory: ", tmp_dir,
                     " with status: ", status.message()));
  }
  std::string filename =
      absl::StrCat(config->filename_prefix, "_", episode_uuid, "_shard",
                   shard_index, ".mcap.tmp");

  std::string tmp_file_path = absl::StrCat(tmp_dir, filename);

  mcap::Status open_status =
      writer->open(tmp_file_path, mcap::McapWriterOptions(""));
  if (!open_status.ok()) {
    return absl::InternalError(
        absl::StrCat("Failed to open MCAP file: ", tmp_file_path,
                     " with status: ", open_status.message));
  }
  return absl::WrapUnique(new McapFileHandle(filename, tmp_file_path,
                                             shard_index, first_publish_time_ns,
                                             config, std::move(writer)));
}

mcap::ChannelId McapFileHandle::RegisterSchemaAndChannels(
    absl::string_view topic, const google::protobuf::Descriptor* descriptor) {
  mcap::Schema schema(descriptor->full_name(), "protobuf",
                      GetFileDescriptorSetString(descriptor));

  writer_->addSchema(schema);

  // Create a channel for the topic.
  mcap::Channel channel(topic, "protobuf", schema.id);
  writer_->addChannel(channel);
  // Channel must be added to the writer before we can get the channel id.
  topic_to_channel_id_map_[topic] = channel.id;

  return channel.id;
}

McapFileHandle::McapFileHandle(absl::string_view filename,
                               absl::string_view tmp_file_path,
                               int64_t shard_index,
                               int64_t first_publish_time_ns,
                               const McapFileConfig* config,
                               std::unique_ptr<mcap::McapWriter> writer)
    : filename_(filename),
      tmp_file_path_(tmp_file_path),
      config_(config),
      writer_(std::move(writer)),
      first_publish_time_ns_(first_publish_time_ns),
      shard_index_(shard_index) {}

McapFileHandle::~McapFileHandle() {
  // Create the file metadata proto and write it to the MCAP file before
  // closing the writer and the file.
  CreateFileMetadataProto();

  // Destroy the flat hash map first before the writer destroys the channels
  // The map stores channel ids which are owned by the writer.
  topic_to_channel_id_map_.clear();

  // Closes the writer, flushing any pending messages to the file.
  // Also moves the file to the final location.
  Close();
}

void McapFileHandle::CreateFileMetadataProto() {
  safari_sdk::protos::logging::FileMetadata file_metadata;
  file_metadata.set_agent_id(config_->agent_id);

  for (const auto& [topic, _] : topic_to_channel_id_map_) {
    safari_sdk::protos::logging::KeyRange* key_range =
        file_metadata.add_stream_coverages();
    key_range->set_topic(topic);
    key_range->mutable_interval()->set_start_nsec(first_publish_time_ns_);
    key_range->mutable_interval()->set_stop_nsec(last_publish_time_ns_);
  }

  McapWriteOp write_op = {
      .serialized_message = file_metadata.SerializeAsString(),
      .descriptor = safari_sdk::protos::logging::FileMetadata::descriptor(),
      .topic = config_->file_metadata_topic,
      .publish_time_ns = last_publish_time_ns_,
      .log_time_ns = absl::ToUnixNanos(absl::Now()),
  };
  absl::Status status = WriteMessage(write_op);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to write file metadata to MCAP file: " << filename_
               << " with status: " << status;
  }
}

void McapFileHandle::Close() {
  if (writer_ != nullptr) {
    writer_->close();
  }

  // Create the final directory for the file.
  std::string final_dir = GetFinalDirectory(config_->output_dir);
  absl::Status status = CreateDirectoryIfNotExist(final_dir);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to create directory: " << final_dir
               << " with status: " << status;
    return;
  }

  std::error_code error_code;
  // Move the file to the final location.
  std::filesystem::path final_filename(filename_);
  final_filename.replace_extension();  // Removes .tmp
  std::string final_file_path =
      std::filesystem::path(final_dir) / final_filename;
  std::filesystem::rename(tmp_file_path_, final_file_path, error_code);
  if (error_code) {
    LOG(ERROR) << "Failed to move file: " << tmp_file_path_
               << " to final location: " << final_file_path
               << " with status: " << error_code.message()
               << " (error code: " << error_code.value() << ")";
    return;
  }

  // Create a mask to remove write permissions for all users.
  std::filesystem::perms current_perms =
      std::filesystem::status(final_file_path).permissions();
  std::filesystem::perms new_perms =
      current_perms & ~(std::filesystem::perms::owner_write |
                        std::filesystem::perms::group_write |
                        std::filesystem::perms::others_write);

  // Set new permissions
  if (new_perms != current_perms) {
    std::filesystem::permissions(final_file_path, new_perms, error_code);
  }
  if (error_code) {
    LOG(ERROR) << "Failed to set permissions for file: " << final_file_path
               << " with status: " << error_code.message()
               << " (error code: " << error_code.value() << ")";
  }
}

absl::Status McapFileHandle::WriteMessage(const McapWriteOp& op) {
  if (writer_ == nullptr) {
    return absl::InternalError(
        absl::StrCat("Failured to write message to MCAP file ", filename_,
                     "The Mcap writer for file is null."));
  }

  mcap::ChannelId channel_id;
  // Find the corresponding channel id for the topic.
  // If the topic is not found, register the schema and corresponding channel.
  auto it = topic_to_channel_id_map_.find(op.topic);
  if (it == topic_to_channel_id_map_.end()) {
    channel_id = RegisterSchemaAndChannels(op.topic, op.descriptor);
  } else {
    channel_id = it->second;
  }

  // Create a message for the channel.
  mcap::Message msg;
  // Must be initialized because this is not done so by default in the MCAP
  // implementation.
  msg.sequence = next_sequence_number_++;
  msg.channelId = channel_id;
  msg.publishTime = op.publish_time_ns;
  msg.logTime = op.log_time_ns;
  msg.data = reinterpret_cast<const std::byte*>(op.serialized_message.data());
  msg.dataSize = op.serialized_message.size();

  // Write the message to the MCAP file.
  mcap::Status write_status = writer_->write(msg);

  if (!write_status.ok()) {
    return absl::InternalError(
        absl::StrCat("Failed to write message to MCAP file: ", filename_,
                     " with status: ", write_status.message,
                     " for topic: ", op.topic, " with channel id: ", channel_id,
                     " and publish time: ", op.publish_time_ns));
  }

  first_publish_time_ns_ = std::min(first_publish_time_ns_, op.publish_time_ns);

  last_publish_time_ns_ = std::max(last_publish_time_ns_, op.publish_time_ns);

  // Update the file size.
  total_messages_size_bytes_ += msg.dataSize;

  VLOG(1) << "Wrote message to MCAP file: " << filename_
          << " for topic: " << op.topic
          << " with sequence number: " << msg.sequence
          << " with channel id: " << msg.channelId
          << " and log time: " << msg.logTime
          << " and publish time: " << msg.publishTime
          << " with size: " << msg.dataSize
          << " first_publish_time_ns: " << first_publish_time_ns_
          << " last_publish_time_ns: " << last_publish_time_ns_;

  return absl::OkStatus();
}

int64_t McapFileHandle::total_messages_size_bytes() const {
  return total_messages_size_bytes_;
}

int64_t McapFileHandle::shard_index() const { return shard_index_; }

int64_t McapFileHandle::last_publish_time_ns() const {
  return last_publish_time_ns_;
}
}  // namespace safari::logging
