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

#ifndef SAFARI_LOGGING_CC_MCAP_WRITE_OP_H_
#define SAFARI_LOGGING_CC_MCAP_WRITE_OP_H_

#include <cstdint>
#include <string>

#include <google/protobuf/descriptor.h>

namespace safari::logging {

// This struct is used to encapsulate the information required to write a
// message to an MCAP file.
struct McapWriteOp {
  // The unique identifier for the episode.
  std::string episode_uuid;
  // Serialized tensorflow.Example or safari.protos.logging.Session proto.
  std::string serialized_message;
  // The message descriptor of the serialized proto message.
  const google::protobuf::Descriptor* descriptor;
  // Topic to write the message to.
  std::string topic;
  // Observation timestamp of the raw data in nanoseconds.
  // This is important for ordering the data in SSOT.
  int64_t publish_time_ns;
  // Log timestamp of the raw data in nanoseconds.
  int64_t log_time_ns;
};

// Configuration for creating an MCAP file.
struct McapFileConfig {
  // The output directory for the MCAP files.
  // The files will be written to a subdirectory of this directory.
  std::string output_dir;
  // The prefix of the filename to use for each MCAP file.
  std::string filename_prefix;
  // The topic to use when writing the file metadata proto.
  std::string file_metadata_topic;
  // The agent id to use when writing the file metadata proto.
  std::string agent_id;
  // The maximum size of a single MCAP file in bytes.
  // If a message is larger than this size, it will be written to a new file.
  int64_t file_shard_size_limit_bytes;
};

}  // namespace safari::logging

#endif  // SAFARI_LOGGING_CC_MCAP_WRITE_OP_H_
