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

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "file/base/filesystem.h"
#include "file/base/options.h"
#include "file/util/temp_path.h"
#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include <absl/log/log.h>
#include <absl/status/status.h>
#include <absl/status/statusor.h>
#include <absl/strings/str_cat.h>
#include <mcap/reader.hpp>
#include <mcap/types.hpp>
#include "safari_sdk/logging/cc/mcap_write_op.h"
#include "safari_sdk/protos/logging/metadata.pb.h"
#include <tensorflow/core/example/example.pb.h>

namespace safari::logging {
namespace {

using ::testing::EqualsProto;
using ::testing::proto::IgnoringRepeatedFieldOrdering;

std::vector<safari_sdk::protos::logging::FileMetadata> GetFileMetadataProtos(
    mcap::McapReader& reader) {
  std::vector<safari_sdk::protos::logging::FileMetadata> file_metadatas;
  auto message_view = reader.readMessages();
  for (auto it = message_view.begin(); it != message_view.end(); it++) {
    auto schema = it->schema.get();
    if (schema->name == "safari.protos.logging.FileMetadata" &&
        schema->encoding == "protobuf") {
      safari_sdk::protos::logging::FileMetadata file_metadata;
      file_metadata.ParseFromArray(it->message.data, it->message.dataSize);
      file_metadatas.push_back(file_metadata);
    }
  }
  return file_metadatas;
}

McapFileConfig CreateDefaultConfig(const std::string& output_dir) {
  return {
      .output_dir = output_dir,
      .filename_prefix = "test_episode",
      .file_metadata_topic = "/file_metadata",
      .agent_id = "test_agent_id",
      .file_shard_size_limit_bytes = 100,
  };
}

class McapFileHandleTest : public ::testing::Test {
 protected:
  McapFileHandleTest()
      : temp_path_(TempPath::Local),
        config_(CreateDefaultConfig(temp_path_.path())) {}

  TempPath temp_path_;
  safari::logging::McapFileConfig config_;
};

TEST_F(McapFileHandleTest, CreateOpensTemporaryFile) {
  std::string episode_uuid = "123";
  int64_t shard_index = 0;

  std::string expected_tmp_file_path =
      absl::StrCat(config_.output_dir, "/tmp/", config_.filename_prefix, "_",
                   episode_uuid, "_shard", shard_index, ".mcap.tmp");
  EXPECT_FALSE(file::Exists(expected_tmp_file_path, file::Defaults()).ok());

  absl::StatusOr<std::unique_ptr<McapFileHandle>> file_handle =
      McapFileHandle::Create(episode_uuid, shard_index,
                             /*first_publish_time_ns=*/0, &config_);

  EXPECT_OK(file_handle);
  EXPECT_OK(file::Exists(expected_tmp_file_path, file::Defaults()));
}

TEST_F(McapFileHandleTest, WriteMessageSucceedsAndUpdatesFileStats) {
  std::string episode_uuid = "123";
  int64_t shard_index = 0;
  std::string expected_tmp_file_path =
      absl::StrCat(config_.output_dir, "/tmp/", config_.filename_prefix, "_",
                   episode_uuid, "_shard", shard_index, ".mcap.tmp");

  absl::StatusOr<std::unique_ptr<McapFileHandle>> file_handle =
      McapFileHandle::Create(episode_uuid, shard_index,
                             /*first_publish_time_ns=*/0, &config_);

  EXPECT_OK(file_handle);
  EXPECT_OK(file::Exists(expected_tmp_file_path, file::Defaults()));

  EXPECT_EQ((*file_handle)->total_messages_size_bytes(), 0);

  safari_sdk::protos::logging::Session session;
  std::string serialized_session = session.SerializeAsString();
  McapWriteOp op = {
      .episode_uuid = episode_uuid,
      .serialized_message = serialized_session,
      .descriptor = safari_sdk::protos::logging::Session::descriptor(),
      .topic = "/test_topic",
      .publish_time_ns = 1234567890,
      .log_time_ns = 987654321,
  };

  EXPECT_OK((*file_handle)->WriteMessage(op));
  EXPECT_EQ((*file_handle)->total_messages_size_bytes(),
            serialized_session.size());
  EXPECT_EQ((*file_handle)->shard_index(), shard_index);
  EXPECT_EQ((*file_handle)->last_publish_time_ns(), 1234567890);
}

TEST_F(McapFileHandleTest, DestructorMovesFileToFinalLocation) {
  std::string episode_uuid = "123";
  int64_t shard_index = 0;

  std::string expected_tmp_filename =
      absl::StrCat(config_.filename_prefix, "_", episode_uuid, "_shard",
                   shard_index, ".mcap.tmp");
  std::string expected_filename =
      absl::StrCat(config_.filename_prefix, "_", episode_uuid, "_shard",
                   shard_index, ".mcap");

  std::string expected_tmp_file_path =
      absl::StrCat(config_.output_dir, "/tmp/", expected_tmp_filename);

  std::string expected_final_file_path =
      absl::StrCat(safari::logging::GetFinalDirectory(config_.output_dir), "/",
                   expected_filename);

  EXPECT_FALSE(file::Exists(expected_tmp_file_path, file::Defaults()).ok());
  EXPECT_FALSE(file::Exists(expected_final_file_path, file::Defaults()).ok());

  absl::StatusOr<std::unique_ptr<McapFileHandle>> file_handle =
      McapFileHandle::Create(episode_uuid, shard_index,
                             /*first_publish_time_ns=*/0, &config_);
  EXPECT_OK(file_handle);
  EXPECT_OK(file::Exists(expected_tmp_file_path, file::Defaults()));
  // The final file should not exist yet.
  EXPECT_FALSE(file::Exists(expected_final_file_path, file::Defaults()).ok());
  EXPECT_EQ((*file_handle)->total_messages_size_bytes(), 0);

  safari_sdk::protos::logging::Session session;
  std::string serialized_session = session.SerializeAsString();

  McapWriteOp op = {
      .episode_uuid = episode_uuid,
      .serialized_message = serialized_session,
      .descriptor = safari_sdk::protos::logging::Session::descriptor(),
      .topic = "/test_topic",
      .publish_time_ns = 1234567890,
      .log_time_ns = 987654321,
  };

  EXPECT_OK((*file_handle)->WriteMessage(op));
  EXPECT_EQ((*file_handle)->total_messages_size_bytes(),
            serialized_session.size());

  file_handle->reset();
  EXPECT_FALSE(file::Exists(expected_tmp_file_path, file::Defaults()).ok());
  EXPECT_OK(file::Exists(expected_final_file_path, file::Defaults()));
}

TEST_F(McapFileHandleTest, WriteMessageCreatesMcapFilesWithCorrectMessages) {
  std::string episode_uuid = "123";
  int64_t shard_index = 0;

  std::string expected_tmp_filename =
      absl::StrCat(config_.filename_prefix, "_", episode_uuid, "_shard",
                   shard_index, ".mcap.tmp");
  std::string expected_filename =
      absl::StrCat(config_.filename_prefix, "_", episode_uuid, "_shard",
                   shard_index, ".mcap");
  std::string expected_tmp_file_path =
      absl::StrCat(config_.output_dir, "/tmp/", expected_tmp_filename);
  std::string expected_final_file_path =
      absl::StrCat(safari::logging::GetFinalDirectory(config_.output_dir), "/",
                   expected_filename);

  absl::StatusOr<std::unique_ptr<McapFileHandle>> file_handle =
      McapFileHandle::Create(episode_uuid, shard_index,
                             /*first_publish_time_ns=*/0, &config_);
  EXPECT_OK(file_handle);
  safari_sdk::protos::logging::Session session;
  std::string serialized_session = session.SerializeAsString();

  McapWriteOp op = {
      .episode_uuid = episode_uuid,
      .serialized_message = serialized_session,
      .descriptor = safari_sdk::protos::logging::Session::descriptor(),
      .topic = "/session_topic",
      .publish_time_ns = 1234567890,
      .log_time_ns = 987654321,
  };

  EXPECT_OK((*file_handle)->WriteMessage(op));

  file_handle->reset();
  EXPECT_OK(file::Exists(expected_final_file_path, file::Defaults()));

  // Read the MCAP file to verify schema and channel
  mcap::McapReader reader;
  ASSERT_TRUE(reader.open(expected_final_file_path).ok());

  // Verify the schemas, topics and messages in the MCAP file.
  std::vector<mcap::Schema> schemas;
  std::vector<std::string> topics;
  std::vector<safari_sdk::protos::logging::Session> sessions;
  std::vector<safari_sdk::protos::logging::FileMetadata> file_metadatas;

  auto message_view = reader.readMessages();
  for (auto it = message_view.begin(); it != message_view.end(); it++) {
    auto schema = it->schema.get();
    schemas.push_back(*schema);
    if (schema->name == "safari.protos.logging.Session" &&
        schema->encoding == "protobuf") {
      safari_sdk::protos::logging::Session session;
      session.ParseFromArray(it->message.data, it->message.dataSize);
      sessions.push_back(session);
    } else if (schema->name == "safari.protos.logging.FileMetadata" &&
               schema->encoding == "protobuf") {
      safari_sdk::protos::logging::FileMetadata file_metadata;
      file_metadata.ParseFromArray(it->message.data, it->message.dataSize);
      file_metadatas.push_back(file_metadata);
    }
    topics.push_back(it->channel->topic);
  }

  EXPECT_THAT(schemas, testing::SizeIs(2));
  EXPECT_THAT(topics, testing::ElementsAre("/session_topic", "/file_metadata"));
  EXPECT_THAT(sessions, testing::SizeIs(1));
  EXPECT_THAT(file_metadatas, testing::SizeIs(1));
  reader.close();
}

TEST_F(McapFileHandleTest,
       WriteMessageCreateMessagesWithIncrementingSequenceNumbers) {
  std::string episode_uuid = "123";
  int64_t shard_index = 0;

  std::string expected_tmp_filename =
      absl::StrCat(config_.filename_prefix, "_", episode_uuid, "_shard",
                   shard_index, ".mcap.tmp");
  std::string expected_filename =
      absl::StrCat(config_.filename_prefix, "_", episode_uuid, "_shard",
                   shard_index, ".mcap");
  std::string expected_tmp_file_path =
      absl::StrCat(config_.output_dir, "/tmp/", expected_tmp_filename);
  std::string expected_final_file_path =
      absl::StrCat(safari::logging::GetFinalDirectory(config_.output_dir), "/",
                   expected_filename);

  absl::StatusOr<std::unique_ptr<McapFileHandle>> file_handle =
      McapFileHandle::Create(episode_uuid, shard_index,
                             /*first_publish_time_ns=*/0, &config_);
  EXPECT_OK(file_handle);

  tensorflow::Example example;
  McapWriteOp op1 = {
      .episode_uuid = episode_uuid,
      .serialized_message = example.SerializeAsString(),
      .descriptor = tensorflow::Example::descriptor(),
      .topic = "/example_topic",
      .publish_time_ns = 1,
      .log_time_ns = 1,
  };
  EXPECT_OK((*file_handle)->WriteMessage(op1));

  tensorflow::Example example2;
  McapWriteOp op2 = {
      .episode_uuid = episode_uuid,
      .serialized_message = example2.SerializeAsString(),
      .descriptor = tensorflow::Example::descriptor(),
      .topic = "/example_topic",
      .publish_time_ns = 2,
      .log_time_ns = 2,
  };
  EXPECT_OK((*file_handle)->WriteMessage(op2));

  file_handle->reset();
  EXPECT_OK(file::Exists(expected_final_file_path, file::Defaults()));

  // Read the MCAP file to verify the message sequence numbers.
  mcap::McapReader reader;
  ASSERT_TRUE(reader.open(expected_final_file_path).ok());

  auto message_view = reader.readMessages();
  int64_t expected_sequence_number = 0;
  for (auto it = message_view.begin(); it != message_view.end(); it++) {
    ASSERT_EQ(it->message.sequence, expected_sequence_number++);
  }
}

TEST_F(McapFileHandleTest,
       FileMetadataProtoTimestampsAreCorrectAfterMultipleWrites) {
  // Tests that the first and last publish timestamps in the file metadata are
  // updated correctly.
  std::string episode_uuid = "123";
  int64_t shard_index = 0;

  std::string expected_tmp_filename =
      absl::StrCat(config_.filename_prefix, "_", episode_uuid, "_shard",
                   shard_index, ".mcap.tmp");
  std::string expected_filename =
      absl::StrCat(config_.filename_prefix, "_", episode_uuid, "_shard",
                   shard_index, ".mcap");
  std::string expected_tmp_file_path =
      absl::StrCat(config_.output_dir, "/tmp/", expected_tmp_filename);
  std::string expected_final_file_path =
      absl::StrCat(safari::logging::GetFinalDirectory(config_.output_dir), "/",
                   expected_filename);

  absl::StatusOr<std::unique_ptr<McapFileHandle>> file_handle =
      McapFileHandle::Create(episode_uuid, shard_index,
                             /*first_publish_time_ns=*/0, &config_);
  EXPECT_OK(file_handle);

  tensorflow::Example example;
  McapWriteOp op1 = {
      .episode_uuid = episode_uuid,
      .serialized_message = example.SerializeAsString(),
      .descriptor = tensorflow::Example::descriptor(),
      .topic = "/example_topic",
      .publish_time_ns = 1,
      .log_time_ns = 1,
  };
  EXPECT_OK((*file_handle)->WriteMessage(op1));

  safari_sdk::protos::logging::Session session;
  McapWriteOp op2 = {
      .episode_uuid = episode_uuid,
      .serialized_message = session.SerializeAsString(),
      .descriptor = safari_sdk::protos::logging::Session::descriptor(),
      .topic = "/session_topic",
      .publish_time_ns = 2,
      .log_time_ns = 2,
  };
  EXPECT_OK((*file_handle)->WriteMessage(op2));

  file_handle->reset();
  EXPECT_OK(file::Exists(expected_final_file_path, file::Defaults()));

  // Read the MCAP file to verify schema and channel
  mcap::McapReader reader;
  ASSERT_TRUE(reader.open(expected_final_file_path).ok());

  std::vector<safari_sdk::protos::logging::FileMetadata> file_metadatas =
      GetFileMetadataProtos(reader);

  ASSERT_THAT(file_metadatas, testing::SizeIs(1));
  EXPECT_THAT(file_metadatas[0], IgnoringRepeatedFieldOrdering(EqualsProto(R"pb(
                agent_id: "test_agent_id"
                stream_coverages {
                  topic: "/session_topic"
                  interval { start_nsec: 0 stop_nsec: 2 }
                }
                stream_coverages {
                  topic: "/example_topic"
                  interval { start_nsec: 0 stop_nsec: 2 }
                }
              )pb")));
}
}  // namespace
}  // namespace safari::logging
