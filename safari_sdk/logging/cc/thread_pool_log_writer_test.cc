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

#include "safari_sdk/logging/cc/thread_pool_log_writer.h"

#include <cstdint>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include <absl/memory/memory.h>
#include <absl/status/status.h>
#include <absl/status/statusor.h>
#include <absl/synchronization/mutex.h>
#include "safari_sdk/logging/cc/base_mcap_file_handle_factory.h"
#include "safari_sdk/logging/cc/episode_data.h"
#include "safari_sdk/logging/cc/mcap_file_handle.h"
#include "safari_sdk/logging/cc/mcap_write_op.h"

namespace safari::logging {

using ::testing::_;
using ::testing::ByMove;
using ::testing::Return;

class MockMcapFileHandle : public BaseMcapFileHandle {
 public:
  MOCK_METHOD(absl::Status, WriteMessage, (const McapWriteOp& request),
              (override));

  MOCK_METHOD(int64_t, total_messages_size_bytes, (), (const, override));
  MOCK_METHOD(int64_t, shard_index, (), (const, override));
  MOCK_METHOD(int64_t, last_publish_time_ns, (), (const, override));

  MOCK_METHOD(void, Die, ());
  ~MockMcapFileHandle() override { Die(); }
};

class MockMcapFileHandleFactory : public BaseMcapFileHandleFactory {
 public:
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<BaseMcapFileHandle>>, Create,
              (const std::string& filename_prefix, int64_t shard_index,
               int64_t first_publish_time_ns, const McapFileConfig* config),
              (override));
};

class ThreadPoolLogWriterTestPeer {
 public:
  static int64_t GetWorkerCount(ThreadPoolLogWriter* writer) {
    return writer->workers_.size();
  }

  static int64_t GetQueueSize(ThreadPoolLogWriter* writer) {
    absl::MutexLock lock(writer->queue_mutex_);
    return writer->queue_.size();
  }

  static int64_t GetEpisodeDataToBeDestroyedQueueSize(
      ThreadPoolLogWriter* writer) {
    absl::MutexLock lock(writer->episode_data_to_be_destroyed_mutex_);
    return writer->episode_data_to_be_destroyed_.size();
  }

  static absl::StatusOr<std::unique_ptr<ThreadPoolLogWriter>> CreateForTest(
      ThreadPoolLogWriterConfig config,
      std::unique_ptr<BaseMcapFileHandleFactory> mcap_file_handle_factory) {
    return absl::WrapUnique(
        new ThreadPoolLogWriter(config, std::move(mcap_file_handle_factory)));
  }
};

namespace {

std::unique_ptr<EpisodeData> CreateEpisodeDataForTesting(int num_timesteps) {
  auto payload = std::make_unique<EpisodeDataPayloadInterface>();
  std::vector<int64_t> timestamps(num_timesteps);
  std::iota(timestamps.begin(), timestamps.end(), 0);
  auto episode_data = EpisodeData::Create(std::move(payload), timestamps);
  return std::move(*episode_data);
}

EnqueueMcapFileOptions CreateEnqueueMcapFileOptionsForTesting(
    std::string episode_uuid = "test_episode_uuid",
    std::string topic = "/example_topic") {
  return EnqueueMcapFileOptions(episode_uuid, topic, /*timestamp_ns=*/1);
}

ThreadPoolLogWriterConfig CreateThreadPoolLogWriterConfigForTesting() {
  return ThreadPoolLogWriterConfig{
      .max_num_workers = 10,
      .image_observation_keys = {"image_observation_key"},
      .mcap_file_config =
          {
              // We mock the McapFileManager, so we can set these values to
              // anything. We will not be doing any file operations.
              .output_dir = "/test_output_dir",
              .file_metadata_topic = "/file_metadata",
              .agent_id = "test_agent_id",
              .file_shard_size_limit_bytes = 1000000,
          },
  };
}

class ThreadPoolLogWriterTest : public ::testing::Test {
 protected:
  void SetUp() override {
    config_ = CreateThreadPoolLogWriterConfigForTesting();
    auto mock_mcap_file_handle_factory =
        std::make_unique<MockMcapFileHandleFactory>();
    mock_mcap_file_handle_factory_ = mock_mcap_file_handle_factory.get();
    ASSERT_OK_AND_ASSIGN(
        writer_, ThreadPoolLogWriterTestPeer::CreateForTest(
                     config_, std::move(mock_mcap_file_handle_factory)));
  }

  ThreadPoolLogWriterConfig config_;
  std::unique_ptr<ThreadPoolLogWriter> writer_;
  MockMcapFileHandleFactory* mock_mcap_file_handle_factory_;
};

TEST_F(ThreadPoolLogWriterTest, CreateSucceedsWithValidConfig) {
  EXPECT_OK(ThreadPoolLogWriter::Create(config_));
}

TEST_F(ThreadPoolLogWriterTest, CreateFailsWithEmptyOutputDir) {
  config_.mcap_file_config.output_dir = "";
  EXPECT_THAT(ThreadPoolLogWriter::Create(config_),
              testing::status::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(ThreadPoolLogWriterTest, CreateWithEmptyFileMetadataTopicFails) {
  config_.mcap_file_config.file_metadata_topic = "";
  EXPECT_THAT(ThreadPoolLogWriter::Create(config_),
              testing::status::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(ThreadPoolLogWriterTest, CreateWithNegativeFileShardSizeLimitFails) {
  config_.mcap_file_config.file_shard_size_limit_bytes = -1;
  EXPECT_THAT(ThreadPoolLogWriter::Create(config_),
              testing::status::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(ThreadPoolLogWriterTest, StartCreatesCorrectNumberOfThreads) {
  writer_->Start();
  EXPECT_EQ(ThreadPoolLogWriterTestPeer::GetWorkerCount(writer_.get()),
            config_.max_num_workers);
}

TEST_F(ThreadPoolLogWriterTest, StartDoesNotCreateThreadsIfAlreadyStarted) {
  writer_->Start();
  int64_t num_workers =
      ThreadPoolLogWriterTestPeer::GetWorkerCount(writer_.get());
  writer_->Start();
  EXPECT_EQ(ThreadPoolLogWriterTestPeer::GetWorkerCount(writer_.get()),
            num_workers);
}

TEST_F(ThreadPoolLogWriterTest,
       EnqueueEpisodeDataAddsCorrectNumberOfItemsToQueue) {
  constexpr int kNumItems = 100;

  for (int i = 0; i < kNumItems; ++i) {
    writer_->EnqueueEpisodeData(
        CreateEpisodeDataForTesting(/*num_timesteps=*/1),
        CreateEnqueueMcapFileOptionsForTesting());
  }

  EXPECT_EQ(ThreadPoolLogWriterTestPeer::GetQueueSize(writer_.get()),
            kNumItems);
}

TEST_F(ThreadPoolLogWriterTest, EnqueueEpisodeDataCreatesNewFileHandle) {
  EXPECT_CALL(*mock_mcap_file_handle_factory_, Create(_, _, _, _)).Times(1);

  writer_->Start();
  writer_->EnqueueEpisodeData(CreateEpisodeDataForTesting(/*num_timesteps=*/1),
                              CreateEnqueueMcapFileOptionsForTesting());
}

TEST_F(ThreadPoolLogWriterTest, EnqueueSessionDataWritesToTheSameFile) {
  // We will only create one file handle for both the EpisodeData and
  // Session data if the file size limit is not reached and we use the same
  // filename prefix.
  EXPECT_CALL(*mock_mcap_file_handle_factory_, Create(_, _, _, _)).Times(1);
  auto mock_mcap_file_handle = std::make_unique<MockMcapFileHandle>();
  EXPECT_CALL(*mock_mcap_file_handle, WriteMessage(_)).Times(2);
  ON_CALL(*mock_mcap_file_handle_factory_, Create(_, _, _, _))
      .WillByDefault(Return(ByMove(std::move(mock_mcap_file_handle))));

  writer_->Start();
  writer_->EnqueueEpisodeData(CreateEpisodeDataForTesting(/*num_timesteps=*/1),
                              CreateEnqueueMcapFileOptionsForTesting());
  writer_->EnqueueSessionData(safari_sdk::protos::logging::Session(),
                              CreateEnqueueMcapFileOptionsForTesting());
}

TEST_F(ThreadPoolLogWriterTest,
       GetOrCreateFileHandleShardsFileWhenSizeLimitIsReached) {
  constexpr int kShards = 100;
  ThreadPoolLogWriterConfig config =
      CreateThreadPoolLogWriterConfigForTesting();
  // Set the file shard size limit to 1 bytes to force the ThreadPoolLogWriter
  // to create a new file for each request.
  config.mcap_file_config.file_shard_size_limit_bytes = 1;
  auto mock_mcap_file_handle_factory =
      std::make_unique<MockMcapFileHandleFactory>();
  EXPECT_CALL(*mock_mcap_file_handle_factory, Create(_, _, _, _))
      .Times(kShards);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<ThreadPoolLogWriter> writer,
                       ThreadPoolLogWriterTestPeer::CreateForTest(
                           config, std::move(mock_mcap_file_handle_factory)));
  writer->Start();
  for (int i = 0; i < kShards; ++i) {
    writer->EnqueueEpisodeData(CreateEpisodeDataForTesting(/*num_timesteps=*/1),
                               CreateEnqueueMcapFileOptionsForTesting());
  }
}

TEST_F(ThreadPoolLogWriterTest, StopWaitsForAllWorkUnitsToComplete) {
  constexpr int kNumItems = 100;
  for (int i = 0; i < kNumItems; ++i) {
    writer_->EnqueueEpisodeData(
        CreateEpisodeDataForTesting(/*num_timesteps=*/1),
        CreateEnqueueMcapFileOptionsForTesting());
  }

  EXPECT_EQ(ThreadPoolLogWriterTestPeer::GetQueueSize(writer_.get()),
            kNumItems);

  writer_->Start();
  writer_->Stop();
  EXPECT_EQ(ThreadPoolLogWriterTestPeer::GetQueueSize(writer_.get()), 0);
  EXPECT_EQ(ThreadPoolLogWriterTestPeer::GetEpisodeDataToBeDestroyedQueueSize(
                writer_.get()),
            0);
}

TEST_F(ThreadPoolLogWriterTest, StopIsNoopIfAlreadyStopped) {
  writer_->Start();
  EXPECT_EQ(ThreadPoolLogWriterTestPeer::GetWorkerCount(writer_.get()),
            config_.max_num_workers);
  writer_->Stop();
  EXPECT_EQ(ThreadPoolLogWriterTestPeer::GetWorkerCount(writer_.get()), 0);
  writer_->Stop();
  EXPECT_EQ(ThreadPoolLogWriterTestPeer::GetWorkerCount(writer_.get()), 0);
}

TEST_F(ThreadPoolLogWriterTest, FinalizeEpisodeWaitsForOpsAndClosesHandle) {
  const std::string uuid = "test-uuid-1";

  auto mock_mcap_file_handle = std::make_unique<MockMcapFileHandle>();
  EXPECT_CALL(*mock_mcap_file_handle, WriteMessage(_))
      .WillRepeatedly(Return(absl::OkStatus()));
  // The file handle should be closed once all pending operations for the
  // episode have been processed.
  EXPECT_CALL(*mock_mcap_file_handle, Die()).Times(1);
  ON_CALL(*mock_mcap_file_handle_factory_, Create(_, _, _, _))
      .WillByDefault(Return(ByMove(std::move(mock_mcap_file_handle))));

  constexpr int kNumItems = 100;
  for (int i = 0; i < kNumItems; ++i) {
    writer_->EnqueueEpisodeData(
        CreateEpisodeDataForTesting(/*num_timesteps=*/1),
        CreateEnqueueMcapFileOptionsForTesting(uuid));
  }
  EXPECT_EQ(ThreadPoolLogWriterTestPeer::GetQueueSize(writer_.get()),
            kNumItems);

  writer_->Start();
  ASSERT_OK(writer_->FinalizeEpisode(uuid));
}

TEST_F(ThreadPoolLogWriterTest, FinalizeEpisodeTwiceIsNoop) {
  const std::string uuid = "test-uuid-1";

  auto mock_mcap_file_handle = std::make_unique<MockMcapFileHandle>();
  EXPECT_CALL(*mock_mcap_file_handle, WriteMessage(_))
      .WillRepeatedly(Return(absl::OkStatus()));
  // The file handle should still only be closed once even if FinalizeEpisode
  // is called twice.
  EXPECT_CALL(*mock_mcap_file_handle, Die()).Times(1);
  ON_CALL(*mock_mcap_file_handle_factory_, Create(_, _, _, _))
      .WillByDefault(Return(ByMove(std::move(mock_mcap_file_handle))));

  constexpr int kNumItems = 100;
  for (int i = 0; i < kNumItems; ++i) {
    writer_->EnqueueEpisodeData(
        CreateEpisodeDataForTesting(/*num_timesteps=*/1),
        CreateEnqueueMcapFileOptionsForTesting(uuid));
  }
  EXPECT_EQ(ThreadPoolLogWriterTestPeer::GetQueueSize(writer_.get()),
            kNumItems);

  writer_->Start();
  ASSERT_OK(writer_->FinalizeEpisode(uuid));
  ASSERT_OK(writer_->FinalizeEpisode(uuid));
}

}  // namespace
}  // namespace safari::logging
