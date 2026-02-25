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
#include <iterator>
#include <memory>
#include <string>
#include <thread>  // NOLINT
#include <utility>

#include <absl/base/thread_annotations.h>
#include <absl/functional/any_invocable.h>
#include <absl/log/log.h>
#include <absl/memory/memory.h>
#include <absl/status/status.h>
#include <absl/strings/string_view.h>
#include <absl/synchronization/mutex.h>
#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <google/protobuf/arena.h>
#include "safari_sdk/logging/cc/base_mcap_file_handle_factory.h"
#include "safari_sdk/logging/cc/episode_data.h"
#include "safari_sdk/logging/cc/log_data_serializer_utils.h"
#include "safari_sdk/logging/cc/mcap_file_handle.h"
#include "safari_sdk/logging/cc/mcap_file_handle_factory.h"
#include "safari_sdk/logging/cc/mcap_write_op.h"
#include "safari_sdk/protos/logging/metadata.pb.h"
#include <tensorflow/core/example/example.pb.h>
#include <tensorflow/core/example/feature.pb.h>

namespace safari::logging {

absl::StatusOr<std::unique_ptr<ThreadPoolLogWriter>>
ThreadPoolLogWriter::Create(ThreadPoolLogWriterConfig config) {
  if (config.mcap_file_config.output_dir.empty()) {
    return absl::InvalidArgumentError("Output directory is empty.");
  }
  if (config.mcap_file_config.file_metadata_topic.empty()) {
    return absl::InvalidArgumentError("File metadata topic is empty.");
  }
  if (config.mcap_file_config.file_shard_size_limit_bytes <= 0) {
    return absl::InvalidArgumentError(
        "File shard size limit bytes is not positive.");
  }
  auto mcap_file_handle_factory = std::make_unique<McapFileHandleFactory>();
  return absl::WrapUnique(new ThreadPoolLogWriter(
      std::move(config), std::move(mcap_file_handle_factory)));
}

ThreadPoolLogWriter::ThreadPoolLogWriter(
    ThreadPoolLogWriterConfig config,
    std::unique_ptr<BaseMcapFileHandleFactory> mcap_file_handle_factory)
    : max_num_workers_(config.max_num_workers),
      image_observation_keys_(
          std::make_move_iterator(config.image_observation_keys.begin()),
          std::make_move_iterator(config.image_observation_keys.end())),
      mcap_file_config_(std::move(config.mcap_file_config)),
      mcap_file_handle_factory_(std::move(mcap_file_handle_factory)) {}

ThreadPoolLogWriter::~ThreadPoolLogWriter() { Stop(); }

void ThreadPoolLogWriter::Start() {
  // If the workers have already been started, then we don't need to start them
  // again.
  if (!workers_.empty()) {
    return;
  }
  // Reset the stop_ flag to false.
  // Do this before starting the workers so that they don't all exit
  // immediately.
  {
    absl::MutexLock lock(queue_mutex_);
    stop_ = false;
  }

  // Create and start the worker threads.
  for (int i = 0; i < max_num_workers_; ++i) {
    workers_.push_back(std::thread(&ThreadPoolLogWriter::WorkerLoop, this));
  }

  VLOG(1) << "ThreadPoolLogWriter started. Number of workers: "
          << workers_.size() << " started";
}

void ThreadPoolLogWriter::Stop() {
  {
    absl::MutexLock lock(queue_mutex_);
    if (stop_) {
      return;
    }
    stop_ = true;
  }

  for (auto& worker : workers_) {
    worker.join();
  }

  workers_.clear();

  // At this point, the queue_ should be empty and all episode data should be
  // serialized and moved to episode_data_to_be_destroyed_
  // Clear the episode_data_to_be_destroyed_ queue.
  ProcessEpisodeDataToBeDestroyed();

  absl::MutexLock lock(file_handles_mutex_);
  file_handles_.clear();
}

absl::Status ThreadPoolLogWriter::FinalizeEpisode(
    absl::string_view episode_uuid) {
  Enqueue([this, episode_uuid_str = std::string(episode_uuid)]() {
    {
      absl::MutexLock lock(pending_ops_mutex_);
      auto all_operations_completed =
          [this, &episode_uuid_str]()
              ABSL_EXCLUSIVE_LOCKS_REQUIRED(pending_ops_mutex_) {
                auto it = pending_ops_.find(episode_uuid_str);
                if (it == pending_ops_.end()) {
                  LOG(INFO) << "No pending operations found for episode "
                            << episode_uuid_str;
                  return true;
                }
                return it->second == 0;
              };
      pending_ops_mutex_.Await(absl::Condition(&all_operations_completed));
    }
    absl::MutexLock lock(file_handles_mutex_);
    auto it = file_handles_.find(episode_uuid_str);
    if (it != file_handles_.end()) {
      // Destruction of the unique_ptr in erase calls ~McapFileHandle,
      // which closes and moves the file.
      file_handles_.erase(it);
      LOG(INFO) << "Enqueued finalization: MCAP file(s) for "
                << episode_uuid_str << " will be closed and moved.";
    } else {
      LOG(WARNING)
          << "FinalizeEpisode: No active file handle found for episode "
          << episode_uuid_str << ". It might have already been finalized.";
    }
  });
  return absl::OkStatus();
}

void ThreadPoolLogWriter::EnqueueEpisodeData(
    std::unique_ptr<EpisodeData> episode_data, EnqueueMcapFileOptions options) {
  {
    absl::MutexLock lock(pending_ops_mutex_);
    pending_ops_[options.episode_uuid] += 1;
  }

  Enqueue([this, episode_data = std::move(episode_data),
           options = std::move(options)]() mutable {
    ProcessEpisodeData(std::move(episode_data), std::move(options));
  });

  ProcessEpisodeDataToBeDestroyed();
}

void ThreadPoolLogWriter::EnqueueSessionData(
    safari_sdk::protos::logging::Session session, EnqueueMcapFileOptions options) {
  {
    absl::MutexLock lock(pending_ops_mutex_);
    pending_ops_[options.episode_uuid] += 1;
  }

  Enqueue([this, session = std::move(session),
           options = std::move(options)]() mutable {
    ProcessSessionData(std::move(session), std::move(options));
  });
}

void ThreadPoolLogWriter::Enqueue(absl::AnyInvocable<void() &&> work_unit) {
  absl::MutexLock lock(queue_mutex_);
  if (stop_) {
    LOG(WARNING) << "Attempted to enqueue a work unit to "
                    "ThreadPoolLogWriter after it has been stopped. Dropping "
                    "the work unit.";
    return;
  }
  queue_.push(std::move(work_unit));
}

void ThreadPoolLogWriter::WorkerLoop() {
  while (true) {
    absl::AnyInvocable<void() &&> work_unit;
    {
      absl::MutexLock lock(queue_mutex_);

      // Wait for a work unit to be enqueued or for the pool to be stopped.
      auto work_unit_available_or_stopped =
          [this]() ABSL_EXCLUSIVE_LOCKS_REQUIRED(queue_mutex_) {
            return stop_ || !queue_.empty();
          };
      queue_mutex_.Await(absl::Condition(&work_unit_available_or_stopped));

      // Exit the thread when we have stopped the pool and all work units have
      // been processed.
      if (stop_ && queue_.empty()) {
        return;
      }

      work_unit = std::move(queue_.front());
      queue_.pop();
    }
    std::move(work_unit)();
  }
}

void ThreadPoolLogWriter::ProcessEpisodeData(
    std::unique_ptr<EpisodeData> episode_data, EnqueueMcapFileOptions options) {
  int64_t num_timesteps = episode_data->timestamps().size();

  if (num_timesteps == 0) {
    LOG(ERROR)
        << "ThreadPoolLogWriter: Received EpisodeData with has no timesteps.";
    return;
  }

  google::protobuf::Arena arena;
  for (int i = 0; i < num_timesteps; ++i) {
    tensorflow::Features* features =
        google::protobuf::Arena::Create<tensorflow::Features>(&arena);

    absl::Status status = FillFeatureMapForTimestep(
        /*timestep=*/i, /*episode_data=*/episode_data.get(),
        /*image_keys=*/image_observation_keys_,
        /*features=*/*features);

    if (!status.ok()) {
      LOG(ERROR) << "Failed to fill feature map for timestep: " << i
                 << " with status: " << status;
      continue;
    }

    // Assign the populated features to the example.
    tensorflow::Example* example =
        google::protobuf::Arena::Create<tensorflow::Example>(&arena);
    *example->mutable_features() = std::move(*features);

    std::string serialized_example = example->SerializeAsString();
    safari::logging::McapWriteOp write_op = {
        .episode_uuid = options.episode_uuid,
        .serialized_message = serialized_example,
        .descriptor = tensorflow::Example::descriptor(),
        .topic = options.topic,
        .publish_time_ns = episode_data->timestamps()[i],
        .log_time_ns = absl::ToUnixNanos(absl::Now()),
    };
    absl::Status write_status = ProcessMcapWriteOp(write_op);
    if (!write_status.ok()) {
      LOG(ERROR) << "Failed to process episode data: " << write_status;
    }
  }
  {
    absl::MutexLock lock(pending_ops_mutex_);
    pending_ops_[options.episode_uuid] -= 1;
  }
  // Once the episode data has been processed, we can destroy the EpisodeData
  // object.
  absl::MutexLock lock(episode_data_to_be_destroyed_mutex_);
  episode_data_to_be_destroyed_.push(std::move(episode_data));
}

void ThreadPoolLogWriter::ProcessSessionData(
    safari_sdk::protos::logging::Session session, EnqueueMcapFileOptions options) {
  std::string serialized_session = session.SerializeAsString();
  safari::logging::McapWriteOp write_op = {
      .episode_uuid = options.episode_uuid,
      .serialized_message = serialized_session,
      .descriptor = safari_sdk::protos::logging::Session::descriptor(),
      .topic = options.topic,
      .publish_time_ns = options.timestamp_ns,
      .log_time_ns = absl::ToUnixNanos(absl::Now()),
  };
  absl::Status status = ProcessMcapWriteOp(write_op);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to process session data: " << status;
  }

  {
    absl::MutexLock lock(pending_ops_mutex_);
    pending_ops_[options.episode_uuid] -= 1;
  }
}

absl::Status ThreadPoolLogWriter::ProcessMcapWriteOp(const McapWriteOp& op) {
  absl::MutexLock lock(file_handles_mutex_);
  // Get the file handle for the request.
  absl::StatusOr<BaseMcapFileHandle*> file_handle = GetOrCreateFileHandle(op);
  if (!file_handle.ok()) {
    return file_handle.status();
  }
  // Write the message to the MCAP file.
  absl::Status write_status = (*file_handle)->WriteMessage(op);
  if (!write_status.ok()) {
    return write_status;
  }
  return absl::OkStatus();
}

absl::StatusOr<BaseMcapFileHandle*> ThreadPoolLogWriter::GetOrCreateFileHandle(
    const McapWriteOp& op) {
  auto it = file_handles_.find(op.episode_uuid);

  int64_t shard_count = 0;
  int64_t first_publish_time_ns = op.publish_time_ns;

  if (it != file_handles_.end()) {
    BaseMcapFileHandle* file_handle = it->second.get();
    // If the current file handle can still fit the message, then we will use
    // the current file handle.
    if ((op.serialized_message.size() +
         file_handle->total_messages_size_bytes()) <=
        mcap_file_config_.file_shard_size_limit_bytes) {
      return file_handle;
    } else {
      // Increment the shard count for the new file handle.
      shard_count = file_handle->shard_index() + 1;
      // Set the first publish time to the last publish time of the previous
      // file handle.
      first_publish_time_ns = file_handle->last_publish_time_ns();

      // Otherwise, destroy the current file handle and create a new one.
      file_handles_.erase(it);
    }
  }

  if (op.serialized_message.size() >
      mcap_file_config_.file_shard_size_limit_bytes) {
    LOG(WARNING) << "Writing message with size=" << op.serialized_message.size()
                 << " which is greater than the file_shard_size_limit_bytes: "
                 << mcap_file_config_.file_shard_size_limit_bytes
                 << " for episode: " << op.episode_uuid;
  }

  absl::StatusOr<std::unique_ptr<BaseMcapFileHandle>> new_file_handle =
      mcap_file_handle_factory_->Create(op.episode_uuid, shard_count,
                                        first_publish_time_ns,
                                        &mcap_file_config_);
  if (!new_file_handle.ok()) {
    return new_file_handle.status();
  }

  BaseMcapFileHandle* file_handle_ptr = (*new_file_handle).get();
  file_handles_[op.episode_uuid] = std::move(*new_file_handle);
  return file_handle_ptr;
}

void ThreadPoolLogWriter::ProcessEpisodeDataToBeDestroyed() {
  absl::MutexLock lock(episode_data_to_be_destroyed_mutex_);
  while (!episode_data_to_be_destroyed_.empty()) {
    std::unique_ptr<EpisodeData> episode_data =
        std::move(episode_data_to_be_destroyed_.front());
    episode_data_to_be_destroyed_.pop();
    episode_data.reset();
  }
}

}  // namespace safari::logging
