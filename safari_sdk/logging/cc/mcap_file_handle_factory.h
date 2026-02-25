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

#ifndef SAFARI_LOGGING_CC_MCAP_FILE_HANDLE_FACTORY_H_
#define SAFARI_LOGGING_CC_MCAP_FILE_HANDLE_FACTORY_H_

#include <cstdint>
#include <memory>
#include <string>

#include <absl/status/statusor.h>
#include "safari_sdk/logging/cc/base_mcap_file_handle_factory.h"
#include "safari_sdk/logging/cc/mcap_file_handle.h"
#include "safari_sdk/logging/cc/mcap_write_op.h"

namespace safari::logging {

// Factory class for creating McapFileHandle objects.
class McapFileHandleFactory : public BaseMcapFileHandleFactory {
 public:
  absl::StatusOr<std::unique_ptr<BaseMcapFileHandle>> Create(
      const std::string& filename_prefix, int64_t shard_index,
      int64_t first_publish_time_ns, const McapFileConfig* config) override;
};

}  // namespace safari::logging

#endif  // SAFARI_LOGGING_CC_MCAP_FILE_HANDLE_FACTORY_H_
