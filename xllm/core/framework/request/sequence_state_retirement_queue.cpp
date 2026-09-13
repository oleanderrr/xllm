/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "core/framework/request/sequence_state_retirement_queue.h"

#include <glog/logging.h>

namespace xllm {

void SequenceStateRetirementQueue::retire(SequenceStateKey key) {
  CHECK_NE(key.sequence_id, 0U);
  std::lock_guard<std::mutex> lock(mutex_);
  pending_.emplace_back(key);
}

std::vector<SequenceStateKey> SequenceStateRetirementQueue::drain() {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<SequenceStateKey> result(pending_);
  pending_.clear();
  return result;
}

}  // namespace xllm
