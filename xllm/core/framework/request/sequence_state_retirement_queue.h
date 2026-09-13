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

#pragma once

#include <mutex>
#include <vector>

#include "core/framework/request/sequence_state_key.h"

namespace xllm {

// CPU lifecycle notifications for one Engine/DP owner. Sequence destruction
// may run on a response thread; the Engine drains on its dispatch thread.
// This queue neither owns Device state nor authorizes reuse of a leased row.
class SequenceStateRetirementQueue final {
 public:
  void retire(SequenceStateKey key);
  // Returns each queued key once and retains the internal vector's capacity.
  std::vector<SequenceStateKey> drain();

 private:
  std::mutex mutex_;
  std::vector<SequenceStateKey> pending_;
};

}  // namespace xllm
