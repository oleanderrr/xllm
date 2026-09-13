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

#include <cstdint>
#include <span>

#include "core/platform/stream.h"
#include "core/runtime/task_pipeline/model_input_storage.h"

namespace xllm {

// Borrowed CPU inputs, including any caller-defined padding rows. Source block
// tables are rectangular with block_table_width columns, padded with zero.
struct ModelInputHostView {
  std::span<const int32_t> token_ids;
  std::span<const int32_t> positions;
  std::span<const int32_t> new_cache_slots;
  std::span<const int32_t> q_seq_lens;
  std::span<const int32_t> kv_seq_lens;
  std::span<const int32_t> q_cu_seq_lens;
  std::span<const int32_t> block_tables;
  uint32_t block_table_width = 0;
};

struct ModelInputTransferInfo {
  uint32_t tokens = 0;
  uint32_t sequences = 0;
  uint32_t h2d_calls = 0;
  uint64_t h2d_bytes = 0;
  uint64_t device_zero_bytes = 0;
};

// Borrows storage, which must outlive this preparer. The caller must finish
// all previous input/model readers and event waits before reuse/destruction.
// No task state or implicit host synchronization is added here.
class ModelInputPreparer final {
 public:
  explicit ModelInputPreparer(ModelInputStorage& storage);
  ModelInputPreparer(const ModelInputPreparer&) = delete;
  ModelInputPreparer& operator=(const ModelInputPreparer&) = delete;

  Status validate(const ModelInputHostView& input) const;

  // Expected validation failures leave storage, event and info unchanged.
  // On success, source spans are no longer borrowed. Unused token/row capacity
  // is untouched and must not be consumed. Device failures are fatal.
  Status prepare(const ModelInputHostView& input,
                 const Stream& stream,
                 ModelInputTransferInfo& info);

  // Valid after a successful prepare. Reused, not a historical task event.
  // Completion permits reading these inputs, not reusing model-owned data.
  const StreamEventPtr& ready_event() const { return ready_event_; }

 private:
  ModelInputStorage& storage_;
  StreamEventPtr ready_event_;
};

}  // namespace xllm
