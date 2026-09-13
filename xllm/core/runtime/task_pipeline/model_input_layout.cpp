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

#include "core/runtime/task_pipeline/model_input_layout.h"

#include <limits>

namespace xllm {
namespace {

constexpr uint64_t kMaxTensorBytes =
    static_cast<uint64_t>(std::numeric_limits<int64_t>::max());

bool append_region(uint64_t elements,
                   ModelInputRegion& region,
                   uint64_t& total_bytes) {
  if (elements > kMaxTensorBytes / sizeof(int32_t)) {
    return false;
  }
  const uint64_t bytes = elements * sizeof(int32_t);
  const uint64_t padding =
      (kModelInputAlignment - bytes % kModelInputAlignment) %
      kModelInputAlignment;
  // total_bytes is aligned and bounded after each successful append. Check
  // both additions before performing them, including the final region's pad.
  if (bytes > kMaxTensorBytes - total_bytes ||
      padding > kMaxTensorBytes - total_bytes - bytes) {
    return false;
  }
  region = {total_bytes, bytes};
  total_bytes += bytes + padding;
  return true;
}

}  // namespace

Status make_model_input_layout(const ModelInputCapacity& capacity,
                               ModelInputLayout& layout) {
  if (capacity.max_tokens == 0 || capacity.max_sequences == 0 ||
      capacity.max_blocks_per_sequence == 0) {
    return Status(StatusCode::INVALID_ARGUMENT,
                  "Model input capacities must all be positive.");
  }

  ModelInputLayout planned;
  planned.capacity = capacity;
  planned.block_table_row_stride_bytes =
      static_cast<uint64_t>(capacity.max_blocks_per_sequence) * sizeof(int32_t);
  // Each dimension is uint32_t, so their product fits after widening both.
  const uint64_t block_elements =
      static_cast<uint64_t>(capacity.max_sequences) *
      static_cast<uint64_t>(capacity.max_blocks_per_sequence);
  if (!append_region(
          capacity.max_tokens, planned.token_ids, planned.total_bytes) ||
      !append_region(
          capacity.max_tokens, planned.positions, planned.total_bytes) ||
      !append_region(
          capacity.max_tokens, planned.new_cache_slots, planned.total_bytes) ||
      !append_region(
          capacity.max_sequences, planned.q_seq_lens, planned.total_bytes) ||
      !append_region(
          capacity.max_sequences, planned.kv_seq_lens, planned.total_bytes) ||
      !append_region(
          capacity.max_sequences, planned.q_cu_seq_lens, planned.total_bytes) ||
      !append_region(
          block_elements, planned.block_tables, planned.total_bytes)) {
    return Status(StatusCode::INVALID_ARGUMENT,
                  "Model input layout exceeds signed 64-bit tensor capacity.");
  }
  layout = planned;
  return Status();
}

}  // namespace xllm
