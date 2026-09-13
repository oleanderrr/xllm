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

#include "core/common/types.h"

namespace xllm {

inline constexpr uint64_t kModelInputAlignment = 16;

// Capacities include any padding required by the owning execution binding.
struct ModelInputCapacity {
  uint32_t max_tokens = 0;
  uint32_t max_sequences = 0;
  uint32_t max_blocks_per_sequence = 0;
};

struct ModelInputRegion {
  uint64_t offset = 0;
  // Data bytes for the full capacity, excluding inter-field alignment padding.
  uint64_t bytes = 0;
};

// Fixed int32 input layout for ordinary NPU LLM paged attention. Positions have
// one axis; q_cu_seq_lens stores cumulative ends without a leading zero.
struct ModelInputLayout {
  ModelInputCapacity capacity;
  ModelInputRegion token_ids;
  ModelInputRegion positions;
  ModelInputRegion new_cache_slots;
  ModelInputRegion q_seq_lens;
  ModelInputRegion kv_seq_lens;
  ModelInputRegion q_cu_seq_lens;
  ModelInputRegion block_tables;
  uint64_t block_table_row_stride_bytes = 0;
  uint64_t total_bytes = 0;
};

// Plans all seven regions without allocating input storage. Reuse the result
// across tasks with different valid lengths. Storage must provide an aligned
// base and preserve block table row stride; total_bytes is not an H2D copy
// size. Rejects zero capacities or layouts exceeding signed 64-bit tensor
// sizes. On failure, layout is unchanged, including when capacity aliases its
// member.
Status make_model_input_layout(const ModelInputCapacity& capacity,
                               ModelInputLayout& layout);

}  // namespace xllm
