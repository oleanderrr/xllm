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

#include <torch/types.h>

#include <cstdint>
#include <memory>

#include "core/common/types.h"
#include "core/platform/stream_event.h"

namespace xllm {

class Stream;

struct TokenResultCapacity {
  uint32_t max_sequences = 0;
  uint32_t max_tokens_per_sequence = 0;
  uint32_t max_top_logprobs = 0;
};

struct TokenResultShape {
  uint32_t sequences = 0;
  uint32_t tokens_per_sequence = 0;
  uint32_t top_logprobs = 0;
  bool logprobs = false;
};

struct TokenResultTensors {
  torch::Tensor tokens;        // [S,W], int64
  torch::Tensor lengths;       // [S], int32, each in [0,W]
  torch::Tensor logprobs;      // [S,W], float32, optional
  torch::Tensor top_tokens;    // [S,W,K], int64, optional
  torch::Tensor top_logprobs;  // [S,W,K], float32, optional
};

struct TokenResultTransferInfo {
  uint64_t d2h_bytes = 0;
  uint32_t d2h_calls = 0;
};

// Owns final token outputs for one Slot. Producers write device() directly,
// including every row's length and all fields of its valid token prefix.
// Retire other readers before rebinding/destruction. Calls on different Host
// threads require the Slot's queue handoff; this is not a concurrent container.
class TokenResultStorage final {
 public:
  static Status create(const TokenResultCapacity& capacity,
                       const torch::Device& device,
                       std::unique_ptr<TokenResultStorage>& output);

  ~TokenResultStorage();
  TokenResultStorage(const TokenResultStorage&) = delete;
  TokenResultStorage& operator=(const TokenResultStorage&) = delete;

  // Establish contiguous active views without Device work. Pending results
  // must first be taken or discarded; expected rejection preserves all views.
  Status bind(TokenResultShape shape);

  // producer_ready must already be recorded after all writers. Installs the
  // dependency, submits bounded D2H and records the owned reusable ready event.
  // An empty result still waits for the producer. No Host/device synchronize.
  Status copy_to_host(const Stream& stream,
                      const StreamEventPtr& producer_ready);

  // Waits only for this result, validates lengths, then returns independent,
  // unpinned CPU tensors. Padding is -1 for token IDs and -inf for logprobs.
  TokenResultTensors take_result();

  // Idempotently retires submitted D2H without allocating a CPU response. The
  // destructor also does this; it cannot drain work not yet submitted here.
  void discard_result();

  const TokenResultTensors& device() const { return device_view_; }
  const TokenResultShape& shape() const { return shape_; }
  const TokenResultTransferInfo& transfer_info() const { return transfer_; }
  uint64_t pinned_bytes() const { return bytes_; }
  uint64_t device_bytes() const { return bytes_; }

 private:
  TokenResultStorage(TokenResultCapacity capacity,
                     torch::Device device,
                     uint64_t bytes);

  TokenResultCapacity capacity_;
  torch::Device device_;
  uint64_t bytes_ = 0;
  TokenResultTensors host_storage_;
  TokenResultTensors device_storage_;
  TokenResultTensors host_view_;
  TokenResultTensors device_view_;
  TokenResultShape shape_;
  TokenResultTransferInfo transfer_;
  std::unique_ptr<StreamEvent> result_ready_;
  bool copy_submitted_ = false;
};

}  // namespace xllm
