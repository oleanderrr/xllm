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

#include "core/framework/sampling/sampling_params.h"
#include "core/runtime/task_pipeline/model_input_preparer.h"
#include "core/runtime/task_pipeline/sequence_state_pool.h"

namespace xllm {

// One Slot's fixed cross-Task token indices, staging and row references.
// The pool outlives the binding. Prepare/release run on the state executor;
// gather/publish run in accepted order on the owner's single task stream.
class SequenceTokenBinding final {
 public:
  static Status create(SequenceStatePool& pool,
                       uint32_t capacity,
                       std::unique_ptr<SequenceTokenBinding>& output);
  SequenceTokenBinding(const SequenceTokenBinding&) = delete;
  SequenceTokenBinding& operator=(const SequenceTokenBinding&) = delete;

  // Model token/sequence layout and sampling indices obey their domain
  // validators; KV fields are unused. This validates the cross-Task mapping,
  // then atomically admits pool access before
  // touching final staging. All borrowed input is released on return.
  // Fully CPU-known independent tasks may omit keys; they never publish state.
  Status prepare(const ModelInputHostView& model,
                 const SamplingParameters& sampling,
                 std::span<const SequenceStateKey> keys,
                 std::span<const SequenceStateKey> retired_keys,
                 const Stream& stream);
  void gather_into(const torch::Tensor& model_tokens);
  void publish(const torch::Tensor& result_tokens);
  // Call after the last Device user, including result D2H, has completed.
  void release();

  uint64_t pinned_bytes() const { return host_indices_.nbytes(); }
  uint64_t device_bytes() const {
    return device_indices_.nbytes() + gathered_int64_.nbytes() +
           gathered_int32_.nbytes();
  }
  const torch::Tensor& host_indices() const { return host_indices_; }
  const torch::Tensor& device_indices() const { return device_indices_; }

 private:
  SequenceTokenBinding(SequenceStatePool& pool, uint32_t capacity);
  Status plan(const ModelInputHostView& model,
              const SamplingParameters& sampling,
              std::span<const SequenceStateKey> keys);

  SequenceStatePool& pool_;
  const uint32_t capacity_;
  SequenceStateLease lease_;
  torch::Tensor host_indices_;
  torch::Tensor device_indices_;
  torch::Tensor gathered_int64_;
  torch::Tensor gathered_int32_;
  torch::Tensor gather_rows_;
  torch::Tensor gather_offsets_;
  torch::Tensor publish_rows_;
  torch::Tensor gather_output_int64_;
  torch::Tensor gather_output_int32_;
  uint32_t gather_count_ = 0;
  uint32_t publish_count_ = 0;
  uint32_t model_tokens_ = 0;
  // Validation scratch is independent of the last successful Device binding.
  std::vector<SequenceStateAccess> accesses_;
  std::vector<uint32_t> gather_sequences_;
  std::vector<uint32_t> token_offsets_;
  std::vector<uint32_t> publish_sequences_;
};

}  // namespace xllm
