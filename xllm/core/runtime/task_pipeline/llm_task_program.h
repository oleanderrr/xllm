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

#include "core/framework/model/causal_lm.h"
#include "core/framework/sampling/prepared_sampler.h"
#include "core/runtime/executor.h"
#include "core/runtime/task_pipeline/model_input_binding.h"
#include "core/runtime/task_pipeline/sampling_input_binding.h"
#include "core/runtime/task_pipeline/sequence_token_binding.h"
#include "core/runtime/task_pipeline/token_result_storage.h"

namespace xllm {

// The owner establishes these model contracts before admission. Persistent
// storage and an LM-head warmup are completed before KV budgeting; KV may be
// populated afterwards.
struct LlmTaskCapacity {
  uint32_t slot_count = 1;
  uint32_t max_live_sequences = 1024;
  ModelInputCapacity model;
  uint32_t max_kv_seq_len = 0;
  uint32_t max_positions = 0;
  uint32_t block_size = 0;
  uint32_t vocab_size = 0;
  uint32_t max_unique_tokens = 0;
  uint32_t max_top_logprobs = 0;
  uint32_t hidden_size = 0;
  torch::ScalarType parameter_dtype = torch::kFloat32;
  bool chunked_prefill = false;
};

// Borrows CPU input only until submit's Prepare Ack. Explicit Sequence keys
// enable cross-Task token state; without them every input token must be known.
struct LlmTaskInput {
  ModelInputHostView model;
  ModelInputBatch batch;
  SamplingParameters sampling;
  std::span<const SequenceStateKey> sequence_state_keys;
  std::span<const SequenceStateKey> retired_sequence_state_keys;
  // Output statistics only; this does not enable graph execution.
  bool is_warmup = false;
};

// Every field is an independent CPU value when Consume returns.
struct LlmTaskOutput {
  TokenResultTensors tokens;
  torch::Tensor do_sample;
  bool is_warmup = false;
};

// One ordinary LLM program with private storage for each admitted Slot.
// Slots share streams, the original model/Executor and sampler workspace.
// The owner retires each Slot before reuse and serializes all Launch calls;
// Prepare and Consume may run beside Launch for a different Slot.
// The model, executor and KV vector outlive every Slot and its readers.
class LlmTaskProgram final {
 public:
  static Status create(CausalLM& model,
                       Executor& executor,
                       std::vector<KVCache>& kv_caches,
                       const LlmTaskCapacity& capacity,
                       std::unique_ptr<LlmTaskProgram>& output);
  ~LlmTaskProgram() = default;
  LlmTaskProgram(const LlmTaskProgram&) = delete;
  LlmTaskProgram& operator=(const LlmTaskProgram&) = delete;

  // Called once after final KV allocation and before the first admission.
  Status warmup_graphs();
  Status prepare(uint32_t slot_id, const LlmTaskInput& input);
  void launch(uint32_t slot_id);
  LlmTaskOutput consume(uint32_t slot_id);
  void discard(uint32_t slot_id);
  uint32_t slot_count() const { return capacity_.slot_count; }
  uint64_t shared_device_bytes() const {
    return sampler_->device_bytes() + sequence_state_pool_->device_bytes();
  }
  uint64_t slot_device_bytes(uint32_t slot_id) const;
  uint64_t slot_pinned_bytes(uint32_t slot_id) const;
  uint64_t pinned_bytes() const;
  uint64_t device_bytes() const;

 private:
  struct Slot {
    bool is_warmup = false;
    StreamEventPtr input_ready;
    StreamEventPtr output_ready;
    std::unique_ptr<ModelInputStorage> storage;
    std::unique_ptr<ModelInputBinding> model_input;
    std::unique_ptr<SequenceTokenBinding> sequence_tokens;
    std::unique_ptr<SamplingInputBinding> sampling_input;
    std::unique_ptr<TokenResultStorage> result;
    std::unique_ptr<PreparedSamplingInvocation> sampling;
    ModelOutput model_output;
    torch::Tensor logits;
  };

  LlmTaskProgram(CausalLM& model,
                 Executor& executor,
                 std::vector<KVCache>& kv_caches,
                 LlmTaskCapacity capacity,
                 torch::Device device);
  Status validate(const Slot& slot, const LlmTaskInput& input) const;
  void release_outputs(Slot& slot);

  CausalLM& model_;
  Executor& executor_;
  std::vector<KVCache>& kv_caches_;
  LlmTaskCapacity capacity_;
  torch::Device device_;
  std::vector<int64_t> graph_batch_sizes_;
  bool graphs_warmed_ = false;
  Stream prepare_stream_;
  Stream task_stream_;
  Stream result_stream_;
  std::unique_ptr<PreparedSampler> sampler_;
  // Destroy every Slot and its row lease before the shared pool.
  std::unique_ptr<SequenceStatePool> sequence_state_pool_;
  std::vector<std::unique_ptr<Slot>> slots_;
};

}  // namespace xllm
