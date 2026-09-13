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
#include "core/runtime/task_pipeline/token_result_storage.h"

namespace xllm {

// The owner establishes these model contracts before admission. Persistent
// storage and an LM-head warmup are completed before KV budgeting; KV may be
// populated afterwards.
struct LlmTaskCapacity {
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

// Borrows CPU input only until submit's Prepare Ack. No Device input or
// previous-task placeholder is admitted by the single-Slot ordinary program.
struct LlmTaskInput {
  ModelInputHostView model;
  ModelInputBatch batch;
  SamplingParameters sampling;
};

// One ordinary LLM Slot. Prepare/Consume use the state executor; launch uses
// the dedicated thread, with exclusive ownership transferred by the pipeline.
// The model, executor and KV vector outlive this object and all of its readers.
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

  Status prepare(const LlmTaskInput& input);
  void launch();
  TokenResultTensors consume();
  void discard();
  uint64_t pinned_bytes() const;
  uint64_t device_bytes() const;

 private:
  LlmTaskProgram(CausalLM& model,
                 Executor& executor,
                 std::vector<KVCache>& kv_caches,
                 LlmTaskCapacity capacity,
                 torch::Device device);
  Status validate(const LlmTaskInput& input) const;
  void release_outputs();

  CausalLM& model_;
  Executor& executor_;
  std::vector<KVCache>& kv_caches_;
  LlmTaskCapacity capacity_;
  torch::Device device_;
  Stream prepare_stream_;
  Stream task_stream_;
  Stream result_stream_;
  StreamEventPtr input_ready_;
  StreamEventPtr output_ready_;
  std::unique_ptr<ModelInputStorage> storage_;
  std::unique_ptr<ModelInputBinding> model_input_;
  std::unique_ptr<SamplingInputBinding> sampling_input_;
  std::unique_ptr<PreparedSampler> sampler_;
  std::unique_ptr<TokenResultStorage> result_;
  std::unique_ptr<PreparedSamplingInvocation> sampling_;
  ModelOutput model_output_;
  torch::Tensor logits_;
};

}  // namespace xllm
