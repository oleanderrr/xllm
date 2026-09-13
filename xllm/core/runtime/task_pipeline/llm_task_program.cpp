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

#include "core/runtime/task_pipeline/llm_task_program.h"

#include <c10/core/DeviceGuard.h>

#include <algorithm>
#include <limits>

namespace xllm {
namespace {

Status invalid(const char* message) {
  return Status(StatusCode::INVALID_ARGUMENT, message);
}

StreamEventPtr reusable_event() {
  aclrtEvent event = nullptr;
  CHECK_EQ(aclrtCreateEventExWithFlag(&event, ACL_EVENT_SYNC), ACL_SUCCESS);
  return std::make_shared<StreamEvent>(event);
}

void record(const Stream& stream, const StreamEventPtr& event) {
  CHECK_EQ(aclrtRecordEvent(event->npu_event(), stream.get_stream()->stream()),
           ACL_SUCCESS);
}

SampleOutput final_views(const TokenResultTensors& tensors) {
  SampleOutput output;
  if (!tensors.tokens.defined()) {
    return output;
  }
  output.next_tokens = tensors.tokens.squeeze(/*dim=*/1);
  if (tensors.logprobs.defined()) {
    output.logprobs = tensors.logprobs.squeeze(/*dim=*/1);
  }
  if (tensors.top_tokens.defined()) {
    output.top_tokens = tensors.top_tokens.squeeze(/*dim=*/1);
    output.top_logprobs = tensors.top_logprobs.squeeze(/*dim=*/1);
  }
  return output;
}

}  // namespace

LlmTaskProgram::LlmTaskProgram(CausalLM& model,
                               Executor& executor,
                               std::vector<KVCache>& kv_caches,
                               LlmTaskCapacity capacity,
                               torch::Device device)
    : model_(model),
      executor_(executor),
      kv_caches_(kv_caches),
      capacity_(std::move(capacity)),
      device_(device),
      prepare_stream_(device),
      task_stream_(device),
      result_stream_(device) {}

Status LlmTaskProgram::create(CausalLM& model,
                              Executor& executor,
                              std::vector<KVCache>& kv_caches,
                              const LlmTaskCapacity& capacity,
                              std::unique_ptr<LlmTaskProgram>& output) {
  if (!executor.supports_prepared_attention_metadata()) {
    return invalid(
        "LLM task pipeline requires supported Python prepared metadata.");
  }
  if (model.device().type() != torch::kPrivateUse1 ||
      !model.device().has_index() || capacity.max_kv_seq_len == 0 ||
      capacity.max_positions == 0 || capacity.block_size == 0 ||
      capacity.hidden_size == 0 ||
      capacity.hidden_size > std::numeric_limits<int32_t>::max() ||
      capacity.max_positions > std::numeric_limits<int32_t>::max() ||
      capacity.block_size > std::numeric_limits<int32_t>::max() ||
      capacity.max_kv_seq_len > capacity.max_positions) {
    return invalid("Invalid ordinary LLM capacity or indexed NPU device.");
  }
  c10::DeviceGuard guard(model.device());
  auto program = std::unique_ptr<LlmTaskProgram>(
      new LlmTaskProgram(model, executor, kv_caches, capacity, model.device()));
  Status status = ModelInputStorage::create(
      capacity.model, model.device(), program->storage_);
  if (!status.ok()) {
    return status;
  }
  program->model_input_ =
      std::make_unique<ModelInputBinding>(*program->storage_);
  const uint32_t rows = capacity.model.max_sequences;
  status = SamplingInputBinding::create({rows,
                                         rows,
                                         capacity.max_unique_tokens,
                                         capacity.vocab_size,
                                         capacity.max_top_logprobs},
                                        model.device(),
                                        capacity.parameter_dtype,
                                        program->sampling_input_);
  if (!status.ok()) {
    return status;
  }
  // Resolve the actual loaded LM head's output contract during initialization.
  // A single deterministic row needs no KV, consumes no RNG and never occurs
  // in admission/Launch. Keep its inputs/output alive through initialization.
  auto hidden = torch::zeros({1, capacity.hidden_size}, model.options());
  auto selected = torch::zeros({1}, model.options().dtype(torch::kInt32));
  auto logits = model.logits(hidden, selected);
  if (logits.dim() != 2 || logits.size(/*dim=*/0) != 1 ||
      logits.size(/*dim=*/1) != capacity.vocab_size ||
      logits.stride(/*dim=*/1) != 1 ||
      logits.stride(/*dim=*/0) < capacity.vocab_size ||
      logits.stride(/*dim=*/0) > std::numeric_limits<uint32_t>::max()) {
    Stream initialization(c10_npu::getCurrentNPUStream(model.device().index()));
    CHECK_EQ(initialization.synchronize(), 0);
    return invalid("LM head output does not match the declared vocabulary.");
  }
  LOG(INFO) << "Prepared LLM logits dtype=" << logits.scalar_type()
            << " vocab=" << logits.size(/*dim=*/1)
            << " row_stride=" << logits.stride(/*dim=*/0);
  Stream warmup(c10_npu::getCurrentNPUStream(model.device().index()));
  CHECK_EQ(warmup.synchronize(), 0);
  status = PreparedSampler::create(
      {{rows, rows, capacity.max_unique_tokens, capacity.vocab_size},
       capacity.max_top_logprobs,
       static_cast<uint32_t>(logits.stride(/*dim=*/0))},
      model.device(),
      logits.scalar_type(),
      capacity.parameter_dtype,
      program->sampler_);
  if (!status.ok()) {
    return status;
  }
  status = TokenResultStorage::create(
      {rows, 1, capacity.max_top_logprobs}, model.device(), program->result_);
  if (!status.ok()) {
    return status;
  }
  program->input_ready_ = reusable_event();
  program->output_ready_ = reusable_event();
  // Initialization is outside serving. Complete allocation-stream writes
  // before handing storage and native plans to either execution thread.
  Stream initialization(c10_npu::getCurrentNPUStream(model.device().index()));
  CHECK_EQ(initialization.synchronize(), 0);
  output = std::move(program);
  return Status();
}

Status LlmTaskProgram::validate(const LlmTaskInput& input) const {
  Status status = model_input_->validate(input.model, input.batch);
  if (!status.ok()) {
    return status;
  }
  status =
      sampling_input_->validate(input.sampling, input.model.token_ids.size());
  if (!status.ok()) {
    return status;
  }
  const auto& host = input.model;
  if (input.batch.num_actual_sequences != host.q_seq_lens.size() ||
      input.batch.is_graph_warmup || input.sampling.return_probs ||
      (!input.sampling.logprobs && input.sampling.max_top_logprobs != 0)) {
    return invalid(
        "Ordinary LLM requires actual eager rows and token results.");
  }
  if (host.token_ids.empty()) {
    return Status();
  }
  if (kv_caches_.empty() || kv_caches_.front().empty()) {
    return Status(StatusCode::UNAVAILABLE, "KV cache is not allocated.");
  }
  const int64_t blocks = kv_caches_.front().get_k_cache().size(/*dim=*/0);
  const int64_t block_size = capacity_.block_size;
  int64_t offset = 0;
  for (uint32_t row = 0; row < host.q_seq_lens.size(); ++row) {
    const int32_t q = host.q_seq_lens[row];
    const int32_t kv = host.kv_seq_lens[row];
    if (static_cast<uint32_t>(kv) > capacity_.max_kv_seq_len ||
        (!capacity_.chunked_prefill && !input.batch.forward_type.is_decode() &&
         q != kv) ||
        (kv + block_size - 1) / block_size > host.block_table_width) {
      return invalid("KV length or prefill mode exceeds the LLM contract.");
    }
    for (int32_t index = 0; index < q; ++index, ++offset) {
      const int32_t position = host.positions[offset];
      const int32_t token = host.token_ids[offset];
      if (position != kv - q + index || position < 0 ||
          static_cast<uint32_t>(position) >= capacity_.max_positions ||
          token < 0 || static_cast<uint32_t>(token) >= capacity_.vocab_size) {
        return invalid("Invalid ordinary token or rotary position.");
      }
      const int32_t block = host.block_tables[static_cast<uint64_t>(row) *
                                                  host.block_table_width +
                                              position / block_size];
      if (host.new_cache_slots[offset] !=
          block * block_size + position % block_size) {
        return invalid("KV write slot does not match the row's page table.");
      }
    }
  }
  if (std::any_of(
          host.block_tables.begin(),
          host.block_tables.end(),
          [blocks](int32_t block) { return block < 0 || block >= blocks; })) {
    return invalid("KV page index is outside allocated cache.");
  }
  return Status();
}

Status LlmTaskProgram::prepare(const LlmTaskInput& input) {
  Status status = validate(input);
  if (!status.ok()) {
    return status;
  }
  c10::DeviceGuard guard(device_);
  // All expected admission failures precede any Slot or Device writes.
  status = model_input_->prepare(input.model, input.batch, prepare_stream_);
  CHECK(status.ok()) << status.message();
  status = sampling_input_->prepare(
      input.sampling, input.model.token_ids.size(), prepare_stream_);
  CHECK(status.ok()) << status.message();
  if (!input.model.token_ids.empty()) {
    executor_.prepare_attention_metadata(kv_caches_, model_input_->params());
  }
  const auto& params = sampling_input_->params();
  const uint32_t samples =
      params.sample_idxes.defined() ? params.sample_idxes.numel() : 0;
  status = result_->bind(
      {samples,
       samples == 0 ? 0U : 1U,
       params.logprobs ? static_cast<uint32_t>(params.max_top_logprobs) : 0U,
       params.logprobs});
  CHECK(status.ok()) << status.message();
  status = sampler_->bind(params,
                          final_views(result_->device()),
                          result_->device().lengths,
                          sampling_);
  CHECK(status.ok()) << status.message();
  record(prepare_stream_, input_ready_);
  return Status();
}

void LlmTaskProgram::launch() {
  c10::DeviceGuard device_guard(device_);
  auto guard = task_stream_.set_stream_guard();
  CHECK_EQ(aclrtStreamWaitEvent(task_stream_.get_stream()->stream(),
                                input_ready_->npu_event()),
           ACL_SUCCESS)
      << "Failed to wait for LLM task input.";
  if (model_input_->tokens().numel() != 0) {
    model_output_ = executor_.forward(model_input_->tokens(),
                                      model_input_->positions(),
                                      kv_caches_,
                                      model_input_->params());
  }
  const auto& params = sampling_input_->params();
  if (params.selected_token_idxes.defined() &&
      params.selected_token_idxes.numel() != 0) {
    logits_ =
        model_.logits(model_output_.hidden_states, params.selected_token_idxes);
    sampling_->run(logits_);
  }
  record(task_stream_, output_ready_);
  const Status status = result_->copy_to_host(result_stream_, output_ready_);
  CHECK(status.ok()) << status.message();
}

TokenResultTensors LlmTaskProgram::consume() {
  c10::DeviceGuard guard(device_);
  TokenResultTensors result = result_->take_result();
  release_outputs();
  return result;
}

void LlmTaskProgram::discard() {
  c10::DeviceGuard guard(device_);
  result_->discard_result();
  release_outputs();
}

void LlmTaskProgram::release_outputs() {
  logits_ = torch::Tensor();
  model_output_ = ModelOutput();
  sampling_.reset();
}

uint64_t LlmTaskProgram::pinned_bytes() const {
  return storage_->host_buffer().nbytes() + sampling_input_->pinned_bytes() +
         result_->pinned_bytes();
}

uint64_t LlmTaskProgram::device_bytes() const {
  return storage_->device_buffer().nbytes() + sampling_input_->device_bytes() +
         sampler_->device_bytes() + result_->device_bytes();
}

}  // namespace xllm
