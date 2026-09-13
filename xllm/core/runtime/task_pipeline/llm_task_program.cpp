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
  if (capacity.slot_count == 0 || capacity.slot_count > 2 ||
      model.device().type() != torch::kPrivateUse1 ||
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
  const uint32_t rows = capacity.model.max_sequences;
  program->slots_.reserve(capacity.slot_count);
  for (uint32_t slot_id = 0; slot_id < capacity.slot_count; ++slot_id) {
    auto slot = std::make_unique<Slot>();
    Status status = ModelInputStorage::create(
        capacity.model, model.device(), slot->storage);
    if (!status.ok()) {
      return status;
    }
    slot->model_input = std::make_unique<ModelInputBinding>(*slot->storage);
    status = SamplingInputBinding::create({rows,
                                           rows,
                                           capacity.max_unique_tokens,
                                           capacity.vocab_size,
                                           capacity.max_top_logprobs},
                                          model.device(),
                                          capacity.parameter_dtype,
                                          slot->sampling_input);
    if (!status.ok()) {
      return status;
    }
    status = TokenResultStorage::create(
        {rows, 1, capacity.max_top_logprobs}, model.device(), slot->result);
    if (!status.ok()) {
      return status;
    }
    slot->input_ready = reusable_event();
    slot->output_ready = reusable_event();
    program->slots_.emplace_back(std::move(slot));
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
  Status status = PreparedSampler::create(
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
  // Initialization is outside serving. Complete allocation-stream writes
  // before handing storage and native plans to either execution thread.
  Stream initialization(c10_npu::getCurrentNPUStream(model.device().index()));
  CHECK_EQ(initialization.synchronize(), 0);
  output = std::move(program);
  return Status();
}

Status LlmTaskProgram::validate(const Slot& slot,
                                const LlmTaskInput& input) const {
  Status status = slot.model_input->validate(input.model, input.batch);
  if (!status.ok()) {
    return status;
  }
  status = slot.sampling_input->validate(input.sampling,
                                         input.model.token_ids.size());
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

Status LlmTaskProgram::prepare(uint32_t slot_id, const LlmTaskInput& input) {
  if (slot_id >= slots_.size()) {
    return invalid("Invalid LLM Slot index.");
  }
  Slot& slot = *slots_[slot_id];
  Status status = validate(slot, input);
  if (!status.ok()) {
    return status;
  }
  c10::DeviceGuard guard(device_);
  // All expected admission failures precede any Slot or Device writes.
  status = slot.model_input->prepare(input.model, input.batch, prepare_stream_);
  CHECK(status.ok()) << status.message();
  status = slot.sampling_input->prepare(
      input.sampling, input.model.token_ids.size(), prepare_stream_);
  CHECK(status.ok()) << status.message();
  if (!input.model.token_ids.empty()) {
    executor_.prepare_attention_metadata(kv_caches_,
                                         slot.model_input->params());
  }
  const auto& params = slot.sampling_input->params();
  const uint32_t samples =
      params.sample_idxes.defined() ? params.sample_idxes.numel() : 0;
  status = slot.result->bind(
      {samples,
       samples == 0 ? 0U : 1U,
       params.logprobs ? static_cast<uint32_t>(params.max_top_logprobs) : 0U,
       params.logprobs});
  CHECK(status.ok()) << status.message();
  status = sampler_->bind(params,
                          final_views(slot.result->device()),
                          slot.result->device().lengths,
                          slot.sampling);
  CHECK(status.ok()) << status.message();
  record(prepare_stream_, slot.input_ready);
  return Status();
}

void LlmTaskProgram::launch(uint32_t slot_id) {
  CHECK_LT(slot_id, slots_.size());
  Slot& slot = *slots_[slot_id];
  c10::DeviceGuard device_guard(device_);
  auto guard = task_stream_.set_stream_guard();
  CHECK_EQ(aclrtStreamWaitEvent(task_stream_.get_stream()->stream(),
                                slot.input_ready->npu_event()),
           ACL_SUCCESS)
      << "Failed to wait for LLM task input.";
  if (slot.model_input->tokens().numel() != 0) {
    slot.model_output = executor_.forward(slot.model_input->tokens(),
                                          slot.model_input->positions(),
                                          kv_caches_,
                                          slot.model_input->params());
  }
  const auto& params = slot.sampling_input->params();
  if (params.selected_token_idxes.defined() &&
      params.selected_token_idxes.numel() != 0) {
    slot.logits = model_.logits(slot.model_output.hidden_states,
                                params.selected_token_idxes);
    slot.sampling->run(slot.logits);
  }
  record(task_stream_, slot.output_ready);
  const Status status =
      slot.result->copy_to_host(result_stream_, slot.output_ready);
  CHECK(status.ok()) << status.message();
}

TokenResultTensors LlmTaskProgram::consume(uint32_t slot_id) {
  CHECK_LT(slot_id, slots_.size());
  Slot& slot = *slots_[slot_id];
  c10::DeviceGuard guard(device_);
  TokenResultTensors result = slot.result->take_result();
  release_outputs(slot);
  return result;
}

void LlmTaskProgram::discard(uint32_t slot_id) {
  CHECK_LT(slot_id, slots_.size());
  Slot& slot = *slots_[slot_id];
  c10::DeviceGuard guard(device_);
  slot.result->discard_result();
  release_outputs(slot);
}

void LlmTaskProgram::release_outputs(Slot& slot) {
  slot.logits = torch::Tensor();
  slot.model_output = ModelOutput();
  slot.sampling.reset();
}

uint64_t LlmTaskProgram::slot_pinned_bytes(uint32_t slot_id) const {
  CHECK_LT(slot_id, slots_.size());
  const Slot& slot = *slots_[slot_id];
  return slot.storage->host_buffer().nbytes() +
         slot.sampling_input->pinned_bytes() + slot.result->pinned_bytes();
}

uint64_t LlmTaskProgram::slot_device_bytes(uint32_t slot_id) const {
  CHECK_LT(slot_id, slots_.size());
  const Slot& slot = *slots_[slot_id];
  return slot.storage->device_buffer().nbytes() +
         slot.sampling_input->device_bytes() + slot.result->device_bytes();
}

uint64_t LlmTaskProgram::pinned_bytes() const {
  uint64_t bytes = 0;
  for (uint32_t slot_id = 0; slot_id < slots_.size(); ++slot_id) {
    bytes += slot_pinned_bytes(slot_id);
  }
  return bytes;
}

uint64_t LlmTaskProgram::device_bytes() const {
  uint64_t bytes = shared_device_bytes();
  for (uint32_t slot_id = 0; slot_id < slots_.size(); ++slot_id) {
    bytes += slot_device_bytes(slot_id);
  }
  return bytes;
}

}  // namespace xllm
