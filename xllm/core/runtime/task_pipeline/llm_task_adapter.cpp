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

#include "core/runtime/task_pipeline/llm_task_adapter.h"

#include <algorithm>
#include <limits>

namespace xllm {
namespace {

bool cpu_int_tensor(const torch::Tensor& tensor, int32_t dimensions) {
  return tensor.defined() && tensor.device().is_cpu() &&
         tensor.scalar_type() == torch::kInt32 && tensor.dim() == dimensions &&
         tensor.is_contiguous();
}

std::span<const int32_t> int_span(const torch::Tensor& tensor) {
  if (!tensor.defined() || tensor.numel() == 0) {
    return {};
  }
  return {tensor.const_data_ptr<int32_t>(),
          static_cast<uint64_t>(tensor.numel())};
}

}  // namespace

Status make_llm_task_input(const ForwardInput& source, LlmTaskInput& output) {
  const auto& params = source.input_params;
  const auto& host = params.attention.host;
  const auto& meta = params.meta;
  const auto& embedding = params.embedding;
  const auto& copy = params.block_copy;
  if (source.device_tensors_ready || source.metadata_ready_event != nullptr ||
      !source.retained_device_tensors.empty() ||
      source.step_decode.has_value() || source.skip_sampling_for_logits_only ||
      source.return_selected_hidden || !source.transfer_kv_infos.empty() ||
      !source.json_object_states.empty() ||
      !source.json_object_state_snapshots.empty() ||
      !source.json_object_invalid_draft.empty() ||
      !source.json_object_errors.empty() || params.is_spec_verify ||
      params.prefill_without_cache || !params.linear_state_cache_ops.empty() ||
      !params.linear_state_validity_mask.empty() ||
      !params.multi_block_tables.empty() || params.mtp_topk_state != nullptr ||
      params.mtp_shifted_token_ids.defined() ||
      params.num_accepted_tokens.defined() ||
      !params.num_accepted_tokens_host.empty() ||
      !std::holds_alternative<std::monostate>(params.rec_params) ||
      embedding.input_embedding.defined() ||
      embedding.mtp_shifted_token_ids.defined() ||
      !embedding.mtp_bootstrap_row_idxes.empty() ||
      embedding.mtp_bootstrap_embeddings.defined() ||
      !copy.swap_blocks.empty() || copy.src_block_indices.defined() ||
      copy.dst_block_indices.defined() || copy.cum_sum.defined() ||
      params.multimodal.mm_data.valid() ||
      !params.multimodal.deep_stacks.empty() ||
      params.parallel.cp_plan.enabled() ||
      params.parallel.layer_wise_load_synchronizer != nullptr ||
      params.parallel.dp_global_token_nums.size() > 1 ||
      params.expert.expert_load_data.defined() ||
      params.expert.expert_array.defined() ||
      params.expert.eplb_decode_token_mask.defined() ||
      !host.ring_cur_seqlen.empty() || !host.ring_cache_seqlen.empty()) {
    return Status(StatusCode::INVALID_ARGUMENT,
                  "Task pipeline requires ordinary CPU LLM input without "
                  "speculative, structured, transfer or recurrent state.");
  }
  // Full-attention rows use -1 to keep this transport field row-aligned.
  // Only those inactive sentinels may be omitted from the prepared program.
  const auto inactive = [](int32_t id) { return id == -1; };
  const auto& linear_ids = embedding.linear_state_ids;
  const auto& linear_indices = embedding.linear_state_indices;
  if ((!linear_ids.empty() &&
       (linear_ids.size() != host.q_seq_lens.size() ||
        !std::all_of(linear_ids.begin(), linear_ids.end(), inactive))) ||
      (linear_indices.defined() &&
       (!cpu_int_tensor(linear_indices, /*dimensions=*/1) ||
        linear_indices.numel() !=
            static_cast<int64_t>(host.q_seq_lens.size()) ||
        !std::all_of(int_span(linear_indices).begin(),
                     int_span(linear_indices).end(),
                     inactive)))) {
    return Status(
        StatusCode::INVALID_ARGUMENT,
        "Task pipeline does not support active linear-attention state.");
  }
  const auto& tokens = source.host_token_ids();
  const auto& slots = params.attention.device.new_cache_slots;
  if (source.input_host_buffer_has_layout ||
      (tokens.defined() && !cpu_int_tensor(tokens, /*dimensions=*/1)) ||
      (source.host_positions().defined() &&
       !cpu_int_tensor(source.host_positions(), /*dimensions=*/1)) ||
      (host.block_tables.defined() &&
       !cpu_int_tensor(host.block_tables, /*dimensions=*/2)) ||
      (host.new_cache_slots.empty() && slots.defined() &&
       !cpu_int_tensor(slots, /*dimensions=*/1)) ||
      params.graph.tiling_data.defined() ||
      params.graph.input_tokens_override.defined() ||
      params.graph.use_expanded_decode_for_spec_verify_attention ||
      params.parallel.layer_synchronizer != nullptr) {
    return Status(
        StatusCode::INVALID_ARGUMENT,
        "Task pipeline requires unpacked CPU input and no graph overrides.");
  }
  const auto& positions = source.host_positions();
  const bool empty = !tokens.defined() || tokens.numel() == 0;
  const int64_t rows = static_cast<int64_t>(host.q_seq_lens.size());
  if (meta.num_sequences < 0 || meta.actual_num_sequences < 0 ||
      meta.num_sequences != rows ||
      source.sequence_state_keys.size() != static_cast<uint64_t>(rows) ||
      (meta.actual_num_sequences != 0 && meta.actual_num_sequences != rows) ||
      (empty && rows != 0) ||
      (!empty && (!cpu_int_tensor(tokens, /*dimensions=*/1) ||
                  !cpu_int_tensor(positions, /*dimensions=*/1) ||
                  !cpu_int_tensor(host.block_tables, /*dimensions=*/2) ||
                  host.block_tables.size(/*dim=*/0) != rows ||
                  host.block_tables.size(/*dim=*/1) >
                      std::numeric_limits<int32_t>::max()))) {
    return Status(StatusCode::INVALID_ARGUMENT,
                  "Task pipeline requires unpadded CPU int32 model input.");
  }
  // BatchInputBuilder leaves actual_num_sequences unset before Worker prepare.
  // ProfileManager also marks ordinary eager warmup as is_graph_warmup.
  // It is an output-metrics marker here, not permission for graph execution.
  // The Slot preserves it in the detached response; this invocation stays
  // eager.
  LlmTaskInput input;
  input.batch = {meta.batch_forward_type,
                 static_cast<uint32_t>(rows),
                 meta.batch_id,
                 false};
  input.model = {
      int_span(tokens),
      int_span(positions),
      host.new_cache_slots.empty()
          ? int_span(slots)
          : std::span<const int32_t>(host.new_cache_slots),
      host.q_seq_lens,
      host.kv_seq_lens,
      host.q_cu_seq_lens,
      int_span(host.block_tables),
      empty ? 0 : static_cast<uint32_t>(host.block_tables.size(/*dim=*/1))};
  // Ordinary BatchInputBuilder also supplies embedding/extra-token IDs and
  // singleton DP summaries. They describe scheduler bookkeeping only: the
  // ordinary program reads token/position/KV views, not those algorithm fields.
  input.sampling = source.sampling_params;
  input.is_warmup = meta.is_graph_warmup;
  input.sequence_state_keys = source.sequence_state_keys;
  input.retired_sequence_state_keys = source.retired_sequence_state_keys;
  output = std::move(input);
  return Status();
}

ForwardOutput make_llm_task_output(LlmTaskOutput result) {
  ForwardOutput output;
  output.cpu_ready = true;
  output.do_sample = std::move(result.do_sample);
  output.is_graph_warmup = result.is_warmup;
  const TokenResultTensors& tokens = result.tokens;
  if (!tokens.tokens.defined()) {
    return output;
  }
  CHECK(tokens.tokens.device().is_cpu());
  CHECK_EQ(tokens.tokens.size(/*dim=*/1), 1);
  output.sample_output.next_tokens = tokens.tokens.squeeze(/*dim=*/1);
  if (tokens.logprobs.defined()) {
    output.logprobs = true;
    output.sample_output.logprobs = tokens.logprobs.squeeze(/*dim=*/1);
  }
  if (tokens.top_tokens.defined()) {
    output.max_top_logprobs = tokens.top_tokens.size(/*dim=*/2);
    output.sample_output.top_tokens = tokens.top_tokens.squeeze(/*dim=*/1);
    output.sample_output.top_logprobs = tokens.top_logprobs.squeeze(/*dim=*/1);
  }
  return output;
}

}  // namespace xllm
