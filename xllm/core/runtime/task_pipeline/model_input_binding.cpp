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

#include "core/runtime/task_pipeline/model_input_binding.h"

#include <algorithm>
#include <limits>
#include <numeric>

#include "core/layers/common/attention_metadata.h"

namespace xllm {
namespace {

Status validate_batch(const ModelInputHostView& input,
                      const ModelInputBatch& batch,
                      uint32_t max_sequences) {
  const uint64_t rows = input.q_seq_lens.size();
  if (rows > max_sequences ||
      rows > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) ||
      input.kv_seq_lens.size() != rows || batch.num_actual_sequences > rows) {
    return Status(StatusCode::INVALID_ARGUMENT,
                  "Invalid physical or actual model input rows.");
  }
  switch (batch.forward_type.value()) {
    case BatchForwardType::EMPTY:
      if (rows != 0 || !input.token_ids.empty()) {
        return Status(StatusCode::INVALID_ARGUMENT,
                      "Empty model input batch contains rows or tokens.");
      }
      return Status();
    case BatchForwardType::PREFILL:
    case BatchForwardType::CHUNKED_PREFILL:
    case BatchForwardType::DECODE:
    case BatchForwardType::MIXED:
      break;
    default:
      return Status(StatusCode::INVALID_ARGUMENT,
                    "Unknown model input forward type.");
  }
  if (rows == 0) {
    return Status(StatusCode::INVALID_ARGUMENT,
                  "Nonempty model input batch requires physical rows.");
  }
  for (uint32_t row = 0; row < batch.num_actual_sequences; ++row) {
    if (input.q_seq_lens[row] <= 0 ||
        (batch.forward_type.is_prefill() &&
         input.kv_seq_lens[row] != input.q_seq_lens[row])) {
      return Status(StatusCode::INVALID_ARGUMENT,
                    "Invalid query or cache lengths for actual model rows.");
    }
  }
  if (batch.forward_type.is_decode() &&
      std::any_of(input.q_seq_lens.begin(),
                  input.q_seq_lens.end(),
                  [](int32_t length) { return length > 1; })) {
    return Status(StatusCode::INVALID_ARGUMENT,
                  "Decode model input rows must contain at most one token.");
  }
  return Status();
}

void assign_prefix(std::vector<int32_t>& output,
                   const torch::Tensor& staging,
                   uint32_t count) {
  const int32_t* values = staging.data_ptr<int32_t>();
  output.assign(values, values + count);
}

int32_t maximum(const std::vector<int32_t>& values) {
  return values.empty() ? 0 : *std::max_element(values.begin(), values.end());
}

}  // namespace

ModelInputBinding::ModelInputBinding(ModelInputStorage& storage)
    : storage_(storage), preparer_(storage) {
  const ModelInputCapacity& capacity = storage.layout().capacity;
  AttentionHostInput& host = params_.attention.host;
  for (std::vector<int32_t>* lengths : {&host.q_seq_lens,
                                        &host.q_cu_seq_lens,
                                        &host.kv_seq_lens,
                                        &host.kv_cache_tokens_nums}) {
    lengths->reserve(capacity.max_sequences);
  }
  host.new_cache_slots.reserve(capacity.max_tokens);
  params_.attn_metadata = std::make_shared<layer::AttentionMetadata>();
  auto& metadata = *params_.attn_metadata;
  metadata.q_seq_lens_vec.reserve(capacity.max_sequences);
  metadata.kv_seq_lens_vec.reserve(capacity.max_sequences);
  metadata.q_cu_seq_lens_host_vec.reserve(capacity.max_sequences);
}

Status ModelInputBinding::prepare(const ModelInputHostView& input,
                                  const ModelInputBatch& batch,
                                  const Stream& stream) {
  Status status =
      validate_batch(input, batch, storage_.layout().capacity.max_sequences);
  if (!status.ok()) {
    return status;
  }
  ModelInputTransferInfo transferred;
  status = preparer_.prepare(input, stream, transferred);
  if (!status.ok()) {
    return status;
  }

  const uint32_t rows = transferred.sequences;
  const uint32_t token_count = transferred.tokens;
  const ModelInputTensors& staging = storage_.host();
  AttentionHostInput& host = params_.attention.host;
  // Read our staging only: input spans may borrow the previous Host metadata.
  assign_prefix(host.q_seq_lens, staging.q_seq_lens, rows);
  assign_prefix(host.q_cu_seq_lens, staging.q_cu_seq_lens, rows);
  assign_prefix(host.kv_seq_lens, staging.kv_seq_lens, rows);
  assign_prefix(host.new_cache_slots, staging.new_cache_slots, token_count);
  host.kv_cache_tokens_nums.resize(rows);
  for (uint32_t row = 0; row < rows; ++row) {
    host.kv_cache_tokens_nums[row] =
        host.kv_seq_lens[row] - host.q_seq_lens[row];
  }
  host.block_tables = staging.block_tables.narrow(/*dim=*/0, /*start=*/0, rows);
  host.graph_q_seq_lens_data = staging.q_seq_lens.data_ptr<int32_t>();
  host.graph_kv_seq_lens_data = staging.kv_seq_lens.data_ptr<int32_t>();

  const ModelInputTensors& device = storage_.device();
  tokens_ = device.token_ids.narrow(/*dim=*/0, /*start=*/0, token_count);
  positions_ = device.positions.narrow(/*dim=*/0, /*start=*/0, token_count);
  AttentionDeviceInput& attention = params_.attention.device;
  attention.q_seq_lens = device.q_seq_lens.narrow(/*dim=*/0, /*start=*/0, rows);
  attention.kv_seq_lens =
      device.kv_seq_lens.narrow(/*dim=*/0, /*start=*/0, rows);
  attention.q_cu_seq_lens =
      device.q_cu_seq_lens.narrow(/*dim=*/0, /*start=*/0, rows);
  attention.new_cache_slots =
      device.new_cache_slots.narrow(/*dim=*/0, /*start=*/0, token_count);
  attention.block_tables =
      device.block_tables.narrow(/*dim=*/0, /*start=*/0, rows);

  params_.meta.batch_forward_type = batch.forward_type;
  params_.meta.num_sequences = static_cast<int32_t>(rows);
  params_.meta.actual_num_sequences =
      static_cast<int32_t>(batch.num_actual_sequences);
  params_.meta.batch_id = batch.batch_id;
  params_.meta.is_graph_warmup = batch.is_graph_warmup;
  params_.meta.q_max_seq_len = maximum(host.q_seq_lens);
  params_.meta.kv_max_seq_len = maximum(host.kv_seq_lens);
  // These are the original Python paged-attention inputs. The Slot owns the
  // tensor views and Host metadata until its last reader retires. No model
  // invocation, rotary computation or attention kernel is created here.
  auto& metadata = *params_.attn_metadata;
  metadata.q_seq_lens = attention.q_seq_lens;
  metadata.kv_seq_lens = attention.kv_seq_lens;
  metadata.q_cu_seq_lens = attention.q_cu_seq_lens;
  metadata.qo_indptr = attention.q_cu_seq_lens;
  metadata.slot_mapping = attention.new_cache_slots;
  metadata.block_table = batch.forward_type.is_prefill()
                             ? torch::Tensor()
                             : attention.block_tables;
  metadata.q_seq_lens_vec.assign(host.q_seq_lens.begin(),
                                 host.q_seq_lens.end());
  metadata.kv_seq_lens_vec.assign(host.kv_seq_lens.begin(),
                                  host.kv_seq_lens.end());
  metadata.q_cu_seq_lens_host_vec.assign(host.q_cu_seq_lens.begin(),
                                         host.q_cu_seq_lens.end());
  metadata.q_seq_lens_host =
      staging.q_seq_lens.narrow(/*dim=*/0, /*start=*/0, rows);
  metadata.kv_seq_lens_host =
      staging.kv_seq_lens.narrow(/*dim=*/0, /*start=*/0, rows);
  metadata.max_query_len = params_.meta.q_max_seq_len;
  metadata.max_seq_len = params_.meta.kv_max_seq_len;
  metadata.total_kv_len = std::accumulate(
      host.kv_seq_lens.begin(), host.kv_seq_lens.end(), int64_t{0});
  metadata.is_prefill = batch.forward_type.is_prefill();
  metadata.is_chunked_prefill =
      batch.forward_type.is_chunked_prefill() || batch.forward_type.is_mixed();
  metadata.is_mixed = batch.forward_type.is_mixed();
  metadata.is_dummy = rows == 0;
  transfer_info_ = transferred;
  return Status();
}

}  // namespace xllm
