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

#include "core/runtime/task_pipeline/sequence_token_binding.h"

#include <c10/core/DeviceGuard.h>
#include <glog/logging.h>

#include <algorithm>
#include <array>

namespace xllm {
namespace {

Status invalid(const char* message) {
  return Status(StatusCode::INVALID_ARGUMENT, message);
}

}  // namespace

Status SequenceTokenBinding::create(
    SequenceStatePool& pool,
    uint32_t capacity,
    std::unique_ptr<SequenceTokenBinding>& output) {
  if (capacity == 0 || capacity > pool.capacity()) {
    return invalid("Invalid fixed Sequence token binding capacity.");
  }
  output = std::unique_ptr<SequenceTokenBinding>(
      new SequenceTokenBinding(pool, capacity));
  return Status();
}

SequenceTokenBinding::SequenceTokenBinding(SequenceStatePool& pool,
                                           uint32_t capacity)
    : pool_(pool), capacity_(capacity), lease_(capacity) {
  c10::DeviceGuard guard(pool.tokens().device());
  const auto options = pool.tokens().options();
  host_indices_ = torch::empty(
      {3, capacity},
      options.device(torch::kCPU).pinned_memory(/*pinned_memory=*/true));
  device_indices_ = torch::empty({3, capacity}, options);
  gathered_int64_ = torch::empty({capacity}, options);
  gathered_int32_ = torch::empty({capacity}, options.dtype(torch::kInt32));
  accesses_.reserve(capacity);
  gather_sequences_.reserve(capacity);
  token_offsets_.reserve(capacity);
  publish_sequences_.reserve(capacity);
}

Status SequenceTokenBinding::plan(const ModelInputHostView& model,
                                  const SamplingParameters& sampling,
                                  std::span<const SequenceStateKey> keys) {
  if (keys.size() > capacity_ ||
      (!keys.empty() && keys.size() != model.q_seq_lens.size())) {
    return invalid("Sequence token identities do not match actual model rows.");
  }
  accesses_.clear();
  gather_sequences_.clear();
  token_offsets_.clear();
  publish_sequences_.clear();
  if (keys.empty()) {
    if (std::any_of(model.token_ids.begin(),
                    model.token_ids.end(),
                    [](int32_t token) { return token < 0; })) {
      return invalid("Unknown tokens require explicit Sequence state keys.");
    }
    return Status();
  }
  CHECK_EQ(model.q_cu_seq_lens.size(), keys.size());
  CHECK_EQ(model.kv_seq_lens.size(), keys.size());
  CHECK_EQ(model.positions.size(), model.token_ids.size());
  uint32_t begin = 0;
  for (uint32_t row = 0; row < keys.size(); ++row) {
    const int32_t end = model.q_cu_seq_lens[row];
    CHECK_GT(end, static_cast<int64_t>(begin));
    CHECK_LE(end, model.token_ids.size());
    accesses_.emplace_back(
        SequenceStateAccess{keys[row], std::nullopt, std::nullopt});
    for (uint32_t offset = begin; offset < static_cast<uint32_t>(end);
         ++offset) {
      if (model.token_ids[offset] >= 0) {
        continue;
      }
      if (offset + 1 != static_cast<uint32_t>(end)) {
        return invalid("Only the last query token may await Sequence state.");
      }
      CHECK_GE(model.positions[offset], 0);
      accesses_.back().read_position =
          static_cast<uint32_t>(model.positions[offset]);
      gather_sequences_.emplace_back(row);
      token_offsets_.emplace_back(offset);
    }
    begin = static_cast<uint32_t>(end);
  }
  CHECK_EQ(begin, model.token_ids.size());
  const int64_t samples =
      sampling.sample_idxes.defined() ? sampling.sample_idxes.numel() : 0;
  if (samples > capacity_) {
    return invalid("Sequence token publication exceeds fixed Slot capacity.");
  }
  if (samples == 0) {
    return Status();
  }
  CHECK(sampling.sample_idxes.device().is_cpu());
  CHECK(sampling.selected_token_idxes.device().is_cpu());
  const int32_t* sample = sampling.sample_idxes.const_data_ptr<int32_t>();
  const int32_t* selected =
      sampling.selected_token_idxes.const_data_ptr<int32_t>();
  for (int64_t output = 0; output < samples; ++output) {
    CHECK_GE(sample[output], 0);
    CHECK_LT(sample[output], sampling.selected_token_idxes.numel());
    const int32_t query = selected[sample[output]];
    CHECK_GE(query, 0);
    const auto end = std::upper_bound(
        model.q_cu_seq_lens.begin(), model.q_cu_seq_lens.end(), query);
    CHECK(end != model.q_cu_seq_lens.end());
    const uint32_t row =
        static_cast<uint32_t>(end - model.q_cu_seq_lens.begin());
    if (query != *end - 1 || accesses_[row].publish_position.has_value()) {
      return invalid("Publish one sampled last query per Sequence.");
    }
    CHECK_GT(model.kv_seq_lens[row], 0);
    accesses_[row].publish_position =
        static_cast<uint32_t>(model.kv_seq_lens[row]);
    publish_sequences_.emplace_back(row);
  }
  return Status();
}

Status SequenceTokenBinding::prepare(
    const ModelInputHostView& model,
    const SamplingParameters& sampling,
    std::span<const SequenceStateKey> keys,
    std::span<const SequenceStateKey> retired_keys,
    const Stream& stream) {
  if (stream.get_stream()->device_index() != pool_.tokens().device().index()) {
    return invalid(
        "Sequence token preparation stream uses a different device.");
  }
  Status status = plan(model, sampling, keys);
  if (!status.ok()) {
    return status;
  }
  status = pool_.admit(accesses_, retired_keys, lease_);
  if (!status.ok()) {
    return status;
  }
  c10::DeviceGuard guard(pool_.tokens().device());
  gather_count_ = static_cast<uint32_t>(gather_sequences_.size());
  publish_count_ = static_cast<uint32_t>(publish_sequences_.size());
  model_tokens_ = static_cast<uint32_t>(model.token_ids.size());
  const auto handles = lease_.handles();
  int64_t* host = host_indices_.data_ptr<int64_t>();
  for (uint32_t index = 0; index < gather_count_; ++index) {
    host[index] = handles[gather_sequences_[index]].row;
    host[capacity_ + index] = token_offsets_[index];
  }
  for (uint32_t index = 0; index < publish_count_; ++index) {
    host[2ULL * capacity_ + index] = handles[publish_sequences_[index]].row;
  }
  int64_t* device = device_indices_.data_ptr<int64_t>();
  const std::array<uint32_t, 3> counts{
      gather_count_, gather_count_, publish_count_};
  for (uint32_t region = 0; region < counts.size(); ++region) {
    if (counts[region] == 0) {
      continue;
    }
    const uint64_t offset = static_cast<uint64_t>(region) * capacity_;
    const uint64_t bytes =
        static_cast<uint64_t>(counts[region]) * sizeof(int64_t);
    CHECK_EQ(aclrtMemcpyAsync(device + offset,
                              bytes,
                              host + offset,
                              bytes,
                              ACL_MEMCPY_HOST_TO_DEVICE,
                              stream.get_stream()->stream()),
             ACL_SUCCESS);
  }
  gather_rows_ = device_indices_.select(/*dim=*/0, /*index=*/0)
                     .narrow(/*dim=*/0, /*start=*/0, gather_count_);
  gather_offsets_ = device_indices_.select(/*dim=*/0, /*index=*/1)
                        .narrow(/*dim=*/0, /*start=*/0, gather_count_);
  publish_rows_ = device_indices_.select(/*dim=*/0, /*index=*/2)
                      .narrow(/*dim=*/0, /*start=*/0, publish_count_);
  gather_output_int64_ =
      gathered_int64_.narrow(/*dim=*/0, /*start=*/0, gather_count_);
  gather_output_int32_ =
      gathered_int32_.narrow(/*dim=*/0, /*start=*/0, gather_count_);
  return Status();
}

void SequenceTokenBinding::gather_into(const torch::Tensor& model_tokens) {
  if (gather_count_ == 0) {
    return;
  }
  CHECK(!lease_.empty());
  CHECK_EQ(model_tokens.device(), pool_.tokens().device());
  CHECK_EQ(model_tokens.scalar_type(), torch::kInt32);
  CHECK_EQ(model_tokens.numel(), model_tokens_);
  CHECK_EQ(model_tokens.dim(), 1);
  CHECK(model_tokens.is_contiguous());
  torch::index_select_out(
      gather_output_int64_, pool_.tokens(), /*dim=*/0, gather_rows_);
  gather_output_int32_.copy_(gather_output_int64_);
  model_tokens.index_copy_(/*dim=*/0, gather_offsets_, gather_output_int32_);
}

void SequenceTokenBinding::publish(const torch::Tensor& result_tokens) {
  if (publish_count_ == 0) {
    return;
  }
  CHECK(!lease_.empty());
  CHECK(result_tokens.defined());
  CHECK_EQ(result_tokens.device(), pool_.tokens().device());
  CHECK_EQ(result_tokens.scalar_type(), torch::kInt64);
  CHECK_EQ(result_tokens.dim(), 2);
  CHECK_EQ(result_tokens.size(/*dim=*/0), publish_count_);
  CHECK_EQ(result_tokens.size(/*dim=*/1), 1);
  CHECK(result_tokens.is_contiguous());
  pool_.tokens().index_copy_(
      /*dim=*/0, publish_rows_, result_tokens.squeeze(/*dim=*/1));
}

void SequenceTokenBinding::release() {
  lease_.reset();
  gather_count_ = 0;
  publish_count_ = 0;
}

}  // namespace xllm
