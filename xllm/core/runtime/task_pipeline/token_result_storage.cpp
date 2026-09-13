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

#include "core/runtime/task_pipeline/token_result_storage.h"

#include <c10/core/DeviceGuard.h>
#include <glog/logging.h>

#include <array>
#include <cstring>
#include <limits>
#include <utility>

#include "core/platform/stream.h"

namespace xllm {
namespace {

enum class ResultDomain : uint8_t { SEQUENCE, TOKEN, TOP_TOKEN };

struct ResultField {
  torch::Tensor TokenResultTensors::* member;
  torch::ScalarType dtype;
  ResultDomain domain;
  double padding;
};

constexpr double kLogprobPadding = -std::numeric_limits<double>::infinity();
constexpr std::array<ResultField, 5> kFields = {{
    {&TokenResultTensors::tokens, torch::kInt64, ResultDomain::TOKEN, -1},
    {&TokenResultTensors::lengths, torch::kInt32, ResultDomain::SEQUENCE, 0},
    {&TokenResultTensors::logprobs,
     torch::kFloat32,
     ResultDomain::TOKEN,
     kLogprobPadding},
    {&TokenResultTensors::top_tokens,
     torch::kInt64,
     ResultDomain::TOP_TOKEN,
     -1},
    {&TokenResultTensors::top_logprobs,
     torch::kFloat32,
     ResultDomain::TOP_TOKEN,
     kLogprobPadding},
}};

uint64_t capacity_elements(const ResultField& field,
                           const TokenResultCapacity& capacity) {
  const uint64_t sequences = capacity.max_sequences;
  switch (field.domain) {
    case ResultDomain::SEQUENCE:
      return sequences;
    case ResultDomain::TOKEN:
      return sequences * capacity.max_tokens_per_sequence;
    case ResultDomain::TOP_TOKEN:
      return sequences * capacity.max_tokens_per_sequence *
             capacity.max_top_logprobs;
  }
  LOG(FATAL) << "Unknown token result domain.";
  return 0;
}

TokenResultTensors bind_views(const TokenResultTensors& storage,
                              const TokenResultShape& shape) {
  TokenResultTensors view;
  if (shape.sequences == 0) {
    return view;
  }
  const int64_t tokens =
      static_cast<int64_t>(shape.sequences) * shape.tokens_per_sequence;
  view.tokens = storage.tokens.narrow(/*dim=*/0, /*start=*/0, tokens)
                    .view({shape.sequences, shape.tokens_per_sequence});
  view.lengths =
      storage.lengths.narrow(/*dim=*/0, /*start=*/0, shape.sequences);
  if (shape.logprobs) {
    view.logprobs = storage.logprobs.narrow(/*dim=*/0, /*start=*/0, tokens)
                        .view({shape.sequences, shape.tokens_per_sequence});
  }
  if (shape.top_logprobs != 0) {
    view.top_tokens =
        storage.top_tokens
            .narrow(/*dim=*/0, /*start=*/0, tokens * shape.top_logprobs)
            .view({shape.sequences,
                   shape.tokens_per_sequence,
                   shape.top_logprobs});
    view.top_logprobs =
        storage.top_logprobs
            .narrow(/*dim=*/0, /*start=*/0, tokens * shape.top_logprobs)
            .view({shape.sequences,
                   shape.tokens_per_sequence,
                   shape.top_logprobs});
  }
  return view;
}

Status pending_result() {
  return Status(StatusCode::RESOURCE_EXHAUSTED,
                "Take or discard the pending token result before reuse.");
}

}  // namespace

TokenResultStorage::TokenResultStorage(TokenResultCapacity capacity,
                                       torch::Device device,
                                       uint64_t bytes)
    : capacity_(std::move(capacity)),
      device_(std::move(device)),
      bytes_(bytes) {
  for (const auto& field : kFields) {
    const int64_t elements =
        static_cast<int64_t>(capacity_elements(field, capacity_));
    if (elements == 0) {
      continue;
    }
    const auto options = torch::TensorOptions().dtype(field.dtype);
    host_storage_.*field.member = torch::empty(
        {elements},
        options.device(torch::kCPU).pinned_memory(/*pinned_memory=*/true));
    device_storage_.*field.member =
        torch::empty({elements}, options.device(device_));
  }
  aclrtEvent event = nullptr;
  CHECK_EQ(aclrtCreateEventExWithFlag(&event, ACL_EVENT_SYNC), ACL_SUCCESS);
  result_ready_ = std::make_unique<StreamEvent>(event);
}

TokenResultStorage::~TokenResultStorage() {
  c10::DeviceGuard guard(device_);
  discard_result();
  result_ready_.reset();
}

Status TokenResultStorage::create(const TokenResultCapacity& capacity,
                                  const torch::Device& device,
                                  std::unique_ptr<TokenResultStorage>& output) {
  constexpr uint64_t kMaxDimension = std::numeric_limits<int32_t>::max();
  if (device.type() != torch::kPrivateUse1 || !device.has_index() ||
      capacity.max_sequences == 0 || capacity.max_tokens_per_sequence == 0 ||
      capacity.max_sequences > kMaxDimension ||
      capacity.max_tokens_per_sequence > kMaxDimension ||
      capacity.max_top_logprobs > kMaxDimension) {
    return Status(StatusCode::INVALID_ARGUMENT,
                  "Invalid token result capacity or NPU device.");
  }
  constexpr uint64_t kMaxBytes = std::numeric_limits<int64_t>::max();
  constexpr uint64_t kTokenBytes = sizeof(int64_t) + sizeof(float);
  const uint64_t tokens = static_cast<uint64_t>(capacity.max_sequences) *
                          capacity.max_tokens_per_sequence;
  uint64_t bytes =
      static_cast<uint64_t>(capacity.max_sequences) * sizeof(int32_t);
  if (tokens > (kMaxBytes - bytes) / kTokenBytes) {
    return Status(StatusCode::INVALID_ARGUMENT,
                  "Token result capacity exceeds int64 bytes.");
  }
  bytes += tokens * kTokenBytes;
  if (capacity.max_top_logprobs != 0) {
    const uint64_t top_bytes = kTokenBytes * capacity.max_top_logprobs;
    if (tokens > (kMaxBytes - bytes) / top_bytes) {
      return Status(StatusCode::INVALID_ARGUMENT,
                    "Top-logprob result capacity exceeds int64 bytes.");
    }
    bytes += tokens * top_bytes;
  }
  c10::DeviceGuard guard(device);
  auto storage = std::unique_ptr<TokenResultStorage>(
      new TokenResultStorage(capacity, device, bytes));
  output = std::move(storage);
  return Status();
}

Status TokenResultStorage::bind(TokenResultShape shape) {
  if (shape.sequences > capacity_.max_sequences ||
      shape.tokens_per_sequence > capacity_.max_tokens_per_sequence ||
      shape.top_logprobs > capacity_.max_top_logprobs ||
      (!shape.logprobs && shape.top_logprobs != 0) ||
      (shape.sequences == 0 && (shape.tokens_per_sequence != 0 ||
                                shape.top_logprobs != 0 || shape.logprobs)) ||
      (shape.sequences != 0 && shape.tokens_per_sequence == 0)) {
    return Status(StatusCode::INVALID_ARGUMENT,
                  "Invalid token result shape or optional outputs.");
  }
  if (copy_submitted_) {
    return pending_result();
  }
  TokenResultTensors host = bind_views(host_storage_, shape);
  TokenResultTensors device = bind_views(device_storage_, shape);
  host_view_ = std::move(host);
  device_view_ = std::move(device);
  shape_ = std::move(shape);
  transfer_ = {};
  return Status();
}

Status TokenResultStorage::copy_to_host(const Stream& stream,
                                        const StreamEventPtr& producer_ready) {
  if (!producer_ready || producer_ready->npu_event() == nullptr ||
      stream.get_stream()->device_index() != device_.index()) {
    return Status(
        StatusCode::INVALID_ARGUMENT,
        "Token result copy requires a matching NPU stream and producer event.");
  }
  if (copy_submitted_) {
    return pending_result();
  }
  c10::DeviceGuard guard(device_);
  const aclrtStream copy_stream = stream.get_stream()->stream();
  CHECK_EQ(aclrtStreamWaitEvent(copy_stream, producer_ready->npu_event()),
           ACL_SUCCESS)
      << "Failed to wait for token result producer.";
  TokenResultTransferInfo transferred;
  for (const auto& field : kFields) {
    const torch::Tensor& source = device_view_.*field.member;
    if (!source.defined()) {
      continue;
    }
    const torch::Tensor& destination = host_view_.*field.member;
    CHECK_EQ(aclrtMemcpyAsync(destination.data_ptr(),
                              destination.nbytes(),
                              source.data_ptr(),
                              source.nbytes(),
                              ACL_MEMCPY_DEVICE_TO_HOST,
                              copy_stream),
             ACL_SUCCESS);
    transferred.d2h_bytes += source.nbytes();
    ++transferred.d2h_calls;
  }
  CHECK_EQ(aclrtRecordEvent(result_ready_->npu_event(), copy_stream),
           ACL_SUCCESS);
  transfer_ = transferred;
  copy_submitted_ = true;
  return Status();
}

TokenResultTensors TokenResultStorage::take_result() {
  CHECK(copy_submitted_) << "No token result copy has been submitted.";
  c10::DeviceGuard guard(device_);
  CHECK(result_ready_->synchronize()) << "Failed to wait for token result D2H.";
  TokenResultTensors result;
  if (shape_.sequences == 0) {
    copy_submitted_ = false;
    return result;
  }
  const int32_t* lengths = host_view_.lengths.data_ptr<int32_t>();
  for (uint32_t row = 0; row < shape_.sequences; ++row) {
    CHECK_GE(lengths[row], 0) << "Invalid token result length.";
    CHECK_LE(static_cast<uint32_t>(lengths[row]), shape_.tokens_per_sequence)
        << "Invalid token result length.";
  }
  for (const auto& field : kFields) {
    const torch::Tensor& source = host_view_.*field.member;
    if (!source.defined()) {
      continue;
    }
    torch::Tensor destination =
        torch::full(source.sizes(),
                    field.padding,
                    source.options().pinned_memory(/*pinned_memory=*/false));
    if (field.domain == ResultDomain::SEQUENCE) {
      std::memcpy(destination.data_ptr(), source.data_ptr(), source.nbytes());
    } else {
      const uint64_t values_per_token =
          field.domain == ResultDomain::TOP_TOKEN ? shape_.top_logprobs : 1;
      const uint64_t token_bytes = values_per_token * source.element_size();
      const uint64_t row_bytes = token_bytes * shape_.tokens_per_sequence;
      const char* from = static_cast<const char*>(source.data_ptr());
      char* to = static_cast<char*>(destination.data_ptr());
      for (uint32_t row = 0; row < shape_.sequences; ++row) {
        std::memcpy(to + row * row_bytes,
                    from + row * row_bytes,
                    lengths[row] * token_bytes);
      }
    }
    result.*field.member = std::move(destination);
  }
  copy_submitted_ = false;
  return result;
}

void TokenResultStorage::discard_result() {
  if (!copy_submitted_) {
    return;
  }
  c10::DeviceGuard guard(device_);
  CHECK(result_ready_->synchronize()) << "Failed to retire token result D2H.";
  copy_submitted_ = false;
}

}  // namespace xllm
