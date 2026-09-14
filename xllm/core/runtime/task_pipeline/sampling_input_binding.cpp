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

#include "core/runtime/task_pipeline/sampling_input_binding.h"

#include <c10/core/DeviceGuard.h>
#include <glog/logging.h>

#include <array>
#include <cmath>
#include <limits>
#include <utility>

namespace xllm {
namespace {

enum class RowDomain : uint8_t { SELECTED, SAMPLE, HISTORY };

struct InputField {
  torch::Tensor SamplingParameters::* member;
  torch::ScalarType dtype;
  RowDomain domain;
};

constexpr std::array<InputField, 12> kFields = {{
    {&SamplingParameters::selected_token_idxes,
     torch::kInt32,
     RowDomain::SELECTED},
    {&SamplingParameters::frequency_penalties,
     torch::kFloat32,
     RowDomain::SELECTED},
    {&SamplingParameters::presence_penalties,
     torch::kFloat32,
     RowDomain::SELECTED},
    {&SamplingParameters::repetition_penalties,
     torch::kFloat32,
     RowDomain::SELECTED},
    {&SamplingParameters::temperatures, torch::kFloat32, RowDomain::SELECTED},
    {&SamplingParameters::top_p, torch::kFloat32, RowDomain::SELECTED},
    {&SamplingParameters::top_k, torch::kInt64, RowDomain::SELECTED},
    {&SamplingParameters::unique_token_ids, torch::kInt64, RowDomain::HISTORY},
    {&SamplingParameters::unique_token_counts,
     torch::kInt32,
     RowDomain::HISTORY},
    {&SamplingParameters::unique_token_ids_lens,
     torch::kInt32,
     RowDomain::SELECTED},
    {&SamplingParameters::sample_idxes, torch::kInt32, RowDomain::SAMPLE},
    {&SamplingParameters::do_sample, torch::kBool, RowDomain::SAMPLE},
}};

bool parameter_type(torch::ScalarType dtype) {
  return dtype == torch::kFloat32 || dtype == torch::kFloat16 ||
         dtype == torch::kBFloat16;
}

uint64_t capacity_elements(const InputField& field,
                           const SamplingInputCapacity& capacity) {
  switch (field.domain) {
    case RowDomain::SELECTED:
      return capacity.max_selected_rows;
    case RowDomain::SAMPLE:
      return capacity.max_sample_rows;
    case RowDomain::HISTORY:
      return static_cast<uint64_t>(capacity.max_selected_rows) *
             capacity.max_unique_tokens;
  }
  LOG(FATAL) << "Unknown sampling row domain.";
  return 0;
}

torch::ScalarType storage_dtype(const InputField& field,
                                torch::ScalarType parameter_dtype) {
  return field.dtype == torch::kFloat32 ? parameter_dtype : field.dtype;
}

template <typename Scalar, typename Predicate>
bool all_values(const torch::Tensor& tensor, Predicate predicate) {
  const Scalar* values = tensor.data_ptr<Scalar>();
  for (int64_t index = 0; index < tensor.numel(); ++index) {
    if (!predicate(values[index])) {
      return false;
    }
  }
  return true;
}

template <typename Predicate>
bool valid_parameters(const torch::Tensor& tensor,
                      torch::ScalarType destination,
                      Predicate predicate) {
  if (!tensor.defined()) {
    return true;
  }
  const auto valid = [destination, predicate](float value) {
    if (!std::isfinite(value) || !predicate(value)) {
      return false;
    }
    if (destination == torch::kFloat16) {
      return std::isfinite(static_cast<float>(static_cast<c10::Half>(value)));
    }
    if (destination == torch::kBFloat16) {
      return std::isfinite(
          static_cast<float>(static_cast<c10::BFloat16>(value)));
    }
    return true;
  };
  switch (tensor.scalar_type()) {
    case torch::kFloat32:
      return all_values<float>(tensor, valid);
    case torch::kFloat16:
      return all_values<c10::Half>(tensor, valid);
    case torch::kBFloat16:
      return all_values<c10::BFloat16>(tensor, valid);
    default:
      return false;
  }
}

Status invalid_input() {
  return Status(StatusCode::INVALID_ARGUMENT,
                "Invalid or unsupported sampling input.");
}

}  // namespace

SamplingInputBinding::SamplingInputBinding(SamplingInputCapacity capacity,
                                           torch::Device device,
                                           torch::ScalarType parameter_dtype,
                                           uint64_t bytes)
    : capacity_(std::move(capacity)),
      device_(std::move(device)),
      parameter_dtype_(parameter_dtype),
      bytes_(bytes) {
  for (const auto& field : kFields) {
    const int64_t elements =
        static_cast<int64_t>(capacity_elements(field, capacity_));
    const auto options =
        torch::TensorOptions().dtype(storage_dtype(field, parameter_dtype));
    host_storage_.*field.member = torch::empty(
        {elements}, options.device(torch::kCPU).pinned_memory(true));
    device_storage_.*field.member =
        torch::empty({elements}, options.device(device_));
  }
}

Status SamplingInputBinding::create(
    const SamplingInputCapacity& capacity,
    const torch::Device& device,
    torch::ScalarType parameter_dtype,
    std::unique_ptr<SamplingInputBinding>& output) {
  constexpr uint64_t kMaxElements = std::numeric_limits<int32_t>::max();
  if (device.type() != torch::kPrivateUse1 || !device.has_index() ||
      !parameter_type(parameter_dtype) || capacity.max_selected_rows == 0 ||
      capacity.max_sample_rows == 0 || capacity.max_unique_tokens == 0 ||
      capacity.vocab_size == 0 || capacity.max_selected_rows > kMaxElements ||
      capacity.max_sample_rows > capacity.max_selected_rows ||
      capacity.max_unique_tokens > kMaxElements ||
      capacity.vocab_size > kMaxElements ||
      capacity.max_top_logprobs > capacity.vocab_size) {
    return invalid_input();
  }
  uint64_t bytes = 0;
  constexpr uint64_t kMaxBytes = std::numeric_limits<int64_t>::max();
  for (const auto& field : kFields) {
    const uint64_t elements = capacity_elements(field, capacity);
    const uint64_t element_bytes =
        torch::elementSize(storage_dtype(field, parameter_dtype));
    if (elements > (kMaxBytes - bytes) / element_bytes) {
      return invalid_input();
    }
    bytes += elements * element_bytes;
  }
  c10::DeviceGuard guard(device);
  auto binding = std::unique_ptr<SamplingInputBinding>(
      new SamplingInputBinding(capacity, device, parameter_dtype, bytes));
  output = std::move(binding);
  return Status();
}

Status SamplingInputBinding::validate(const SamplingParameters& input,
                                      uint32_t model_tokens) const {
  if (input.filter_mask.defined() || input.filter_bitmask.defined() ||
      input.acc_logprob.defined() || input.is_embeddings ||
      input.use_beam_search || input.num_return_sequences != 0 ||
      input.max_top_logprobs < 0 ||
      static_cast<uint64_t>(input.max_top_logprobs) >
          capacity_.max_top_logprobs ||
      model_tokens >
          static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
    return invalid_input();
  }
  const int64_t rows = input.selected_token_idxes.defined()
                           ? input.selected_token_idxes.numel()
                           : 0;
  const int64_t samples =
      input.sample_idxes.defined() ? input.sample_idxes.numel() : 0;
  if (rows > capacity_.max_selected_rows ||
      samples > capacity_.max_sample_rows || samples > rows ||
      (rows > 0 && (samples == 0 || model_tokens == 0))) {
    return invalid_input();
  }
  int64_t width = 0;
  for (const auto& field : kFields) {
    const torch::Tensor& tensor = input.*field.member;
    if (!tensor.defined()) {
      continue;
    }
    const bool dtype_matches = field.dtype == torch::kFloat32
                                   ? parameter_type(tensor.scalar_type())
                                   : tensor.scalar_type() == field.dtype;
    const int64_t expected_rows =
        field.domain == RowDomain::SAMPLE ? samples : rows;
    const bool history = field.domain == RowDomain::HISTORY;
    if (!tensor.device().is_cpu() || !tensor.is_contiguous() ||
        !dtype_matches || tensor.dim() != (history ? 2 : 1) ||
        tensor.size(0) != expected_rows) {
      return invalid_input();
    }
    if (history) {
      if (tensor.size(1) > capacity_.max_unique_tokens ||
          (width != 0 && width != tensor.size(1)) ||
          (rows > 0 && tensor.size(1) == 0)) {
        return invalid_input();
      }
      width = tensor.size(1);
    }
  }
  const bool history = input.unique_token_ids.defined();
  if (history != input.unique_token_counts.defined() ||
      history != input.unique_token_ids_lens.defined() ||
      input.frequency_penalties.defined() !=
          input.presence_penalties.defined() ||
      ((input.frequency_penalties.defined() ||
        input.repetition_penalties.defined()) &&
       !history) ||
      (samples > 0 && !input.do_sample.defined())) {
    return invalid_input();
  }
  if (rows > 0) {
    if (!all_values<int32_t>(
            input.selected_token_idxes, [model_tokens](int32_t value) {
              return value >= 0 && static_cast<uint32_t>(value) < model_tokens;
            })) {
      return invalid_input();
    }
    int32_t previous = -1;
    if (!all_values<int32_t>(
            input.sample_idxes, [rows, &previous](int32_t value) {
              const bool valid = value > previous && value < rows;
              previous = value;
              return valid;
            })) {
      return invalid_input();
    }
  }
  const auto finite = [](float /*value*/) { return true; };
  if (!valid_parameters(input.frequency_penalties, parameter_dtype_, finite) ||
      !valid_parameters(input.presence_penalties, parameter_dtype_, finite) ||
      !valid_parameters(input.repetition_penalties,
                        parameter_dtype_,
                        [](float value) { return value > 0; }) ||
      !valid_parameters(input.temperatures,
                        parameter_dtype_,
                        [](float value) { return value >= 0; }) ||
      !valid_parameters(input.top_p, parameter_dtype_, [](float value) {
        return value >= 0 && value <= 1;
      })) {
    return invalid_input();
  }
  if (input.top_k.defined() &&
      !all_values<int64_t>(input.top_k, [](int64_t value) {
        return value <= std::numeric_limits<int32_t>::max();
      })) {
    return invalid_input();
  }
  if (history &&
      (!all_values<int64_t>(input.unique_token_ids,
                            [this](int64_t value) {
                              return value >= 0 &&
                                     static_cast<uint64_t>(value) <
                                         capacity_.vocab_size;
                            }) ||
       !all_values<int32_t>(input.unique_token_counts,
                            [](int32_t value) { return value >= 0; }) ||
       !all_values<int32_t>(
           input.unique_token_ids_lens,
           [width](int32_t value) { return value >= 0 && value <= width; }))) {
    return invalid_input();
  }
  return Status();
}

Status SamplingInputBinding::prepare(const SamplingParameters& input,
                                     uint32_t model_tokens,
                                     const Stream& stream) {
  Status status = validate(input, model_tokens);
  if (!status.ok()) {
    return status;
  }
  if (stream.get_stream()->device_index() != device_.index()) {
    return invalid_input();
  }
  c10::DeviceGuard guard(device_);
  SamplingParameters prepared;
  SamplingInputTransferInfo transferred;
  const bool empty = !input.selected_token_idxes.defined() ||
                     input.selected_token_idxes.numel() == 0;
  if (!empty) {
    const aclrtStream transfer_stream = stream.get_stream()->stream();
    for (const auto& field : kFields) {
      const torch::Tensor& source = input.*field.member;
      if (!source.defined()) {
        continue;
      }
      torch::Tensor host = (host_storage_.*field.member)
                               .narrow(/*dim=*/0, /*start=*/0, source.numel());
      torch::Tensor device =
          (device_storage_.*field.member)
              .narrow(/*dim=*/0, /*start=*/0, source.numel());
      host.copy_(source.view({-1}));
      CHECK_EQ(aclrtMemcpyAsync(device.data_ptr(),
                                device.nbytes(),
                                host.data_ptr(),
                                host.nbytes(),
                                ACL_MEMCPY_HOST_TO_DEVICE,
                                transfer_stream),
               ACL_SUCCESS);
      prepared.*field.member = device.view(source.sizes());
      transferred.h2d_bytes += host.nbytes();
      ++transferred.h2d_calls;
    }
    prepared.all_random_sample =
        all_values<bool>(input.do_sample, [](bool value) { return value; });
    prepared.all_greedy_sample =
        all_values<bool>(input.do_sample, [](bool value) { return !value; });
    prepared.logprobs = input.logprobs;
    prepared.return_probs = input.return_probs;
    prepared.max_top_logprobs = input.max_top_logprobs;
  }
  cpu_do_sample_ = input.do_sample.defined()
                       ? host_storage_.do_sample.narrow(
                             /*dim=*/0, /*start=*/0, input.do_sample.numel())
                       : torch::Tensor();
  params_ = std::move(prepared);
  transfer_ = transferred;
  return Status();
}

torch::Tensor SamplingInputBinding::copy_cpu_do_sample() const {
  if (!cpu_do_sample_.defined()) {
    return {};
  }
  auto output = torch::empty(cpu_do_sample_.sizes(),
                             torch::TensorOptions()
                                 .dtype(torch::kBool)
                                 .device(torch::kCPU)
                                 .pinned_memory(/*pinned_memory=*/false));
  output.copy_(cpu_do_sample_);
  return output;
}

}  // namespace xllm
