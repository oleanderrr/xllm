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

#include "core/framework/sampling/prepared_sampler.h"

#include <c10/core/DeviceGuard.h>
#include <glog/logging.h>

#include <algorithm>
#include <array>
#include <limits>
#include <utility>

#include "core/framework/sampling/logits_utils.h"

namespace xllm {
namespace {

Status invalid_input() {
  return Status(
      StatusCode::INVALID_ARGUMENT,
      "Invalid prepared sampler capacity, input or final output views.");
}

bool supported_dtype(torch::ScalarType dtype) {
  return dtype == torch::kFloat16 || dtype == torch::kBFloat16 ||
         dtype == torch::kFloat32;
}

torch::Tensor prefix(const torch::Tensor& buffer, int64_t elements) {
  return buffer.narrow(/*dim=*/0, /*start=*/0, elements);
}

torch::Tensor matrix(const torch::Tensor& buffer,
                     int64_t rows,
                     int64_t columns) {
  return prefix(buffer, rows * columns).view({rows, columns});
}

constexpr std::array<torch::Tensor SamplingParameters::*, 5> kFloatRows = {
    &SamplingParameters::frequency_penalties,
    &SamplingParameters::presence_penalties,
    &SamplingParameters::repetition_penalties,
    &SamplingParameters::temperatures,
    &SamplingParameters::top_p};

}  // namespace

Status PreparedSampler::create(const PreparedSamplerCapacity& capacity,
                               const torch::Device& device,
                               torch::ScalarType logits_dtype,
                               torch::ScalarType parameter_dtype,
                               std::unique_ptr<PreparedSampler>& output) {
  const auto& model = capacity.logits;
  const uint32_t stride = capacity.logits_row_stride == 0
                              ? model.vocab_size
                              : capacity.logits_row_stride;
  constexpr uint32_t kMaxDimension = std::numeric_limits<int32_t>::max();
  if (device.type() != torch::kPrivateUse1 || !device.has_index() ||
      !supported_dtype(logits_dtype) || !supported_dtype(parameter_dtype) ||
      model.max_selected_rows == 0 || model.max_sample_rows == 0 ||
      model.max_sample_rows > model.max_selected_rows ||
      model.vocab_size == 0 || model.max_selected_rows > kMaxDimension ||
      model.max_unique_tokens > kMaxDimension || stride < model.vocab_size ||
      stride > kMaxDimension || capacity.max_top_logprobs > model.vocab_size) {
    return invalid_input();
  }
  const uint64_t samples = model.max_sample_rows;
  const uint64_t native = torch::elementSize(logits_dtype);
  const uint64_t limit = std::numeric_limits<int64_t>::max();
  // The logits template exists only while constructing native filter plans.
  if (samples * stride > limit / native) {
    return invalid_input();
  }
  const uint64_t bytes = samples * (sizeof(int32_t) + native);
  PreparedSamplerCapacity normalized = capacity;
  normalized.logits_row_stride = stride;
  c10::DeviceGuard guard(device);
  output = std::unique_ptr<PreparedSampler>(new PreparedSampler(
      normalized, device, logits_dtype, parameter_dtype, bytes));
  return Status();
}

PreparedSampler::PreparedSampler(PreparedSamplerCapacity capacity,
                                 torch::Device device,
                                 torch::ScalarType logits_dtype,
                                 torch::ScalarType parameter_dtype,
                                 uint64_t bytes)
    : capacity_(capacity),
      device_(std::move(device)),
      logits_dtype_(logits_dtype),
      parameter_dtype_(parameter_dtype),
      bytes_(bytes) {
  const int64_t samples = capacity_.logits.max_sample_rows;
  const int64_t vocab = capacity_.logits.vocab_size;
  const auto options = torch::TensorOptions().device(device_);
  native_top_k_ = torch::full({samples}, vocab, options.dtype(torch::kInt32));
  native_top_p_ = torch::ones({samples}, options.dtype(logits_dtype_));
  auto logits_template = torch::empty({samples * capacity_.logits_row_stride},
                                      options.dtype(logits_dtype_));
  dense_plans_.reserve(samples);
  const bool padded = capacity_.logits_row_stride != vocab;
  if (padded) {
    padded_plans_.reserve(samples);
  }
  uint64_t workspace_bytes = 0;
  for (int64_t rows = 1; rows <= samples; ++rows) {
    auto top_k = prefix(native_top_k_, rows);
    auto top_p = prefix(native_top_p_, rows);
    auto dense = matrix(logits_template, rows, vocab);
    auto plan = kernel::npu::TopKTopPPlan::create(dense, top_k, top_p);
    workspace_bytes = std::max(workspace_bytes, plan->workspace_bytes());
    dense_plans_.emplace_back(std::move(plan));
    if (padded) {
      auto input = matrix(logits_template, rows, capacity_.logits_row_stride)
                       .narrow(/*dim=*/1, /*start=*/0, vocab);
      auto padded_plan = kernel::npu::TopKTopPPlan::create(input, top_k, top_p);
      workspace_bytes =
          std::max(workspace_bytes, padded_plan->workspace_bytes());
      padded_plans_.emplace_back(std::move(padded_plan));
    }
  }
  CHECK_LE(workspace_bytes,
           static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) - bytes_);
  native_workspace_ = torch::empty({static_cast<int64_t>(workspace_bytes)},
                                   options.dtype(torch::kUInt8));
  bytes_ += workspace_bytes;
}

Status PreparedSampler::validate_input(const SamplingParameters& params) const {
  if (params.filter_mask.defined() || params.filter_bitmask.defined() ||
      params.acc_logprob.defined() || params.use_beam_search ||
      params.is_embeddings || params.num_return_sequences != 0) {
    return invalid_input();
  }
  const auto vector_matches = [this](const torch::Tensor& tensor,
                                     torch::ScalarType dtype,
                                     int64_t rows) {
    return tensor.defined() && tensor.device() == device_ &&
           tensor.scalar_type() == dtype && tensor.is_contiguous() &&
           tensor.dim() == 1 && tensor.size(/*dim=*/0) == rows;
  };
  const int64_t rows = params.selected_token_idxes.defined()
                           ? params.selected_token_idxes.numel()
                           : 0;
  const int64_t samples =
      params.sample_idxes.defined() ? params.sample_idxes.numel() : 0;
  if (rows == 0) {
    // The canonical empty binding has no tensor fields. Do not admit partially
    // populated inputs that would silently discard a producer's data.
    const std::array<const torch::Tensor*, 12> fields = {
        &params.selected_token_idxes,
        &params.sample_idxes,
        &params.do_sample,
        &params.frequency_penalties,
        &params.presence_penalties,
        &params.repetition_penalties,
        &params.temperatures,
        &params.top_k,
        &params.top_p,
        &params.unique_token_ids,
        &params.unique_token_counts,
        &params.unique_token_ids_lens};
    for (const auto* tensor : fields) {
      if (tensor->defined()) {
        return invalid_input();
      }
    }
    return Status();
  }
  if (rows > capacity_.logits.max_selected_rows || samples == 0 ||
      samples > rows || samples > capacity_.logits.max_sample_rows ||
      !vector_matches(params.selected_token_idxes, torch::kInt32, rows) ||
      !vector_matches(params.sample_idxes, torch::kInt32, samples) ||
      !vector_matches(params.do_sample, torch::kBool, samples) ||
      (params.all_random_sample && params.all_greedy_sample)) {
    return invalid_input();
  }
  for (const auto member : kFloatRows) {
    const torch::Tensor& tensor = params.*member;
    if (tensor.defined() && !vector_matches(tensor, parameter_dtype_, rows)) {
      return invalid_input();
    }
  }
  if ((params.top_k.defined() &&
       !vector_matches(params.top_k, torch::kInt64, rows)) ||
      params.frequency_penalties.defined() !=
          params.presence_penalties.defined()) {
    return invalid_input();
  }
  const bool history = params.unique_token_ids.defined();
  if (history != params.unique_token_counts.defined() ||
      history != params.unique_token_ids_lens.defined() ||
      ((params.frequency_penalties.defined() ||
        params.repetition_penalties.defined()) &&
       !history)) {
    return invalid_input();
  }
  if (history) {
    const auto& ids = params.unique_token_ids;
    const auto& counts = params.unique_token_counts;
    if (ids.device() != device_ || ids.scalar_type() != torch::kInt64 ||
        !ids.is_contiguous() || ids.dim() != 2 || ids.size(/*dim=*/0) != rows ||
        ids.size(/*dim=*/1) == 0 ||
        ids.size(/*dim=*/1) > capacity_.logits.max_unique_tokens ||
        counts.device() != device_ || counts.scalar_type() != torch::kInt32 ||
        !counts.is_contiguous() || counts.sizes() != ids.sizes() ||
        !vector_matches(params.unique_token_ids_lens, torch::kInt32, rows)) {
      return invalid_input();
    }
  }
  return Status();
}

Status PreparedSampler::validate_output(const SamplingParameters& params,
                                        const SampleOutput& output,
                                        const torch::Tensor& lengths) const {
  const int64_t samples =
      params.sample_idxes.defined() ? params.sample_idxes.numel() : 0;
  const int64_t top = params.max_top_logprobs;
  if (samples > capacity_.logits.max_sample_rows || top < 0 ||
      top > capacity_.max_top_logprobs || (!params.logprobs && top != 0) ||
      output.probs.defined() || output.embeddings.defined() ||
      output.selected_embeddings.defined() || !output.mm_embeddings.empty() ||
      !output.speculative_token_stats.empty()) {
    return invalid_input();
  }
  const auto matches = [this](const torch::Tensor& tensor,
                              torch::ScalarType dtype,
                              torch::IntArrayRef shape,
                              bool required) {
    if (!required) {
      return !tensor.defined();
    }
    return tensor.defined() && tensor.device() == device_ &&
           tensor.scalar_type() == dtype && tensor.is_contiguous() &&
           tensor.sizes() == shape;
  };
  if (!matches(output.next_tokens, torch::kInt64, {samples}, samples > 0) ||
      !matches(lengths, torch::kInt32, {samples}, samples > 0) ||
      !matches(output.logprobs,
               torch::kFloat32,
               {samples},
               samples > 0 && params.logprobs) ||
      !matches(output.top_tokens,
               torch::kInt64,
               {samples, top},
               samples > 0 && top > 0) ||
      !matches(output.top_logprobs,
               torch::kFloat32,
               {samples, top},
               samples > 0 && top > 0)) {
    return invalid_input();
  }
  const std::array<const torch::Tensor*, 6> fields = {&output.next_tokens,
                                                      &lengths,
                                                      &output.logprobs,
                                                      &output.top_tokens,
                                                      &output.top_logprobs,
                                                      &output.probs};
  for (uint32_t i = 0; i < fields.size(); ++i) {
    if (!fields[i]->defined()) {
      continue;
    }
    const uintptr_t first = reinterpret_cast<uintptr_t>(fields[i]->data_ptr());
    for (uint32_t j = i + 1; j < fields.size(); ++j) {
      if (!fields[j]->defined()) {
        continue;
      }
      const uintptr_t second =
          reinterpret_cast<uintptr_t>(fields[j]->data_ptr());
      if (first < second + fields[j]->nbytes() &&
          second < first + fields[i]->nbytes()) {
        return invalid_input();
      }
    }
  }
  return Status();
}

Status PreparedSampler::bind(
    SamplingParameters params,
    SampleOutput final_views,
    torch::Tensor lengths,
    std::unique_ptr<PreparedSamplingInvocation>& output) const {
  Status status = validate_output(params, final_views, lengths);
  if (!status.ok()) {
    return status;
  }
  status = validate_input(params);
  if (!status.ok()) {
    return status;
  }
  output = std::unique_ptr<PreparedSamplingInvocation>(
      new PreparedSamplingInvocation(*this,
                                     std::move(params),
                                     std::move(final_views),
                                     std::move(lengths)));
  return Status();
}

PreparedSamplingInvocation::PreparedSamplingInvocation(
    const PreparedSampler& sampler,
    SamplingParameters params,
    SampleOutput output,
    torch::Tensor lengths)
    : params_(std::move(params)),
      device_(sampler.device_),
      output_(std::move(output)),
      lengths_(std::move(lengths)),
      vocab_(sampler.capacity_.logits.vocab_size),
      row_stride_(sampler.capacity_.logits_row_stride),
      logits_dtype_(sampler.logits_dtype_) {
  selected_rows_ = params_.selected_token_idxes.defined()
                       ? params_.selected_token_idxes.numel()
                       : 0;
  samples_ = params_.sample_idxes.defined() ? params_.sample_idxes.numel() : 0;
  if (samples_ == 0) {
    return;
  }
  native_top_k_ = prefix(sampler.native_top_k_, samples_);
  native_top_p_ = prefix(sampler.native_top_p_, samples_);
  native_workspace_ = sampler.native_workspace_;
  dense_plan_ = sampler.dense_plans_[samples_ - 1].get();
  if (!sampler.padded_plans_.empty()) {
    padded_plan_ = sampler.padded_plans_[samples_ - 1].get();
  }
  token_columns_ = output_.next_tokens.view({samples_, 1});
  if (params_.logprobs) {
    logprob_columns_ = output_.logprobs.view({samples_, 1});
  }
}

void PreparedSamplingInvocation::filter(torch::Tensor& sample,
                                        const torch::Tensor& temperatures,
                                        const torch::Tensor& top_k,
                                        const torch::Tensor& top_p) {
  if (!top_k.defined() || !top_p.defined()) {
    apply_top_k_top_p(sample, temperatures, top_k, top_p);
    return;
  }
  if (temperatures.defined()) {
    apply_temperatures(sample, temperatures);
  }
  // Keep raw ACL inputs alive across asynchronous execution. Ordinary Torch
  // intermediates use the original allocator; only native resources persist.
  native_top_k_.copy_(
      torch::where(top_k <= 0, std::numeric_limits<int64_t>::max(), top_k)
          .to(torch::kInt32));
  native_top_p_.copy_(top_p);
  auto* plan = sample.stride(/*dim=*/0) == vocab_ ? dense_plan_ : padded_plan_;
  CHECK(plan != nullptr);
  plan->run(sample, native_top_k_, native_top_p_, native_workspace_);
}

const SampleOutput& PreparedSamplingInvocation::run(
    torch::Tensor& selected_logits) {
  if (samples_ == 0) {
    CHECK(!selected_logits.defined() || selected_logits.numel() == 0);
    return output_;
  }
  CHECK(selected_logits.defined());
  CHECK(selected_logits.device() == device_);
  CHECK(selected_logits.scalar_type() == logits_dtype_);
  CHECK_EQ(selected_logits.dim(), 2);
  CHECK_EQ(selected_logits.size(/*dim=*/0), selected_rows_);
  CHECK_EQ(selected_logits.size(/*dim=*/1), vocab_);
  CHECK_EQ(selected_logits.stride(/*dim=*/1), 1);
  CHECK(selected_logits.stride(/*dim=*/0) == vocab_ ||
        selected_logits.stride(/*dim=*/0) == row_stride_);
  c10::DeviceGuard guard(selected_logits.device());
  const auto& params = params_;
  if (params.frequency_penalties.defined()) {
    apply_frequency_presence_penalties(selected_logits,
                                       params.unique_token_ids,
                                       params.unique_token_counts,
                                       params.frequency_penalties,
                                       params.presence_penalties);
  }
  if (params.repetition_penalties.defined()) {
    apply_repetition_penalties(
        selected_logits, params.unique_token_ids, params.repetition_penalties);
  }
  const bool subset = selected_rows_ != samples_;
  torch::Tensor sample_logits =
      subset ? selected_logits.index_select(/*dim=*/0, params.sample_idxes)
             : selected_logits;
  if (params.all_greedy_sample && !params.logprobs && !subset) {
    torch::argmax_out(
        output_.next_tokens, sample_logits, /*dim=*/-1, /*keepdim=*/false);
    if (params.return_probs) {
      auto selected =
          sample_logits.gather(/*dim=*/-1, token_columns_).to(torch::kFloat32);
      auto log_probs = selected - torch::logsumexp(sample_logits,
                                                   /*dim=*/-1,
                                                   /*keepdim=*/true);
      output_.probs = log_probs.exp().view({-1}).to(logits_dtype_);
    }
    lengths_.fill_(/*value=*/1);
    return output_;
  }
  torch::Tensor temperatures = params.temperatures;
  torch::Tensor top_k = params.top_k;
  torch::Tensor top_p = params.top_p;
  if (subset) {
    if (temperatures.defined()) {
      temperatures = temperatures.index_select(/*dim=*/0, params.sample_idxes);
    }
    if (top_k.defined()) {
      top_k = top_k.index_select(/*dim=*/0, params.sample_idxes);
    }
    if (top_p.defined()) {
      top_p = top_p.index_select(/*dim=*/0, params.sample_idxes);
    }
  }
  filter(sample_logits, temperatures, top_k, top_p);
  if (subset) {
    selected_logits.index_copy_(/*dim=*/0, params.sample_idxes, sample_logits);
  }
  auto probs =
      torch::softmax(sample_logits, /*dim=*/-1, /*dtype=*/torch::kFloat32);
  if (!params.all_greedy_sample) {
    torch::multinomial_out(token_columns_,
                           probs,
                           /*num_samples=*/1,
                           /*replacement=*/false);
  }
  if (params.all_greedy_sample) {
    torch::argmax_out(
        output_.next_tokens, probs, /*dim=*/-1, /*keepdim=*/false);
  } else if (!params.all_random_sample) {
    auto greedy = probs.argmax(/*dim=*/-1);
    torch::where_out(
        output_.next_tokens, params.do_sample, output_.next_tokens, greedy);
  }
  output_.probs = probs.to(logits_dtype_);
  if (params.logprobs) {
    const auto logprobs = torch::log_softmax(
        sample_logits, /*dim=*/-1, /*dtype=*/torch::kFloat32);
    torch::gather_out(logprob_columns_, logprobs, /*dim=*/-1, token_columns_);
    if (params.max_top_logprobs > 0) {
      torch::topk_out(output_.top_logprobs,
                      output_.top_tokens,
                      logprobs,
                      params.max_top_logprobs,
                      /*dim=*/-1,
                      /*largest=*/true,
                      /*sorted=*/true);
    }
  }
  lengths_.fill_(/*value=*/1);
  return output_;
}

}  // namespace xllm
