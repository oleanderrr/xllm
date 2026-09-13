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

#include <cstdint>
#include <memory>
#include <vector>

#include "core/common/types.h"
#include "core/framework/sampling/sampling_params.h"
#include "core/kernels/npu/xllm_ops/top_k_top_p_plan.h"

namespace xllm {

struct PreparedSamplerCapacity {
  struct LogitsCapacity {
    uint32_t max_selected_rows = 0;
    uint32_t max_sample_rows = 0;
    uint32_t max_unique_tokens = 0;
    uint32_t vocab_size = 0;
  };
  LogitsCapacity logits;
  uint32_t max_top_logprobs = 0;
  // Zero means vocab_size. Dense sample subsets are always supported too.
  uint32_t logits_row_stride = 0;
};

class PreparedSamplingInvocation;

// One owner per serialized Launch stream. Initialize every admitted native
// filter plan and backing buffer before serving, then hand off initialization
// to Launch. Retire all readers before destruction; no implicit
// synchronization. This is ordinary token sampling, not a
// reranker/beam/embedding implementation.
class PreparedSampler final {
 public:
  static Status create(const PreparedSamplerCapacity& capacity,
                       const torch::Device& device,
                       torch::ScalarType logits_dtype,
                       torch::ScalarType parameter_dtype,
                       std::unique_ptr<PreparedSampler>& output);
  PreparedSampler(const PreparedSampler&) = delete;
  PreparedSampler& operator=(const PreparedSampler&) = delete;

  // Input values/indices are already CPU-validated by SamplingInputBinding.
  // Only binds metadata; no Device work or RNG. Expected rejection leaves
  // output intact. Final tokens/logprobs are [S], top fields [S,K], lengths
  // int32 [S]. Unrequested fields and caller-supplied probs must be undefined.
  Status bind(SamplingParameters params,
              SampleOutput final_views,
              torch::Tensor lengths,
              std::unique_ptr<PreparedSamplingInvocation>& output) const;
  uint64_t device_bytes() const { return bytes_; }

 private:
  friend class PreparedSamplingInvocation;
  PreparedSampler(PreparedSamplerCapacity capacity,
                  torch::Device device,
                  torch::ScalarType logits_dtype,
                  torch::ScalarType parameter_dtype,
                  uint64_t bytes);
  Status validate_input(const SamplingParameters& params) const;
  Status validate_output(const SamplingParameters& params,
                         const SampleOutput& final_views,
                         const torch::Tensor& lengths) const;

  PreparedSamplerCapacity capacity_;
  torch::Device device_;
  torch::ScalarType logits_dtype_;
  torch::ScalarType parameter_dtype_;
  uint64_t bytes_ = 0;
  torch::Tensor native_top_k_;
  torch::Tensor native_top_p_;
  torch::Tensor native_workspace_;
  std::vector<std::unique_ptr<kernel::npu::TopKTopPPlan>> dense_plans_;
  std::vector<std::unique_ptr<kernel::npu::TopKTopPPlan>> padded_plans_;
};

class PreparedSamplingInvocation final {
 public:
  PreparedSamplingInvocation(const PreparedSamplingInvocation&) = delete;
  PreparedSamplingInvocation& operator=(const PreparedSamplingInvocation&) =
      delete;

  // On the current Launch stream after input-ready. Writes final token fields
  // directly and fills lengths=1; caller records producer-ready and owns D2H.
  // Ordinary intermediates and returned probs use Torch-owned storage.
  // No event creation or explicit Device wait. Native stream access may flush
  // Host submission work.
  const SampleOutput& run(torch::Tensor& selected_logits);

 private:
  friend class PreparedSampler;
  PreparedSamplingInvocation(const PreparedSampler& sampler,
                             SamplingParameters params,
                             SampleOutput output,
                             torch::Tensor lengths);
  void filter(torch::Tensor& sample,
              const torch::Tensor& temperatures,
              const torch::Tensor& top_k,
              const torch::Tensor& top_p);

  SamplingParameters params_;
  torch::Device device_;
  SampleOutput output_;
  torch::Tensor lengths_;
  int64_t selected_rows_ = 0;
  int64_t samples_ = 0;
  int64_t vocab_ = 0;
  int64_t row_stride_ = 0;
  torch::ScalarType logits_dtype_;
  kernel::npu::TopKTopPPlan* dense_plan_ = nullptr;
  kernel::npu::TopKTopPPlan* padded_plan_ = nullptr;
  torch::Tensor native_top_k_;
  torch::Tensor native_top_p_;
  torch::Tensor native_workspace_;
  torch::Tensor token_columns_;
  torch::Tensor logprob_columns_;
};

}  // namespace xllm
