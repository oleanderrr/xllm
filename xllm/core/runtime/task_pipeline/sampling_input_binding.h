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

#include "core/common/types.h"
#include "core/framework/sampling/sampling_params.h"
#include "core/platform/stream.h"

namespace xllm {

struct SamplingInputCapacity {
  uint32_t max_selected_rows = 0;
  uint32_t max_sample_rows = 0;
  uint32_t max_unique_tokens = 0;
  uint32_t vocab_size = 0;
  uint32_t max_top_logprobs = 0;
};

struct SamplingInputTransferInfo {
  uint64_t h2d_bytes = 0;
  uint32_t h2d_calls = 0;
};

// Owns one Slot's sampling input domain. Model input addresses and capacities
// are independent. Retire all readers before reuse or destruction; initialize
// and hand off the allocations to other streams before the first prepare.
class SamplingInputBinding final {
 public:
  static Status create(const SamplingInputCapacity& capacity,
                       const torch::Device& device,
                       torch::ScalarType parameter_dtype,
                       std::unique_ptr<SamplingInputBinding>& output);

  SamplingInputBinding(const SamplingInputBinding&) = delete;
  SamplingInputBinding& operator=(const SamplingInputBinding&) = delete;

  Status validate(const SamplingParameters& input, uint32_t model_tokens) const;

  // Reads only contiguous CPU tensors. After success, source storage may be
  // released or overwritten: H2D reads owned pinned staging. Record the Task's
  // final ready event after this and every other preparation, then wait on it
  // before consuming params(). No RNG, event creation or Host/device wait.
  // Expected rejection changes neither the previous binding nor its storage.
  Status prepare(const SamplingParameters& input,
                 uint32_t model_tokens,
                 const Stream& stream);

  const SamplingParameters& params() const { return params_; }
  const SamplingInputTransferInfo& transfer_info() const { return transfer_; }
  const SamplingInputCapacity& capacity() const { return capacity_; }
  uint64_t pinned_bytes() const { return bytes_; }
  uint64_t device_bytes() const { return bytes_; }

 private:
  SamplingInputBinding(SamplingInputCapacity capacity,
                       torch::Device device,
                       torch::ScalarType parameter_dtype,
                       uint64_t bytes);

  SamplingInputCapacity capacity_;
  torch::Device device_;
  torch::ScalarType parameter_dtype_;
  uint64_t bytes_ = 0;
  SamplingParameters host_storage_;
  SamplingParameters device_storage_;
  SamplingParameters params_;
  SamplingInputTransferInfo transfer_;
};

}  // namespace xllm
