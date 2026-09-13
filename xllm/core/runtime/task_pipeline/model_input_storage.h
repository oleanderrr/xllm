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

#include <torch/types.h>

#include <memory>

#include "core/runtime/task_pipeline/model_input_layout.h"

namespace xllm {

// Capacity views share ownership of their backing tensor's storage. They do
// not specify this task's valid lengths or authorize reading unused capacity.
struct ModelInputTensors {
  torch::Tensor token_ids;
  torch::Tensor positions;
  torch::Tensor new_cache_slots;
  torch::Tensor q_seq_lens;
  torch::Tensor kv_seq_lens;
  torch::Tensor q_cu_seq_lens;
  torch::Tensor block_tables;
};

// Fixed model-input storage for one ordinary NPU LLM invocation. Transfer
// ownership through unique_ptr. All asynchronous readers must finish before
// reuse or destruction; this owner does not synchronize or track task state.
class ModelInputStorage final {
 public:
  // Validates capacity and an explicitly indexed NPU device before allocating.
  // Expected validation failures preserve output. Allocation errors propagate.
  static Status create(const ModelInputCapacity& capacity,
                       const torch::Device& device,
                       std::unique_ptr<ModelInputStorage>& output);

  ModelInputStorage(const ModelInputStorage&) = delete;
  ModelInputStorage& operator=(const ModelInputStorage&) = delete;

  const ModelInputLayout& layout() const { return layout_; }
  const torch::Tensor& host_buffer() const { return host_buffer_; }
  const torch::Tensor& device_buffer() const { return device_buffer_; }
  const ModelInputTensors& host() const { return host_; }
  const ModelInputTensors& device() const { return device_; }

 private:
  ModelInputStorage(ModelInputLayout layout,
                    torch::Tensor host_buffer,
                    torch::Tensor device_buffer);

  ModelInputLayout layout_;
  torch::Tensor host_buffer_;
  torch::Tensor device_buffer_;
  ModelInputTensors host_;
  ModelInputTensors device_;
};

}  // namespace xllm
