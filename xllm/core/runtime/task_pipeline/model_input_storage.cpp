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

#include "core/runtime/task_pipeline/model_input_storage.h"

#include <c10/core/DeviceGuard.h>
#include <glog/logging.h>

#include <utility>

namespace xllm {
namespace {

torch::Tensor field_view(const torch::Tensor& buffer,
                         const ModelInputRegion& region) {
  return buffer.narrow(/*dim=*/0,
                       static_cast<int64_t>(region.offset / sizeof(int32_t)),
                       static_cast<int64_t>(region.bytes / sizeof(int32_t)));
}

ModelInputTensors bind_views(const torch::Tensor& buffer,
                             const ModelInputLayout& layout) {
  ModelInputTensors views;
  views.token_ids = field_view(buffer, layout.token_ids);
  views.positions = field_view(buffer, layout.positions);
  views.new_cache_slots = field_view(buffer, layout.new_cache_slots);
  views.q_seq_lens = field_view(buffer, layout.q_seq_lens);
  views.kv_seq_lens = field_view(buffer, layout.kv_seq_lens);
  views.q_cu_seq_lens = field_view(buffer, layout.q_cu_seq_lens);
  views.block_tables = field_view(buffer, layout.block_tables)
                           .view({layout.capacity.max_sequences,
                                  layout.capacity.max_blocks_per_sequence});
  return views;
}

}  // namespace

ModelInputStorage::ModelInputStorage(ModelInputLayout layout,
                                     torch::Tensor host_buffer,
                                     torch::Tensor device_buffer)
    : layout_(std::move(layout)),
      host_buffer_(std::move(host_buffer)),
      device_buffer_(std::move(device_buffer)),
      host_(bind_views(host_buffer_, layout_)),
      device_(bind_views(device_buffer_, layout_)) {}

Status ModelInputStorage::create(const ModelInputCapacity& capacity,
                                 const torch::Device& device,
                                 std::unique_ptr<ModelInputStorage>& output) {
  ModelInputLayout layout;
  Status status = make_model_input_layout(capacity, layout);
  if (!status.ok()) {
    return status;
  }
  if (device.type() != torch::kPrivateUse1 || !device.has_index()) {
    return Status(StatusCode::INVALID_ARGUMENT,
                  "Model input storage requires an explicitly indexed NPU.");
  }
  c10::DeviceGuard guard(device);
  const int64_t elements =
      static_cast<int64_t>(layout.total_bytes / sizeof(int32_t));
  torch::Tensor host_buffer = torch::empty({elements},
                                           torch::TensorOptions()
                                               .dtype(torch::kInt32)
                                               .device(torch::kCPU)
                                               .pinned_memory(true));
  torch::Tensor device_buffer = torch::empty(
      {elements}, torch::TensorOptions().dtype(torch::kInt32).device(device));
  CHECK_EQ(reinterpret_cast<uintptr_t>(host_buffer.data_ptr()) %
               kModelInputAlignment,
           0);
  CHECK_EQ(reinterpret_cast<uintptr_t>(device_buffer.data_ptr()) %
               kModelInputAlignment,
           0);
  auto storage = std::unique_ptr<ModelInputStorage>(new ModelInputStorage(
      std::move(layout), std::move(host_buffer), std::move(device_buffer)));
  output = std::move(storage);
  return Status();
}

}  // namespace xllm
