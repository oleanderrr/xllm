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

#include "core/runtime/task_pipeline/model_input_preparer.h"

#include <c10/core/DeviceGuard.h>
#include <glog/logging.h>

#include <array>
#include <cstring>
#include <limits>

namespace xllm {
namespace {

bool overlaps_host(std::span<const int32_t> source,
                   const torch::Tensor& destination) {
  if (source.empty()) {
    return false;
  }
  const uintptr_t start = reinterpret_cast<uintptr_t>(source.data());
  const uintptr_t base = reinterpret_cast<uintptr_t>(destination.data_ptr());
  return start >= base ? start - base < destination.nbytes()
                       : base - start < source.size_bytes();
}

Status validate(const ModelInputHostView& input,
                const ModelInputStorage& storage) {
  const ModelInputCapacity& capacity = storage.layout().capacity;
  const uint64_t tokens = input.token_ids.size();
  const uint64_t rows = input.q_seq_lens.size();
  if (tokens > capacity.max_tokens || rows > capacity.max_sequences ||
      tokens > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) ||
      input.block_table_width > capacity.max_blocks_per_sequence ||
      input.positions.size() != tokens ||
      input.new_cache_slots.size() != tokens ||
      input.kv_seq_lens.size() != rows || input.q_cu_seq_lens.size() != rows ||
      input.block_tables.size() != rows * input.block_table_width ||
      (rows == 0 && (tokens != 0 || input.block_table_width != 0)) ||
      (rows != 0 && (tokens == 0 || input.block_table_width == 0))) {
    return Status(StatusCode::INVALID_ARGUMENT,
                  "Invalid model input sizes or capacity.");
  }
  int64_t cumulative = 0;
  for (uint64_t row = 0; row < rows; ++row) {
    cumulative += input.q_seq_lens[row];
    if (input.q_seq_lens[row] < 0 ||
        input.kv_seq_lens[row] < input.q_seq_lens[row] ||
        cumulative != input.q_cu_seq_lens[row]) {
      return Status(StatusCode::INVALID_ARGUMENT,
                    "Inconsistent model input sequence lengths.");
    }
  }
  if (static_cast<uint64_t>(cumulative) != tokens) {
    return Status(StatusCode::INVALID_ARGUMENT,
                  "Query lengths do not cover input tokens.");
  }
  for (const auto source : {input.token_ids,
                            input.positions,
                            input.new_cache_slots,
                            input.q_seq_lens,
                            input.kv_seq_lens,
                            input.q_cu_seq_lens,
                            input.block_tables}) {
    if (overlaps_host(source, storage.host_buffer())) {
      return Status(StatusCode::INVALID_ARGUMENT,
                    "Input aliases destination staging storage.");
    }
  }
  return Status();
}

}  // namespace

ModelInputPreparer::ModelInputPreparer(ModelInputStorage& storage)
    : storage_(storage) {
  c10::DeviceGuard guard(storage_.device_buffer().device());
  aclrtEvent event = nullptr;
  // Ordinary ACL events retain completion until reset. The Ex event binds
  // each wait to a record generation and supports reuse without a host reset.
  CHECK_EQ(aclrtCreateEventExWithFlag(&event, ACL_EVENT_SYNC), ACL_SUCCESS);
  ready_event_ = std::make_shared<StreamEvent>(event);
}

Status ModelInputPreparer::prepare(const ModelInputHostView& input,
                                   const Stream& stream,
                                   ModelInputTransferInfo& info) {
  Status status = validate(input, storage_);
  if (!status.ok()) {
    return status;
  }
  if (stream.get_stream()->device_index() !=
      storage_.device_buffer().device().index()) {
    return Status(StatusCode::INVALID_ARGUMENT,
                  "Input stream and storage device differ.");
  }
  c10::DeviceGuard guard(storage_.device_buffer().device());
  const aclrtStream transfer_stream = stream.get_stream()->stream();
  const ModelInputLayout& layout = storage_.layout();
  ModelInputTransferInfo transferred;
  transferred.tokens = static_cast<uint32_t>(input.token_ids.size());
  transferred.sequences = static_cast<uint32_t>(input.q_seq_lens.size());
  int32_t* host = storage_.host_buffer().data_ptr<int32_t>();
  int32_t* device = storage_.device_buffer().data_ptr<int32_t>();
  const std::array<std::span<const int32_t>, 6> sources = {
      input.token_ids,
      input.positions,
      input.new_cache_slots,
      input.q_seq_lens,
      input.kv_seq_lens,
      input.q_cu_seq_lens};
  const std::array<ModelInputRegion, 6> regions = {layout.token_ids,
                                                   layout.positions,
                                                   layout.new_cache_slots,
                                                   layout.q_seq_lens,
                                                   layout.kv_seq_lens,
                                                   layout.q_cu_seq_lens};
  for (uint32_t i = 0; i < sources.size(); ++i) {
    if (sources[i].empty()) {
      continue;
    }
    const uint64_t offset = regions[i].offset / sizeof(int32_t);
    std::memcpy(host + offset, sources[i].data(), sources[i].size_bytes());
    CHECK_EQ(aclrtMemcpyAsync(device + offset,
                              regions[i].bytes,
                              host + offset,
                              sources[i].size_bytes(),
                              ACL_MEMCPY_HOST_TO_DEVICE,
                              transfer_stream),
             ACL_SUCCESS);
    transferred.h2d_bytes += sources[i].size_bytes();
    ++transferred.h2d_calls;
  }
  if (transferred.sequences != 0) {
    const uint64_t offset = layout.block_tables.offset / sizeof(int32_t);
    const uint64_t stride = layout.block_table_row_stride_bytes;
    const uint64_t width =
        static_cast<uint64_t>(input.block_table_width) * sizeof(int32_t);
    const uint64_t bytes = stride * transferred.sequences;
    if (width == stride) {
      std::memcpy(host + offset, input.block_tables.data(), bytes);
    } else {
      std::memset(host + offset, 0, bytes);
      for (uint32_t row = 0; row < transferred.sequences; ++row) {
        std::memcpy(host + offset +
                        static_cast<uint64_t>(row) *
                            layout.capacity.max_blocks_per_sequence,
                    input.block_tables.data() +
                        static_cast<uint64_t>(row) * input.block_table_width,
                    width);
      }
      CHECK_EQ(aclrtMemsetAsync(device + offset,
                                layout.block_tables.bytes,
                                /*value=*/0,
                                bytes,
                                transfer_stream),
               ACL_SUCCESS);
      transferred.device_zero_bytes = bytes;
    }
    CHECK_EQ(aclrtMemcpy2dAsync(device + offset,
                                stride,
                                host + offset,
                                stride,
                                width,
                                transferred.sequences,
                                ACL_MEMCPY_HOST_TO_DEVICE,
                                transfer_stream),
             ACL_SUCCESS);
    transferred.h2d_bytes += width * transferred.sequences;
    ++transferred.h2d_calls;
  }
  CHECK_EQ(aclrtRecordEvent(ready_event_->npu_event(), transfer_stream),
           ACL_SUCCESS);
  info = transferred;
  return Status();
}

}  // namespace xllm
