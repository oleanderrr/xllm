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

#include "core/framework/model/model_input_params.h"
#include "core/runtime/task_pipeline/model_input_preparer.h"

namespace xllm {

// Physical rows include caller-provided padding; actual rows form a prefix.
struct ModelInputBatch {
  BatchForwardType forward_type;
  uint32_t num_actual_sequences = 0;
  uint64_t batch_id = 0;
  bool is_graph_warmup = false;
};

// Binds ordinary Python non-MLA input and paged-attention metadata. Shared
// KV/state dependencies are established by the caller before execution.
// Borrows storage until all model readers and event waits have completed.
class ModelInputBinding final {
 public:
  explicit ModelInputBinding(ModelInputStorage& storage);
  ModelInputBinding(const ModelInputBinding&) = delete;
  ModelInputBinding& operator=(const ModelInputBinding&) = delete;

  // Validation failures preserve the previous binding, storage and event.
  // Successful return stabilizes source data, but H2D can remain in flight.
  Status prepare(const ModelInputHostView& input,
                 const ModelInputBatch& batch,
                 const Stream& stream);

  // Valid after successful prepare, until the next prepare or destruction.
  // Views keep the fixed storage alive; reuse still requires reader retirement.
  const torch::Tensor& tokens() const { return tokens_; }
  const torch::Tensor& positions() const { return positions_; }
  const ModelInputParams& params() const { return params_; }
  const ModelInputTransferInfo& transfer_info() const { return transfer_info_; }
  const StreamEventPtr& ready_event() const { return preparer_.ready_event(); }

 private:
  ModelInputStorage& storage_;
  ModelInputPreparer preparer_;
  ModelInputParams params_;
  torch::Tensor tokens_;
  torch::Tensor positions_;
  ModelInputTransferInfo transfer_info_;
};

}  // namespace xllm
