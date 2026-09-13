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

#include <aclnn/acl_meta.h>
#include <torch/types.h>

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace xllm::kernel::npu {

// A fixed shape/stride/dtype plan. Create all admitted plans during
// initialization, then allocate one shared workspace of their maximum size.
// Logits are filtered in place. Threshold values must already be validated:
// top_k is int32 and top_p has the logits dtype. No implicit dtype conversion.
class TopKTopPPlan final {
 public:
  static std::unique_ptr<TopKTopPPlan> create(const torch::Tensor& logits,
                                              const torch::Tensor& top_k,
                                              const torch::Tensor& top_p);
  ~TopKTopPPlan();
  TopKTopPPlan(const TopKTopPPlan&) = delete;
  TopKTopPPlan& operator=(const TopKTopPPlan&) = delete;

  uint64_t workspace_bytes() const { return workspace_bytes_; }

  // Metadata must match the templates; storage offsets and addresses may
  // change. Uses the caller's current NPU stream. Serialize runs and every
  // workspace reader on one Launch stream, and retire them before destruction.
  // No Device allocation, workspace growth, dtype cast, or Host/device wait.
  void run(torch::Tensor& logits,
           const torch::Tensor& top_k,
           const torch::Tensor& top_p,
           const torch::Tensor& workspace);

 private:
  struct TensorSpec {
    torch::ScalarType dtype;
    std::vector<int64_t> sizes;
    std::vector<int64_t> strides;
  };

  TopKTopPPlan(const torch::Tensor& logits,
               const torch::Tensor& top_k,
               const torch::Tensor& top_p);
  void check_binding(const torch::Tensor& tensor, uint32_t index) const;

  torch::Device device_;
  // ACL input order: logits, p, k; final descriptor is the in-place output.
  std::array<aclTensor*, 4> tensors_{};
  std::array<TensorSpec, 3> specs_;
  aclOpExecutor* executor_ = nullptr;
  uint64_t workspace_bytes_ = 0;
};

}  // namespace xllm::kernel::npu
