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

#include "core/kernels/npu/xllm_ops/top_k_top_p_plan.h"

#include <aclnnop/aclnn_apply_top_k_top_p.h>
#include <c10/core/DeviceGuard.h>
#include <glog/logging.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <limits>

#include "core/kernels/npu/utils.h"

namespace xllm::kernel::npu {
namespace {

aclTensor* make_descriptor(const torch::Tensor& tensor) {
  // Use the logical start as the base so a later binding can have a different
  // storage offset. Only a contiguous vector or positive-stride row slice is
  // admitted. The storage span includes row padding, without retaining the
  // template tensor's allocation or assuming its total storage size.
  const int64_t span = tensor.dim() == 1 ? tensor.numel()
                                         : (tensor.size(/*dim=*/0) - 1) *
                                                   tensor.stride(/*dim=*/0) +
                                               tensor.size(/*dim=*/1);
  aclTensor* descriptor =
      aclCreateTensor(tensor.sizes().data(),
                      tensor.dim(),
                      type_info::get_acl_type(tensor.scalar_type()),
                      tensor.strides().data(),
                      /*offset=*/0,
                      ACL_FORMAT_ND,
                      &span,
                      /*storageDimsNum=*/1,
                      tensor.data_ptr());
  CHECK(descriptor != nullptr);
  return descriptor;
}

}  // namespace

std::unique_ptr<TopKTopPPlan> TopKTopPPlan::create(const torch::Tensor& logits,
                                                   const torch::Tensor& top_k,
                                                   const torch::Tensor& top_p) {
  CHECK(logits.defined());
  CHECK(logits.device().type() == torch::kPrivateUse1);
  CHECK(logits.device().has_index());
  CHECK(logits.scalar_type() == torch::kFloat32 ||
        logits.scalar_type() == torch::kFloat16 ||
        logits.scalar_type() == torch::kBFloat16);
  CHECK_EQ(logits.dim(), 2);
  CHECK_GT(logits.size(/*dim=*/0), 0);
  CHECK_GT(logits.size(/*dim=*/1), 0);
  CHECK_LE(logits.size(/*dim=*/0), std::numeric_limits<int32_t>::max());
  CHECK_LE(logits.size(/*dim=*/1), std::numeric_limits<int32_t>::max());
  CHECK_EQ(logits.stride(/*dim=*/1), 1);
  CHECK_GE(logits.stride(/*dim=*/0), logits.size(/*dim=*/1));
  CHECK_LE(logits.stride(/*dim=*/0), std::numeric_limits<int32_t>::max());
  for (const auto* threshold : {&top_k, &top_p}) {
    CHECK(threshold->defined());
    CHECK(threshold->device() == logits.device());
    CHECK_EQ(threshold->dim(), 1);
    CHECK_EQ(threshold->size(0), logits.size(/*dim=*/0));
    CHECK(threshold->is_contiguous());
  }
  CHECK(top_k.scalar_type() == torch::kInt32);
  CHECK(top_p.scalar_type() == logits.scalar_type());
  c10::DeviceGuard guard(logits.device());
  return std::unique_ptr<TopKTopPPlan>(new TopKTopPPlan(logits, top_k, top_p));
}

TopKTopPPlan::TopKTopPPlan(const torch::Tensor& logits,
                           const torch::Tensor& top_k,
                           const torch::Tensor& top_p)
    : device_(logits.device()) {
  const std::array<const torch::Tensor*, 3> inputs = {&logits, &top_p, &top_k};
  for (uint32_t index = 0; index < inputs.size(); ++index) {
    const auto& tensor = *inputs[index];
    specs_[index] = {
        tensor.scalar_type(), tensor.sizes().vec(), tensor.strides().vec()};
    tensors_[index] = make_descriptor(tensor);
  }
  tensors_[3] = make_descriptor(logits);
  CHECK_EQ(aclnnApplyTopKTopPGetWorkspaceSize(tensors_[0],
                                              tensors_[1],
                                              tensors_[2],
                                              tensors_[3],
                                              &workspace_bytes_,
                                              &executor_),
           0);
  CHECK(executor_ != nullptr);
  CHECK_EQ(aclSetAclOpExecutorRepeatable(executor_), 0);
}

TopKTopPPlan::~TopKTopPPlan() {
  c10::DeviceGuard guard(device_);
  CHECK_EQ(aclDestroyAclOpExecutor(executor_), 0);
  for (auto* tensor : tensors_) {
    CHECK_EQ(aclDestroyTensor(tensor), 0);
  }
}

void TopKTopPPlan::check_binding(const torch::Tensor& tensor,
                                 uint32_t index) const {
  CHECK(tensor.defined());
  CHECK(tensor.device() == device_);
  const auto& spec = specs_[index];
  CHECK(tensor.scalar_type() == spec.dtype);
  CHECK(tensor.sizes() == torch::IntArrayRef(spec.sizes));
  CHECK(tensor.strides() == torch::IntArrayRef(spec.strides));
}

void TopKTopPPlan::run(torch::Tensor& logits,
                       const torch::Tensor& top_k,
                       const torch::Tensor& top_p,
                       const torch::Tensor& workspace) {
  check_binding(logits, /*index=*/0);
  check_binding(top_p, /*index=*/1);
  check_binding(top_k, /*index=*/2);
  void* workspace_addr = nullptr;
  if (workspace_bytes_ > 0) {
    CHECK(workspace.defined());
    CHECK(workspace.device() == device_);
    CHECK(workspace.scalar_type() == torch::kUInt8);
    CHECK(workspace.is_contiguous());
    CHECK_GE(workspace.nbytes(), workspace_bytes_);
    workspace_addr = workspace.data_ptr();
  }
  c10::DeviceGuard guard(device_);
  CHECK_EQ(aclSetInputTensorAddr(
               executor_, /*index=*/0, tensors_[0], logits.data_ptr()),
           0);
  CHECK_EQ(aclSetInputTensorAddr(
               executor_, /*index=*/1, tensors_[1], top_p.data_ptr()),
           0);
  CHECK_EQ(aclSetInputTensorAddr(
               executor_, /*index=*/2, tensors_[2], top_k.data_ptr()),
           0);
  CHECK_EQ(aclSetOutputTensorAddr(
               executor_, /*index=*/0, tensors_[3], logits.data_ptr()),
           0);
  const aclrtStream stream =
      c10_npu::getCurrentNPUStream(device_.index()).stream();
  CHECK_EQ(
      aclnnApplyTopKTopP(workspace_addr, workspace_bytes_, executor_, stream),
      0);
}

}  // namespace xllm::kernel::npu
