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

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <limits>
#include <thread>
#include <vector>

#include "core/kernels/npu/xllm_ops/xllm_ops_api.h"
#include "core/platform/stream.h"

namespace xllm::kernel::npu {
namespace {

struct FilterCase {
  torch::Tensor logits;
  torch::Tensor reference;
  torch::Tensor top_k;
  torch::Tensor top_p;
  const void* logits_address = nullptr;
  uint32_t plan = 0;
};

class TopKTopPPlanTest : public ::testing::TestWithParam<torch::ScalarType> {
 protected:
  void TearDown() override { EXPECT_EQ(aclrtSynchronizeDevice(), ACL_SUCCESS); }
  const torch::Device device_{torch::kPrivateUse1, 0};
};

TEST_P(TopKTopPPlanTest, RepeatedAddressUpdatesShareWorkspaceAcrossShapes) {
  constexpr int64_t kVocab = 128;
  const auto options = torch::TensorOptions().dtype(GetParam()).device(device_);
  std::vector<std::unique_ptr<TopKTopPPlan>> plans;
  plans.reserve(/*new_cap=*/8);
  uint64_t max_workspace = 0;
  for (const int64_t padding : {0, 8}) {
    for (int64_t rows = 1; rows <= 4; ++rows) {
      auto logits = torch::zeros({rows, kVocab + padding}, options)
                        .narrow(/*dim=*/1, /*start=*/0, kVocab);
      auto top_k = torch::full({rows}, kVocab, options.dtype(torch::kInt32));
      auto top_p = torch::ones({rows}, options);
      auto plan = TopKTopPPlan::create(logits, top_k, top_p);
      max_workspace = std::max(max_workspace, plan->workspace_bytes());
      plans.emplace_back(std::move(plan));
      // All template tensors die here; run must bind the supplied Slot data.
    }
  }
  ASSERT_LE(max_workspace,
            static_cast<uint64_t>(std::numeric_limits<int64_t>::max()));
  auto workspace = torch::empty({static_cast<int64_t>(max_workspace)},
                                options.dtype(torch::kUInt8));
  const void* workspace_address = workspace.data_ptr();
  std::vector<FilterCase> cases;
  cases.reserve(/*new_cap=*/64);
  constexpr std::array<int32_t, 4> kThresholds = {-1, 1, 13, 128};
  constexpr std::array<float, 4> kMasses = {0.0f, 0.01f, 0.8f, 1.0f};
  for (uint32_t round = 0; round < 64; ++round) {
    const int64_t rows = 1 + round % 4;
    const int64_t padding = round % 8 >= 4 ? 8 : 0;
    const int64_t offset_row = round % 3;
    const int64_t offset_column = padding > 0 ? round % 5 : 0;
    auto values =
        ((torch::arange((rows + 2) * (kVocab + padding)).to(torch::kFloat32) +
          17 * round)
             .remainder(/*other=*/131) -
         65)
            .div(/*other=*/7.31)
            .view({rows + 2, kVocab + padding});
    auto logits = values.to(options)
                      .narrow(/*dim=*/0, offset_row, rows)
                      .narrow(/*dim=*/1, offset_column, kVocab);
    auto expected = values.to(options)
                        .narrow(/*dim=*/0, offset_row, rows)
                        .narrow(/*dim=*/1, offset_column, kVocab);
    auto k_host = torch::empty({rows + 2}, torch::kInt32);
    auto p_host = torch::empty({rows + 2}, torch::kFloat32);
    for (int64_t row = 0; row < rows + 2; ++row) {
      k_host[row] = kThresholds[(row + round) % kThresholds.size()];
      p_host[row] = kMasses[(3 * row + round) % kMasses.size()];
    }
    auto top_k = k_host.to(device_).narrow(/*dim=*/0, round % 3, rows);
    auto top_p = p_host.to(options).narrow(/*dim=*/0, round % 3, rows);
    top_k_top_p(expected, top_k, top_p);
    const void* address = logits.data_ptr();
    cases.emplace_back(FilterCase{std::move(logits),
                                  std::move(expected),
                                  std::move(top_k),
                                  std::move(top_p),
                                  address,
                                  round % 8});
  }
  Stream launch(device_);
  ASSERT_EQ(aclrtSynchronizeDevice(), ACL_SUCCESS);
  // Initialize on the state thread and run all shapes/addresses from a separate
  // Launch thread. No per-invocation Host wait or workspace allocation.
  std::thread runner([&] {
    auto guard = launch.set_stream_guard();
    for (auto& item : cases) {
      plans[item.plan]->run(item.logits, item.top_k, item.top_p, workspace);
    }
  });
  runner.join();
  ASSERT_EQ(aclrtSynchronizeStream(launch.get_stream()->stream()), ACL_SUCCESS);
  for (uint32_t index = 0; index < cases.size(); ++index) {
    SCOPED_TRACE(index);
    const auto& item = cases[index];
    EXPECT_EQ(item.logits.data_ptr(), item.logits_address);
    EXPECT_TRUE(torch::equal(item.reference.cpu(), item.logits.cpu()));
  }
  EXPECT_EQ(workspace.data_ptr(), workspace_address);
  LOG(INFO) << "TOP_K_TOP_P_PLAN dtype=" << GetParam()
            << " plans=" << plans.size() << " submissions=" << cases.size()
            << " shared_workspace_bytes=" << max_workspace;
}

TEST_P(TopKTopPPlanTest, RecreatesAndRetiresPlansWithoutRetainingOldData) {
  const auto options = torch::TensorOptions().dtype(GetParam()).device(device_);
  Stream launch(device_);
  auto guard = launch.set_stream_guard();
  for (int32_t round = 0; round < 4; ++round) {
    SCOPED_TRACE(round);
    auto initial = torch::zeros({2, 256}, options);
    auto top_k = torch::full({2}, 1 + round, options.dtype(torch::kInt32));
    auto top_p = torch::full({2}, /*fill_value=*/0.9, options);
    auto plan = TopKTopPPlan::create(initial, top_k, top_p);
    auto workspace =
        torch::empty({static_cast<int64_t>(plan->workspace_bytes())},
                     options.dtype(torch::kUInt8));
    auto logits = torch::arange(/*end=*/512, options).view({2, 256});
    auto expected = logits.clone();
    top_k_top_p(expected, top_k, top_p);
    plan->run(logits, top_k, top_p, workspace);
    EXPECT_TRUE(torch::equal(expected.cpu(), logits.cpu()));
    EXPECT_EQ(aclrtSynchronizeStream(launch.get_stream()->stream()),
              ACL_SUCCESS);
    plan.reset();
    EXPECT_TRUE(torch::equal(expected.cpu(), logits.cpu()));
  }
}

INSTANTIATE_TEST_SUITE_P(Dtypes,
                         TopKTopPPlanTest,
                         ::testing::Values(torch::kFloat16,
                                           torch::kBFloat16,
                                           torch::kFloat32));

}  // namespace
}  // namespace xllm::kernel::npu
