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

#include "core/runtime/task_pipeline/sequence_token_binding.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <array>
#include <chrono>
#include <cstdlib>
#include <numeric>
#include <vector>

namespace xllm {
namespace {

struct BindingBatch {
  std::vector<int32_t> tokens;
  std::vector<int32_t> positions;
  std::vector<int32_t> q;
  std::vector<int32_t> kv;
  std::vector<int32_t> ends;
};

ModelInputHostView view(const BindingBatch& input) {
  // KV page fields are irrelevant to token binding; its caller validates them.
  return {
      input.tokens, input.positions, {}, input.q, input.kv, input.ends, {}, 0};
}

SamplingParameters sampling(std::vector<int32_t> selected,
                            std::vector<int32_t> sample_rows) {
  SamplingParameters params;
  params.selected_token_idxes = torch::tensor(selected, torch::kInt32);
  params.sample_idxes = torch::tensor(sample_rows, torch::kInt32);
  params.do_sample =
      torch::zeros({static_cast<int64_t>(sample_rows.size())}, torch::kBool);
  params.all_greedy_sample = true;
  return params;
}

class SequenceTokenBindingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_TRUE(SequenceStatePool::create(/*capacity=*/8, device_, pool_).ok());
    ASSERT_TRUE(
        SequenceTokenBinding::create(*pool_, /*capacity=*/4, first_).ok());
    ASSERT_TRUE(
        SequenceTokenBinding::create(*pool_, /*capacity=*/4, second_).ok());
    first_->host_indices().fill_(/*value=*/-77);
    second_->host_indices().fill_(/*value=*/-77);
    ASSERT_EQ(aclrtSynchronizeDevice(), ACL_SUCCESS);
  }
  void TearDown() override {
    EXPECT_EQ(aclrtSynchronizeDevice(), ACL_SUCCESS);
    first_->release();
    second_->release();
    first_.reset();
    second_.reset();
    pool_.reset();
  }
  const torch::Device device_{torch::kPrivateUse1, 0};
  std::unique_ptr<SequenceStatePool> pool_;
  std::unique_ptr<SequenceTokenBinding> first_;
  std::unique_ptr<SequenceTokenBinding> second_;
};

TEST_F(SequenceTokenBindingTest, FactoryAndIndependentCpuInputUseFixedStorage) {
  const SequenceTokenBinding* original = first_.get();
  EXPECT_FALSE(
      SequenceTokenBinding::create(*pool_, /*capacity=*/0, first_).ok());
  EXPECT_FALSE(
      SequenceTokenBinding::create(*pool_, /*capacity=*/9, first_).ok());
  EXPECT_EQ(first_.get(), original);
  EXPECT_EQ(first_->pinned_bytes(), 4 * 24);
  EXPECT_EQ(first_->device_bytes(), 4 * 36);
  EXPECT_NE(first_->host_indices().data_ptr(),
            second_->host_indices().data_ptr());
  EXPECT_NE(first_->device_indices().data_ptr(),
            second_->device_indices().data_ptr());
  Stream prepare(device_);
  BindingBatch input{{11, 12}, {0, 1}, {2}, {2}, {2}};
  const auto params = sampling({1}, {0});
  ASSERT_TRUE(first_->prepare(view(input), params, {}, {}, prepare).ok());
  EXPECT_EQ(pool_->available_rows(), 8U);
  first_->release();
  input.tokens.back() = -17;
  EXPECT_FALSE(first_->prepare(view(input), params, {}, {}, prepare).ok());
  EXPECT_EQ(pool_->available_rows(), 8U);
}

TEST_F(SequenceTokenBindingTest,
       MapsSelectedSampleRowsAndBorrowsOnlyUntilPrepare) {
  Stream prepare(device_);
  Stream task(device_);
  BindingBatch first{{11, 12, 21}, {0, 1, 0}, {2, 1}, {2, 1}, {2, 3}};
  BindingBatch second{{-7, -901}, {1, 2}, {1, 1}, {2, 3}, {1, 2}};
  std::array<SequenceStateKey, 2> first_keys{{{11, 0}, {22, 0}}};
  std::array<SequenceStateKey, 2> second_keys{{{22, 0}, {11, 0}}};
  // Selected order and model-row order intentionally differ.
  auto params_a = sampling({2, 1}, {0, 1});
  auto params_b = sampling({1, 0}, {0, 1});
  const void* host = second_->host_indices().data_ptr();
  const void* device = second_->device_indices().data_ptr();
  auto model_tokens = torch::tensor(second.tokens, torch::kInt32).to(device_);
  auto output_a = torch::tensor(std::vector<int64_t>{71, 93}, torch::kInt64)
                      .to(device_)
                      .view({2, 1});
  auto output_b = torch::tensor(std::vector<int64_t>{103, 81}, torch::kInt64)
                      .to(device_)
                      .view({2, 1});
  ASSERT_EQ(aclrtSynchronizeDevice(), ACL_SUCCESS);
  ASSERT_TRUE(
      first_->prepare(view(first), params_a, first_keys, {}, prepare).ok());
  const auto ready_a = prepare.record_event();
  ASSERT_TRUE(
      second_->prepare(view(second), params_b, second_keys, {}, prepare).ok());
  const auto ready_b = prepare.record_event();
  params_a.selected_token_idxes.zero_();
  params_b.selected_token_idxes.zero_();
  first_keys = {};
  second_keys = {};
  {
    auto guard = task.set_stream_guard();
    ASSERT_TRUE(task.wait_event(ready_a));
    first_->publish(output_a);
    ASSERT_TRUE(task.wait_event(ready_b));
    second_->gather_into(model_tokens);
    second_->publish(output_b);
  }
  ASSERT_EQ(task.synchronize(), ACL_SUCCESS);
  EXPECT_TRUE(
      torch::equal(model_tokens.cpu(), torch::tensor({71, 93}, torch::kInt32)));
  EXPECT_EQ(second_->host_indices().data_ptr(), host);
  EXPECT_EQ(second_->device_indices().data_ptr(), device);
  first_->release();
  const std::array<SequenceStateKey, 2> retired{{{11, 0}, {22, 0}}};
  BindingBatch empty;
  ASSERT_TRUE(first_->prepare(view(empty), {}, {}, retired, prepare).ok());
  EXPECT_EQ(pool_->available_rows(), 6U);
  second_->release();
  EXPECT_EQ(pool_->available_rows(), 8U);
}

TEST_F(SequenceTokenBindingTest, MappingRejectionPreservesFinalStagingAndPool) {
  Stream prepare(device_);
  BindingBatch input{{11, 12, 21}, {0, 1, 0}, {2, 1}, {2, 1}, {2, 3}};
  const std::array<SequenceStateKey, 2> keys{{{11, 0}, {22, 0}}};
  const auto valid = sampling({1, 2}, {0, 1});
  ASSERT_TRUE(first_->prepare(view(input), valid, keys, {}, prepare).ok());
  ASSERT_EQ(prepare.synchronize(), ACL_SUCCESS);
  auto before = first_->host_indices().clone();
  first_->release();
  const auto check_rejected = [&](const BindingBatch& batch,
                                  const SamplingParameters& params) {
    EXPECT_FALSE(first_->prepare(view(batch), params, keys, {}, prepare).ok());
    EXPECT_TRUE(torch::equal(before, first_->host_indices()));
    EXPECT_EQ(pool_->available_rows(), 6U);
  };
  check_rejected(input, sampling({0, 2}, {0, 1}));
  check_rejected(input, sampling({1, 1}, {0, 1}));
  input.tokens.front() = -1;
  check_rejected(input, valid);
  input.tokens.front() = 11;
  // Correct mapping still rejects a duplicate publication position.
  check_rejected(input, valid);
  BindingBatch next{{-1, -2}, {2, 1}, {1, 1}, {3, 2}, {1, 2}};
  ASSERT_TRUE(
      first_->prepare(view(next), sampling({0, 1}, {0, 1}), keys, {}, prepare)
          .ok());
  ASSERT_EQ(prepare.synchronize(), ACL_SUCCESS);
}

TEST_F(SequenceTokenBindingTest,
       EmptySamplingLeavesRowsUninitializedAndEpochsIndependent) {
  Stream prepare(device_);
  BindingBatch first{{11, 12}, {0, 1}, {2}, {2}, {2}};
  const std::array<SequenceStateKey, 1> old_key{{{11, 0}}};
  const std::array<SequenceStateKey, 1> new_key{{{11, 1}}};
  ASSERT_TRUE(first_->prepare(view(first), {}, old_key, {}, prepare).ok());
  first_->release();
  BindingBatch read{{-1}, {2}, {1}, {3}, {1}};
  EXPECT_FALSE(
      second_->prepare(view(read), sampling({0}, {0}), old_key, {}, prepare)
          .ok());
  EXPECT_FALSE(
      second_
          ->prepare(view(read), sampling({0}, {0}), new_key, old_key, prepare)
          .ok());
  // Failed replacement preserved old state so its retirement remains valid.
  ASSERT_TRUE(
      second_
          ->prepare(view(first), sampling({1}, {0}), new_key, old_key, prepare)
          .ok());
  ASSERT_EQ(prepare.synchronize(), ACL_SUCCESS);
  EXPECT_EQ(pool_->available_rows(), 7U);
}

TEST_F(SequenceTokenBindingTest, GatherPublishPerformance) {
  if (std::getenv(/*name=*/"XLLM_PIPELINE_PERF") == nullptr) {
    GTEST_SKIP() << "Run separately with XLLM_PIPELINE_PERF=1.";
  }
  first_.reset();
  second_.reset();
  ASSERT_TRUE(SequenceStatePool::create(/*capacity=*/128, device_, pool_).ok());
  ASSERT_TRUE(
      SequenceTokenBinding::create(*pool_, /*capacity=*/128, first_).ok());
  ASSERT_TRUE(
      SequenceTokenBinding::create(*pool_, /*capacity=*/128, second_).ok());
  Stream prepare(device_);
  Stream task(device_);
  constexpr int32_t kIterations = 128;
  for (uint32_t rows : std::array<uint32_t, 4>{1, 4, 32, 128}) {
    BindingBatch input;
    input.tokens.assign(rows, 13);
    input.positions.assign(rows, 0);
    input.q.assign(rows, 1);
    input.kv.assign(rows, 1);
    input.ends.resize(rows);
    std::iota(input.ends.begin(), input.ends.end(), 1);
    std::vector<SequenceStateKey> keys;
    std::vector<int32_t> indices;
    keys.reserve(rows);
    indices.reserve(rows);
    for (uint32_t row = 0; row < rows; ++row) {
      keys.emplace_back(SequenceStateKey{row + 1U, rows});
      indices.emplace_back(static_cast<int32_t>(row));
    }
    const auto params = sampling(indices, indices);
    auto tokens = torch::full(
        {rows},
        -1,
        torch::TensorOptions().dtype(torch::kInt32).device(device_));
    auto published = torch::full({rows, 1}, 13, pool_->tokens().options());
    ASSERT_EQ(aclrtSynchronizeDevice(), ACL_SUCCESS);
    ASSERT_TRUE(first_->prepare(view(input), params, keys, {}, prepare).ok());
    auto ready = prepare.record_event();
    {
      auto guard = task.set_stream_guard();
      ASSERT_TRUE(task.wait_event(ready));
      first_->publish(published);
    }
    ASSERT_EQ(task.synchronize(), ACL_SUCCESS);
    first_->release();
    input.tokens.assign(rows, -1);
    input.positions.assign(rows, 1);
    input.kv.assign(rows, 2);
    ASSERT_TRUE(first_->prepare(view(input), params, keys, {}, prepare).ok());
    ready = prepare.record_event();
    auto guard = task.set_stream_guard();
    ASSERT_TRUE(task.wait_event(ready));
    const auto cycle = [&]() {
      first_->gather_into(tokens);
      first_->publish(published);
    };
    for (int32_t round = 0; round < 3; ++round) {
      for (int32_t warmup = 0; warmup < 8; ++warmup) {
        cycle();
      }
      ASSERT_EQ(task.synchronize(), ACL_SUCCESS);
      const auto begin = std::chrono::steady_clock::now();
      for (int32_t step = 0; step < kIterations; ++step) {
        cycle();
      }
      const auto submitted = std::chrono::steady_clock::now();
      ASSERT_EQ(task.synchronize(), ACL_SUCCESS);
      const auto completed = std::chrono::steady_clock::now();
      EXPECT_TRUE(
          torch::equal(tokens.cpu(), torch::full({rows}, 13, torch::kInt32)));
      LOG(INFO) << "TOKEN_BINDING_PERF rows=" << rows << " round=" << round
                << " iterations=" << kIterations << " host_us="
                << std::chrono::duration<double, std::micro>(submitted - begin)
                           .count() /
                       kIterations
                << " completion_us="
                << std::chrono::duration<double, std::micro>(completed - begin)
                           .count() /
                       kIterations
                << " pinned_bytes=" << first_->pinned_bytes()
                << " device_bytes=" << first_->device_bytes();
    }
    first_->release();
    BindingBatch empty;
    ASSERT_TRUE(first_->prepare(view(empty), {}, {}, keys, prepare).ok());
    EXPECT_EQ(pool_->available_rows(), 128U);
  }
}

}  // namespace
}  // namespace xllm
