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

#include "core/runtime/task_pipeline/sequence_state_pool.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <array>
#include <chrono>
#include <cstdlib>
#include <initializer_list>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include "core/platform/stream.h"

namespace xllm {
namespace {

static_assert(!std::is_copy_constructible_v<SequenceStateLease>);
static_assert(!std::is_copy_assignable_v<SequenceStateLease>);
static_assert(std::is_nothrow_move_constructible_v<SequenceStateLease>);
static_assert(std::is_nothrow_move_assignable_v<SequenceStateLease>);

torch::Tensor row_indices(const SequenceStateLease& lease,
                          const torch::Device& device) {
  std::vector<int64_t> rows;
  rows.reserve(lease.handles().size());
  for (const auto& handle : lease.handles()) {
    rows.emplace_back(handle.row);
  }
  return torch::tensor(rows, torch::kInt64).to(device);
}

class SequenceStatePoolTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_TRUE(SequenceStatePool::create(/*capacity=*/4, device_, pool_).ok());
    ASSERT_EQ(aclrtSynchronizeDevice(), ACL_SUCCESS);
  }

  void TearDown() override {
    EXPECT_EQ(aclrtSynchronizeDevice(), ACL_SUCCESS);
    pool_.reset();
  }

  Status acquire(std::initializer_list<SequenceStateKey> keys,
                 SequenceStateLease& lease) {
    std::vector<SequenceStateAccess> accesses;
    accesses.reserve(keys.size());
    for (const SequenceStateKey& key : keys) {
      accesses.emplace_back(
          SequenceStateAccess{key, std::nullopt, std::nullopt});
    }
    return pool_->admit(accesses, {}, lease);
  }

  Status retire(std::initializer_list<SequenceStateKey> keys) {
    SequenceStateLease empty(/*capacity=*/4);
    return pool_->admit({}, {keys.begin(), keys.size()}, empty);
  }

  Status admit(std::initializer_list<SequenceStateAccess> accesses,
               std::initializer_list<SequenceStateKey> retired_keys,
               SequenceStateLease& lease) {
    return pool_->admit({accesses.begin(), accesses.size()},
                        {retired_keys.begin(), retired_keys.size()},
                        lease);
  }

  const torch::Device device_{torch::kPrivateUse1, 0};
  std::unique_ptr<SequenceStatePool> pool_;
};

TEST_F(SequenceStatePoolTest, FactoryRejectsBeforeReplacingOutput) {
  const SequenceStatePool* original = pool_.get();
  EXPECT_FALSE(SequenceStatePool::create(/*capacity=*/0, device_, pool_).ok());
  EXPECT_FALSE(SequenceStatePool::create(
                   std::numeric_limits<uint32_t>::max(), device_, pool_)
                   .ok());
  EXPECT_FALSE(SequenceStatePool::create(
                   /*capacity=*/4, torch::Device(torch::kCPU), pool_)
                   .ok());
  EXPECT_FALSE(SequenceStatePool::create(
                   /*capacity=*/4, torch::Device(torch::kPrivateUse1), pool_)
                   .ok());
  EXPECT_EQ(pool_.get(), original);
  EXPECT_EQ(pool_->capacity(), 4U);
  EXPECT_EQ(pool_->available_rows(), 4U);
  EXPECT_EQ(pool_->device_bytes(), 4 * sizeof(int64_t));
  EXPECT_EQ(pool_->tokens().device(), device_);
  EXPECT_EQ(pool_->tokens().scalar_type(), torch::kInt64);
  EXPECT_EQ(pool_->tokens().sizes(), torch::IntArrayRef({4}));
  EXPECT_FALSE(pool_->is_current({0, 0}));
  EXPECT_FALSE(pool_->is_current({4, 1}));
}

TEST_F(SequenceStatePoolTest, BatchRejectionPreservesRowsAndLease) {
  SequenceStateLease held(/*capacity=*/4);
  ASSERT_TRUE(acquire({{10, 0}, {20, 0}, {30, 0}}, held).ok());
  const auto first = held.handles().front();
  const int64_t* storage = pool_->tokens().const_data_ptr<int64_t>();
  SequenceStateLease rejected(/*capacity=*/4);
  EXPECT_FALSE(acquire({{10, 0}, {40, 0}, {50, 0}}, rejected).ok());
  EXPECT_FALSE(acquire({{40, 0}, {0, 0}}, rejected).ok());
  EXPECT_FALSE(acquire({{40, 0}, {40, 0}}, rejected).ok());
  EXPECT_FALSE(
      acquire({{1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}}, rejected).ok());
  EXPECT_FALSE(acquire({{40, 0}}, held).ok());
  EXPECT_FALSE(retire({{10, 0}, {99, 0}}).ok());
  EXPECT_FALSE(retire({{10, 0}, {10, 0}}).ok());
  EXPECT_TRUE(rejected.empty());
  ASSERT_EQ(held.handles().size(), 3U);
  EXPECT_EQ(held.handles().front().row, first.row);
  EXPECT_EQ(held.handles().front().generation, first.generation);
  EXPECT_EQ(pool_->available_rows(), 1U);
  EXPECT_EQ(pool_->tokens().const_data_ptr<int64_t>(), storage);
  ASSERT_TRUE(acquire({{10, 0}}, rejected).ok());
  ASSERT_TRUE(retire({{10, 0}, {20, 0}, {30, 0}}).ok());
  held.reset();
  EXPECT_EQ(pool_->available_rows(), 3U);
  EXPECT_TRUE(pool_->is_current(first));
  rejected.reset();
  EXPECT_EQ(pool_->available_rows(), 4U);
  EXPECT_FALSE(pool_->is_current(first));
  EXPECT_FALSE(retire({{10, 0}}).ok());
}

TEST_F(SequenceStatePoolTest, RetirementWaitsForBothTasksAndSeparatesEpochs) {
  SequenceStateLease first(/*capacity=*/4);
  SequenceStateLease second(/*capacity=*/4);
  ASSERT_TRUE(acquire({{7, 0}}, first).ok());
  ASSERT_TRUE(acquire({{7, 0}}, second).ok());
  const auto old = first.handles().front();
  ASSERT_TRUE(retire({{7, 0}}).ok());
  ASSERT_TRUE(retire({{7, 0}}).ok());
  SequenceStateLease rejected(/*capacity=*/4);
  EXPECT_FALSE(acquire({{7, 0}}, rejected).ok());
  SequenceStateLease recomputed(/*capacity=*/4);
  ASSERT_TRUE(acquire({{7, 1}}, recomputed).ok());
  EXPECT_NE(recomputed.handles().front().row, old.row);
  first.reset();
  EXPECT_TRUE(pool_->is_current(old));
  EXPECT_EQ(pool_->available_rows(), 2U);
  second.reset();
  EXPECT_FALSE(pool_->is_current(old));
  EXPECT_EQ(pool_->available_rows(), 3U);
  SequenceStateLease replacement(/*capacity=*/4);
  ASSERT_TRUE(acquire({{8, 0}}, replacement).ok());
  EXPECT_EQ(replacement.handles().front().row, old.row);
  EXPECT_GT(replacement.handles().front().generation, old.generation);
  EXPECT_FALSE(pool_->is_current(old));
  EXPECT_TRUE(pool_->is_current(replacement.handles().front()));
}

TEST_F(SequenceStatePoolTest, MoveTransfersExactlyOneReferenceSet) {
  SequenceStateLease original(/*capacity=*/4);
  SequenceStateLease target(/*capacity=*/4);
  ASSERT_TRUE(acquire({{1, 0}}, original).ok());
  ASSERT_TRUE(acquire({{2, 0}}, target).ok());
  const auto one = original.handles().front();
  const auto two = target.handles().front();
  ASSERT_TRUE(retire({{1, 0}, {2, 0}}).ok());
  SequenceStateLease moved(std::move(original));
  EXPECT_TRUE(original.empty());
  original.reset();
  EXPECT_TRUE(pool_->is_current(one));
  target = std::move(moved);
  EXPECT_TRUE(moved.empty());
  EXPECT_FALSE(pool_->is_current(two));
  EXPECT_TRUE(pool_->is_current(one));
  target.reset();
  target.reset();
  EXPECT_EQ(pool_->available_rows(), 4U);
  EXPECT_FALSE(pool_->is_current(one));
}

TEST_F(SequenceStatePoolTest, NonAdjacentSequencesKeepRowsAndTokens) {
  Stream stream(device_);
  auto guard = stream.set_stream_guard();
  SequenceStateLease initial(/*capacity=*/4);
  ASSERT_TRUE(acquire({{1, 0}, {2, 0}, {3, 0}}, initial).ok());
  const auto first = initial.handles().front();
  const torch::Tensor index = row_indices(initial, device_);
  const torch::Tensor values =
      torch::tensor({101, 202, 303}, torch::kInt64).to(device_);
  pool_->tokens().index_copy_(/*dim=*/0, index, values);
  ASSERT_EQ(stream.synchronize(), ACL_SUCCESS);
  initial.reset();
  EXPECT_EQ(pool_->available_rows(), 1U);
  {
    SequenceStateLease unrelated(/*capacity=*/4);
    ASSERT_TRUE(acquire({{4, 0}}, unrelated).ok());
    ASSERT_TRUE(retire({{4, 0}}).ok());
  }
  SequenceStateLease reordered(/*capacity=*/4);
  ASSERT_TRUE(acquire({{3, 0}, {1, 0}}, reordered).ok());
  EXPECT_EQ(reordered.handles().back().row, first.row);
  EXPECT_EQ(reordered.handles().back().generation, first.generation);
  const torch::Tensor gather_index = row_indices(reordered, device_);
  torch::Tensor output = torch::empty({2}, values.options());
  torch::index_select_out(output, pool_->tokens(), /*dim=*/0, gather_index);
  ASSERT_EQ(stream.synchronize(), ACL_SUCCESS);
  EXPECT_TRUE(
      torch::equal(output.cpu(), torch::tensor({303, 101}, torch::kInt64)));
}

TEST_F(SequenceStatePoolTest, BatchedPublishPrecedesNextTaskGather) {
  SequenceStateLease task_a(/*capacity=*/4);
  SequenceStateLease task_b(/*capacity=*/4);
  ASSERT_TRUE(admit({{{11, 0}, std::nullopt, 1}, {{22, 0}, std::nullopt, 1}},
                    {},
                    task_a)
                  .ok());
  // Accept the read before A actually publishes on the Device stream.
  ASSERT_TRUE(admit({{{22, 0}, 1, 2}, {{11, 0}, 1, 2}}, {}, task_b).ok());
  Stream stream(device_);
  auto guard = stream.set_stream_guard();
  const torch::Tensor rows_a = row_indices(task_a, device_);
  const torch::Tensor rows_b = row_indices(task_b, device_);
  const torch::Tensor values =
      torch::tensor({71, 93}, torch::kInt64).to(device_);
  torch::Tensor gathered = torch::empty({2}, values.options());
  torch::Tensor result = torch::empty({2}, values.options());
  const int64_t* storage = pool_->tokens().const_data_ptr<int64_t>();
  // Both Tasks hold independent leases before either produces a CPU result.
  // All Device state operations are submitted on one stream.
  pool_->tokens().index_copy_(/*dim=*/0, rows_a, values);
  torch::index_select_out(gathered, pool_->tokens(), /*dim=*/0, rows_b);
  gathered.add_(/*other=*/5);
  pool_->tokens().index_copy_(/*dim=*/0, rows_b, gathered);
  torch::index_select_out(result, pool_->tokens(), /*dim=*/0, rows_a);
  ASSERT_TRUE(retire({{11, 0}, {22, 0}}).ok());
  EXPECT_EQ(pool_->available_rows(), 2U);
  SequenceStateLease rejected(/*capacity=*/4);
  EXPECT_FALSE(acquire({{11, 0}}, rejected).ok());
  ASSERT_EQ(stream.synchronize(), ACL_SUCCESS);
  EXPECT_TRUE(
      torch::equal(result.cpu(), torch::tensor({76, 98}, torch::kInt64)));
  task_a.reset();
  EXPECT_EQ(pool_->available_rows(), 2U);
  task_b.reset();
  EXPECT_EQ(pool_->available_rows(), 4U);
  EXPECT_EQ(pool_->tokens().const_data_ptr<int64_t>(), storage);
}

TEST_F(SequenceStatePoolTest, RepeatedEpochsReuseFixedStorage) {
  const int64_t* storage = pool_->tokens().const_data_ptr<int64_t>();
  SequenceStateHandle previous;
  for (uint64_t epoch = 0; epoch < 256; ++epoch) {
    SequenceStateLease lease(/*capacity=*/4);
    ASSERT_TRUE(acquire({{1, epoch}}, lease).ok());
    const auto current = lease.handles().front();
    EXPECT_FALSE(pool_->is_current(previous));
    EXPECT_TRUE(pool_->is_current(current));
    ASSERT_TRUE(retire({{1, epoch}}).ok());
    previous = current;
  }
  EXPECT_EQ(pool_->available_rows(), 4U);
  EXPECT_EQ(pool_->tokens().const_data_ptr<int64_t>(), storage);
  EXPECT_EQ(pool_->device_bytes(), 4 * sizeof(int64_t));
  SequenceStateLease empty(/*capacity=*/4);
  EXPECT_TRUE(acquire({}, empty).ok());
  EXPECT_TRUE(retire({}).ok());
  EXPECT_TRUE(empty.empty());
}

TEST_F(SequenceStatePoolTest, FullPoolReplacementCommitsAfterAllValidation) {
  SequenceStateLease initial(/*capacity=*/4);
  ASSERT_TRUE(acquire({{1, 0}, {2, 0}, {3, 0}, {4, 0}}, initial).ok());
  const auto old = initial.handles().front();
  initial.reset();
  SequenceStateLease replacement(/*capacity=*/4);
  EXPECT_FALSE(admit({{{5, 0}, 7, std::nullopt}}, {{1, 0}}, replacement).ok());
  EXPECT_TRUE(replacement.empty());
  EXPECT_TRUE(pool_->is_current(old));
  EXPECT_EQ(pool_->available_rows(), 0U);
  ASSERT_TRUE(admit({{{5, 0}, std::nullopt, 7}}, {{1, 0}}, replacement).ok());
  ASSERT_EQ(replacement.handles().size(), 1U);
  EXPECT_EQ(replacement.handles().front().row, old.row);
  EXPECT_GT(replacement.handles().front().generation, old.generation);
  EXPECT_FALSE(pool_->is_current(old));
  EXPECT_EQ(pool_->available_rows(), 0U);
}

TEST_F(SequenceStatePoolTest, PendingLeasesCannotFundRejectedReplacement) {
  SequenceStateLease held(/*capacity=*/4);
  ASSERT_TRUE(acquire({{1, 0}, {2, 0}, {3, 0}, {4, 0}}, held).ok());
  SequenceStateLease replacement(/*capacity=*/4);
  const Status rejected =
      admit({{{5, 0}, std::nullopt, 1}}, {{1, 0}}, replacement);
  EXPECT_FALSE(rejected.ok());
  EXPECT_TRUE(replacement.empty());
  // Failed capacity admission did not even mark the old identity retired.
  SequenceStateLease still_live(/*capacity=*/1);
  ASSERT_TRUE(acquire({{1, 0}}, still_live).ok());
  still_live.reset();
  held.reset();
  ASSERT_TRUE(admit({{{5, 0}, std::nullopt, 1}}, {{1, 0}}, replacement).ok());
}

TEST_F(SequenceStatePoolTest, ReadsRequireExactAcceptedPublication) {
  SequenceStateLease first(/*capacity=*/4);
  ASSERT_TRUE(
      admit({{{1, 0}, std::nullopt, 7}, {{2, 0}, std::nullopt, std::nullopt}},
            {},
            first)
          .ok());
  SequenceStateLease next(/*capacity=*/4);
  EXPECT_FALSE(admit({{{2, 0}, 0, std::nullopt}}, {}, next).ok());
  EXPECT_FALSE(admit({{{1, 0}, 6, 8}}, {}, next).ok());
  EXPECT_FALSE(admit({{{1, 0}, 8, 9}}, {}, next).ok());
  EXPECT_FALSE(admit({{{1, 0}, 7, 7}}, {}, next).ok());
  ASSERT_TRUE(admit({{{1, 0}, 7, 8}}, {}, next).ok());
  SequenceStateLease observer(/*capacity=*/1);
  EXPECT_FALSE(admit({{{1, 0}, 7, std::nullopt}}, {}, observer).ok());
  ASSERT_TRUE(admit({{{1, 0}, 8, std::nullopt}}, {}, observer).ok());
  EXPECT_EQ(first.handles().front().row, next.handles().front().row);
}

TEST_F(SequenceStatePoolTest, LaterRejectionPreservesEarlierPublication) {
  SequenceStateLease initial(/*capacity=*/4);
  ASSERT_TRUE(admit({{{1, 0}, std::nullopt, 3},
                     {{2, 0}, std::nullopt, 3},
                     {{3, 0}, std::nullopt, 3}},
                    {},
                    initial)
                  .ok());
  const auto retired = initial.handles().back();
  initial.reset();
  SequenceStateLease next(/*capacity=*/4);
  EXPECT_FALSE(admit({{{1, 0}, 3, 4}, {{2, 0}, 2, 4}}, {{3, 0}}, next).ok());
  EXPECT_TRUE(pool_->is_current(retired));
  EXPECT_EQ(pool_->available_rows(), 1U);
  EXPECT_TRUE(next.empty());
  ASSERT_TRUE(admit({{{1, 0}, 3, 4}, {{2, 0}, 3, 4}}, {{3, 0}}, next).ok());
  EXPECT_FALSE(pool_->is_current(retired));
  EXPECT_EQ(pool_->available_rows(), 2U);
}

TEST_F(SequenceStatePoolTest, FixedLeaseCapacitySurvivesResetAndRejection) {
  SequenceStateLease lease(/*capacity=*/1);
  const SequenceStateHandle* storage = lease.handles().data();
  for (uint64_t epoch = 0; epoch < 256; ++epoch) {
    EXPECT_FALSE(acquire({{1, epoch}, {2, epoch}}, lease).ok());
    EXPECT_TRUE(lease.empty());
    EXPECT_EQ(lease.handles().data(), storage);
    EXPECT_EQ(pool_->available_rows(), 4U);
    ASSERT_TRUE(admit({{{1, epoch}, std::nullopt, 7}}, {}, lease).ok());
    EXPECT_EQ(lease.handles().data(), storage);
    ASSERT_TRUE(retire({{1, epoch}}).ok());
    lease.reset();
    EXPECT_EQ(lease.handles().data(), storage);
    SequenceStateLease uninitialized(/*capacity=*/1);
    EXPECT_FALSE(
        admit({{{1, epoch + 1}, 7, std::nullopt}}, {}, uninitialized).ok());
  }
  SequenceStateLease no_capacity;
  EXPECT_FALSE(acquire({{9, 0}}, no_capacity).ok());
  EXPECT_TRUE(acquire({}, no_capacity).ok());
  EXPECT_EQ(pool_->available_rows(), 4U);
}

TEST_F(SequenceStatePoolTest, TransactionRejectsConflictingKeysAndPositions) {
  SequenceStateLease held(/*capacity=*/4);
  ASSERT_TRUE(admit({{{1, 0}, std::nullopt, 1}}, {}, held).ok());
  const auto handle = held.handles().front();
  held.reset();
  SequenceStateLease next(/*capacity=*/4);
  EXPECT_FALSE(admit({{{1, 0}, 1, 2}}, {{1, 0}}, next).ok());
  EXPECT_FALSE(admit({{{1, 0}, 1, 2}}, {{9, 0}}, next).ok());
  EXPECT_FALSE(admit({{{2, 0}, std::nullopt, 1}, {{2, 0}, std::nullopt, 1}},
                     {{1, 0}},
                     next)
                   .ok());
  EXPECT_FALSE(admit({{{0, 0}, std::nullopt, 1}}, {{1, 0}}, next).ok());
  EXPECT_FALSE(
      admit({{{2, 0}, std::nullopt, std::numeric_limits<uint32_t>::max()}},
            {{1, 0}},
            next)
          .ok());
  EXPECT_FALSE(
      admit({{{1, 0}, std::numeric_limits<uint32_t>::max(), std::nullopt}},
            {},
            next)
          .ok());
  EXPECT_FALSE(admit({}, {{1, 0}, {1, 0}}, next).ok());
  EXPECT_TRUE(pool_->is_current(handle));
  EXPECT_TRUE(next.empty());
  EXPECT_EQ(pool_->available_rows(), 3U);
  ASSERT_TRUE(admit({}, {{1, 0}}, next).ok());
  EXPECT_FALSE(pool_->is_current(handle));
  ASSERT_TRUE(
      admit({{{2, 0}, std::nullopt, std::numeric_limits<int32_t>::max()}},
            {},
            next)
          .ok());
  next.reset();
  ASSERT_TRUE(
      admit({{{2, 0}, std::numeric_limits<int32_t>::max(), std::nullopt}},
            {},
            next)
          .ok());
}

TEST_F(SequenceStatePoolTest, TransactionAdmissionPerformance) {
  if (std::getenv(/*name=*/"XLLM_PIPELINE_PERF") == nullptr) {
    GTEST_SKIP() << "Run separately with XLLM_PIPELINE_PERF=1.";
  }
  ASSERT_TRUE(SequenceStatePool::create(/*capacity=*/128, device_, pool_).ok());
  constexpr int32_t kIterations = 4096;
  for (uint32_t batch : std::array<uint32_t, 4>{1, 4, 32, 128}) {
    std::vector<SequenceStateAccess> accesses;
    std::vector<SequenceStateKey> retired;
    accesses.reserve(batch);
    retired.reserve(batch);
    for (uint32_t row = 0; row < batch; ++row) {
      accesses.emplace_back(
          SequenceStateAccess{{row + 1U, 0}, std::nullopt, 1});
      retired.emplace_back(SequenceStateKey{row + 1U, 0});
    }
    SequenceStateLease lease(/*capacity=*/128);
    ASSERT_TRUE(pool_->admit(accesses, {}, lease).ok());
    lease.reset();
    const SequenceStateHandle* storage = lease.handles().data();
    const auto cycle = [&]() {
      for (uint32_t row = 0; row < batch; ++row) {
        retired[row] = accesses[row].key;
        ++accesses[row].key.epoch;
      }
      const Status status = pool_->admit(accesses, retired, lease);
      CHECK(status.ok()) << status.message();
      lease.reset();
    };
    for (int32_t round = 0; round < 3; ++round) {
      for (int32_t warmup = 0; warmup < 32; ++warmup) {
        cycle();
      }
      const auto begin = std::chrono::steady_clock::now();
      for (int32_t step = 0; step < kIterations; ++step) {
        cycle();
      }
      const double us = std::chrono::duration<double, std::micro>(
                            std::chrono::steady_clock::now() - begin)
                            .count() /
                        kIterations;
      EXPECT_EQ(lease.handles().data(), storage);
      EXPECT_EQ(pool_->available_rows(), 128U - batch);
      LOG(INFO) << "POOL_ADMISSION_PERF rows=" << batch << " round=" << round
                << " us=" << us << " iterations=" << kIterations;
    }
    for (uint32_t row = 0; row < batch; ++row) {
      retired[row] = accesses[row].key;
    }
    ASSERT_TRUE(pool_->admit({}, retired, lease).ok());
    EXPECT_EQ(pool_->available_rows(), 128U);
  }
}

TEST_F(SequenceStatePoolTest, BatchedTokenTransferPerformance) {
  if (std::getenv(/*name=*/"XLLM_PIPELINE_PERF") == nullptr) {
    GTEST_SKIP() << "Run separately with XLLM_PIPELINE_PERF=1.";
  }
  ASSERT_TRUE(SequenceStatePool::create(/*capacity=*/128, device_, pool_).ok());
  Stream stream(device_);
  auto guard = stream.set_stream_guard();
  constexpr int32_t kIterations = 256;
  for (uint32_t batch : std::array<uint32_t, 4>{1, 4, 32, 128}) {
    std::vector<SequenceStateKey> keys;
    keys.reserve(batch);
    for (uint32_t row = 0; row < batch; ++row) {
      keys.emplace_back(SequenceStateKey{row + 1U, batch});
    }
    std::vector<SequenceStateAccess> accesses;
    accesses.reserve(batch);
    for (const SequenceStateKey& key : keys) {
      accesses.emplace_back(
          SequenceStateAccess{key, std::nullopt, std::nullopt});
    }
    SequenceStateLease lease(/*capacity=*/128);
    ASSERT_TRUE(pool_->admit(accesses, {}, lease).ok());
    const torch::Tensor index = row_indices(lease, device_);
    torch::Tensor values =
        torch::full({batch}, /*fill_value=*/13, pool_->tokens().options());
    torch::Tensor gathered = torch::empty_like(values);
    const auto cycle = [&]() {
      pool_->tokens().index_copy_(/*dim=*/0, index, values);
      torch::index_select_out(gathered, pool_->tokens(), /*dim=*/0, index);
      values.add_(/*other=*/1);
    };
    for (int32_t round = 0; round < 3; ++round) {
      for (int32_t warmup = 0; warmup < 8; ++warmup) {
        cycle();
      }
      values.fill_(/*value=*/13);
      ASSERT_EQ(stream.synchronize(), ACL_SUCCESS);
      const auto begin = std::chrono::steady_clock::now();
      for (int32_t step = 0; step < kIterations; ++step) {
        cycle();
      }
      const auto submitted = std::chrono::steady_clock::now();
      ASSERT_EQ(stream.synchronize(), ACL_SUCCESS);
      const auto completed = std::chrono::steady_clock::now();
      const double host_us =
          std::chrono::duration<double, std::micro>(submitted - begin).count() /
          kIterations;
      const double completion_us =
          std::chrono::duration<double, std::micro>(completed - begin).count() /
          kIterations;
      EXPECT_TRUE(torch::equal(
          gathered.cpu(),
          torch::full(
              {batch}, /*fill_value=*/13 + kIterations - 1, torch::kInt64)));
      LOG(INFO) << "POOL_PERF batch=" << batch << " round=" << round
                << " iterations=" << kIterations << " host_us=" << host_us
                << " completion_us=" << completion_us
                << " device_bytes=" << pool_->device_bytes();
    }
    SequenceStateLease retire_only;
    ASSERT_TRUE(pool_->admit({}, keys, retire_only).ok());
    lease.reset();
    EXPECT_EQ(pool_->available_rows(), 128U);
  }
}

}  // namespace
}  // namespace xllm
