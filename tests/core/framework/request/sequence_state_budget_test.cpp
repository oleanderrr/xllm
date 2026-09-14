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

#include "core/framework/request/sequence_state_budget.h"

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

#include "core/framework/config/execution_config.h"
#include "core/util/json_reader.h"

namespace xllm {
namespace {

TEST(SequenceStateBudgetTest, RejectsWithoutChangingReservationsOrQuota) {
  auto budget = std::make_shared<SequenceStateBudget>(/*capacity=*/3);
  SequenceStateReservation held;
  ASSERT_TRUE(held.acquire(budget, /*count=*/2));
  EXPECT_EQ(budget->available(), 1U);
  SequenceStateReservation rejected;
  EXPECT_FALSE(rejected.acquire(budget, /*count=*/0));
  EXPECT_FALSE(rejected.acquire(budget, /*count=*/2));
  EXPECT_FALSE(rejected.acquire(budget, std::numeric_limits<uint64_t>::max()));
  EXPECT_FALSE(rejected.acquire(/*budget=*/nullptr, /*count=*/1));
  EXPECT_TRUE(rejected.empty());
  EXPECT_EQ(budget->available(), 1U);
  auto other = std::make_shared<SequenceStateBudget>(/*capacity=*/4);
  EXPECT_FALSE(held.acquire(other, /*count=*/1));
  EXPECT_EQ(other->available(), 4U);
  EXPECT_EQ(held.count(), 2U);
  held.reset();
  ASSERT_TRUE(rejected.acquire(budget, /*count=*/3));
  EXPECT_EQ(budget->available(), 0U);
  EXPECT_FALSE(held.acquire(budget, /*count=*/1));
  rejected.reset();
  EXPECT_EQ(budget->available(), budget->capacity());
}

TEST(SequenceStateBudgetTest, MovesAndRepeatedResetReturnQuotaExactlyOnce) {
  auto first = std::make_shared<SequenceStateBudget>(/*capacity=*/5);
  auto second = std::make_shared<SequenceStateBudget>(/*capacity=*/7);
  SequenceStateReservation left;
  SequenceStateReservation right;
  ASSERT_TRUE(left.acquire(first, /*count=*/3));
  ASSERT_TRUE(right.acquire(second, /*count=*/2));
  SequenceStateReservation moved(std::move(left));
  EXPECT_TRUE(left.empty());
  right = std::move(moved);
  EXPECT_TRUE(moved.empty());
  EXPECT_EQ(second->available(), 7U);
  EXPECT_EQ(first->available(), 2U);
  auto& alias = right;
  right = std::move(alias);
  EXPECT_EQ(right.count(), 3U);
  right.reset();
  right.reset();
  left.reset();
  moved.reset();
  EXPECT_EQ(first->available(), 5U);
  {
    SequenceStateReservation scoped;
    ASSERT_TRUE(scoped.acquire(first, /*count=*/5));
    EXPECT_EQ(first->available(), 0U);
  }
  EXPECT_EQ(first->available(), 5U);
}

TEST(SequenceStateBudgetTest, ReservationOutlivesOwnerAtMaximumCount) {
  constexpr uint32_t kCapacity = std::numeric_limits<uint32_t>::max();
  auto budget = std::make_shared<SequenceStateBudget>(kCapacity);
  std::weak_ptr<SequenceStateBudget> observer = budget;
  SequenceStateReservation held;
  ASSERT_TRUE(held.acquire(budget, kCapacity));
  EXPECT_EQ(budget->available(), 0U);
  EXPECT_EQ(held.count(), kCapacity);
  budget.reset();
  EXPECT_FALSE(observer.expired());
  held.reset();
  EXPECT_TRUE(observer.expired());
}

TEST(SequenceStateBudgetTest, ZeroCapacityViolatesConstructorContract) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_DEATH(SequenceStateBudget(/*capacity=*/0), "capacity > 0U");
}

TEST(SequenceStateBudgetTest, ConcurrentReservationsNeverExceedCapacity) {
  constexpr uint32_t kThreads = 4;
  constexpr uint32_t kCapacity = 3;
  constexpr uint32_t kRounds = 128;
  auto budget = std::make_shared<SequenceStateBudget>(kCapacity);
  std::barrier start(kThreads + 1);
  std::barrier acquired(kThreads + 1);
  std::barrier release(kThreads + 1);
  std::barrier finished(kThreads + 1);
  std::atomic<uint32_t> accepted{0};
  std::vector<std::thread> workers;
  workers.reserve(kThreads);
  for (uint32_t thread = 0; thread < kThreads; ++thread) {
    workers.emplace_back([&]() {
      for (uint32_t round = 0; round < kRounds; ++round) {
        start.arrive_and_wait();
        SequenceStateReservation held;
        if (held.acquire(budget, /*count=*/1)) {
          accepted.fetch_add(/*arg=*/1, std::memory_order_relaxed);
        }
        acquired.arrive_and_wait();
        release.arrive_and_wait();
        held.reset();
        finished.arrive_and_wait();
      }
    });
  }
  for (uint32_t round = 0; round < kRounds; ++round) {
    accepted.store(/*desired=*/0, std::memory_order_relaxed);
    start.arrive_and_wait();
    acquired.arrive_and_wait();
    EXPECT_EQ(accepted.load(std::memory_order_relaxed), kCapacity);
    EXPECT_EQ(budget->available(), 0U);
    release.arrive_and_wait();
    finished.arrive_and_wait();
    EXPECT_EQ(budget->available(), kCapacity);
  }
  for (std::thread& worker : workers) {
    worker.join();
  }
}

TEST(SequenceStateBudgetTest, ConfigurationRoundTripIncludesHelpCategory) {
  ExecutionConfig defaults;
  EXPECT_EQ(defaults.task_pipeline_max_live_sequences(), 1024);
  nlohmann::ordered_json initial;
  defaults.append_config_json(initial);
  const std::string option = "task_pipeline_max_live_sequences";
  EXPECT_FALSE(initial.contains(option));
  gflags::FlagSaver restore_test_flags;
  nlohmann::ordered_json serialized;
  {
    // Config loading writes resolved values back to gflags. A second startup
    // must begin with fresh defaults so CLI precedence cannot mask its JSON.
    gflags::FlagSaver restore_startup_flags;
    JsonReader reader;
    ASSERT_TRUE(reader.parse_text(
        R"({"task_pipeline_slots":1,"task_pipeline_max_live_sequences":7})"));
    ExecutionConfig configured;
    configured.from_json(reader);
    EXPECT_EQ(configured.task_pipeline_slots(), 1);
    EXPECT_EQ(configured.task_pipeline_max_live_sequences(), 7);
    configured.append_config_json(serialized);
    ASSERT_EQ(serialized[option], 7);
  }
  JsonReader restored_reader;
  ASSERT_TRUE(restored_reader.parse_text(serialized.dump()));
  ExecutionConfig restored;
  restored.from_json(restored_reader);
  EXPECT_EQ(restored.task_pipeline_max_live_sequences(), 7);
  const auto& options = ExecutionConfig::option_category().option_names;
  EXPECT_NE(std::find(options.begin(), options.end(), option), options.end());
}

TEST(SequenceStateBudgetTest, ReservationPerformance) {
  if (std::getenv(/*name=*/"XLLM_PIPELINE_PERF") == nullptr) {
    GTEST_SKIP() << "Run separately with XLLM_PIPELINE_PERF=1.";
  }
  constexpr uint32_t kIterations = 32768;
  auto budget = std::make_shared<SequenceStateBudget>(/*capacity=*/128);
  SequenceStateReservation reservation;
  for (uint32_t rows : std::array<uint32_t, 4>{1, 4, 32, 128}) {
    for (uint32_t round = 0; round < 3; ++round) {
      const auto cycle = [&]() {
        CHECK(reservation.acquire(budget, rows));
        reservation.reset();
      };
      for (uint32_t warmup = 0; warmup < 128; ++warmup) {
        cycle();
      }
      const auto begin = std::chrono::steady_clock::now();
      for (uint32_t step = 0; step < kIterations; ++step) {
        cycle();
      }
      const double us = std::chrono::duration<double, std::micro>(
                            std::chrono::steady_clock::now() - begin)
                            .count() /
                        kIterations;
      EXPECT_TRUE(reservation.empty());
      EXPECT_EQ(budget->available(), budget->capacity());
      LOG(INFO) << "STATE_BUDGET_PERF rows=" << rows << " round=" << round
                << " us=" << us << " iterations=" << kIterations;
    }
  }
}

}  // namespace
}  // namespace xllm
