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

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <span>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/framework/batch/batch.h"
#include "core/framework/batch/batch_input_builder.h"
#include "core/framework/block/block_manager_impl.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/model/model_args.h"
#include "core/framework/request/sequence.h"
#include "core/framework/request/sequence_state_retirement_queue.h"
#include "core/runtime/params_utils.h"

namespace xllm {
namespace {

void expect_keys(std::span<const SequenceStateKey> actual,
                 std::span<const SequenceStateKey> expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (uint64_t row = 0; row < actual.size(); ++row) {
    EXPECT_EQ(actual[row].sequence_id, expected[row].sequence_id);
    EXPECT_EQ(actual[row].epoch, expected[row].epoch);
  }
}

ForwardInput unpack(const proto::PackedForwardInput& packed) {
  ForwardInput lazy;
  packed_proto_to_forward_input(
      packed, lazy, torch::Device(torch::kCPU), /*stream=*/nullptr);
  ForwardInput result;
  CHECK(detail::unpack_from_input_host_buffer(
      lazy,
      torch::Device(torch::kCPU),
      torch::kFloat32,
      result,
      /*materialize_device_buffer=*/false));
  return result;
}

uint64_t descriptor_bytes(const proto::PackedForwardInput& packed) {
  uint64_t bytes = 0;
  CHECK_GE(packed.payload().size(), sizeof(bytes));
  std::memcpy(&bytes, packed.payload().data(), sizeof(bytes));
  return bytes;
}

class SequenceStateInputTest : public ::testing::Test {
 protected:
  void SetUp() override {
    saved_config_ = ExecutionConfig::get_instance();
    ExecutionConfig::get_instance().task_pipeline_slots(/*value=*/1);
    ExecutionConfig::get_instance().use_contiguous_input_buffer(
        /*value=*/false);
    BlockManager::Options options;
    options.num_blocks(/*value=*/8192).block_size(/*value=*/16);
    manager_ = std::make_unique<BlockManagerImpl>(options);
    stopping_.set_max_generated_tokens(/*tokens=*/8);
  }

  void TearDown() override { ExecutionConfig::get_instance() = saved_config_; }

  std::unique_ptr<Sequence> make_sequence(uint32_t index,
                                          uint32_t prompt_size = 4,
                                          bool allocate_blocks = true) {
    SequenceParams params;
    params.seq_capacity = prompt_size + 8;
    params.request_id = "reused-external-request-id";
    params.sampling_param = &sampling_;
    params.stopping_checker = &stopping_;
    std::vector<int32_t> tokens(prompt_size, /*value=*/7);
    IncrementalDecoder decoder(/*prompt=*/"",
                               prompt_size,
                               /*echo=*/false,
                               /*skip_special_tokens=*/true);
    auto sequence = std::make_unique<Sequence>(
        index, tokens, torch::Tensor(), MMData(), decoder, params);
    if (allocate_blocks) {
      sequence->add_blocks(BlockType::KV,
                           manager_->allocate((prompt_size + 15) / 16));
    }
    return sequence;
  }

  ForwardInput build(std::vector<Sequence*> sequences,
                     ThreadPool* threads = nullptr,
                     bool include_sequence_state_keys = true) {
    std::vector<uint32_t> budgets;
    budgets.reserve(sequences.size());
    for (Sequence* sequence : sequences) {
      // Rebuild this CPU-only prefill fixture without changing its identity.
      sequence->kv_state().set_kv_cache_tokens_num(/*num=*/0);
      budgets.emplace_back(static_cast<uint32_t>(sequence->num_tokens()));
    }
    const std::vector<torch::Tensor> embeddings;
    BatchInputBuilder builder(sequences,
                              budgets,
                              embeddings,
                              /*mm_data_vec=*/{},
                              /*swap_block_transfer_infos=*/nullptr,
                              /*batch_id=*/1,
                              /*args=*/nullptr,
                              BatchForwardType::PREFILL,
                              /*cp_size=*/1,
                              threads,
                              include_sequence_state_keys);
    return builder.build_forward_input(/*num_decoding_tokens=*/1,
                                       /*min_decoding_batch_size=*/0);
  }

  ExecutionConfig saved_config_;
  RequestSamplingParam sampling_;
  StoppingChecker stopping_;
  std::unique_ptr<BlockManagerImpl> manager_;
};

TEST_F(SequenceStateInputTest, CopyForkAndRequestReuseHaveDistinctIdentities) {
  auto first = make_sequence(/*index=*/0);
  auto reused = make_sequence(/*index=*/0);
  const auto initial = first->sequence_state_key();
  EXPECT_NE(initial.sequence_id, 0U);
  EXPECT_EQ(initial.epoch, 0U);
  EXPECT_NE(initial.sequence_id, reused->sequence_state_key().sequence_id);
  first->reset();
  const auto reset = first->sequence_state_key();
  EXPECT_EQ(reset.sequence_id, initial.sequence_id);
  EXPECT_EQ(reset.epoch, initial.epoch + 1);
  EXPECT_EQ(first->num_tokens(), 4U);
  EXPECT_EQ(first->kv_cache_tokens_num(), 0U);
  Sequence copied(*first);
  auto forked = first->fork(/*index=*/1);
  std::unordered_set<uint64_t> identities = {
      initial.sequence_id,
      reused->sequence_state_key().sequence_id,
      copied.sequence_state_key().sequence_id,
      forked->sequence_state_key().sequence_id};
  EXPECT_EQ(identities.size(), 4U);
  EXPECT_EQ(copied.sequence_state_key().epoch, 0U);
  EXPECT_EQ(forked->sequence_state_key().epoch, 0U);
  first->reset();
  EXPECT_EQ(first->sequence_state_key().epoch, 2U);
  EXPECT_EQ(copied.sequence_state_key().epoch, 0U);
}

TEST_F(SequenceStateInputTest, ConcurrentCreationHasUniqueNonzeroIdentities) {
  std::array<std::vector<uint64_t>, 4> identities;
  std::vector<std::thread> threads;
  threads.reserve(identities.size());
  for (auto& ids : identities) {
    threads.emplace_back([this, &ids]() {
      ids.reserve(/*new_cap=*/128);
      for (uint32_t row = 0; row < 128; ++row) {
        auto sequence = make_sequence(row,
                                      /*prompt_size=*/4,
                                      /*allocate_blocks=*/false);
        ids.emplace_back(sequence->sequence_state_key().sequence_id);
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  std::unordered_set<uint64_t> unique;
  for (const auto& ids : identities) {
    unique.insert(ids.begin(), ids.end());
  }
  EXPECT_EQ(unique.size(), 512U);
  EXPECT_FALSE(unique.contains(/*key=*/0));
}

TEST_F(SequenceStateInputTest, BuilderKeepsModelRowIdentityAcrossOrdering) {
  auto first = make_sequence(/*index=*/0);
  auto second = make_sequence(/*index=*/1);
  const std::array<SequenceStateKey, 2> expected = {
      second->sequence_state_key(), first->sequence_state_key()};
  ForwardInput input = build({second.get(), first.get()});
  expect_keys(input.sequence_state_keys, expected);
  ForwardInput copied = input.to(torch::Device(torch::kCPU), torch::kFloat32);
  input.sequence_state_keys.clear();
  expect_keys(copied.sequence_state_keys, expected);
  ExecutionConfig::get_instance().task_pipeline_slots(/*value=*/0);
  expect_keys(build({second.get(), first.get()}).sequence_state_keys, expected);
  ExecutionConfig::get_instance().task_pipeline_slots(/*value=*/1);
  EXPECT_TRUE(build({first.get(), second.get()},
                    /*threads=*/nullptr,
                    /*include_sequence_state_keys=*/false)
                  .sequence_state_keys.empty());
  EXPECT_TRUE(build({}).sequence_state_keys.empty());
}

TEST_F(SequenceStateInputTest, DisabledDefaultAndPaddingKeepLogicalKeys) {
  auto first = make_sequence(/*index=*/0);
  auto second = make_sequence(/*index=*/1);
  std::vector<Sequence*> sequences{first.get(), second.get()};
  const std::vector<uint32_t> budgets{4, 4};
  const std::vector<torch::Tensor> embeddings;
  BatchInputBuilder ordinary(sequences,
                             budgets,
                             embeddings,
                             {},
                             /*swap_block_transfer_infos=*/nullptr,
                             /*batch_id=*/1,
                             /*args=*/nullptr,
                             BatchForwardType::PREFILL);
  auto ordinary_input = ordinary.build_forward_input(
      /*num_decoding_tokens=*/1, /*min_decoding_batch_size=*/0);
  EXPECT_TRUE(ordinary_input.sequence_state_keys.empty());
  first->kv_state().set_kv_cache_tokens_num(/*num=*/3);
  second->kv_state().set_kv_cache_tokens_num(/*num=*/3);
  BatchInputBuilder padded(sequences,
                           budgets,
                           embeddings,
                           {},
                           /*swap_block_transfer_infos=*/nullptr,
                           /*batch_id=*/2,
                           /*args=*/nullptr,
                           BatchForwardType::DECODE,
                           /*cp_size=*/1,
                           /*thread_pool=*/nullptr,
                           /*include_sequence_state_keys=*/true);
  auto input = padded.build_forward_input(/*num_decoding_tokens=*/1,
                                          /*min_decoding_batch_size=*/4);
  EXPECT_EQ(input.input_params.attention.host.q_seq_lens.size(), 4U);
  const std::array<SequenceStateKey, 2> expected{first->sequence_state_key(),
                                                 second->sequence_state_key()};
  expect_keys(input.sequence_state_keys, expected);
}

TEST_F(SequenceStateInputTest, BatchEntryPointsUseExplicitIdentityOption) {
  ModelArgs args;
  auto first = make_sequence(/*index=*/0);
  auto second = make_sequence(/*index=*/1);
  Batch batch(std::vector<Sequence*>{second.get(), first.get()});
  for (bool include_keys : std::array<bool, 2>{false, true}) {
    first->kv_state().set_kv_cache_tokens_num(/*num=*/0);
    second->kv_state().set_kv_cache_tokens_num(/*num=*/0);
    ForwardInput local = batch.prepare_forward_input(
        /*num_decoding_tokens=*/1,
        /*min_decoding_batch_size=*/0,
        args,
        /*cp_size=*/1,
        include_keys);
    std::vector<SequenceStateKey> expected;
    if (include_keys) {
      expected = {batch[0]->sequence_state_key(),
                  batch[1]->sequence_state_key()};
    }
    expect_keys(local.sequence_state_keys, expected);
    first->kv_state().set_kv_cache_tokens_num(/*num=*/0);
    second->kv_state().set_kv_cache_tokens_num(/*num=*/0);
    ForwardInput distributed = batch.prepare_forward_input(
        args, /*thread_pool=*/nullptr, /*cp_size=*/1, include_keys);
    if (include_keys) {
      expected = {batch[0]->sequence_state_key(),
                  batch[1]->sequence_state_key()};
    }
    expect_keys(distributed.sequence_state_keys, expected);
    EXPECT_TRUE(
        torch::equal(local.host_token_ids(), distributed.host_token_ids()));
  }
}

TEST_F(SequenceStateInputTest, ParallelBuilderMatchesSingleThreadRows) {
  auto first = make_sequence(/*index=*/0, /*prompt_size=*/32768);
  auto second = make_sequence(/*index=*/1, /*prompt_size=*/32768);
  ForwardInput single = build({second.get(), first.get()});
  ThreadPool threads(/*num_threads=*/2);
  ForwardInput parallel = build({second.get(), first.get()}, &threads);
  expect_keys(parallel.sequence_state_keys, single.sequence_state_keys);
  EXPECT_TRUE(torch::equal(parallel.host_token_ids(), single.host_token_ids()));
  EXPECT_TRUE(torch::equal(parallel.host_positions(), single.host_positions()));
  EXPECT_EQ(parallel.input_params.attention.host.q_seq_lens,
            single.input_params.attention.host.q_seq_lens);
  EXPECT_TRUE(torch::equal(parallel.sampling_params.selected_token_idxes,
                           single.sampling_params.selected_token_idxes));
}

TEST_F(SequenceStateInputTest, PackedSuffixCarriesRowsAndExplicitRetirements) {
  auto sequence = make_sequence(/*index=*/0);
  ForwardInput input = build({sequence.get()});
  input.retired_sequence_state_keys = {{99, 4}, {99, 5}};
  proto::PackedForwardInput packed;
  ASSERT_TRUE(forward_input_to_packed_proto(input, &packed));
  ForwardInput actual = unpack(packed);
  expect_keys(actual.sequence_state_keys, input.sequence_state_keys);
  expect_keys(actual.retired_sequence_state_keys,
              input.retired_sequence_state_keys);
  EXPECT_TRUE(torch::equal(actual.host_token_ids(), input.host_token_ids()));
  EXPECT_TRUE(torch::equal(actual.host_positions(), input.host_positions()));
  ForwardInput copied = input.to(torch::Device(torch::kCPU), torch::kFloat32);
  expect_keys(copied.retired_sequence_state_keys,
              input.retired_sequence_state_keys);
  ForwardInput control;
  control.retired_sequence_state_keys = {{77, 9}};
  ASSERT_TRUE(forward_input_to_packed_proto(control, &packed));
  actual = unpack(packed);
  EXPECT_TRUE(actual.sequence_state_keys.empty());
  expect_keys(actual.retired_sequence_state_keys,
              control.retired_sequence_state_keys);
}

TEST_F(SequenceStateInputTest, EmptySuffixPreservesLegacyDescriptor) {
  auto sequence = make_sequence(/*index=*/0);
  ForwardInput input = build({sequence.get()});
  proto::PackedForwardInput with_keys;
  ASSERT_TRUE(forward_input_to_packed_proto(input, &with_keys));
  input.sequence_state_keys.clear();
  proto::PackedForwardInput legacy;
  ASSERT_TRUE(forward_input_to_packed_proto(input, &legacy));
  const uint64_t old_bytes = descriptor_bytes(legacy);
  EXPECT_EQ(descriptor_bytes(with_keys) - old_bytes, 40U);
  constexpr uint64_t kHeaderBytes = 3 * sizeof(uint64_t);
  EXPECT_EQ(with_keys.payload().substr(kHeaderBytes, old_bytes),
            legacy.payload().substr(kHeaderBytes, old_bytes));
  EXPECT_TRUE(unpack(legacy).sequence_state_keys.empty());
  // The tensor arena is independently located by the header. Removing the
  // suffix from the descriptor leaves valid padding before that arena.
  std::memcpy(
      with_keys.mutable_payload()->data(), &old_bytes, sizeof(old_bytes));
  ForwardInput old_shape = unpack(with_keys);
  EXPECT_TRUE(old_shape.sequence_state_keys.empty());
  EXPECT_TRUE(torch::equal(old_shape.host_token_ids(), input.host_token_ids()));
}

TEST_F(SequenceStateInputTest, RejectsTruncatedAndUnknownSuffixBeforeKeyRead) {
  // cc_test initializes NPU/Python before every test, including CPU input
  // tests. Re-exec each death-test child instead of inheriting that runtime.
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  ForwardInput input;
  input.sequence_state_keys = {{1, 0}};
  proto::PackedForwardInput valid;
  ASSERT_TRUE(forward_input_to_packed_proto(input, &valid));
  constexpr uint64_t kHeaderBytes = 3 * sizeof(uint64_t);
  const uint64_t full_bytes = descriptor_bytes(valid);
  const uint64_t suffix_offset = kHeaderBytes + full_bytes - 40;
  proto::PackedForwardInput unknown = valid;
  unknown.mutable_payload()->at(suffix_offset) ^= 1;
  EXPECT_DEATH(unpack(unknown), "unknown forward-input descriptor extension");
  proto::PackedForwardInput truncated = valid;
  const uint64_t short_bytes = full_bytes - 1;
  std::memcpy(
      truncated.mutable_payload()->data(), &short_bytes, sizeof(short_bytes));
  EXPECT_DEATH(unpack(truncated), "truncated sequence-state count");
  proto::PackedForwardInput count_overflow = valid;
  const uint64_t huge = std::numeric_limits<uint64_t>::max();
  std::memcpy(count_overflow.mutable_payload()->data() + suffix_offset + 8,
              &huge,
              sizeof(huge));
  EXPECT_DEATH(unpack(count_overflow), "truncated sequence-state keys");
}

TEST_F(SequenceStateInputTest, RetiresOnlyDispatchedEpochExactlyOnce) {
  auto queue = std::make_shared<SequenceStateRetirementQueue>();
  auto sequence = make_sequence(/*index=*/0);
  sequence->reset();
  EXPECT_TRUE(queue->drain().empty());
  const auto first = sequence->sequence_state_key();
  sequence->track_sequence_state(queue);
  sequence->track_sequence_state(queue);
  // Omitting the Sequence from any number of subsequent batches is not end.
  EXPECT_TRUE(queue->drain().empty());
  {
    Sequence copied(*sequence);
    auto forked = sequence->fork(/*index=*/1);
    copied.reset();
    forked->reset();
  }
  EXPECT_TRUE(queue->drain().empty());
  sequence->reset();
  expect_keys(queue->drain(), std::array{first});
  sequence->reset();
  EXPECT_TRUE(queue->drain().empty());
  const auto next = sequence->sequence_state_key();
  sequence->track_sequence_state(queue);
  sequence.reset();
  expect_keys(queue->drain(), std::array{next});
  EXPECT_TRUE(queue->drain().empty());
  auto never_dispatched = make_sequence(/*index=*/0);
  never_dispatched.reset();
  EXPECT_TRUE(queue->drain().empty());
}

TEST_F(SequenceStateInputTest, OwnerChangesRequireResetAndQueueOutlivesEngine) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  auto first_owner = std::make_shared<SequenceStateRetirementQueue>();
  auto second_owner = std::make_shared<SequenceStateRetirementQueue>();
  auto sequence = make_sequence(/*index=*/0);
  auto unrelated = make_sequence(/*index=*/0);
  const auto first_key = sequence->sequence_state_key();
  const auto unrelated_key = unrelated->sequence_state_key();
  sequence->track_sequence_state(first_owner);
  unrelated->track_sequence_state(second_owner);
  EXPECT_DEATH(sequence->track_sequence_state(second_owner),
               "reset Sequence before changing state owner");
  sequence->reset();
  expect_keys(first_owner->drain(), std::array{first_key});
  EXPECT_TRUE(second_owner->drain().empty());
  const auto next_key = sequence->sequence_state_key();
  sequence->track_sequence_state(second_owner);
  unrelated.reset();
  sequence.reset();
  expect_keys(second_owner->drain(), std::array{unrelated_key, next_key});
  EXPECT_TRUE(first_owner->drain().empty());
  // A response retains its Sequence after Engine's owning reference is gone.
  auto late = make_sequence(/*index=*/1);
  auto engine_queue = std::make_shared<SequenceStateRetirementQueue>();
  std::weak_ptr<SequenceStateRetirementQueue> observer = engine_queue;
  late->track_sequence_state(engine_queue);
  engine_queue.reset();
  EXPECT_FALSE(observer.expired());
  late.reset();
  EXPECT_TRUE(observer.expired());
}

TEST_F(SequenceStateInputTest, ConcurrentDestructionAndDrainLoseNoIdentities) {
  constexpr uint32_t kThreads = 4;
  constexpr uint32_t kPerThread = 128;
  auto queue = std::make_shared<SequenceStateRetirementQueue>();
  std::barrier start(kThreads + 1);
  std::atomic<uint32_t> finished{0};
  std::vector<std::thread> producers;
  producers.reserve(kThreads);
  std::unordered_set<uint64_t> expected;
  expected.reserve(kThreads * kPerThread);
  for (uint32_t thread = 0; thread < kThreads; ++thread) {
    std::vector<std::unique_ptr<Sequence>> owned;
    owned.reserve(kPerThread);
    for (uint32_t row = 0; row < kPerThread; ++row) {
      auto sequence = make_sequence(row,
                                    /*prompt_size=*/4,
                                    /*allocate_blocks=*/false);
      sequence->track_sequence_state(queue);
      expected.emplace(sequence->sequence_state_key().sequence_id);
      owned.emplace_back(std::move(sequence));
    }
    producers.emplace_back([&, owned = std::move(owned)]() mutable {
      start.arrive_and_wait();
      owned.clear();
      finished.fetch_add(/*arg=*/1, std::memory_order_release);
    });
  }
  std::vector<SequenceStateKey> actual;
  actual.reserve(kThreads * kPerThread);
  start.arrive_and_wait();
  while (finished.load(std::memory_order_acquire) < kThreads) {
    const auto drained = queue->drain();
    actual.insert(actual.end(), drained.begin(), drained.end());
    std::this_thread::yield();
  }
  for (std::thread& producer : producers) {
    producer.join();
  }
  const auto remaining = queue->drain();
  actual.insert(actual.end(), remaining.begin(), remaining.end());
  ASSERT_EQ(actual.size(), expected.size());
  for (const SequenceStateKey& key : actual) {
    EXPECT_EQ(key.epoch, 0U);
    EXPECT_EQ(expected.erase(key.sequence_id), 1U);
  }
  EXPECT_TRUE(expected.empty());
  EXPECT_TRUE(queue->drain().empty());
}

TEST_F(SequenceStateInputTest, PackedInputSeparatesRetiredAndRecomputedEpochs) {
  auto queue = std::make_shared<SequenceStateRetirementQueue>();
  auto first = make_sequence(/*index=*/0);
  auto second = make_sequence(/*index=*/1);
  ForwardInput previous = build({first.get(), second.get()});
  first->track_sequence_state(queue);
  second->track_sequence_state(queue);
  const auto old_key = first->sequence_state_key();
  first->reset();
  first->add_blocks(BlockType::KV, manager_->allocate(/*num_blocks=*/1));
  ForwardInput next = build({second.get(), first.get()});
  next.retired_sequence_state_keys = queue->drain();
  proto::PackedForwardInput packed;
  ASSERT_TRUE(forward_input_to_packed_proto(next, &packed));
  const ForwardInput received = unpack(packed);
  expect_keys(
      received.sequence_state_keys,
      std::array{second->sequence_state_key(), first->sequence_state_key()});
  expect_keys(received.retired_sequence_state_keys, std::array{old_key});
  expect_keys(previous.sequence_state_keys,
              std::array{old_key, second->sequence_state_key()});
  first->track_sequence_state(queue);
  const auto next_key = first->sequence_state_key();
  const auto second_key = second->sequence_state_key();
  first.reset();
  second.reset();
  ForwardInput idle;
  idle.retired_sequence_state_keys = queue->drain();
  ASSERT_TRUE(forward_input_to_packed_proto(idle, &packed));
  const ForwardInput control = unpack(packed);
  EXPECT_TRUE(control.sequence_state_keys.empty());
  expect_keys(control.retired_sequence_state_keys,
              std::array{next_key, second_key});
  EXPECT_TRUE(queue->drain().empty());
}

TEST_F(SequenceStateInputTest, SequenceRetirementPerformance) {
  if (std::getenv(/*name=*/"XLLM_PIPELINE_PERF") == nullptr) {
    GTEST_SKIP() << "Run separately with XLLM_PIPELINE_PERF=1.";
  }
  constexpr int32_t kIterations = 256;
  for (uint32_t rows : std::array<uint32_t, 4>{1, 4, 32, 128}) {
    auto queue = std::make_shared<SequenceStateRetirementQueue>();
    std::vector<std::unique_ptr<Sequence>> sequences;
    sequences.reserve(rows);
    for (uint32_t row = 0; row < rows; ++row) {
      sequences.emplace_back(make_sequence(row,
                                           /*prompt_size=*/4,
                                           /*allocate_blocks=*/false));
    }
    std::vector<SequenceStateKey> retired;
    const auto cycle = [&]() {
      for (const auto& sequence : sequences) {
        sequence->track_sequence_state(queue);
        sequence->reset();
      }
      retired = queue->drain();
    };
    for (int32_t round = 0; round < 3; ++round) {
      for (int32_t warmup = 0; warmup < 8; ++warmup) {
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
      ASSERT_EQ(retired.size(), rows);
      for (uint32_t row = 0; row < rows; ++row) {
        const auto key = sequences[row]->sequence_state_key();
        EXPECT_EQ(retired[row].sequence_id, key.sequence_id);
        EXPECT_EQ(retired[row].epoch + 1, key.epoch);
      }
      EXPECT_TRUE(queue->drain().empty());
      LOG(INFO) << "STATE_RETIREMENT_PERF rows=" << rows << " round=" << round
                << " us=" << us << " iterations=" << kIterations;
    }
  }
}

TEST_F(SequenceStateInputTest, PackedIdentityPerformance) {
  if (std::getenv(/*name=*/"XLLM_PIPELINE_PERF") == nullptr) {
    GTEST_SKIP() << "Run separately with XLLM_PIPELINE_PERF=1.";
  }
  constexpr int32_t kIterations = 256;
  for (uint32_t rows : std::array<uint32_t, 4>{1, 4, 32, 128}) {
    std::vector<std::unique_ptr<Sequence>> owned;
    std::vector<Sequence*> sequences;
    owned.reserve(rows);
    sequences.reserve(rows);
    for (uint32_t row = 0; row < rows; ++row) {
      owned.emplace_back(make_sequence(row));
      sequences.emplace_back(owned.back().get());
    }
    ForwardInput input = build(std::move(sequences));
    const auto keys = input.sequence_state_keys;
    for (int32_t pair = 0; pair < 3; ++pair) {
      for (int32_t order = 0; order < 2; ++order) {
        const bool with_keys = (pair + order) % 2 == 0;
        input.sequence_state_keys =
            with_keys ? keys : std::vector<SequenceStateKey>();
        proto::PackedForwardInput packed;
        ForwardInput result;
        const auto cycle = [&]() {
          CHECK(forward_input_to_packed_proto(input, &packed));
          result = unpack(packed);
        };
        for (int32_t warmup = 0; warmup < 8; ++warmup) {
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
        expect_keys(result.sequence_state_keys, input.sequence_state_keys);
        EXPECT_TRUE(
            torch::equal(result.host_token_ids(), input.host_token_ids()));
        LOG(INFO) << "STATE_INPUT_PERF rows=" << rows << " pair=" << pair
                  << " with_keys=" << with_keys << " us=" << us
                  << " descriptor_bytes=" << descriptor_bytes(packed)
                  << " payload_bytes=" << packed.payload().size();
      }
    }
  }
}

}  // namespace
}  // namespace xllm
