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

#include "core/runtime/task_pipeline/model_input_binding.h"

#include <gtest/gtest.h>

#include <array>
#include <numeric>
#include <utility>
#include <vector>

namespace xllm {
namespace {

struct InputData {
  std::vector<int32_t> tokens;
  std::vector<int32_t> positions;
  std::vector<int32_t> slots;
  std::vector<int32_t> query;
  std::vector<int32_t> kv;
  std::vector<int32_t> cumulative;
  std::vector<int32_t> blocks;
  uint32_t width = 2;
};

InputData make_input(std::vector<int32_t> query, std::vector<int32_t> kv) {
  InputData input;
  input.query = std::move(query);
  input.kv = std::move(kv);
  input.cumulative.resize(input.query.size());
  std::partial_sum(
      input.query.begin(), input.query.end(), input.cumulative.begin());
  const int32_t tokens = input.cumulative.empty() ? 0 : input.cumulative.back();
  input.tokens.resize(tokens);
  input.positions.resize(tokens);
  input.slots.resize(tokens);
  input.blocks.resize(input.query.size() * input.width);
  std::iota(input.tokens.begin(), input.tokens.end(), 101);
  std::iota(input.positions.begin(), input.positions.end(), 201);
  std::iota(input.slots.begin(), input.slots.end(), 301);
  std::iota(input.blocks.begin(), input.blocks.end(), 401);
  return input;
}

ModelInputHostView view(const InputData& input) {
  return {input.tokens,
          input.positions,
          input.slots,
          input.query,
          input.kv,
          input.cumulative,
          input.blocks,
          input.width};
}

class ModelInputBindingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_TRUE(ModelInputStorage::create(
                    {16, 4, 5}, torch::Device(torch::kPrivateUse1, 0), storage_)
                    .ok());
    binding_ = std::make_unique<ModelInputBinding>(*storage_);
    transfer_ = std::make_unique<Stream>(storage_->device_buffer().device());
    consumer_ = std::make_unique<Stream>(storage_->device_buffer().device());
    storage_->host_buffer().fill_(17);
    storage_->device_buffer().fill_(-23);
    ASSERT_EQ(aclrtSynchronizeDevice(), ACL_SUCCESS);
  }

  void expect_device_input(const InputData& input) {
    CHECK(consumer_->wait_event(binding_->ready_event()));
    auto guard = consumer_->set_stream_guard();
    const AttentionDeviceInput& device = binding_->params().attention.device;
    const std::array<torch::Tensor, 6> tensors = {binding_->tokens(),
                                                  binding_->positions(),
                                                  device.new_cache_slots,
                                                  device.q_seq_lens,
                                                  device.kv_seq_lens,
                                                  device.q_cu_seq_lens};
    const std::array<std::span<const int32_t>, 6> expected = {input.tokens,
                                                              input.positions,
                                                              input.slots,
                                                              input.query,
                                                              input.kv,
                                                              input.cumulative};
    for (uint32_t field = 0; field < tensors.size(); ++field) {
      torch::Tensor host = tensors[field].cpu();
      EXPECT_EQ(host.numel(), expected[field].size());
      const int32_t* data = host.data_ptr<int32_t>();
      for (uint32_t index = 0; index < expected[field].size(); ++index) {
        EXPECT_EQ(data[index], expected[field][index]);
      }
    }
    torch::Tensor blocks = device.block_tables.cpu();
    EXPECT_EQ(blocks.size(0), input.query.size());
    EXPECT_EQ(blocks.size(1), 5);
    const int32_t* values = blocks.data_ptr<int32_t>();
    for (uint32_t row = 0; row < input.query.size(); ++row) {
      for (uint32_t column = 0; column < input.width; ++column) {
        EXPECT_EQ(values[row * 5 + column],
                  input.blocks[row * input.width + column]);
      }
      for (uint32_t column = input.width; column < 5; ++column) {
        EXPECT_EQ(values[row * 5 + column], 0);
      }
    }
  }

  void expect_rejected(const ModelInputHostView& input,
                       const ModelInputBatch& batch) {
    const torch::Tensor before = storage_->host_buffer().clone();
    const uint64_t batch_id = binding_->params().meta.batch_id;
    const std::vector<int32_t> lengths =
        binding_->params().attention.host.kv_seq_lens;
    const ModelInputTransferInfo info = binding_->transfer_info();
    const aclrtEvent event = binding_->ready_event()->npu_event();
    const void* tokens = binding_->tokens().data_ptr();
    EXPECT_FALSE(binding_->prepare(input, batch, *transfer_).ok());
    EXPECT_TRUE(torch::equal(before, storage_->host_buffer()));
    EXPECT_EQ(binding_->params().meta.batch_id, batch_id);
    EXPECT_EQ(binding_->params().attention.host.kv_seq_lens, lengths);
    EXPECT_EQ(binding_->transfer_info().h2d_bytes, info.h2d_bytes);
    EXPECT_EQ(binding_->transfer_info().h2d_calls, info.h2d_calls);
    EXPECT_EQ(binding_->ready_event()->npu_event(), event);
    EXPECT_EQ(binding_->tokens().data_ptr(), tokens);
  }

  std::unique_ptr<ModelInputStorage> storage_;
  std::unique_ptr<ModelInputBinding> binding_;
  std::unique_ptr<Stream> transfer_;
  std::unique_ptr<Stream> consumer_;
};

TEST_F(ModelInputBindingTest, PrefillUsesFixedInputsAndUncachedLengths) {
  InputData input = make_input({3, 2}, {3, 2});
  ASSERT_TRUE(binding_
                  ->prepare(view(input),
                            {BatchForwardType::PREFILL, 2, 41, true},
                            *transfer_)
                  .ok());
  expect_device_input(input);
  const ModelInputParams& params = binding_->params();
  EXPECT_EQ(params.attention.host.kv_cache_tokens_nums,
            (std::vector<int32_t>{0, 0}));
  EXPECT_EQ(params.attention.host.q_cu_seq_lens, (std::vector<int32_t>{3, 5}));
  EXPECT_EQ(params.meta.num_sequences, 2);
  EXPECT_EQ(params.meta.actual_num_sequences, 2);
  EXPECT_EQ(params.meta.batch_id, 41);
  EXPECT_TRUE(params.meta.is_graph_warmup);
  EXPECT_EQ(params.meta.q_max_seq_len, 3);
  EXPECT_EQ(params.meta.kv_max_seq_len, 3);
  EXPECT_EQ(binding_->tokens().data_ptr(),
            storage_->device().token_ids.data_ptr());
  EXPECT_EQ(params.attention.device.block_tables.data_ptr(),
            storage_->device().block_tables.data_ptr());
  EXPECT_EQ(params.attention.host.block_tables.data_ptr(),
            storage_->host().block_tables.data_ptr());
  EXPECT_EQ(params.attention.device.block_tables.stride(0), 5);
  EXPECT_EQ(params.attention.host.graph_q_seq_lens_data,
            storage_->host().q_seq_lens.data_ptr<int32_t>());
  EXPECT_EQ(params.attention.host.graph_kv_seq_lens_data,
            storage_->host().kv_seq_lens.data_ptr<int32_t>());
  EXPECT_FALSE(params.attention.device.kv_cache_tokens_nums.defined());
  EXPECT_EQ(binding_->transfer_info().h2d_calls, 7);
}

TEST_F(ModelInputBindingTest, MixedBatchKeepsActualAndPaddingRowsDistinct) {
  InputData input = make_input({2, 1, 0}, {5, 7, 0});
  ASSERT_TRUE(
      binding_
          ->prepare(view(input), {BatchForwardType::MIXED, 2, 42}, *transfer_)
          .ok());
  expect_device_input(input);
  EXPECT_EQ(binding_->params().meta.num_sequences, 3);
  EXPECT_EQ(binding_->params().meta.actual_num_sequences, 2);
  EXPECT_EQ(binding_->params().meta.q_max_seq_len, 2);
  EXPECT_EQ(binding_->params().meta.kv_max_seq_len, 7);
  EXPECT_EQ(binding_->params().attention.host.kv_cache_tokens_nums,
            (std::vector<int32_t>{3, 6, 0}));
  EXPECT_EQ(binding_->transfer_info().h2d_bytes, 96);
}

TEST_F(ModelInputBindingTest, VaryingSizesReuseAddressesAndHostCapacity) {
  InputData first = make_input({2, 2, 2, 2}, {7, 7, 7, 7});
  ASSERT_TRUE(binding_
                  ->prepare(view(first),
                            {BatchForwardType::CHUNKED_PREFILL, 4},
                            *transfer_)
                  .ok());
  expect_device_input(first);
  const AttentionHostInput& host = binding_->params().attention.host;
  const int32_t* lengths = host.q_seq_lens.data();
  const int32_t* slots = host.new_cache_slots.data();
  const size_t lengths_capacity = host.q_seq_lens.capacity();
  const size_t slots_capacity = host.new_cache_slots.capacity();
  const aclrtEvent event = binding_->ready_event()->npu_event();
  for (uint32_t iteration = 0; iteration < 16; ++iteration) {
    const uint32_t rows = iteration % 4 + 1;
    InputData input = make_input(std::vector<int32_t>(rows, 1),
                                 std::vector<int32_t>(rows, iteration + 2));
    input.tokens[0] += static_cast<int32_t>(iteration);
    ASSERT_TRUE(binding_
                    ->prepare(view(input),
                              {BatchForwardType::DECODE, rows, iteration},
                              *transfer_)
                    .ok());
    expect_device_input(input);
    EXPECT_EQ(host.q_seq_lens.data(), lengths);
    EXPECT_EQ(host.new_cache_slots.data(), slots);
    EXPECT_EQ(host.q_seq_lens.capacity(), lengths_capacity);
    EXPECT_EQ(host.new_cache_slots.capacity(), slots_capacity);
    EXPECT_EQ(binding_->tokens().data_ptr(),
              storage_->device().token_ids.data_ptr());
    EXPECT_EQ(binding_->ready_event()->npu_event(), event);
    EXPECT_EQ(binding_->params().meta.q_max_seq_len, 1);
    EXPECT_EQ(binding_->params().meta.kv_max_seq_len, iteration + 2);
  }
}

TEST_F(ModelInputBindingTest,
       EmptyBatchClearsViewsAndMetadataWithoutReallocation) {
  InputData input = make_input({3, 2}, {8, 9});
  ASSERT_TRUE(binding_
                  ->prepare(view(input),
                            {BatchForwardType::CHUNKED_PREFILL, 2, 51, true},
                            *transfer_)
                  .ok());
  expect_device_input(input);
  const size_t capacity =
      binding_->params().attention.host.q_seq_lens.capacity();
  ASSERT_TRUE(
      binding_->prepare({}, {BatchForwardType::EMPTY, 0, 52}, *transfer_).ok());
  CHECK(consumer_->wait_event(binding_->ready_event()));
  ASSERT_EQ(consumer_->synchronize(), 0);
  EXPECT_EQ(binding_->tokens().numel(), 0);
  EXPECT_EQ(binding_->params().attention.device.block_tables.size(0), 0);
  EXPECT_TRUE(binding_->params().attention.host.q_seq_lens.empty());
  EXPECT_TRUE(binding_->params().attention.host.kv_cache_tokens_nums.empty());
  EXPECT_TRUE(binding_->params().attention.host.new_cache_slots.empty());
  EXPECT_EQ(binding_->params().attention.host.q_seq_lens.capacity(), capacity);
  EXPECT_EQ(binding_->params().meta.q_max_seq_len, 0);
  EXPECT_EQ(binding_->params().meta.kv_max_seq_len, 0);
  EXPECT_EQ(binding_->params().meta.actual_num_sequences, 0);
  EXPECT_EQ(binding_->params().meta.batch_id, 52);
  EXPECT_FALSE(binding_->params().meta.is_graph_warmup);
  EXPECT_EQ(binding_->transfer_info().h2d_calls, 0);
}

TEST_F(ModelInputBindingTest, InvalidBatchSemanticsPreservePreviousBinding) {
  InputData input = make_input({2, 1}, {5, 7});
  ASSERT_TRUE(
      binding_
          ->prepare(view(input), {BatchForwardType::MIXED, 2, 63}, *transfer_)
          .ok());
  expect_device_input(input);
  for (const ModelInputBatch& batch :
       {ModelInputBatch{BatchForwardType(999), 2},
        ModelInputBatch{BatchForwardType::DECODE, 2},
        ModelInputBatch{BatchForwardType::PREFILL, 2},
        ModelInputBatch{BatchForwardType::MIXED, 3},
        ModelInputBatch{BatchForwardType::EMPTY, 0}}) {
    expect_rejected(view(input), batch);
  }
  InputData zero_query = make_input({2, 0}, {5, 7});
  expect_rejected(view(zero_query), {BatchForwardType::MIXED, 2});
  expect_rejected({}, {BatchForwardType::DECODE, 0});
}

TEST_F(ModelInputBindingTest, InvalidTransferInputPreservesPreviousBinding) {
  InputData input = make_input({1}, {4});
  ASSERT_TRUE(
      binding_
          ->prepare(view(input), {BatchForwardType::DECODE, 1, 71}, *transfer_)
          .ok());
  expect_device_input(input);
  InputData wrong_shape = make_input({2}, {5});
  wrong_shape.positions.pop_back();
  expect_rejected(view(wrong_shape), {BatchForwardType::CHUNKED_PREFILL, 1});
  InputData oversized = make_input({17}, {17});
  expect_rejected(view(oversized), {BatchForwardType::PREFILL, 1});
}

TEST_F(ModelInputBindingTest, NextInputCanBorrowPreviousHostMetadata) {
  InputData first = make_input({2, 1}, {4, 3});
  ASSERT_TRUE(binding_
                  ->prepare(view(first),
                            {BatchForwardType::CHUNKED_PREFILL, 2},
                            *transfer_)
                  .ok());
  expect_device_input(first);
  InputData next = make_input({2, 2}, {2, 3});
  ModelInputHostView borrowed = view(next);
  borrowed.q_seq_lens = binding_->params().attention.host.kv_cache_tokens_nums;
  borrowed.kv_seq_lens = binding_->params().attention.host.q_cu_seq_lens;
  ASSERT_TRUE(binding_
                  ->prepare(borrowed,
                            {BatchForwardType::CHUNKED_PREFILL, 2},
                            *transfer_)
                  .ok());
  expect_device_input(next);
  EXPECT_EQ(binding_->params().attention.host.kv_seq_lens, next.kv);
  EXPECT_EQ(binding_->params().attention.host.kv_cache_tokens_nums,
            (std::vector<int32_t>{0, 1}));
}

TEST_F(ModelInputBindingTest, CallerSuppliedDummyRowHasNoActualSequence) {
  InputData input = make_input({1}, {1});
  ASSERT_TRUE(
      binding_->prepare(view(input), {BatchForwardType::DECODE, 0}, *transfer_)
          .ok());
  expect_device_input(input);
  EXPECT_EQ(binding_->params().meta.num_sequences, 1);
  EXPECT_EQ(binding_->params().meta.actual_num_sequences, 0);
  EXPECT_EQ(binding_->tokens().numel(), 1);
}

}  // namespace
}  // namespace xllm
