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

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
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
  uint32_t width = 0;
};

InputData make_input(std::vector<int32_t> query, uint32_t width) {
  InputData data;
  data.query = std::move(query);
  data.width = width;
  data.kv.reserve(data.query.size());
  data.cumulative.reserve(data.query.size());
  int32_t total = 0;
  for (const int32_t length : data.query) {
    total += length;
    data.kv.emplace_back(length + 5);
    data.cumulative.emplace_back(total);
  }
  data.tokens.reserve(total);
  data.positions.reserve(total);
  data.slots.reserve(total);
  for (int32_t i = 0; i < total; ++i) {
    data.tokens.emplace_back(101 + i);
    data.positions.emplace_back(11 + i);
    data.slots.emplace_back(201 + i);
  }
  data.blocks.reserve(data.query.size() * width);
  for (uint32_t i = 0; i < data.query.size() * width; ++i) {
    data.blocks.emplace_back(301 + i);
  }
  return data;
}

ModelInputHostView view(const InputData& data) {
  return {data.tokens,
          data.positions,
          data.slots,
          data.query,
          data.kv,
          data.cumulative,
          data.blocks,
          data.width};
}

class ModelInputPreparerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_TRUE(ModelInputStorage::create(
                    {8, 3, 5}, torch::Device(torch::kPrivateUse1, 0), storage_)
                    .ok());
    preparer_ = std::make_unique<ModelInputPreparer>(*storage_);
    transfer_ = std::make_unique<Stream>(storage_->device_buffer().device());
    consumer_ = std::make_unique<Stream>(storage_->device_buffer().device());
    ASSERT_NE(transfer_->get_stream()->stream(),
              consumer_->get_stream()->stream());
    storage_->host_buffer().fill_(71);
    storage_->device_buffer().fill_(-33);
    ASSERT_EQ(aclrtSynchronizeDevice(), ACL_SUCCESS);
  }

  torch::Tensor readback() {
    CHECK(consumer_->wait_event(preparer_->ready_event()));
    auto guard = consumer_->set_stream_guard();
    return storage_->device_buffer().cpu();
  }

  void expect_contents(const InputData& data, const torch::Tensor& copy) {
    const int32_t* values = copy.data_ptr<int32_t>();
    const ModelInputLayout& layout = storage_->layout();
    const std::array<ModelInputRegion, 6> regions = {layout.token_ids,
                                                     layout.positions,
                                                     layout.new_cache_slots,
                                                     layout.q_seq_lens,
                                                     layout.kv_seq_lens,
                                                     layout.q_cu_seq_lens};
    const std::array<std::span<const int32_t>, 6> sources = {data.tokens,
                                                             data.positions,
                                                             data.slots,
                                                             data.query,
                                                             data.kv,
                                                             data.cumulative};
    for (uint32_t i = 0; i < sources.size(); ++i) {
      for (uint32_t j = 0; j < sources[i].size(); ++j) {
        EXPECT_EQ(values[regions[i].offset / sizeof(int32_t) + j],
                  sources[i][j]);
      }
    }
    const int32_t* blocks =
        values + layout.block_tables.offset / sizeof(int32_t);
    for (uint32_t row = 0; row < data.query.size(); ++row) {
      for (uint32_t col = 0; col < data.width; ++col) {
        EXPECT_EQ(blocks[row * 5 + col], data.blocks[row * data.width + col]);
      }
      for (uint32_t col = data.width; col < 5; ++col) {
        EXPECT_EQ(blocks[row * 5 + col], 0);
      }
    }
  }

  void expect_rejected(const ModelInputHostView& input) {
    torch::Tensor before = storage_->host_buffer().clone();
    ModelInputTransferInfo info{17, 19, 23, 29, 31};
    EXPECT_FALSE(preparer_->prepare(input, *transfer_, info).ok());
    EXPECT_TRUE(torch::equal(before, storage_->host_buffer()));
    EXPECT_EQ(info.tokens, 17);
    EXPECT_EQ(info.sequences, 19);
    EXPECT_EQ(info.h2d_calls, 23);
    EXPECT_EQ(info.h2d_bytes, 29);
    EXPECT_EQ(info.device_zero_bytes, 31);
    EXPECT_TRUE(storage_->device_buffer().cpu().eq(-33).all().item<bool>());
  }

  std::unique_ptr<ModelInputStorage> storage_;
  std::unique_ptr<ModelInputPreparer> preparer_;
  std::unique_ptr<Stream> transfer_;
  std::unique_ptr<Stream> consumer_;
};

TEST_F(ModelInputPreparerTest, MixedPrefillCopiesExactDataAcrossStreams) {
  InputData data = make_input({2, 3}, 2);
  ModelInputTransferInfo info;
  ASSERT_TRUE(preparer_->prepare(view(data), *transfer_, info).ok());
  torch::Tensor copy = readback();
  expect_contents(data, copy);
  EXPECT_EQ(info.tokens, 5);
  EXPECT_EQ(info.sequences, 2);
  EXPECT_EQ(info.h2d_calls, 7);
  EXPECT_EQ(info.h2d_bytes, 100);
  EXPECT_EQ(info.device_zero_bytes, 40);
  const int32_t* values = copy.data_ptr<int32_t>();
  EXPECT_EQ(values[5], -33);
  EXPECT_EQ(values[storage_->layout().block_tables.offset / 4 + 10], -33);
}

TEST_F(ModelInputPreparerTest,
       DecodeReusesAddressesAndEventAcrossChangingWidths) {
  const void* host = storage_->host_buffer().data_ptr();
  const void* device = storage_->device().block_tables.data_ptr();
  const aclrtEvent event = preparer_->ready_event()->npu_event();
  const std::array<uint32_t, 4> widths = {5, 1, 3, 5};
  for (uint32_t iteration = 0; iteration < 16; ++iteration) {
    SCOPED_TRACE(iteration);
    const uint32_t width = widths[iteration % widths.size()];
    InputData data = make_input(
        width == 1 ? std::vector<int32_t>{1} : std::vector<int32_t>{1, 1, 1},
        width);
    data.tokens[0] += static_cast<int32_t>(iteration);
    ModelInputTransferInfo info;
    ASSERT_TRUE(preparer_->prepare(view(data), *transfer_, info).ok());
    expect_contents(data, readback());
    EXPECT_EQ(storage_->host_buffer().data_ptr(), host);
    EXPECT_EQ(storage_->device().block_tables.data_ptr(), device);
    EXPECT_EQ(preparer_->ready_event()->npu_event(), event);
    EXPECT_EQ(info.h2d_calls, 7);
    EXPECT_EQ(info.device_zero_bytes, width == 5 ? 0 : data.query.size() * 20);
  }
}

TEST_F(ModelInputPreparerTest, SourceCanChangeImmediatelyAfterPrepareReturns) {
  InputData data = make_input({2, 1}, 3);
  ModelInputTransferInfo info;
  ASSERT_TRUE(preparer_->prepare(view(data), *transfer_, info).ok());
  std::fill(data.tokens.begin(), data.tokens.end(), -99);
  std::fill(data.blocks.begin(), data.blocks.end(), -99);
  InputData expected = make_input({2, 1}, 3);
  expect_contents(expected, readback());
}

TEST_F(ModelInputPreparerTest, EmptyInputRecordsReadyWithoutTouchingBuffers) {
  ModelInputTransferInfo info{1, 2, 3, 4, 5};
  ASSERT_TRUE(preparer_->prepare({}, *transfer_, info).ok());
  EXPECT_TRUE(readback().eq(-33).all().item<bool>());
  EXPECT_TRUE(storage_->host_buffer().eq(71).all().item<bool>());
  EXPECT_EQ(info.tokens, 0);
  EXPECT_EQ(info.sequences, 0);
  EXPECT_EQ(info.h2d_calls, 0);
  EXPECT_EQ(info.h2d_bytes, 0);
  EXPECT_EQ(info.device_zero_bytes, 0);
}

TEST_F(ModelInputPreparerTest, FullCapacityHasNoBlockTableClear) {
  InputData data = make_input({3, 3, 2}, 5);
  ModelInputTransferInfo info;
  ASSERT_TRUE(preparer_->prepare(view(data), *transfer_, info).ok());
  expect_contents(data, readback());
  EXPECT_EQ(info.h2d_bytes, 192);
  EXPECT_EQ(info.device_zero_bytes, 0);
}

TEST_F(ModelInputPreparerTest, RejectsShapeMismatchesBeforeWriting) {
  InputData data = make_input({2, 3}, 2);
  ModelInputHostView input = view(data);
  input.positions = input.positions.first(4);
  expect_rejected(input);
  input = view(data);
  input.kv_seq_lens = input.kv_seq_lens.first(1);
  expect_rejected(input);
  input = view(data);
  input.block_table_width = 3;
  expect_rejected(input);
  input = view(data);
  input.new_cache_slots = {};
  expect_rejected(input);
}

TEST_F(ModelInputPreparerTest, RejectsCapacityOverflowBeforeWriting) {
  for (const InputData& data :
       {make_input({9}, 1), make_input({1, 1, 1, 1}, 1), make_input({1}, 6)}) {
    expect_rejected(view(data));
  }
}

TEST_F(ModelInputPreparerTest, RejectsInconsistentLengthsBeforeWriting) {
  InputData data = make_input({2, 3}, 2);
  data.cumulative[1] = 4;
  expect_rejected(view(data));
  data = make_input({2, 3}, 2);
  data.query[0] = -1;
  expect_rejected(view(data));
  data = make_input({2, 3}, 2);
  data.kv[1] = 2;
  expect_rejected(view(data));
  data = make_input({2, 2}, 2);
  data.tokens.emplace_back(99);
  data.positions.emplace_back(99);
  data.slots.emplace_back(99);
  expect_rejected(view(data));
}

TEST_F(ModelInputPreparerTest, RejectsStagingAliasBeforeAnyFieldIsOverwritten) {
  InputData data = make_input({2, 3}, 2);
  ModelInputHostView input = view(data);
  input.block_tables = std::span<const int32_t>(
      storage_->host().block_tables.data_ptr<int32_t>(), data.blocks.size());
  expect_rejected(input);
}

TEST_F(ModelInputPreparerTest, RejectsIncompleteEmptyDescriptor) {
  ModelInputHostView input;
  input.block_table_width = 1;
  expect_rejected(input);
  InputData data = make_input({1}, 0);
  expect_rejected(view(data));
}

}  // namespace
}  // namespace xllm
