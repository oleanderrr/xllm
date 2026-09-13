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

#include "core/runtime/task_pipeline/model_input_layout.h"

#include <gtest/gtest.h>

#include <array>
#include <limits>

namespace xllm {
namespace {

std::array<ModelInputRegion, 7> regions(const ModelInputLayout& layout) {
  return {layout.token_ids,
          layout.positions,
          layout.new_cache_slots,
          layout.q_seq_lens,
          layout.kv_seq_lens,
          layout.q_cu_seq_lens,
          layout.block_tables};
}

void expect_same_layout(const ModelInputLayout& actual,
                        const ModelInputLayout& expected) {
  EXPECT_EQ(actual.capacity.max_tokens, expected.capacity.max_tokens);
  EXPECT_EQ(actual.capacity.max_sequences, expected.capacity.max_sequences);
  EXPECT_EQ(actual.capacity.max_blocks_per_sequence,
            expected.capacity.max_blocks_per_sequence);
  EXPECT_EQ(actual.block_table_row_stride_bytes,
            expected.block_table_row_stride_bytes);
  EXPECT_EQ(actual.total_bytes, expected.total_bytes);
  const auto actual_regions = regions(actual);
  const auto expected_regions = regions(expected);
  for (uint32_t i = 0; i < actual_regions.size(); ++i) {
    SCOPED_TRACE(i);
    EXPECT_EQ(actual_regions[i].offset, expected_regions[i].offset);
    EXPECT_EQ(actual_regions[i].bytes, expected_regions[i].bytes);
  }
}

TEST(ModelInputLayoutTest, PlansExactNonAlignedCapacities) {
  ModelInputLayout layout;
  ASSERT_TRUE(make_model_input_layout({5, 2, 3}, layout).ok());
  const std::array<uint64_t, 7> offsets = {0, 32, 64, 96, 112, 128, 144};
  const std::array<uint64_t, 7> bytes = {20, 20, 20, 8, 8, 8, 24};
  const auto fields = regions(layout);
  for (uint32_t i = 0; i < fields.size(); ++i) {
    SCOPED_TRACE(i);
    EXPECT_EQ(fields[i].offset, offsets[i]);
    EXPECT_EQ(fields[i].bytes, bytes[i]);
  }
  EXPECT_EQ(layout.total_bytes, 176);
  EXPECT_EQ(layout.block_table_row_stride_bytes, 12);
  EXPECT_EQ(layout.capacity.max_tokens, 5);
  EXPECT_EQ(layout.capacity.max_sequences, 2);
  EXPECT_EQ(layout.capacity.max_blocks_per_sequence, 3);
}

TEST(ModelInputLayoutTest, KeepsAllRegionsAlignedBoundedAndDisjoint) {
  const std::array<ModelInputCapacity, 5> cases = {{{1, 1, 1},
                                                    {4, 4, 4},
                                                    {1024, 32, 128},
                                                    {4097, 257, 513},
                                                    {65537, 513, 1025}}};
  for (const auto& capacity : cases) {
    ModelInputLayout layout;
    ASSERT_TRUE(make_model_input_layout(capacity, layout).ok());
    uint64_t previous_end = 0;
    for (const auto& field : regions(layout)) {
      EXPECT_EQ(field.offset % kModelInputAlignment, 0);
      EXPECT_GE(field.offset, previous_end);
      EXPECT_GT(field.bytes, 0);
      ASSERT_LE(field.offset, layout.total_bytes);
      EXPECT_LE(field.bytes, layout.total_bytes - field.offset);
      previous_end = field.offset + field.bytes;
    }
    EXPECT_EQ(layout.total_bytes % kModelInputAlignment, 0);
    EXPECT_LT(layout.total_bytes - previous_end, kModelInputAlignment);
  }
}

TEST(ModelInputLayoutTest, WidensTokenCapacityBeforeComputingBytes) {
  ModelInputLayout layout;
  ASSERT_TRUE(make_model_input_layout(
                  {std::numeric_limits<uint32_t>::max(), 1, 1}, layout)
                  .ok());
  EXPECT_EQ(layout.token_ids.bytes, 17179869180ULL);
  EXPECT_EQ(layout.positions.offset, 17179869184ULL);
  EXPECT_EQ(layout.total_bytes, 51539607616ULL);
}

TEST(ModelInputLayoutTest, ReusesTheSameLayoutForTheSameCapacity) {
  ModelInputLayout first;
  ModelInputLayout second;
  const ModelInputCapacity capacity{1024, 32, 128};
  ASSERT_TRUE(make_model_input_layout(capacity, first).ok());
  ASSERT_TRUE(make_model_input_layout(capacity, second).ok());
  expect_same_layout(second, first);
  EXPECT_EQ(first.positions.offset, 4096);
}

TEST(ModelInputLayoutTest, ReplacesAnExistingLayoutWithoutAccumulatingOffsets) {
  ModelInputLayout reused;
  ModelInputLayout expected;
  ASSERT_TRUE(make_model_input_layout({4096, 128, 1024}, reused).ok());
  ASSERT_TRUE(make_model_input_layout({5, 2, 3}, expected).ok());
  ASSERT_TRUE(make_model_input_layout({5, 2, 3}, reused).ok());
  expect_same_layout(reused, expected);
}

TEST(ModelInputLayoutTest, SupportsCapacityAliasingTheOutput) {
  ModelInputLayout layout;
  ASSERT_TRUE(make_model_input_layout({5, 2, 3}, layout).ok());
  const ModelInputLayout original = layout;
  ASSERT_TRUE(make_model_input_layout(layout.capacity, layout).ok());
  expect_same_layout(layout, original);
}

TEST(ModelInputLayoutTest, PreservesAliasedOutputOnFailure) {
  ModelInputLayout layout;
  ASSERT_TRUE(make_model_input_layout({5, 2, 3}, layout).ok());
  layout.capacity.max_tokens = 0;
  const ModelInputLayout original = layout;
  EXPECT_EQ(make_model_input_layout(layout.capacity, layout).code(),
            StatusCode::INVALID_ARGUMENT);
  expect_same_layout(layout, original);
}

TEST(ModelInputLayoutTest, WidensBlockTableRowStrideBeforeComputingBytes) {
  ModelInputLayout layout;
  ASSERT_TRUE(make_model_input_layout(
                  {1, 1, std::numeric_limits<uint32_t>::max()}, layout)
                  .ok());
  EXPECT_EQ(layout.block_table_row_stride_bytes, 17179869180ULL);
  EXPECT_EQ(layout.block_tables.bytes, 17179869180ULL);
  EXPECT_EQ(layout.total_bytes, 17179869280ULL);
}

TEST(ModelInputLayoutTest, RejectsEachZeroCapacityWithoutChangingOutput) {
  const std::array<ModelInputCapacity, 3> invalid = {
      {{0, 2, 3}, {5, 0, 3}, {5, 2, 0}}};
  ModelInputLayout layout;
  ASSERT_TRUE(make_model_input_layout({5, 2, 3}, layout).ok());
  const ModelInputLayout original = layout;
  for (const auto& capacity : invalid) {
    const Status status = make_model_input_layout(capacity, layout);
    EXPECT_EQ(status.code(), StatusCode::INVALID_ARGUMENT);
    EXPECT_FALSE(status.message().empty());
    expect_same_layout(layout, original);
  }
}

TEST(ModelInputLayoutTest, AcceptsTheLargestAlignedTotalBelowTheSignedLimit) {
  ModelInputCapacity capacity{715827880, 1073741822, 2147483647};
  ModelInputLayout layout;
  ASSERT_TRUE(make_model_input_layout(capacity, layout).ok());
  EXPECT_EQ(layout.total_bytes, 9223372036854775792ULL);
  const ModelInputLayout original = layout;
  // One more token rounds up each of the three token regions by 16 bytes.
  ++capacity.max_tokens;
  EXPECT_EQ(make_model_input_layout(capacity, layout).code(),
            StatusCode::INVALID_ARGUMENT);
  expect_same_layout(layout, original);
}

TEST(ModelInputLayoutTest,
     RejectsOversizedProductsAndTotalsWithoutChangingOutput) {
  constexpr uint32_t kMax = std::numeric_limits<uint32_t>::max();
  // Respectively: int32 bytes overflow uint64, bytes exceed int64, and the
  // block field fits int64 but adding the preceding regions does not.
  const std::array<ModelInputCapacity, 3> invalid = {
      {{1, kMax, kMax},
       {1, 2147483648U, 1073741824U},
       {1, 2147483647U, 1073741824U}}};
  ModelInputLayout layout;
  ASSERT_TRUE(make_model_input_layout({5, 2, 3}, layout).ok());
  const ModelInputLayout original = layout;
  for (const auto& capacity : invalid) {
    const Status status = make_model_input_layout(capacity, layout);
    EXPECT_EQ(status.code(), StatusCode::INVALID_ARGUMENT);
    EXPECT_FALSE(status.message().empty());
    expect_same_layout(layout, original);
  }
}

}  // namespace
}  // namespace xllm
