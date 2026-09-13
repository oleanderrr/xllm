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

#include "core/runtime/task_pipeline/token_result_storage.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "core/platform/stream.h"

namespace xllm {
namespace {

constexpr std::array<torch::Tensor TokenResultTensors::*, 5> kFields = {
    &TokenResultTensors::tokens,
    &TokenResultTensors::lengths,
    &TokenResultTensors::logprobs,
    &TokenResultTensors::top_tokens,
    &TokenResultTensors::top_logprobs};

void expect_result(const TokenResultTensors& values,
                   const TokenResultTensors& actual) {
  for (const auto member : kFields) {
    const torch::Tensor& expected_field = values.*member;
    const torch::Tensor& actual_field = actual.*member;
    ASSERT_EQ(expected_field.defined(), actual_field.defined());
    if (!actual_field.defined()) {
      continue;
    }
    EXPECT_TRUE(actual_field.device().is_cpu());
    EXPECT_FALSE(actual_field.is_pinned());
    EXPECT_TRUE(actual_field.is_contiguous());
    EXPECT_EQ(actual_field.sizes(), expected_field.sizes());
    EXPECT_EQ(actual_field.scalar_type(), expected_field.scalar_type());
  }
  ASSERT_TRUE(torch::equal(values.lengths, actual.lengths));
  for (int64_t row = 0; row < values.tokens.size(/*dim=*/0); ++row) {
    const int32_t length = values.lengths[row].item<int32_t>();
    for (int64_t column = 0; column < values.tokens.size(/*dim=*/1); ++column) {
      if (column < length) {
        EXPECT_EQ(actual.tokens[row][column].item<int64_t>(),
                  values.tokens[row][column].item<int64_t>());
        if (values.logprobs.defined()) {
          EXPECT_EQ(actual.logprobs[row][column].item<float>(),
                    values.logprobs[row][column].item<float>());
        }
        if (values.top_tokens.defined()) {
          EXPECT_TRUE(torch::equal(values.top_tokens[row][column],
                                   actual.top_tokens[row][column]));
          EXPECT_TRUE(torch::equal(values.top_logprobs[row][column],
                                   actual.top_logprobs[row][column]));
        }
      } else {
        EXPECT_EQ(actual.tokens[row][column].item<int64_t>(), -1);
        if (values.logprobs.defined()) {
          EXPECT_EQ(actual.logprobs[row][column].item<float>(),
                    -std::numeric_limits<float>::infinity());
        }
        if (values.top_tokens.defined()) {
          EXPECT_TRUE(
              (actual.top_tokens[row][column] == -1).all().item<bool>());
          EXPECT_TRUE((actual.top_logprobs[row][column] ==
                       -std::numeric_limits<float>::infinity())
                          .all()
                          .item<bool>());
        }
      }
    }
  }
}

class TokenResultStorageTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_TRUE(TokenResultStorage::create({4, 4, 5}, device_, storage_).ok());
    producer_ = std::make_unique<Stream>(device_);
    copy_ = std::make_unique<Stream>(device_);
    aclrtEvent event = nullptr;
    ASSERT_EQ(aclrtCreateEventExWithFlag(&event, ACL_EVENT_SYNC), ACL_SUCCESS);
    producer_ready_ = std::make_shared<StreamEvent>(event);
    ASSERT_EQ(aclrtSynchronizeDevice(), ACL_SUCCESS);
  }

  void TearDown() override {
    storage_.reset();
    EXPECT_EQ(aclrtSynchronizeDevice(), ACL_SUCCESS);
  }

  TokenResultTensors produce(const std::vector<int32_t>& lengths,
                             int32_t seed) {
    auto guard = producer_->set_stream_guard();
    TokenResultTensors expected;
    for (const auto member : kFields) {
      const torch::Tensor& target = storage_->device().*member;
      if (!target.defined()) {
        continue;
      }
      torch::Tensor values;
      if (member == &TokenResultTensors::lengths) {
        values = torch::tensor(lengths, torch::kInt32);
      } else if (target.scalar_type() == torch::kInt64) {
        values = (torch::arange(target.numel(), torch::kInt64) + seed)
                     .view(target.sizes());
      } else {
        values =
            (torch::arange(target.numel(), torch::kFloat32) * -0.125f - seed)
                .view(target.sizes());
      }
      target.copy_(values);
      expected.*member = std::move(values);
    }
    // A final Device write remains ordered by producer_ready, beyond the CPU
    // fixture transfers. The copy stream must observe this updated value.
    storage_->device().tokens.add_(/*other=*/17);
    expected.tokens.add_(/*other=*/17);
    record_producer();
    return expected;
  }

  void record_producer() {
    CHECK_EQ(aclrtRecordEvent(producer_ready_->npu_event(),
                              producer_->get_stream()->stream()),
             ACL_SUCCESS);
  }

  void expect_producer_complete() {
    aclrtEventRecordedStatus status = ACL_EVENT_RECORDED_STATUS_NOT_READY;
    ASSERT_EQ(aclrtQueryEventStatus(producer_ready_->npu_event(), &status),
              ACL_SUCCESS);
    EXPECT_EQ(status, ACL_EVENT_RECORDED_STATUS_COMPLETE);
  }

  const torch::Device device_{torch::kPrivateUse1, 0};
  std::unique_ptr<TokenResultStorage> storage_;
  std::unique_ptr<Stream> producer_;
  std::unique_ptr<Stream> copy_;
  StreamEventPtr producer_ready_;
};

TEST_F(TokenResultStorageTest, CapacityValidationAndByteAccounting) {
  EXPECT_EQ(storage_->pinned_bytes(), 1168);
  EXPECT_EQ(storage_->device_bytes(), 1168);
  const TokenResultStorage* original = storage_.get();
  for (const auto capacity : {TokenResultCapacity{0, 4, 5},
                              TokenResultCapacity{4, 0, 5},
                              TokenResultCapacity{2147483648U, 1, 0},
                              TokenResultCapacity{2147483647, 2147483647, 0},
                              TokenResultCapacity{2147483647, 1, 2147483647}}) {
    EXPECT_FALSE(TokenResultStorage::create(capacity, device_, storage_).ok());
    EXPECT_EQ(storage_.get(), original);
  }
  EXPECT_FALSE(TokenResultStorage::create(
                   {4, 4, 5}, torch::Device(torch::kCPU), storage_)
                   .ok());
  EXPECT_FALSE(TokenResultStorage::create(
                   {4, 4, 5}, torch::Device(torch::kPrivateUse1), storage_)
                   .ok());
  EXPECT_EQ(storage_.get(), original);
  storage_->discard_result();
  storage_->discard_result();
}

TEST_F(TokenResultStorageTest, VariableLengthsHidePaddingAndDetachCpuStorage) {
  ASSERT_TRUE(storage_->bind({3, 4, 3, true}).ok());
  const TokenResultTensors expected = produce({0, 2, 4}, /*seed=*/100);
  ASSERT_TRUE(storage_->copy_to_host(*copy_, producer_ready_).ok());
  EXPECT_EQ(storage_->transfer_info().d2h_calls, 5);
  EXPECT_EQ(storage_->transfer_info().d2h_bytes, 588);
  const TokenResultTensors result = storage_->take_result();
  expect_producer_complete();
  storage_.reset();
  expect_result(expected, result);
}

TEST_F(TokenResultStorageTest, FixedAddressesAcrossShapeAndFeatureChanges) {
  ASSERT_TRUE(storage_->bind({4, 4, 5, true}).ok());
  std::array<const void*, kFields.size()> addresses{};
  for (uint32_t i = 0; i < kFields.size(); ++i) {
    addresses[i] = (storage_->device().*kFields[i]).data_ptr();
  }
  std::vector<TokenResultTensors> results;
  std::vector<TokenResultTensors> expected;
  results.reserve(/*new_cap=*/16);
  expected.reserve(/*new_cap=*/16);
  for (uint32_t round = 0; round < 16; ++round) {
    SCOPED_TRACE(round);
    const uint32_t rows = 1 + round % 4;
    const uint32_t width = 1 + (round / 2) % 4;
    const bool logprobs = round % 3 != 0;
    const uint32_t top = logprobs ? round % 5 : 0;
    ASSERT_TRUE(storage_->bind({rows, width, top, logprobs}).ok());
    for (uint32_t i = 0; i < kFields.size(); ++i) {
      const torch::Tensor& tensor = storage_->device().*kFields[i];
      if (tensor.defined()) {
        EXPECT_EQ(tensor.data_ptr(), addresses[i]);
        EXPECT_TRUE(tensor.is_contiguous());
      }
    }
    std::vector<int32_t> lengths;
    lengths.reserve(rows);
    for (uint32_t row = 0; row < rows; ++row) {
      lengths.emplace_back(static_cast<int32_t>((round + row) % (width + 1)));
    }
    expected.emplace_back(produce(lengths, static_cast<int32_t>(100 * round)));
    ASSERT_TRUE(storage_->copy_to_host(*copy_, producer_ready_).ok());
    const uint64_t bytes =
        rows * sizeof(int32_t) + rows * width * sizeof(int64_t) +
        (logprobs ? rows * width * sizeof(float) : 0) +
        rows * width * top * (sizeof(int64_t) + sizeof(float));
    EXPECT_EQ(storage_->transfer_info().d2h_bytes, bytes);
    results.emplace_back(storage_->take_result());
    expect_result(expected.back(), results.back());
  }
  storage_.reset();
  for (uint32_t index = 0; index < results.size(); ++index) {
    expect_result(expected[index], results[index]);
  }
}

TEST_F(TokenResultStorageTest, PendingResultsRejectOverwriteAndInvalidShape) {
  ASSERT_TRUE(storage_->bind({2, 2, 2, true}).ok());
  const TokenResultTensors expected = produce({1, 2}, /*seed=*/23);
  ASSERT_TRUE(storage_->copy_to_host(*copy_, producer_ready_).ok());
  const TokenResultTransferInfo info = storage_->transfer_info();
  const void* tokens = storage_->device().tokens.data_ptr();
  EXPECT_EQ(storage_->bind({1, 1, 0, false}).code(),
            StatusCode::RESOURCE_EXHAUSTED);
  EXPECT_EQ(storage_->copy_to_host(*copy_, producer_ready_).code(),
            StatusCode::RESOURCE_EXHAUSTED);
  for (const auto shape : {TokenResultShape{5, 1, 0, false},
                           TokenResultShape{1, 5, 0, false},
                           TokenResultShape{1, 1, 6, true},
                           TokenResultShape{1, 1, 1, false},
                           TokenResultShape{0, 1, 0, false},
                           TokenResultShape{1, 0, 0, false},
                           TokenResultShape{0, 0, 0, true}}) {
    EXPECT_EQ(storage_->bind(shape).code(), StatusCode::INVALID_ARGUMENT);
  }
  EXPECT_EQ(storage_->copy_to_host(*copy_, {}).code(),
            StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(storage_->device().tokens.data_ptr(), tokens);
  EXPECT_EQ(storage_->shape().sequences, 2);
  EXPECT_EQ(storage_->shape().tokens_per_sequence, 2);
  EXPECT_EQ(storage_->transfer_info().d2h_bytes, info.d2h_bytes);
  EXPECT_EQ(storage_->transfer_info().d2h_calls, info.d2h_calls);
  expect_result(expected, storage_->take_result());
  ASSERT_TRUE(storage_->bind({1, 1, 0, false}).ok());
  EXPECT_EQ(storage_->copy_to_host(*copy_, {}).code(),
            StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(storage_->transfer_info().d2h_calls, 0);
  const TokenResultTensors next = produce({1}, /*seed=*/321);
  ASSERT_TRUE(storage_->copy_to_host(*copy_, producer_ready_).ok());
  expect_result(next, storage_->take_result());
}

TEST_F(TokenResultStorageTest, ZeroTopCapacityAndOptionalLogprobs) {
  ASSERT_TRUE(TokenResultStorage::create({2, 3, 0}, device_, storage_).ok());
  ASSERT_EQ(aclrtSynchronizeDevice(), ACL_SUCCESS);
  EXPECT_EQ(storage_->pinned_bytes(), 80);
  ASSERT_TRUE(storage_->bind({2, 3, 0, true}).ok());
  const TokenResultTensors expected = produce({3, 1}, /*seed=*/9);
  ASSERT_TRUE(storage_->copy_to_host(*copy_, producer_ready_).ok());
  EXPECT_EQ(storage_->transfer_info().d2h_calls, 3);
  EXPECT_EQ(storage_->transfer_info().d2h_bytes, 80);
  expect_result(expected, storage_->take_result());
  ASSERT_TRUE(storage_->bind({2, 1, 0, false}).ok());
  const TokenResultTensors greedy = produce({1, 1}, /*seed=*/11);
  ASSERT_TRUE(storage_->copy_to_host(*copy_, producer_ready_).ok());
  EXPECT_EQ(storage_->transfer_info().d2h_calls, 2);
  EXPECT_EQ(storage_->transfer_info().d2h_bytes, 24);
  expect_result(greedy, storage_->take_result());
}

TEST_F(TokenResultStorageTest, EmptyResultStillRetiresProducerWork) {
  ASSERT_TRUE(storage_->bind({}).ok());
  auto guard = producer_->set_stream_guard();
  torch::Tensor scratch =
      torch::ones({1024 * 1024}, torch::TensorOptions().device(device_));
  for (uint32_t i = 0; i < 32; ++i) {
    scratch.sin_();
  }
  record_producer();
  ASSERT_TRUE(storage_->copy_to_host(*copy_, producer_ready_).ok());
  EXPECT_EQ(storage_->transfer_info().d2h_calls, 0);
  EXPECT_EQ(storage_->transfer_info().d2h_bytes, 0);
  const TokenResultTensors result = storage_->take_result();
  for (const auto member : kFields) {
    EXPECT_FALSE((result.*member).defined());
    EXPECT_FALSE((storage_->device().*member).defined());
  }
  expect_producer_complete();
}

TEST_F(TokenResultStorageTest, DiscardAndDestructionRetireUnclaimedCopies) {
  for (uint32_t round = 0; round < 4; ++round) {
    ASSERT_TRUE(storage_->bind({4, 4, 5, true}).ok());
    produce({4, 3, 2, 1}, static_cast<int32_t>(round));
    auto guard = producer_->set_stream_guard();
    torch::Tensor scratch =
        torch::ones({1024 * 1024}, torch::TensorOptions().device(device_));
    for (uint32_t i = 0; i < 32; ++i) {
      scratch.sin_();
    }
    record_producer();
    ASSERT_TRUE(storage_->copy_to_host(*copy_, producer_ready_).ok());
    if (round == 3) {
      storage_.reset();
    } else {
      storage_->discard_result();
      storage_->discard_result();
    }
    expect_producer_complete();
  }
}

TEST_F(TokenResultStorageTest, NpuSamplingKernelsWriteFinalDeviceViews) {
  for (const int32_t modulus : {131, 73}) {
    SCOPED_TRACE(modulus);
    ASSERT_TRUE(storage_->bind({3, 1, 5, true}).ok());
    auto guard = producer_->set_stream_guard();
    const auto options =
        torch::TensorOptions().device(device_).dtype(torch::kFloat32);
    torch::Tensor logits =
        (torch::arange(/*end=*/3 * 128, options).remainder(modulus) * 0.03125f -
         1.0f)
            .view({3, 128});
    const torch::Tensor logprobs = torch::log_softmax(logits, /*dim=*/-1);
    torch::Tensor tokens = storage_->device().tokens.view({3});
    torch::Tensor selected = storage_->device().logprobs;
    torch::Tensor top_values = storage_->device().top_logprobs.view({3, 5});
    torch::Tensor top_indices = storage_->device().top_tokens.view({3, 5});
    const void* token_address = tokens.data_ptr();
    const void* top_address = top_values.data_ptr();
    torch::argmax_out(tokens, logits, /*dim=*/-1, /*keepdim=*/false);
    torch::gather_out(selected, logprobs, /*dim=*/1, storage_->device().tokens);
    torch::topk_out(top_values,
                    top_indices,
                    logprobs,
                    /*k=*/5,
                    /*dim=*/-1,
                    /*largest=*/true,
                    /*sorted=*/true);
    storage_->device().lengths.fill_(/*value=*/1);
    EXPECT_EQ(tokens.data_ptr(), token_address);
    EXPECT_EQ(top_values.data_ptr(), top_address);
    record_producer();
    ASSERT_TRUE(storage_->copy_to_host(*copy_, producer_ready_).ok());
    const TokenResultTensors result = storage_->take_result();
    const torch::Tensor reference_tokens =
        logits.argmax(/*dim=*/-1).cpu().view({3, 1});
    const torch::Tensor reference_logprobs = logprobs.cpu();
    const auto [reference_values, reference_indices] =
        reference_logprobs.topk(/*k=*/5, /*dim=*/-1);
    EXPECT_TRUE(torch::equal(result.tokens, reference_tokens));
    EXPECT_TRUE(
        torch::equal(result.logprobs,
                     reference_logprobs.gather(/*dim=*/1, reference_tokens)));
    const torch::Tensor returned_indices = result.top_tokens.view({3, 5});
    ASSERT_GE(returned_indices.min().item<int64_t>(), 0);
    ASSERT_LT(returned_indices.max().item<int64_t>(), 128);
    if (modulus == 131) {
      // No tied logits within a row: exact IDs must match the CPU reference.
      EXPECT_TRUE(torch::equal(returned_indices, reference_indices));
    }
    // Tied scores need not choose the same ID on different backends. Verify
    // score/ID correspondence and distinct candidates as well as top-k scores.
    EXPECT_TRUE(
        torch::equal(reference_logprobs.gather(/*dim=*/1, returned_indices),
                     result.top_logprobs.view({3, 5})));
    for (int32_t row = 0; row < 3; ++row) {
      std::array<int64_t, 5> ids;
      std::copy_n(
          returned_indices[row].data_ptr<int64_t>(), /*count=*/5, ids.begin());
      std::sort(ids.begin(), ids.end());
      EXPECT_EQ(std::adjacent_find(ids.begin(), ids.end()), ids.end());
    }
    EXPECT_TRUE(
        torch::equal(result.top_logprobs.view({3, 5}), reference_values));
    EXPECT_TRUE(torch::equal(result.lengths, torch::ones({3}, torch::kInt32)));
  }
}

class TokenResultStorageDeathTest : public TokenResultStorageTest {};

TEST_F(TokenResultStorageDeathTest, InvalidDeviceLengthsFailFastBeforeCpuCopy) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  for (const int32_t invalid_length : {-1, 3}) {
    ASSERT_TRUE(storage_->bind({2, 2, 1, true}).ok());
    produce({invalid_length, 1}, /*seed=*/7);
    ASSERT_TRUE(storage_->copy_to_host(*copy_, producer_ready_).ok());
    EXPECT_DEATH(storage_->take_result(), "Invalid token result length");
    storage_->discard_result();
  }
}

}  // namespace
}  // namespace xllm
