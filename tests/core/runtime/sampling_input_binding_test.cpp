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

#include "core/runtime/task_pipeline/sampling_input_binding.h"

#include <gtest/gtest.h>
#include <torch_npu/csrc/aten/NPUGeneratorImpl.h>

#include <array>
#include <limits>
#include <mutex>

#include "core/framework/sampling/sampler.h"

namespace xllm {
namespace {

constexpr std::array<torch::Tensor SamplingParameters::*, 12> kInputs = {
    &SamplingParameters::selected_token_idxes,
    &SamplingParameters::frequency_penalties,
    &SamplingParameters::presence_penalties,
    &SamplingParameters::repetition_penalties,
    &SamplingParameters::temperatures,
    &SamplingParameters::top_p,
    &SamplingParameters::top_k,
    &SamplingParameters::unique_token_ids,
    &SamplingParameters::unique_token_counts,
    &SamplingParameters::unique_token_ids_lens,
    &SamplingParameters::sample_idxes,
    &SamplingParameters::do_sample};

SamplingParameters make_input(int32_t rows, int32_t width) {
  SamplingParameters input;
  input.selected_token_idxes = torch::arange(rows, torch::kInt32);
  input.sample_idxes = torch::arange(rows, torch::kInt32);
  input.do_sample =
      torch::arange(rows, torch::kInt32).remainder(/*other=*/2) == 0;
  input.frequency_penalties = torch::full({rows}, /*fill_value=*/0.15f);
  input.presence_penalties = torch::full({rows}, /*fill_value=*/0.2f);
  input.repetition_penalties = torch::full({rows}, /*fill_value=*/1.15f);
  input.temperatures = torch::full({rows}, /*fill_value=*/0.85f);
  input.top_p = torch::full({rows}, /*fill_value=*/0.9f);
  input.top_k = torch::full({rows}, /*fill_value=*/16, torch::kInt64);
  input.unique_token_ids =
      torch::arange(width, torch::kInt64).expand({rows, width}).clone();
  input.unique_token_counts =
      torch::full({rows, width}, /*fill_value=*/2, torch::kInt32);
  input.unique_token_ids_lens = torch::full({rows}, width, torch::kInt32);
  input.unique_token_ids_lens[0] = width - 1;
  input.unique_token_ids[0][width - 1] = 0;
  input.unique_token_counts[0][width - 1] = 0;
  input.all_random_sample = input.do_sample.all().item<bool>();
  input.all_greedy_sample = !input.do_sample.any().item<bool>();
  input.logprobs = true;
  input.max_top_logprobs = 5;
  return input;
}

torch::Tensor rng_state() {
  auto generator = at_npu::detail::getDefaultNPUGenerator(/*device_index=*/0);
  std::lock_guard<std::mutex> lock(generator.mutex());
  return generator.get_state();
}

void set_rng_state(const torch::Tensor& state) {
  auto generator = at_npu::detail::getDefaultNPUGenerator(/*device_index=*/0);
  std::lock_guard<std::mutex> lock(generator.mutex());
  generator.set_state(state);
}

void expect_equal(const torch::Tensor& expected, const torch::Tensor& actual) {
  ASSERT_EQ(expected.defined(), actual.defined());
  if (expected.defined()) {
    EXPECT_EQ(expected.scalar_type(), actual.scalar_type());
    EXPECT_EQ(expected.sizes(), actual.sizes());
    EXPECT_TRUE(torch::equal(expected.cpu(), actual.cpu()));
  }
}

class SamplingInputBindingTest
    : public ::testing::TestWithParam<torch::ScalarType> {
 protected:
  void SetUp() override {
    ASSERT_TRUE(
        SamplingInputBinding::create(capacity_, device_, GetParam(), binding_)
            .ok());
    prepare_ = std::make_unique<Stream>(device_);
    launch_ = std::make_unique<Stream>(device_);
    aclrtEvent event = nullptr;
    ASSERT_EQ(aclrtCreateEventExWithFlag(&event, ACL_EVENT_SYNC), ACL_SUCCESS);
    ready_ = std::make_shared<StreamEvent>(event);
    ASSERT_EQ(aclrtSynchronizeDevice(), ACL_SUCCESS);
  }

  void TearDown() override { EXPECT_EQ(aclrtSynchronizeDevice(), ACL_SUCCESS); }

  void handoff() {
    ASSERT_EQ(
        aclrtRecordEvent(ready_->npu_event(), prepare_->get_stream()->stream()),
        ACL_SUCCESS);
    ASSERT_TRUE(launch_->wait_event(ready_));
  }

  void expect_inputs(const SamplingParameters& reference) {
    auto guard = launch_->set_stream_guard();
    for (const auto member : kInputs) {
      expect_equal(reference.*member, binding_->params().*member);
    }
    EXPECT_EQ(reference.all_greedy_sample,
              binding_->params().all_greedy_sample);
    EXPECT_EQ(reference.all_random_sample,
              binding_->params().all_random_sample);
    EXPECT_EQ(reference.logprobs, binding_->params().logprobs);
    EXPECT_EQ(reference.return_probs, binding_->params().return_probs);
    EXPECT_EQ(reference.max_top_logprobs, binding_->params().max_top_logprobs);
    ASSERT_EQ(aclrtSynchronizeStream(launch_->get_stream()->stream()),
              ACL_SUCCESS);
  }

  void expect_rejected(const SamplingParameters& input, uint32_t tokens = 8) {
    const SamplingInputTransferInfo before = binding_->transfer_info();
    const SamplingParameters old = binding_->params();
    std::array<torch::Tensor, kInputs.size()> values;
    auto guard = launch_->set_stream_guard();
    for (uint32_t index = 0; index < kInputs.size(); ++index) {
      const torch::Tensor& tensor = old.*kInputs[index];
      values[index] = tensor.defined() ? tensor.cpu() : torch::Tensor();
    }
    EXPECT_FALSE(binding_->prepare(input, tokens, *prepare_).ok());
    EXPECT_EQ(before.h2d_bytes, binding_->transfer_info().h2d_bytes);
    EXPECT_EQ(before.h2d_calls, binding_->transfer_info().h2d_calls);
    for (uint32_t index = 0; index < kInputs.size(); ++index) {
      const torch::Tensor& tensor = binding_->params().*kInputs[index];
      expect_equal(values[index], tensor);
      if (tensor.defined()) {
        EXPECT_EQ(tensor.data_ptr(), (old.*kInputs[index]).data_ptr());
      }
    }
  }

  const torch::Device device_{torch::kPrivateUse1, 0};
  const SamplingInputCapacity capacity_{8, 8, 16, 128, 8};
  std::unique_ptr<SamplingInputBinding> binding_;
  std::unique_ptr<Stream> prepare_;
  std::unique_ptr<Stream> launch_;
  std::shared_ptr<StreamEvent> ready_;
};

TEST_P(SamplingInputBindingTest, FixedAddressesSurviveShapeAndFeatureChanges) {
  std::array<const void*, kInputs.size()> addresses{};
  for (int32_t round = 0; round < 16; ++round) {
    SCOPED_TRACE(round);
    SamplingParameters input = make_input(1 + round % 4, 1 + round % 7);
    if (round == 15) {
      input.top_k.fill_(/*value=*/-7);
    }
    const SamplingParameters reference = input.to(device_, GetParam());
    ASSERT_EQ(aclrtSynchronizeDevice(), ACL_SUCCESS);
    input.all_random_sample = !input.all_random_sample;
    input.all_greedy_sample = !input.all_greedy_sample;
    ASSERT_TRUE(binding_->prepare(input, /*model_tokens=*/8, *prepare_).ok());
    uint64_t bytes = 0;
    for (uint32_t index = 0; index < kInputs.size(); ++index) {
      const torch::Tensor& tensor = binding_->params().*kInputs[index];
      if (round == 0) {
        addresses[index] = tensor.data_ptr();
      }
      EXPECT_EQ(addresses[index], tensor.data_ptr());
      EXPECT_TRUE(tensor.is_contiguous());
      bytes += tensor.nbytes();
      (input.*kInputs[index]).zero_();
    }
    EXPECT_EQ(binding_->transfer_info().h2d_calls, 12);
    EXPECT_EQ(binding_->transfer_info().h2d_bytes, bytes);
    handoff();
    expect_inputs(reference);

    SamplingParameters greedy;
    greedy.selected_token_idxes = torch::tensor({0}, torch::kInt32);
    greedy.sample_idxes = torch::tensor({0}, torch::kInt32);
    greedy.do_sample = torch::tensor({false}, torch::kBool);
    ASSERT_TRUE(binding_->prepare(greedy, /*model_tokens=*/8, *prepare_).ok());
    handoff();
    expect_inputs(greedy.to(device_, GetParam()));
    EXPECT_EQ(binding_->transfer_info().h2d_calls, 3);
    EXPECT_EQ(binding_->transfer_info().h2d_bytes, 9);
  }
}

TEST_P(SamplingInputBindingTest, InvalidInputPreservesPriorBindingAndValues) {
  SamplingParameters valid = make_input(/*rows=*/4, /*width=*/6);
  ASSERT_TRUE(binding_->prepare(valid, /*model_tokens=*/8, *prepare_).ok());
  handoff();
  expect_inputs(valid.to(device_, GetParam()));

  auto invalid = valid;
  invalid.selected_token_idxes = torch::tensor({0, 1, -1, 3}, torch::kInt32);
  expect_rejected(invalid);
  invalid.selected_token_idxes = torch::tensor({0, 1, 8, 3}, torch::kInt32);
  expect_rejected(invalid);
  invalid = valid;
  invalid.sample_idxes = torch::tensor({0, 2, 1, 3}, torch::kInt32);
  expect_rejected(invalid);
  invalid.sample_idxes = torch::tensor({0, 1, 1, 3}, torch::kInt32);
  expect_rejected(invalid);
  invalid = valid;
  invalid.temperatures = torch::tensor({1.0f, 1.0f, -1.0f, 1.0f});
  expect_rejected(invalid);
  invalid.temperatures =
      torch::full({4}, std::numeric_limits<float>::quiet_NaN());
  expect_rejected(invalid);
  invalid = valid;
  invalid.top_p = torch::full({4}, /*fill_value=*/1.1f);
  expect_rejected(invalid);
  invalid = valid;
  invalid.unique_token_ids =
      torch::full({4, 6}, /*fill_value=*/128, torch::kInt64);
  expect_rejected(invalid);
  invalid = valid;
  invalid.unique_token_counts =
      torch::full({4, 5}, /*fill_value=*/1, torch::kInt32);
  expect_rejected(invalid);
  invalid = valid;
  invalid.unique_token_ids_lens =
      torch::full({4}, /*fill_value=*/7, torch::kInt32);
  expect_rejected(invalid);
  invalid = valid;
  invalid.presence_penalties = torch::Tensor();
  expect_rejected(invalid);
  invalid = valid;
  invalid.do_sample = torch::Tensor();
  expect_rejected(invalid);
  invalid = valid;
  invalid.top_k = torch::zeros({4}, torch::kInt32);
  expect_rejected(invalid);
  invalid = valid;
  invalid.frequency_penalties =
      torch::zeros({4, 2}).select(/*dim=*/1, /*index=*/0);
  expect_rejected(invalid);
  invalid = valid;
  invalid.max_top_logprobs = 9;
  expect_rejected(invalid);
  invalid = valid;
  invalid.filter_bitmask = torch::zeros({4, 4}, torch::kInt32);
  expect_rejected(invalid);
  invalid = valid;
  invalid.use_beam_search = true;
  expect_rejected(invalid);
  invalid = valid;
  invalid.is_embeddings = true;
  expect_rejected(invalid);
  expect_rejected(valid.to(device_, GetParam()));
  expect_rejected(valid, /*tokens=*/0);
  expect_rejected(make_input(/*rows=*/9, /*width=*/3));
  expect_rejected(make_input(/*rows=*/4, /*width=*/17));
}

TEST_P(SamplingInputBindingTest, EmptyClearsBindingAndTransfersNothing) {
  auto input = make_input(/*rows=*/4, /*width=*/6);
  ASSERT_TRUE(binding_->prepare(input, /*model_tokens=*/8, *prepare_).ok());
  handoff();
  expect_inputs(input.to(device_, GetParam()));
  ASSERT_TRUE(binding_->prepare({}, /*model_tokens=*/0, *prepare_).ok());
  EXPECT_EQ(binding_->transfer_info().h2d_calls, 0);
  EXPECT_EQ(binding_->transfer_info().h2d_bytes, 0);
  for (const auto member : kInputs) {
    EXPECT_FALSE((binding_->params().*member).defined());
  }
  EXPECT_FALSE(binding_->params().logprobs);
  EXPECT_FALSE(binding_->params().return_probs);
  EXPECT_EQ(binding_->params().max_top_logprobs, 0);
}

TEST_P(SamplingInputBindingTest,
       CapacityRejectsBeforeReplacingOwnerAndAccountsBytes) {
  const uint64_t float_bytes = GetParam() == torch::kFloat32 ? 4 : 2;
  const uint64_t expected_bytes =
      8 * (16 + 5 * float_bytes) + 8 * 16 * 12 + 8 * 5;
  EXPECT_EQ(binding_->pinned_bytes(), expected_bytes);
  EXPECT_EQ(binding_->device_bytes(), expected_bytes);
  const SamplingInputBinding* original = binding_.get();
  for (const auto capacity :
       {SamplingInputCapacity{0, 8, 16, 128, 8},
        SamplingInputCapacity{8, 9, 16, 128, 8},
        SamplingInputCapacity{8, 8, 0, 128, 8},
        SamplingInputCapacity{8, 8, 16, 128, 129},
        SamplingInputCapacity{2147483647, 1, 2147483647, 128, 8}}) {
    EXPECT_FALSE(
        SamplingInputBinding::create(capacity, device_, GetParam(), binding_)
            .ok());
    EXPECT_EQ(binding_.get(), original);
  }
  EXPECT_FALSE(SamplingInputBinding::create(
                   capacity_, torch::Device(torch::kCPU), GetParam(), binding_)
                   .ok());
  EXPECT_FALSE(
      SamplingInputBinding::create(
          capacity_, torch::Device(torch::kPrivateUse1), GetParam(), binding_)
          .ok());
  EXPECT_FALSE(SamplingInputBinding::create(
                   capacity_, device_, torch::kFloat64, binding_)
                   .ok());
  EXPECT_EQ(binding_.get(), original);
}

TEST_P(SamplingInputBindingTest,
       RealSamplerMatchesTokensProbabilitiesAndRngProgress) {
  const torch::Tensor original_rng = rng_state();
  for (int32_t mode = 0; mode < 7; ++mode) {
    SCOPED_TRACE(mode);
    SamplingParameters input = make_input(/*rows=*/4, /*width=*/6);
    if (mode == 0 || mode == 5) {
      input = SamplingParameters();
      input.selected_token_idxes = torch::arange(/*end=*/4, torch::kInt32);
      input.sample_idxes = torch::arange(/*end=*/4, torch::kInt32);
      input.do_sample = torch::zeros({4}, torch::kBool);
      input.return_probs = mode == 5;
    } else if (mode == 1) {
      input.do_sample.zero_();
    } else if (mode == 6) {
      input.do_sample.fill_(/*value=*/true);
      input.top_k = torch::Tensor();
      input.top_p.zero_();
    } else if (mode == 3) {
      input.sample_idxes = torch::tensor({1, 3}, torch::kInt32);
      input.do_sample = torch::tensor({false, true}, torch::kBool);
      input.top_k = torch::Tensor();
    } else {
      input.do_sample.fill_(/*value=*/true);
      if (mode == 4) {
        input.top_p = torch::Tensor();
      }
    }
    input.all_random_sample = input.do_sample.all().item<bool>();
    input.all_greedy_sample = !input.do_sample.any().item<bool>();
    const SamplingParameters legacy = input.to(device_, GetParam());
    const torch::Tensor initial_logits =
        (torch::arange(/*end=*/4 * 128, torch::kFloat32)
                 .remainder(/*other=*/73) *
             0.037f -
         1.4f)
            .view({4, 128})
            .to(device_, GetParam());
    ASSERT_EQ(aclrtSynchronizeDevice(), ACL_SUCCESS);
    auto guard = launch_->set_stream_guard();
    torch::Tensor old_logits = initial_logits.clone();
    const torch::Tensor start_rng = rng_state();
    const SampleOutput expected = Sampler().forward(old_logits, legacy);
    ASSERT_EQ(aclrtSynchronizeStream(launch_->get_stream()->stream()),
              ACL_SUCCESS);
    const torch::Tensor end_rng = rng_state();
    set_rng_state(start_rng);
    ASSERT_TRUE(binding_->prepare(input, /*model_tokens=*/8, *prepare_).ok());
    EXPECT_TRUE(torch::equal(start_rng, rng_state()));
    handoff();
    torch::Tensor new_logits = initial_logits.clone();
    const SampleOutput actual =
        Sampler().forward(new_logits, binding_->params());
    ASSERT_EQ(aclrtSynchronizeStream(launch_->get_stream()->stream()),
              ACL_SUCCESS);
    EXPECT_TRUE(torch::equal(end_rng, rng_state()));
    expect_equal(old_logits, new_logits);
    expect_equal(expected.next_tokens, actual.next_tokens);
    expect_equal(expected.probs, actual.probs);
    expect_equal(expected.logprobs, actual.logprobs);
    expect_equal(expected.top_logprobs, actual.top_logprobs);
    expect_equal(expected.top_tokens, actual.top_tokens);
  }
  set_rng_state(original_rng);
}

INSTANTIATE_TEST_SUITE_P(ParameterTypes,
                         SamplingInputBindingTest,
                         ::testing::Values(torch::kFloat16,
                                           torch::kBFloat16,
                                           torch::kFloat32));

}  // namespace
}  // namespace xllm
