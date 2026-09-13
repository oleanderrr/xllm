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

#include "core/framework/sampling/prepared_sampler.h"

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch_npu/csrc/aten/NPUGeneratorImpl.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <limits>
#include <mutex>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "core/framework/sampling/sampler.h"
#include "core/runtime/task_pipeline/sampling_input_binding.h"
#include "core/runtime/task_pipeline/token_result_storage.h"

namespace xllm {
namespace {

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

void expect_equal(const torch::Tensor& expected,
                  const torch::Tensor& actual,
                  const char* field) {
  SCOPED_TRACE(field);
  ASSERT_EQ(expected.defined(), actual.defined());
  if (expected.defined()) {
    ASSERT_EQ(expected.scalar_type(), actual.scalar_type());
    ASSERT_EQ(expected.sizes(), actual.sizes());
    const auto expected_cpu = expected.cpu();
    const auto actual_cpu = actual.cpu();
    const bool equal = torch::equal(expected_cpu, actual_cpu);
    if (!equal) {
      const auto mismatch =
          (expected_cpu != actual_cpu).flatten().nonzero().flatten();
      const auto first = mismatch.narrow(
          /*dim=*/0, /*start=*/0, std::min<int64_t>(8, mismatch.numel()));
      LOG(ERROR) << field << " mismatch_count=" << mismatch.numel()
                 << " indices=" << first << " expected="
                 << expected_cpu.flatten().index_select(/*dim=*/0, first)
                 << " actual="
                 << actual_cpu.flatten().index_select(/*dim=*/0, first);
    }
    EXPECT_TRUE(equal);
  }
}

SamplingParameters make_params(int32_t rows, int32_t width, int32_t mode) {
  SamplingParameters p;
  p.selected_token_idxes = torch::arange(rows, torch::kInt32);
  p.sample_idxes =
      (mode == 4 || mode == 8 || mode == 12 || mode == 13)
          ? torch::arange(/*start=*/1, rows, /*step=*/2, torch::kInt32)
          : torch::arange(rows, torch::kInt32);
  const int64_t samples = p.sample_idxes.numel();
  p.do_sample = torch::ones({samples}, torch::kBool);
  p.frequency_penalties = torch::full({rows}, /*fill_value=*/0.15f);
  p.presence_penalties = torch::full({rows}, /*fill_value=*/0.2f);
  p.repetition_penalties = torch::full({rows}, /*fill_value=*/1.15f);
  p.temperatures = torch::full({rows}, /*fill_value=*/0.85f);
  p.temperatures[0] = 0;
  p.top_k = torch::full({rows}, /*fill_value=*/16, torch::kInt64);
  p.top_p = torch::full({rows}, /*fill_value=*/0.9f);
  p.unique_token_ids =
      torch::arange(width, torch::kInt64).expand({rows, width}).clone();
  p.unique_token_counts = torch::arange(rows * width, torch::kInt32)
                              .remainder(/*other=*/5)
                              .view({rows, width});
  p.unique_token_ids_lens = torch::full({rows}, width, torch::kInt32);
  p.unique_token_ids_lens[0] = width - 1;
  p.unique_token_counts[0][width - 1] = 0;
  p.logprobs = true;
  p.max_top_logprobs = 5;
  if (mode <= 2 || mode == 8 || mode == 15) {
    p.do_sample.zero_();
  }
  if (mode == 4 || mode == 13 || mode == 14) {
    p.do_sample = torch::arange(samples).remainder(/*other=*/2) == 0;
  }
  if (mode == 0 || mode == 1 || mode == 7 || mode == 8) {
    p.logprobs = false;
    p.max_top_logprobs = 0;
  }
  p.return_probs = mode == 1;
  if (mode == 0) {
    p.frequency_penalties = torch::Tensor();
    p.presence_penalties = torch::Tensor();
    p.repetition_penalties = torch::Tensor();
    p.temperatures = torch::Tensor();
    p.unique_token_ids = torch::Tensor();
    p.unique_token_counts = torch::Tensor();
    p.unique_token_ids_lens = torch::Tensor();
  }
  if (mode == 0 || mode == 6 || mode == 7 || mode == 10 || mode == 12 ||
      mode == 15) {
    p.top_k = torch::Tensor();
  }
  if (mode == 0 || mode == 5 || mode == 7 || mode == 10 || mode == 13) {
    p.top_p = torch::Tensor();
  }
  if (mode == 5 || mode == 9 || mode == 13) {
    p.top_k = torch::tensor({int64_t{-7},
                             std::numeric_limits<int64_t>::min(),
                             int64_t{0},
                             int64_t{128}},
                            torch::kInt64);
  }
  if (mode == 6 || mode == 12 || mode == 15) {
    p.top_p = torch::tensor({0.0f, 0.01f, 0.8f, 1.0f});
  }
  if (mode == 7) {
    p.temperatures.zero_();
  }
  if (mode == 10) {
    p.max_top_logprobs = 0;
  }
  p.all_random_sample = p.do_sample.all().item<bool>();
  p.all_greedy_sample = !p.do_sample.any().item<bool>();
  return p;
}

SamplingParameters legacy_params(const SamplingParameters& host,
                                 const torch::Device& device,
                                 torch::ScalarType parameter_dtype,
                                 torch::ScalarType logits_dtype) {
  auto params = host.to(device, parameter_dtype);
  if (params.top_k.defined() && params.top_p.defined()) {
    // Legacy's raw ACL wrapper takes a descriptor from top_p.to(logits_dtype)
    // after acquiring its stream. Hold that conversion before calling Sampler
    // so the reference cannot read a temporary after its owner has gone away.
    // The conversion and threshold values are exactly the same as Legacy's;
    // standalone top-p retains parameter dtype. Prepared still receives the
    // original independent parameter dtype through SamplingInputBinding.
    params.top_p = params.top_p.to(logits_dtype);
  }
  return params;
}

SampleOutput final_views(const TokenResultTensors& tensors,
                         int64_t samples,
                         int64_t top) {
  SampleOutput result;
  if (samples == 0) {
    return result;
  }
  result.next_tokens = tensors.tokens.view({samples});
  if (tensors.logprobs.defined()) {
    result.logprobs = tensors.logprobs.view({samples});
  }
  if (top > 0) {
    result.top_tokens = tensors.top_tokens.view({samples, top});
    result.top_logprobs = tensors.top_logprobs.view({samples, top});
  }
  return result;
}

TokenResultTensors cpu_reference(const SampleOutput& output) {
  const int64_t rows = output.next_tokens.numel();
  TokenResultTensors result;
  result.tokens = output.next_tokens.cpu().view({rows, 1});
  result.lengths = torch::ones({rows}, torch::kInt32);
  if (output.logprobs.defined()) {
    result.logprobs = output.logprobs.cpu().view({rows, 1});
  }
  if (output.top_tokens.defined()) {
    const int64_t top = output.top_tokens.size(/*dim=*/-1);
    result.top_tokens = output.top_tokens.cpu().view({rows, 1, top});
    result.top_logprobs = output.top_logprobs.cpu().view({rows, 1, top});
  }
  return result;
}

void expect_result(const TokenResultTensors& expected,
                   const TokenResultTensors& actual) {
  expect_equal(expected.tokens, actual.tokens, "tokens");
  expect_equal(expected.lengths, actual.lengths, "lengths");
  expect_equal(expected.logprobs, actual.logprobs, "logprobs");
  expect_equal(expected.top_tokens, actual.top_tokens, "top_tokens");
  expect_equal(expected.top_logprobs, actual.top_logprobs, "top_logprobs");
  EXPECT_FALSE(actual.tokens.is_pinned());
}

using Dtypes = std::tuple<torch::ScalarType, torch::ScalarType>;
class PreparedSamplerTest : public ::testing::TestWithParam<Dtypes> {
 protected:
  void SetUp() override {
    ASSERT_TRUE(PreparedSampler::create(capacity_,
                                        device_,
                                        std::get<0>(GetParam()),
                                        std::get<1>(GetParam()),
                                        sampler_)
                    .ok());
    for (uint32_t slot = 0; slot < 2; ++slot) {
      ASSERT_TRUE(
          SamplingInputBinding::create(
              {4, 4, 8, 128, 5}, device_, std::get<1>(GetParam()), input_[slot])
              .ok());
      ASSERT_TRUE(
          TokenResultStorage::create({4, 1, 5}, device_, result_[slot]).ok());
      aclrtEvent input = nullptr;
      aclrtEvent output = nullptr;
      ASSERT_EQ(aclrtCreateEventExWithFlag(&input, ACL_EVENT_SYNC),
                ACL_SUCCESS);
      ASSERT_EQ(aclrtCreateEventExWithFlag(&output, ACL_EVENT_SYNC),
                ACL_SUCCESS);
      ready_[slot] = std::make_shared<StreamEvent>(input);
      produced_[slot] = std::make_shared<StreamEvent>(output);
    }
    prepare_ = std::make_unique<Stream>(device_);
    launch_ = std::make_unique<Stream>(device_);
    copy_ = std::make_unique<Stream>(device_);
    ASSERT_EQ(aclrtSynchronizeDevice(), ACL_SUCCESS);
  }
  void TearDown() override { EXPECT_EQ(aclrtSynchronizeDevice(), ACL_SUCCESS); }

  torch::Tensor source(int32_t rows, int32_t seed, bool padded) {
    const int64_t columns = padded ? 136 : 128;
    return ((torch::arange(rows * columns).to(torch::kFloat32) + 17 * seed)
                .remainder(/*other=*/131) -
            65)
        .div(/*other=*/7.31)
        .view({rows, columns})
        .to(std::get<0>(GetParam()))
        .to(device_)
        .narrow(/*dim=*/1, /*start=*/0, 128);
  }

  void bind(uint32_t slot, const SamplingParameters& params) {
    const uint32_t samples = static_cast<uint32_t>(params.sample_idxes.numel());
    ASSERT_TRUE(
        input_[slot]->prepare(params, /*model_tokens=*/4, *prepare_).ok());
    ASSERT_TRUE(result_[slot]
                    ->bind({samples,
                            1,
                            static_cast<uint32_t>(params.max_top_logprobs),
                            params.logprobs})
                    .ok());
    ASSERT_TRUE(sampler_
                    ->bind(input_[slot]->params(),
                           final_views(result_[slot]->device(),
                                       samples,
                                       params.max_top_logprobs),
                           result_[slot]->device().lengths,
                           invocation_[slot])
                    .ok());
    ASSERT_EQ(aclrtRecordEvent(ready_[slot]->npu_event(),
                               prepare_->get_stream()->stream()),
              ACL_SUCCESS);
    ASSERT_TRUE(launch_->wait_event(ready_[slot]));
  }

  void copy_result(uint32_t slot) {
    ASSERT_EQ(aclrtRecordEvent(produced_[slot]->npu_event(),
                               launch_->get_stream()->stream()),
              ACL_SUCCESS);
    ASSERT_TRUE(result_[slot]->copy_to_host(*copy_, produced_[slot]).ok());
  }

  const torch::Device device_{torch::kPrivateUse1, 0};
  const PreparedSamplerCapacity capacity_{{4, 4, 8, 128}, 5, 136};
  std::unique_ptr<PreparedSampler> sampler_;
  std::array<std::unique_ptr<SamplingInputBinding>, 2> input_;
  std::array<std::unique_ptr<TokenResultStorage>, 2> result_;
  std::array<std::unique_ptr<PreparedSamplingInvocation>, 2> invocation_;
  std::array<StreamEventPtr, 2> ready_;
  std::array<StreamEventPtr, 2> produced_;
  std::unique_ptr<Stream> prepare_;
  std::unique_ptr<Stream> launch_;
  std::unique_ptr<Stream> copy_;
};

TEST_P(PreparedSamplerTest, MatchesActualSamplerOutputsAndRngInEveryBranch) {
  for (int32_t mode = 0; mode < 17; ++mode) {
    SCOPED_TRACE(mode);
    const int32_t rows = mode == 11 ? 1 : (mode == 16 ? 3 : 4);
    auto params = make_params(rows, 1 + mode % 7, mode);
    auto reference_params = legacy_params(
        params, device_, std::get<1>(GetParam()), std::get<0>(GetParam()));
    auto expected_logits = source(rows, mode, /*padded=*/mode % 2 == 0);
    auto actual_logits = source(rows, mode, /*padded=*/mode % 2 == 0);
    ASSERT_EQ(aclrtSynchronizeDevice(), ACL_SUCCESS);
    auto guard = launch_->set_stream_guard();
    const auto initial_rng = rng_state();
    const auto expected = Sampler().forward(expected_logits, reference_params);
    const auto expected_rng = rng_state();
    set_rng_state(initial_rng);
    bind(/*slot=*/0, params);
    expect_equal(initial_rng, rng_state(), "prepare/bind RNG");
    if (params.frequency_penalties.defined()) {
      params.frequency_penalties.fill_(/*value=*/99);
    }
    params.selected_token_idxes.zero_();
    const auto& actual = invocation_[0]->run(actual_logits);
    expect_equal(expected_rng, rng_state(), "sampling RNG");
    expect_equal(expected_logits, actual_logits, "processed model logits");
    expect_equal(expected.next_tokens, actual.next_tokens, "Device tokens");
    expect_equal(expected.probs, actual.probs, "Device probabilities");
    expect_equal(expected.logprobs, actual.logprobs, "Device logprobs");
    expect_equal(expected.top_tokens, actual.top_tokens, "Device top tokens");
    expect_equal(
        expected.top_logprobs, actual.top_logprobs, "Device top logprobs");
    EXPECT_EQ(actual.next_tokens.data_ptr(),
              result_[0]->device().tokens.data_ptr());
    copy_result(/*slot=*/0);
    const auto cpu = result_[0]->take_result();
    expect_result(cpu_reference(expected), cpu);
  }
}

TEST_P(PreparedSamplerTest,
       TwoSlotsKeepFinalOutputsAndOldCpuResultsIndependent) {
  std::vector<std::pair<TokenResultTensors, TokenResultTensors>> retained;
  retained.reserve(/*new_cap=*/16);
  std::array<const void*, 2> token_addresses{};
  for (int32_t round = 0; round < 8; ++round) {
    SCOPED_TRACE(round);
    const int32_t rows = round % 2 == 0 ? 2 : 4;
    std::array<SamplingParameters, 2> params;
    std::array<SampleOutput, 2> expected;
    std::array<torch::Tensor, 2> actual_logits;
    std::array<torch::Tensor, 2> expected_logits;
    std::array<torch::Tensor, 2> expected_rng;
    for (uint32_t slot = 0; slot < 2; ++slot) {
      params[slot] = make_params(rows, 1 + round % 7, slot == 0 ? 3 : 4);
      params[slot].max_top_logprobs = round % 6;
      actual_logits[slot] = source(rows, round + slot, /*padded=*/slot == 0);
      expected_logits[slot] = source(rows, round + slot, /*padded=*/slot == 0);
    }
    ASSERT_EQ(aclrtSynchronizeDevice(), ACL_SUCCESS);
    auto guard = launch_->set_stream_guard();
    const auto initial_rng = rng_state();
    for (uint32_t slot = 0; slot < 2; ++slot) {
      const auto reference_params = legacy_params(params[slot],
                                                  device_,
                                                  std::get<1>(GetParam()),
                                                  std::get<0>(GetParam()));
      expected[slot] =
          Sampler().forward(expected_logits[slot], reference_params);
      expected_rng[slot] = rng_state();
    }
    set_rng_state(initial_rng);
    std::array<torch::Tensor, 2> probability_snapshots;
    for (uint32_t slot = 0; slot < 2; ++slot) {
      bind(slot, params[slot]);
    }
    std::thread runner([&] {
      auto runner_guard = launch_->set_stream_guard();
      for (uint32_t slot = 0; slot < 2; ++slot) {
        const auto& actual = invocation_[slot]->run(actual_logits[slot]);
        expect_equal(expected_rng[slot], rng_state(), "queued sampling RNG");
        // Test-only consumer on Launch: complete its read before the next run
        // reuses shared probability scratch. Production has no such staging.
        probability_snapshots[slot] = actual.probs.clone();
        const void* address = actual.next_tokens.data_ptr();
        if (round == 0) {
          token_addresses[slot] = address;
        }
        EXPECT_EQ(address, token_addresses[slot]);
        copy_result(slot);
      }
    });
    runner.join();
    for (uint32_t slot = 0; slot < 2; ++slot) {
      auto cpu = result_[slot]->take_result();
      auto reference = cpu_reference(expected[slot]);
      expect_result(reference, cpu);
      expect_equal(expected[slot].probs,
                   probability_snapshots[slot],
                   "queued probabilities");
      expect_equal(
          expected_logits[slot], actual_logits[slot], "queued model logits");
      retained.emplace_back(std::move(reference), std::move(cpu));
    }
  }
  result_[0].reset();
  result_[1].reset();
  for (const auto& [expected, actual] : retained) {
    expect_result(expected, actual);
  }
}

TEST_P(PreparedSamplerTest,
       RejectsOutputMetadataAndPreservesBindingThenHandlesEmpty) {
  auto params = make_params(/*rows=*/4, /*width=*/5, /*mode=*/3);
  bind(/*slot=*/0, params);
  auto views = final_views(result_[0]->device(), /*samples=*/4, /*top=*/5);
  auto lengths = result_[0]->device().lengths;
  const auto* old = invocation_[0].get();
  const auto reject = [&](SampleOutput bad, torch::Tensor bad_lengths) {
    EXPECT_FALSE(sampler_
                     ->bind(input_[0]->params(),
                            std::move(bad),
                            std::move(bad_lengths),
                            invocation_[0])
                     .ok());
    EXPECT_EQ(old, invocation_[0].get());
  };
  auto bad = views;
  bad.next_tokens = torch::zeros({4}, torch::kInt64);
  reject(bad, lengths);
  bad = views;
  // Correct shape/contiguity, but overlapping output storage must be rejected.
  bad.next_tokens = bad.top_tokens.flatten().narrow(/*dim=*/0, /*start=*/0, 4);
  reject(bad, lengths);
  bad = views;
  bad.next_tokens = bad.top_tokens.select(/*dim=*/1, /*index=*/0);
  reject(bad, lengths);
  bad = views;
  bad.top_tokens = bad.top_tokens.narrow(/*dim=*/1, /*start=*/0, 3);
  reject(bad, lengths);
  bad = views;
  bad.top_logprobs = bad.logprobs.unsqueeze(/*dim=*/1).expand({4, 5});
  reject(bad, lengths);
  bad = views;
  bad.probs = bad.logprobs;
  reject(bad, lengths);
  reject(views, lengths.view({4, 1}));
  auto invalid = input_[0]->params();
  invalid.max_top_logprobs = 6;
  EXPECT_FALSE(
      sampler_->bind(std::move(invalid), views, lengths, invocation_[0]).ok());
  EXPECT_EQ(old, invocation_[0].get());
  const auto reject_input = [&](SamplingParameters bad_params) {
    EXPECT_FALSE(
        sampler_->bind(std::move(bad_params), views, lengths, invocation_[0])
            .ok());
    EXPECT_EQ(old, invocation_[0].get());
  };
  reject_input(params);
  invalid = input_[0]->params();
  invalid.temperatures = params.temperatures;
  reject_input(invalid);
  invalid = input_[0]->params();
  invalid.sample_idxes = invalid.sample_idxes.view({1, -1});
  reject_input(invalid);
  invalid = input_[0]->params();
  invalid.unique_token_counts =
      invalid.unique_token_counts.narrow(/*dim=*/1, /*start=*/0, 2);
  reject_input(invalid);
  invalid = input_[0]->params();
  invalid.presence_penalties = torch::Tensor();
  reject_input(invalid);
  invalid = input_[0]->params();
  invalid.all_random_sample = true;
  invalid.all_greedy_sample = true;
  reject_input(invalid);
  auto guard = launch_->set_stream_guard();
  auto logits = source(/*rows=*/4, /*seed=*/31, /*padded=*/true);
  invocation_[0]->run(logits);
  copy_result(/*slot=*/0);
  result_[0]->take_result();
  ASSERT_TRUE(sampler_
                  ->bind(SamplingParameters(),
                         SampleOutput(),
                         torch::Tensor(),
                         invocation_[0])
                  .ok());
  const auto before_rng = rng_state();
  torch::Tensor empty;
  const auto& output = invocation_[0]->run(empty);
  EXPECT_FALSE(output.next_tokens.defined());
  EXPECT_FALSE(output.probs.defined());
  expect_equal(before_rng, rng_state(), "empty RNG");
}

INSTANTIATE_TEST_SUITE_P(
    IndependentDtypes,
    PreparedSamplerTest,
    ::testing::Combine(
        ::testing::Values(torch::kFloat16, torch::kBFloat16, torch::kFloat32),
        ::testing::Values(torch::kFloat16, torch::kBFloat16, torch::kFloat32)));

TEST(PreparedSamplerCapacityTest, RejectsInvalidCapacityBeforeAllocation) {
  std::unique_ptr<PreparedSampler> output;
  const torch::Device device(torch::kPrivateUse1, 0);
  const PreparedSamplerCapacity valid{{4, 4, 8, 128}, 5, 136};
  EXPECT_FALSE(PreparedSampler::create(
                   {}, device, torch::kFloat32, torch::kFloat32, output)
                   .ok());
  EXPECT_FALSE(PreparedSampler::create(valid,
                                       torch::Device(torch::kCPU),
                                       torch::kFloat32,
                                       torch::kFloat32,
                                       output)
                   .ok());
  EXPECT_FALSE(PreparedSampler::create(valid,
                                       torch::Device(torch::kPrivateUse1),
                                       torch::kFloat32,
                                       torch::kFloat32,
                                       output)
                   .ok());
  EXPECT_FALSE(PreparedSampler::create(
                   valid, device, torch::kFloat64, torch::kFloat32, output)
                   .ok());
  auto bad = valid;
  bad.logits_row_stride = 127;
  EXPECT_FALSE(PreparedSampler::create(
                   bad, device, torch::kFloat32, torch::kFloat32, output)
                   .ok());
  bad = valid;
  bad.max_top_logprobs = 129;
  EXPECT_FALSE(PreparedSampler::create(
                   bad, device, torch::kFloat32, torch::kFloat32, output)
                   .ok());
  constexpr uint32_t kMax = std::numeric_limits<int32_t>::max();
  EXPECT_FALSE(PreparedSampler::create({{kMax, kMax, kMax, kMax}, 5, kMax},
                                       device,
                                       torch::kFloat32,
                                       torch::kFloat32,
                                       output)
                   .ok());
  EXPECT_EQ(output, nullptr);
}

TEST(PreparedSamplerPerformanceTest, AlternatesLegacyAndPreparedSampling) {
  const torch::Device device(torch::kPrivateUse1, 0);
  constexpr int32_t kRows = 4;
  constexpr int32_t kVocab = 151936;
  constexpr int32_t kIterations = 32;
  std::unique_ptr<PreparedSampler> sampler;
  ASSERT_TRUE(PreparedSampler::create({{kRows, kRows, 128, kVocab}, 5, 0},
                                      device,
                                      torch::kFloat32,
                                      torch::kBFloat16,
                                      sampler)
                  .ok());
  std::unique_ptr<TokenResultStorage> storage;
  ASSERT_TRUE(TokenResultStorage::create({kRows, 1, 5}, device, storage).ok());
  Stream launch(device);
  ASSERT_EQ(aclrtSynchronizeDevice(), ACL_SUCCESS);
  auto guard = launch.set_stream_guard();
  for (const bool random : {false, true}) {
    auto host = make_params(kRows, /*width=*/128, random ? 3 : 1);
    host.frequency_penalties.zero_();
    host.presence_penalties.zero_();
    host.repetition_penalties.fill_(/*value=*/1);
    host.temperatures.fill_(/*value=*/1);
    host.top_k.fill_(kVocab);
    host.top_p.fill_(/*value=*/1);
    auto params = host.to(device, torch::kBFloat16);
    const uint32_t top = static_cast<uint32_t>(host.max_top_logprobs);
    ASSERT_TRUE(storage->bind({kRows, 1, top, host.logprobs}).ok());
    std::unique_ptr<PreparedSamplingInvocation> invocation;
    ASSERT_TRUE(sampler
                    ->bind(params,
                           final_views(storage->device(), kRows, top),
                           storage->device().lengths,
                           invocation)
                    .ok());
    auto legacy_logits =
        torch::zeros({kRows, kVocab}, torch::TensorOptions().device(device));
    auto prepared_logits = legacy_logits.clone();
    ASSERT_EQ(aclrtSynchronizeStream(launch.get_stream()->stream()),
              ACL_SUCCESS);
    const auto reference_params =
        legacy_params(host, device, torch::kBFloat16, torch::kFloat32);
    const auto initial_rng = rng_state();
    auto legacy_output = Sampler().forward(legacy_logits, reference_params);
    const auto expected_rng = rng_state();
    set_rng_state(initial_rng);
    const auto& actual = invocation->run(prepared_logits);
    expect_equal(expected_rng, rng_state(), "benchmark RNG");
    expect_equal(
        legacy_output.next_tokens, actual.next_tokens, "benchmark tokens");
    expect_equal(legacy_output.probs, actual.probs, "benchmark probs");
    expect_equal(legacy_output.logprobs, actual.logprobs, "benchmark logprobs");
    const auto stable_logits = legacy_logits.clone();
    for (int32_t warmup = 0; warmup < 16; ++warmup) {
      legacy_output = Sampler().forward(legacy_logits, reference_params);
      invocation->run(prepared_logits);
    }
    // Neutral parameters plus k=V/p=1 keep this repeated-workload fixture
    // stationary. No extra logits reset/D2D is included in timed iterations.
    expect_equal(
        stable_logits, legacy_logits, "benchmark stationary Legacy input");
    expect_equal(
        stable_logits, prepared_logits, "benchmark stationary prepared input");
    ASSERT_EQ(aclrtSynchronizeStream(launch.get_stream()->stream()),
              ACL_SUCCESS);
    for (int32_t pair = 0; pair < 3; ++pair) {
      for (int32_t order = 0; order < 2; ++order) {
        const bool prepared = (pair + order) % 2 == 1;
        for (const bool closed_loop : {false, true}) {
          const auto start = std::chrono::steady_clock::now();
          for (int32_t iteration = 0; iteration < kIterations; ++iteration) {
            if (prepared) {
              invocation->run(prepared_logits);
            } else {
              legacy_output =
                  Sampler().forward(legacy_logits, reference_params);
            }
            if (closed_loop) {
              ASSERT_EQ(aclrtSynchronizeStream(launch.get_stream()->stream()),
                        ACL_SUCCESS);
            }
          }
          const auto submitted = std::chrono::steady_clock::now();
          ASSERT_EQ(aclrtSynchronizeStream(launch.get_stream()->stream()),
                    ACL_SUCCESS);
          const auto completed = std::chrono::steady_clock::now();
          const double submit_us =
              std::chrono::duration<double, std::micro>(submitted - start)
                  .count() /
              kIterations;
          const double complete_us =
              std::chrono::duration<double, std::micro>(completed - start)
                  .count() /
              kIterations;
          LOG(INFO) << "C10B_BENCHMARK random=" << random << " pair=" << pair
                    << " prepared=" << prepared
                    << " closed_loop=" << closed_loop
                    << " iterations=" << kIterations
                    << " submit_us=" << submit_us
                    << " complete_us=" << complete_us
                    << " sampler_bytes=" << sampler->device_bytes();
        }
      }
    }
    expect_equal(
        stable_logits, legacy_logits, "stationary Legacy after timing");
    expect_equal(
        stable_logits, prepared_logits, "stationary prepared after timing");
  }
  ASSERT_EQ(aclrtSynchronizeStream(launch.get_stream()->stream()), ACL_SUCCESS);
}

}  // namespace
}  // namespace xllm
