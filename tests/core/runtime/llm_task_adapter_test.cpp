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

#include "core/runtime/task_pipeline/llm_task_adapter.h"

#include <gtest/gtest.h>

namespace xllm {
namespace {

ForwardInput ordinary_input() {
  ForwardInput input;
  input.sequence_state_keys = {{17, 3}};
  input.token_ids = torch::tensor({11, 12, 13}, torch::kInt32);
  input.positions = torch::tensor({0, 1, 2}, torch::kInt32);
  input.input_params.meta = {BatchForwardType::PREFILL, 1, 0, 3, 3, 42, false};
  auto& host = input.input_params.attention.host;
  host.q_seq_lens = {3};
  host.kv_seq_lens = {3};
  host.q_cu_seq_lens = {3};
  host.block_tables = torch::tensor({{0}}, torch::kInt32);
  input.input_params.attention.device.new_cache_slots =
      torch::tensor({0, 1, 2}, torch::kInt32);
  input.sampling_params.selected_token_idxes =
      torch::tensor({2}, torch::kInt32);
  return input;
}

TEST(LlmTaskAdapterTest, BorrowsOrdinaryBuilderViewsUntilPrepareAck) {
  auto source = ordinary_input();
  source.input_params.embedding.extra_token_ids = {-1};
  source.input_params.embedding.embedding_ids = {-1};
  source.input_params.embedding.linear_state_ids = {-1};
  source.input_params.embedding.linear_state_indices =
      torch::tensor({-1}, torch::kInt32);
  source.input_params.parallel.dp_global_token_nums = {3};
  source.input_params.meta.is_graph_warmup = true;
  LlmTaskInput task;
  ASSERT_TRUE(make_llm_task_input(source, task).ok());
  EXPECT_EQ(task.sequence_state_keys.data(), source.sequence_state_keys.data());
  EXPECT_EQ(task.sequence_state_keys.size(), 1U);
  source.retired_sequence_state_keys = {{16, 2}};
  ASSERT_TRUE(make_llm_task_input(source, task).ok());
  EXPECT_EQ(task.retired_sequence_state_keys.data(),
            source.retired_sequence_state_keys.data());
  EXPECT_EQ(task.batch.num_actual_sequences, 1);
  EXPECT_EQ(task.batch.batch_id, 42);
  EXPECT_FALSE(task.batch.is_graph_warmup);
  EXPECT_TRUE(task.is_warmup);
  EXPECT_EQ(task.model.token_ids.data(),
            source.token_ids.const_data_ptr<int32_t>());
  EXPECT_EQ(task.model.new_cache_slots.data(),
            source.input_params.attention.device.new_cache_slots
                .const_data_ptr<int32_t>());
  EXPECT_EQ(task.model.q_seq_lens.data(),
            source.input_params.attention.host.q_seq_lens.data());
  EXPECT_EQ(task.sampling.selected_token_idxes.data_ptr(),
            source.sampling_params.selected_token_idxes.data_ptr());
  source.input_params.attention.host.new_cache_slots = {0, 1, 2};
  ASSERT_TRUE(make_llm_task_input(source, task).ok());
  EXPECT_EQ(task.model.new_cache_slots.data(),
            source.input_params.attention.host.new_cache_slots.data());
}

TEST(LlmTaskAdapterTest, EmptyInputAndAbsentSamplingAreValid) {
  LlmTaskInput task;
  ASSERT_TRUE(make_llm_task_input(ForwardInput{}, task).ok());
  EXPECT_TRUE(task.model.token_ids.empty());
  EXPECT_EQ(task.batch.num_actual_sequences, 0);
  EXPECT_FALSE(task.sampling.selected_token_idxes.defined());
  ForwardInput retired_only;
  retired_only.retired_sequence_state_keys = {{19, 4}};
  ASSERT_TRUE(make_llm_task_input(retired_only, task).ok());
  EXPECT_TRUE(task.sequence_state_keys.empty());
  ASSERT_EQ(task.retired_sequence_state_keys.size(), 1U);
  auto source = ordinary_input();
  source.sampling_params = {};
  ASSERT_TRUE(make_llm_task_input(source, task).ok());
  EXPECT_EQ(task.model.token_ids.size(), 3);
  EXPECT_FALSE(task.sampling.selected_token_idxes.defined());
}

TEST(LlmTaskAdapterTest,
     RejectsInvalidTransportAndAlgorithmBeforeReplacingView) {
  auto source = ordinary_input();
  LlmTaskInput task;
  ASSERT_TRUE(make_llm_task_input(source, task).ok());
  const int32_t* original = task.model.token_ids.data();
  const auto rejected = [&](ForwardInput invalid) {
    EXPECT_FALSE(make_llm_task_input(invalid, task).ok());
    EXPECT_EQ(task.model.token_ids.data(), original);
  };
  auto invalid = ordinary_input();
  invalid.sequence_state_keys.clear();
  rejected(invalid);
  invalid = ordinary_input();
  invalid.sequence_state_keys.emplace_back(SequenceStateKey{18, 3});
  rejected(invalid);
  invalid = ordinary_input();
  invalid.token_ids = invalid.token_ids.to(torch::kInt64);
  rejected(invalid);
  invalid = ordinary_input();
  invalid.positions = torch::zeros({6}, torch::kInt32)
                          .slice(/*dim=*/0, /*start=*/0, /*end=*/6, /*step=*/2);
  rejected(invalid);
  invalid = ordinary_input();
  invalid.input_params.attention.host.block_tables =
      torch::zeros({2, 1}, torch::kInt32);
  rejected(invalid);
  invalid = ordinary_input();
  invalid.input_params.meta.actual_num_sequences = 2;
  rejected(invalid);
  invalid = ordinary_input();
  invalid.input_host_buffer_has_layout = true;
  rejected(invalid);
  invalid = ordinary_input();
  invalid.device_tensors_ready = true;
  rejected(invalid);
  invalid = ordinary_input();
  invalid.input_params.is_spec_verify = true;
  rejected(invalid);
  invalid = ordinary_input();
  invalid.input_params.block_copy.src_block_indices =
      torch::tensor({0}, torch::kInt32);
  rejected(invalid);
  invalid = ordinary_input();
  invalid.input_params.embedding.linear_state_ids = {0};
  rejected(invalid);
  invalid = ordinary_input();
  invalid.input_params.embedding.linear_state_indices =
      torch::tensor({0}, torch::kInt32);
  rejected(invalid);
  invalid = ordinary_input();
  invalid.input_params.parallel.dp_global_token_nums = {3, 3};
  rejected(invalid);
  invalid = ordinary_input();
  invalid.skip_sampling_for_logits_only = true;
  rejected(invalid);
}

TEST(LlmTaskAdapterTest, PublishesCpuViewsWithNoCopyAndKeepsOwnership) {
  TokenResultTensors tokens;
  tokens.tokens = torch::tensor({{11}, {12}}, torch::kInt64);
  tokens.lengths = torch::tensor({1, 1}, torch::kInt32);
  tokens.logprobs = torch::tensor({{-0.2f}, {-0.3f}}, torch::kFloat32);
  tokens.top_tokens = torch::tensor({{{11, 21}}, {{12, 22}}}, torch::kInt64);
  tokens.top_logprobs =
      torch::tensor({{{-0.2f, -0.5f}}, {{-0.3f, -0.6f}}}, torch::kFloat32);
  const void* original = tokens.tokens.data_ptr();
  auto mask = torch::tensor({false, true}, torch::kBool);
  const void* mask_address = mask.data_ptr();
  auto output =
      make_llm_task_output({std::move(tokens), std::move(mask), true});
  EXPECT_EQ(output.do_sample.data_ptr(), mask_address);
  EXPECT_TRUE(output.is_graph_warmup);
  EXPECT_TRUE(output.cpu_ready);
  EXPECT_EQ(output.ready_event, nullptr);
  EXPECT_EQ(output.sample_output.next_tokens.data_ptr(), original);
  EXPECT_EQ(output.sample_output.next_tokens.sizes(), (torch::IntArrayRef{2}));
  EXPECT_EQ(output.sample_output.top_tokens.sizes(),
            (torch::IntArrayRef{2, 2}));
  EXPECT_TRUE(output.logprobs);
  EXPECT_EQ(output.max_top_logprobs, 2);
  EXPECT_EQ(output.sample_output.next_tokens[1].item<int64_t>(), 12);
  auto empty = make_llm_task_output({{}, {}, true});
  EXPECT_TRUE(empty.is_graph_warmup);
  EXPECT_FALSE(empty.do_sample.defined());
  EXPECT_TRUE(empty.cpu_ready);
  EXPECT_FALSE(empty.sample_output.next_tokens.defined());
}

}  // namespace
}  // namespace xllm
