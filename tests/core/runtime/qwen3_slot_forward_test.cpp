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

#include <acl/acl.h>
#include <gtest/gtest.h>
#include <torch_npu/csrc/aten/NPUGeneratorImpl.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <mutex>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "core/framework/config/execution_config.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/model_config.h"
#include "core/framework/config/scheduler_config.h"
#include "core/framework/model/causal_lm.h"
#include "core/framework/model_context.h"
#include "core/framework/parallel_state/process_group.h"
#include "core/framework/sampling/sampler.h"
#include "core/framework/tokenizer/tokenizer.h"
#include "core/runtime/executor.h"
#include "core/runtime/task_pipeline/model_input_binding.h"
#include "core/runtime/task_pipeline/task_execution_pipeline.h"
#include "models/model_registry.h"

namespace xllm {
namespace {

torch::Tensor pipeline_rng_state() {
  auto generator = at_npu::detail::getDefaultNPUGenerator(/*device_index=*/0);
  std::lock_guard<std::mutex> lock(generator.mutex());
  return generator.get_state();
}

void restore_pipeline_rng(const torch::Tensor& state) {
  auto generator = at_npu::detail::getDefaultNPUGenerator(/*device_index=*/0);
  std::lock_guard<std::mutex> lock(generator.mutex());
  generator.set_state(state);
}

struct BatchData {
  std::vector<int32_t> tokens;
  std::vector<int32_t> positions;
  std::vector<int32_t> slots;
  std::vector<int32_t> q;
  std::vector<int32_t> kv;
  std::vector<int32_t> cumulative;
  std::vector<int32_t> blocks;
};

BatchData make_batch(std::vector<int32_t> q, std::vector<int32_t> kv) {
  BatchData batch;
  batch.q = std::move(q);
  batch.kv = std::move(kv);
  batch.cumulative.resize(batch.q.size());
  std::partial_sum(batch.q.begin(), batch.q.end(), batch.cumulative.begin());
  const int32_t tokens = batch.cumulative.empty() ? 0 : batch.cumulative.back();
  batch.tokens.reserve(tokens);
  batch.positions.reserve(tokens);
  batch.slots.reserve(tokens);
  batch.blocks.reserve(batch.q.size());
  for (uint32_t row = 0; row < batch.q.size(); ++row) {
    batch.blocks.emplace_back(row);
    const int32_t past = batch.kv[row] - batch.q[row];
    for (int32_t index = 0; index < batch.q[row]; ++index) {
      const int32_t position = past + index;
      batch.positions.emplace_back(position);
      batch.tokens.emplace_back(100 + 17 * row + position);
      batch.slots.emplace_back(row * 128 + position);
    }
  }
  return batch;
}

ModelInputHostView view(const BatchData& batch) {
  return {batch.tokens,
          batch.positions,
          batch.slots,
          batch.q,
          batch.kv,
          batch.cumulative,
          batch.blocks,
          batch.q.empty()
              ? 0U
              : static_cast<uint32_t>(batch.blocks.size() / batch.q.size())};
}

class Qwen3SlotForwardTest : public ::testing::TestWithParam<bool> {
 protected:
  void SetUp() override {
    const char* model_path = std::getenv("XLLM_QWEN3_TEST_MODEL");
    if (model_path == nullptr) {
      GTEST_SKIP() << "Set XLLM_QWEN3_TEST_MODEL to Qwen3-0.6B weights.";
    }
    old_model_impl_ = ModelConfig::get_instance().model_impl();
    ModelConfig::get_instance().model_impl("python");
    previous_threads_ = torch::get_num_threads();
    torch::set_num_threads(/*num_threads=*/1);
    auto& scheduler = SchedulerConfig::get_instance();
    old_chunked_ = scheduler.enable_chunked_prefill();
    old_max_tokens_ = scheduler.max_tokens_per_batch();
    old_graph_ = ExecutionConfig::get_instance().enable_graph();
    old_prefix_ = KVCacheConfig::get_instance().enable_prefix_cache();
    old_block_size_ = KVCacheConfig::get_instance().block_size();
    scheduler.enable_chunked_prefill(GetParam());
    scheduler.max_tokens_per_batch(32);
    ExecutionConfig::get_instance().enable_graph(false);
    KVCacheConfig::get_instance().enable_prefix_cache(false);
    KVCacheConfig::get_instance().block_size(128);

    auto loader = ModelLoader::create(model_path);
    ASSERT_NE(loader, nullptr);
    tokenizer_ = loader->tokenizer();
    ASSERT_NE(tokenizer_, nullptr);
    args_ = loader->model_args();
    ASSERT_EQ(args_.model_type(), "qwen3");
    args_.dtype("bfloat16");
    process_group_ =
        std::make_unique<ProcessGroup>(/*rank=*/0, /*world_size=*/1, device_);
    ParallelArgs parallel(/*rank=*/0, /*world_size=*/1, process_group_.get());
    parallel.moe_tp_group_ = process_group_.get();
    parallel.python_rendezvous_host_ = "127.0.0.1";
    // Single-rank Python creates no network communicator.
    parallel.python_rendezvous_port_ = 29500;
    context_ = std::make_unique<ModelContext>(
        parallel, args_, loader->quant_args(), options_);
    context_->set_model_impl("python");
    model_ = create_llm_model(*context_);
    ASSERT_NE(model_, nullptr);
    model_->load_model(std::move(loader));
    runtime::Options execution;
    execution.max_seqs_per_batch(3);
    execution.max_tokens_per_batch(32);
    execution.enable_graph(false);
    execution.enable_chunked_prefill(GetParam());
    eager_executor_ =
        std::make_unique<Executor>(model_.get(), args_, device_, execution);
    slot_executor_ =
        std::make_unique<Executor>(model_.get(), args_, device_, execution);
    for (uint32_t slot = 0; slot < bindings_.size(); ++slot) {
      ASSERT_TRUE(
          ModelInputStorage::create({32, 3, 2}, device_, storage_[slot]).ok());
      bindings_[slot] = std::make_unique<ModelInputBinding>(*storage_[slot]);
    }
    prepare_ = std::make_unique<Stream>(device_);
    launch_ = std::make_unique<Stream>(device_);
    KVCacheCapacity capacity;
    capacity.n_blocks(4);
    capacity.block_size(128);
    const KVCacheShape shape(capacity, args_, /*world_size=*/1);
    KVCacheCreateOptions cache_options;
    cache_options.device(device_);
    cache_options.dtype(torch::kBFloat16);
    cache_options.num_layers(args_.n_layers());
    cache_options.model_type(args_.model_type());
    eager_kv_.reserve(args_.n_layers());
    prepared_kv_.reserve(args_.n_layers());
    for (int32_t layer = 0; layer < args_.n_layers(); ++layer) {
      eager_kv_.emplace_back(shape, cache_options, layer);
      prepared_kv_.emplace_back(shape, cache_options, layer);
      eager_kv_.back().get_k_cache().zero_();
      eager_kv_.back().get_v_cache().zero_();
      prepared_kv_.back().get_k_cache().zero_();
      prepared_kv_.back().get_v_cache().zero_();
    }
    ASSERT_EQ(aclrtSynchronizeDevice(), ACL_SUCCESS);
  }

  void TearDown() override {
    if (previous_threads_ == 0) {
      return;
    }
    EXPECT_EQ(aclrtSynchronizeDevice(), ACL_SUCCESS);
    slot_executor_.reset();
    eager_executor_.reset();
    model_.reset();
    auto& scheduler = SchedulerConfig::get_instance();
    scheduler.enable_chunked_prefill(old_chunked_);
    scheduler.max_tokens_per_batch(old_max_tokens_);
    ExecutionConfig::get_instance().enable_graph(old_graph_);
    KVCacheConfig::get_instance().enable_prefix_cache(old_prefix_);
    KVCacheConfig::get_instance().block_size(old_block_size_);
    ModelConfig::get_instance().model_impl(old_model_impl_);
    torch::set_num_threads(previous_threads_);
  }

  void compare(BatchData batch,
               BatchForwardType phase,
               uint32_t slot,
               int32_t actual = -1) {
    auto& binding = *bindings_[slot];
    if (actual < 0) {
      actual = batch.q.size();
    }
    ASSERT_TRUE(
        binding
            .prepare(view(batch),
                     {phase, static_cast<uint32_t>(actual), batch_id_++, false},
                     *prepare_)
            .ok());
    slot_executor_->prepare_attention_metadata(prepared_kv_, binding.params());
    std::fill(batch.positions.begin(), batch.positions.end(), -99);
    const StreamEventPtr ready = prepare_->record_event();
    ASSERT_NE(ready, nullptr);
    // The reference uses the original Python Executor metadata construction
    // and an independent KV cache. Slot input storage is shared read-only.
    Stream original(c10_npu::getCurrentNPUStream(device_.index()));
    ASSERT_TRUE(original.wait_event(ready));
    ModelInputParams reference = binding.params();
    reference.attn_metadata.reset();
    reference.python_attention_metadata.reset();
    ModelOutput eager = eager_executor_->forward(
        binding.tokens(), binding.positions(), eager_kv_, reference);
    std::vector<int32_t> selected;
    selected.reserve(batch.q.size());
    for (uint32_t row = 0; row < batch.q.size(); ++row) {
      if (batch.q[row] != 0) {
        selected.emplace_back(batch.cumulative[row] - 1);
      }
    }
    torch::Tensor indices = torch::tensor(selected, torch::kInt32).to(device_);
    torch::Tensor eager_logits =
        model_->logits(eager.hidden_states, indices).cpu();
    torch::Tensor eager_hidden = eager.hidden_states.cpu();
    ASSERT_EQ(original.synchronize(), 0);
    ASSERT_TRUE(launch_->wait_event(ready));
    {
      auto guard = launch_->set_stream_guard();
      ModelOutput prepared = slot_executor_->forward(binding.tokens(),
                                                     binding.positions(),
                                                     prepared_kv_,
                                                     binding.params());
      torch::Tensor prepared_logits =
          model_->logits(prepared.hidden_states, indices).cpu();
      torch::Tensor prepared_hidden = prepared.hidden_states.cpu();
      ASSERT_EQ(eager_hidden.sizes(), prepared_hidden.sizes());
      EXPECT_TRUE(torch::equal(eager_hidden, prepared_hidden));
      EXPECT_TRUE(torch::equal(eager_logits, prepared_logits));
      EXPECT_TRUE(torch::equal(eager_logits.argmax(/*dim=*/-1),
                               prepared_logits.argmax(/*dim=*/-1)));
    }
    ASSERT_EQ(launch_->synchronize(), 0);
  }

  void compare_kv() {
    for (uint32_t layer = 0; layer < eager_kv_.size(); ++layer) {
      EXPECT_TRUE(torch::equal(eager_kv_[layer].get_k_cache().cpu(),
                               prepared_kv_[layer].get_k_cache().cpu()))
          << layer;
      EXPECT_TRUE(torch::equal(eager_kv_[layer].get_v_cache().cpu(),
                               prepared_kv_[layer].get_v_cache().cpu()))
          << layer;
    }
  }

  std::unique_ptr<LlmTaskProgram> make_program(
      uint32_t slot_count = 1,
      torch::ScalarType parameter_dtype = torch::kBFloat16) {
    auto rng_before = pipeline_rng_state();
    LlmTaskCapacity capacity;
    capacity.slot_count = slot_count;
    capacity.model = {32, 3, 2};
    capacity.max_kv_seq_len = 128;
    capacity.max_positions = 128;
    capacity.block_size = 128;
    capacity.vocab_size = args_.vocab_size();
    capacity.hidden_size = args_.hidden_size();
    capacity.max_unique_tokens = 32;
    capacity.max_top_logprobs = 5;
    capacity.parameter_dtype = parameter_dtype;
    capacity.chunked_prefill = GetParam();
    std::unique_ptr<LlmTaskProgram> program;
    Status status = LlmTaskProgram::create(
        *model_, *slot_executor_, prepared_kv_, capacity, program);
    EXPECT_TRUE(status.ok()) << status.message();
    if (!status.ok()) {
      return nullptr;
    }
    EXPECT_TRUE(torch::equal(rng_before, pipeline_rng_state()));
    return program;
  }

  std::unique_ptr<TaskExecutionPipeline> make_pipeline(
      ThreadPool& state,
      torch::ScalarType parameter_dtype = torch::kBFloat16,
      uint32_t slot_count = 1) {
    auto program = make_program(slot_count, parameter_dtype);
    if (program == nullptr) {
      return nullptr;
    }
    std::unique_ptr<TaskExecutionPipeline> pipeline;
    const Status status =
        TaskExecutionPipeline::create(state, std::move(program), pipeline);
    EXPECT_TRUE(status.ok()) << status.message();
    return pipeline;
  }

  SamplingParameters sampling_for(const BatchData& batch, int32_t mode) {
    const int64_t rows = batch.q.size();
    std::vector<int32_t> selected;
    selected.reserve(rows);
    for (int32_t end : batch.cumulative) {
      selected.emplace_back(end - 1);
    }
    SamplingParameters params;
    params.selected_token_idxes = torch::tensor(selected, torch::kInt32);
    params.sample_idxes = torch::arange(rows, torch::kInt32);
    params.do_sample = torch::full({rows}, mode != 0, torch::kBool);
    params.all_greedy_sample = mode == 0;
    params.all_random_sample = mode == 1;
    if (mode == 2) {
      params.do_sample[0] = false;
    }
    params.logprobs = true;
    params.max_top_logprobs = 5;
    params.temperatures = torch::full({rows}, 0.8F);
    params.top_k = torch::full({rows}, /*fill_value=*/16, torch::kInt64);
    params.top_p = torch::full({rows}, 0.9F);
    params.frequency_penalties = torch::full({rows}, 0.1F);
    params.presence_penalties = torch::full({rows}, 0.2F);
    params.repetition_penalties = torch::full({rows}, 1.1F);
    params.unique_token_ids =
        torch::full({rows, 1}, /*fill_value=*/100, torch::kInt64);
    params.unique_token_counts = torch::ones({rows, 1}, torch::kInt32);
    params.unique_token_ids_lens = torch::ones({rows}, torch::kInt32);
    return params;
  }

  SampleOutput reference(const BatchData& batch,
                         BatchForwardType phase,
                         const SamplingParameters& params,
                         torch::ScalarType parameter_dtype = torch::kBFloat16) {
    // Build the reference with the original variable-width input preparation,
    // independently of the Slot layout and its padded page-table capacity.
    ModelInputParams reference_params;
    auto& host = reference_params.attention.host;
    host.q_seq_lens = batch.q;
    host.kv_seq_lens = batch.kv;
    host.q_cu_seq_lens = batch.cumulative;
    host.new_cache_slots = batch.slots;
    host.kv_cache_tokens_nums.resize(batch.q.size());
    for (uint32_t row = 0; row < batch.q.size(); ++row) {
      host.kv_cache_tokens_nums[row] = batch.kv[row] - batch.q[row];
    }
    host.block_tables =
        torch::tensor(batch.blocks, torch::kInt32)
            .view({static_cast<int64_t>(batch.q.size()),
                   static_cast<int64_t>(batch.blocks.size() / batch.q.size())});
    reference_params.meta.batch_forward_type = phase;
    reference_params.meta.num_sequences = static_cast<int32_t>(batch.q.size());
    reference_params.meta.actual_num_sequences =
        static_cast<int32_t>(batch.q.size());
    reference_params.meta.q_max_seq_len =
        *std::max_element(batch.q.begin(), batch.q.end());
    reference_params.meta.kv_max_seq_len =
        *std::max_element(batch.kv.begin(), batch.kv.end());
    EXPECT_TRUE(reference_params.attention.rebuild_device_buffer(device_));
    auto tokens = torch::tensor(batch.tokens, torch::kInt32).to(device_);
    auto positions = torch::tensor(batch.positions, torch::kInt32).to(device_);
    Stream original(c10_npu::getCurrentNPUStream(device_.index()));
    ModelOutput model_output = eager_executor_->forward(
        tokens, positions, eager_kv_, reference_params);
    if (!params.selected_token_idxes.defined()) {
      EXPECT_EQ(original.synchronize(), 0);
      return {};
    }
    auto device_params = params.to(device_, parameter_dtype);
    auto logits = model_->logits(model_output.hidden_states,
                                 device_params.selected_token_idxes);
    // Retain the native top-p conversion across Legacy's raw ACL submission,
    // as in the existing C10b reference; no Legacy source change is made.
    device_params.top_p = device_params.top_p.to(logits.scalar_type());
    SampleOutput output = Sampler().forward(logits, device_params);
    for (torch::Tensor* field : {&output.next_tokens,
                                 &output.logprobs,
                                 &output.top_tokens,
                                 &output.top_logprobs}) {
      if (field->defined()) {
        *field = field->cpu();
      }
    }
    output.probs = torch::Tensor();
    EXPECT_EQ(original.synchronize(), 0);
    return output;
  }

  void compare_result(const TaskResult& actual, const SampleOutput& expected) {
    ASSERT_TRUE(actual.status.ok()) << actual.status.message();
    EXPECT_TRUE(torch::equal(actual.output.tokens.tokens.squeeze(/*dim=*/1),
                             expected.next_tokens));
    EXPECT_TRUE(torch::equal(actual.output.tokens.logprobs.squeeze(/*dim=*/1),
                             expected.logprobs));
    EXPECT_TRUE(torch::equal(actual.output.tokens.top_tokens.squeeze(/*dim=*/1),
                             expected.top_tokens));
    EXPECT_TRUE(
        torch::equal(actual.output.tokens.top_logprobs.squeeze(/*dim=*/1),
                     expected.top_logprobs));
    EXPECT_TRUE(torch::equal(actual.output.tokens.lengths,
                             torch::ones_like(actual.output.tokens.lengths)));
    EXPECT_TRUE(actual.output.tokens.tokens.device().is_cpu());
    EXPECT_FALSE(actual.output.tokens.tokens.is_pinned());
  }

  const torch::Device device_{torch::kPrivateUse1, 0};
  const torch::TensorOptions options_ =
      torch::dtype(torch::kBFloat16).device(device_);
  ModelArgs args_;
  std::unique_ptr<Tokenizer> tokenizer_;
  std::unique_ptr<ProcessGroup> process_group_;
  std::unique_ptr<ModelContext> context_;
  std::unique_ptr<CausalLM> model_;
  std::array<std::unique_ptr<ModelInputStorage>, 2> storage_;
  std::array<std::unique_ptr<ModelInputBinding>, 2> bindings_;
  std::unique_ptr<Executor> eager_executor_;
  std::unique_ptr<Executor> slot_executor_;
  std::string old_model_impl_;
  std::unique_ptr<Stream> prepare_;
  std::unique_ptr<Stream> launch_;
  std::vector<KVCache> eager_kv_;
  std::vector<KVCache> prepared_kv_;
  uint64_t batch_id_ = 1;
  int32_t previous_threads_ = 0;
  int32_t old_max_tokens_ = 0;
  int32_t old_block_size_ = 0;
  bool old_chunked_ = false;
  bool old_graph_ = false;
  bool old_prefix_ = false;
};

TEST_P(Qwen3SlotForwardTest, RealWeightsPrefillAndMultiStepDecode) {
  compare(make_batch({5, 3}, {5, 3}), BatchForwardType::PREFILL, 0);
  if (GetParam()) {
    compare(make_batch({3, 1}, {8, 4}), BatchForwardType::MIXED, 1);
    compare(make_batch({2, 2}, {10, 6}), BatchForwardType::CHUNKED_PREFILL, 0);
  }
  const int32_t past0 = GetParam() ? 10 : 5;
  const int32_t past1 = GetParam() ? 6 : 3;
  for (int32_t step = 1; step <= 8; ++step) {
    compare(make_batch({1, 1}, {past0 + step, past1 + step}),
            BatchForwardType::DECODE,
            step % 2);
  }
  compare_kv();
}

TEST_P(Qwen3SlotForwardTest, PipelinePrefillDecodeSamplingAndInputOwnership) {
  ThreadPool state(/*num_threads=*/1);
  auto pipeline = make_pipeline(state);
  ASSERT_NE(pipeline, nullptr);
  std::vector<TokenResultTensors> retained;
  std::vector<torch::Tensor> snapshots;
  retained.reserve(/*new_cap=*/9);
  snapshots.reserve(/*new_cap=*/9);
  for (int32_t step = 0; step < 9; ++step) {
    auto batch = step == 0 ? make_batch({5, 3}, {5, 3})
                           : make_batch({1, 1}, {5 + step, 3 + step});
    const BatchForwardType phase =
        step == 0 ? BatchForwardType::PREFILL : BatchForwardType::DECODE;
    if (step != 0) {
      const int64_t* previous = retained.back().tokens.data_ptr<int64_t>();
      batch.tokens[0] = static_cast<int32_t>(previous[0]);
      batch.tokens[1] = static_cast<int32_t>(previous[1]);
    }
    auto params = sampling_for(batch, step % 3);
    SCOPED_TRACE(step);
    auto rng_before = pipeline_rng_state();
    auto expected = reference(batch, phase, params);
    auto expected_rng = pipeline_rng_state();
    ASSERT_EQ(aclrtSynchronizeDevice(), ACL_SUCCESS);
    restore_pipeline_rng(rng_before);
    const auto submitted = pipeline->submit({view(batch), {phase, 2}, params});
    ASSERT_TRUE(submitted.status.ok()) << submitted.status.message();
    EXPECT_EQ(pipeline->submit({view(batch), {phase, 2}, params}).status.code(),
              StatusCode::RESOURCE_EXHAUSTED);
    std::fill(batch.tokens.begin(), batch.tokens.end(), -99);
    std::fill(batch.positions.begin(), batch.positions.end(), -99);
    params.selected_token_idxes.fill_(/*value=*/-99);
    params.top_p.fill_(/*value=*/0);
    params.unique_token_counts.fill_(/*value=*/99);
    params.do_sample.logical_not_();
    auto result = pipeline->take_result_async(submitted.task_id).get();
    compare_result(result, expected);
    EXPECT_TRUE(torch::equal(pipeline_rng_state(), expected_rng));
    EXPECT_EQ(
        pipeline->take_result_async(submitted.task_id).get().status.code(),
        StatusCode::INVALID_ARGUMENT);
    snapshots.emplace_back(result.output.tokens.tokens.clone());
    retained.emplace_back(std::move(result.output.tokens));
  }
  pipeline.reset();
  for (uint32_t index = 0; index < retained.size(); ++index) {
    EXPECT_TRUE(torch::equal(retained[index].tokens, snapshots[index]));
  }
  compare_kv();
}

TEST_P(Qwen3SlotForwardTest, TwoSlotsShareWorkspaceAndKeepPrivateInputs) {
  for (torch::ScalarType parameter_dtype :
       {torch::kBFloat16, torch::kFloat32}) {
    SCOPED_TRACE(parameter_dtype);
    auto single = make_program(/*slot_count=*/1, parameter_dtype);
    ASSERT_NE(single, nullptr);
    const uint64_t shared_bytes = single->shared_device_bytes();
    const uint64_t private_bytes = single->slot_device_bytes(/*slot_id=*/0);
    const uint64_t pinned_bytes = single->pinned_bytes();
    single.reset();
    auto program = make_program(/*slot_count=*/2, parameter_dtype);
    ASSERT_NE(program, nullptr);
    EXPECT_EQ(program->slot_count(), 2);
    EXPECT_GT(shared_bytes, 0);
    // A second Slot adds three conservative in-flight epoch rows.
    EXPECT_EQ(program->shared_device_bytes(),
              shared_bytes + 3 * sizeof(int64_t));
    EXPECT_EQ(program->device_bytes(),
              program->shared_device_bytes() + 2 * private_bytes);
    EXPECT_EQ(program->pinned_bytes(), 2 * pinned_bytes);
    EXPECT_EQ(program->slot_device_bytes(/*slot_id=*/1), private_bytes);
    std::vector<LlmTaskOutput> retained;
    std::vector<SampleOutput> snapshots;
    retained.reserve(/*new_cap=*/12);
    snapshots.reserve(/*new_cap=*/12);
    for (int32_t pair = 0; pair < 6; ++pair) {
      SCOPED_TRACE(pair);
      auto first = make_batch({5, 3}, {5, 3});
      auto second = make_batch({1, 1}, {6, 4});
      auto first_params = sampling_for(first, pair % 3);
      auto second_params = sampling_for(second, pair % 3);
      auto rng_before = pipeline_rng_state();
      auto expected_first = reference(
          first, BatchForwardType::PREFILL, first_params, parameter_dtype);
      auto expected_second = reference(
          second, BatchForwardType::DECODE, second_params, parameter_dtype);
      auto expected_rng = pipeline_rng_state();
      restore_pipeline_rng(rng_before);
      ASSERT_TRUE(
          program
              ->prepare(
                  /*slot_id=*/0,
                  {view(first), {BatchForwardType::PREFILL, 2}, first_params})
              .ok());
      ASSERT_TRUE(
          program
              ->prepare(
                  /*slot_id=*/1,
                  {view(second), {BatchForwardType::DECODE, 2}, second_params})
              .ok());
      EXPECT_EQ(program->prepare(/*slot_id=*/2, {}).code(),
                StatusCode::INVALID_ARGUMENT);
      EXPECT_TRUE(torch::equal(rng_before, pipeline_rng_state()));
      // Both Slots are prepared before either is launched. Mutating both
      // caller inputs must affect neither Slot's metadata nor its sampling.
      first.tokens.assign(first.tokens.size(), /*value=*/-99);
      second.tokens.assign(second.tokens.size(), /*value=*/-99);
      for (BatchData* batch : {&first, &second}) {
        std::fill(
            batch->positions.begin(), batch->positions.end(), /*value=*/-99);
        std::fill(batch->q.begin(), batch->q.end(), /*value=*/0);
        std::fill(batch->kv.begin(), batch->kv.end(), /*value=*/0);
        std::fill(
            batch->cumulative.begin(), batch->cumulative.end(), /*value=*/0);
        std::fill(batch->blocks.begin(), batch->blocks.end(), /*value=*/-99);
      }
      for (SamplingParameters* params : {&first_params, &second_params}) {
        params->selected_token_idxes.fill_(/*value=*/-99);
        params->unique_token_counts.fill_(/*value=*/99);
        params->do_sample.logical_not_();
      }
      first_params.top_p.zero_();
      second_params.top_p.zero_();
      std::thread launch([&program] {
        program->launch(/*slot_id=*/0);
        program->launch(/*slot_id=*/1);
      });
      launch.join();
      auto first_result = program->consume(/*slot_id=*/0);
      auto second_result = program->consume(/*slot_id=*/1);
      compare_result({Status(), first_result}, expected_first);
      compare_result({Status(), second_result}, expected_second);
      EXPECT_TRUE(torch::equal(expected_rng, pipeline_rng_state()));
      snapshots.emplace_back(std::move(expected_first));
      snapshots.emplace_back(std::move(expected_second));
      retained.emplace_back(std::move(first_result));
      retained.emplace_back(std::move(second_result));
    }
    program.reset();
    for (uint32_t index = 0; index < retained.size(); ++index) {
      compare_result({Status(), retained[index]}, snapshots[index]);
    }
    compare_kv();
  }
}

TEST_P(Qwen3SlotForwardTest, TwoSlotPipelineRetiresInOrderAndReusesStorage) {
  for (torch::ScalarType parameter_dtype :
       {torch::kBFloat16, torch::kFloat32}) {
    SCOPED_TRACE(parameter_dtype);
    ThreadPool state(/*num_threads=*/1);
    auto pipeline = make_pipeline(state, parameter_dtype, /*slot_count=*/2);
    ASSERT_NE(pipeline, nullptr);
    auto first = make_batch({5, 3}, {5, 3});
    auto second = make_batch({1, 1}, {6, 4});
    auto third = make_batch({1, 1}, {7, 5});
    auto first_params = sampling_for(first, /*mode=*/0);
    auto second_params = sampling_for(second, /*mode=*/1);
    auto third_params = sampling_for(third, /*mode=*/2);
    auto rng_before = pipeline_rng_state();
    auto expected_first = reference(
        first, BatchForwardType::PREFILL, first_params, parameter_dtype);
    auto expected_second = reference(
        second, BatchForwardType::DECODE, second_params, parameter_dtype);
    auto expected_third = reference(
        third, BatchForwardType::DECODE, third_params, parameter_dtype);
    auto expected_rng = pipeline_rng_state();
    restore_pipeline_rng(rng_before);
    const auto a = pipeline->submit(
        {view(first), {BatchForwardType::PREFILL, 2}, first_params});
    ASSERT_TRUE(a.status.ok());
    const auto b = pipeline->submit(
        {view(second), {BatchForwardType::DECODE, 2}, second_params});
    ASSERT_TRUE(b.status.ok());
    EXPECT_GT(b.task_id, a.task_id);
    EXPECT_EQ(
        pipeline
            ->submit({view(third), {BatchForwardType::DECODE, 2}, third_params})
            .status.code(),
        StatusCode::RESOURCE_EXHAUSTED);
    // Prepare B completed without requesting Consume A. A later ticket must
    // not steal the oldest completion, even when both kernels were submitted.
    EXPECT_EQ(pipeline->take_result_async(b.task_id).get().status.code(),
              StatusCode::INVALID_ARGUMENT);
    auto result_a = pipeline->take_result_async(a.task_id).get();
    compare_result(result_a, expected_first);
    auto invalid = third;
    invalid.positions[0] = -1;
    const auto rejected = pipeline->submit(
        {view(invalid), {BatchForwardType::DECODE, 2}, third_params});
    EXPECT_EQ(rejected.status.code(), StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(rejected.task_id, 0);
    const auto c = pipeline->submit(
        {view(third), {BatchForwardType::DECODE, 2}, third_params});
    ASSERT_TRUE(c.status.ok());
    EXPECT_EQ(c.task_id, b.task_id + 1);
    EXPECT_EQ(pipeline->submit({}).status.code(),
              StatusCode::RESOURCE_EXHAUSTED);
    EXPECT_EQ(pipeline->take_result_async(a.task_id).get().status.code(),
              StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(pipeline->take_result_async(c.task_id).get().status.code(),
              StatusCode::INVALID_ARGUMENT);
    auto result_b = pipeline->take_result_async(b.task_id).get();
    auto result_c = pipeline->take_result_async(c.task_id).get();
    compare_result(result_b, expected_second);
    compare_result(result_c, expected_third);
    EXPECT_TRUE(torch::equal(expected_rng, pipeline_rng_state()));
    BatchData empty;
    const auto last =
        pipeline->submit({view(empty), {BatchForwardType::EMPTY, 0}, {}});
    ASSERT_TRUE(last.status.ok());
    auto empty_result = pipeline->take_result_async(last.task_id).get();
    EXPECT_TRUE(empty_result.status.ok());
    EXPECT_FALSE(empty_result.output.tokens.tokens.defined());
    pipeline.reset();
    compare_result(result_a, expected_first);
    compare_result(result_b, expected_second);
    compare_result(result_c, expected_third);
    compare_kv();
  }
}

TEST_P(Qwen3SlotForwardTest, TwoSlotPipelineDrainsUnclaimedAndPendingResults) {
  for (torch::ScalarType parameter_dtype :
       {torch::kBFloat16, torch::kFloat32}) {
    SCOPED_TRACE(parameter_dtype);
    ThreadPool state(/*num_threads=*/1);
    auto first = make_batch({5, 3}, {5, 3});
    auto second = make_batch({1, 1}, {6, 4});
    auto first_params = sampling_for(first, /*mode=*/1);
    auto second_params = sampling_for(second, /*mode=*/2);
    for (int32_t pending = 0; pending <= 2; ++pending) {
      SCOPED_TRACE(pending);
      auto pipeline = make_pipeline(state, parameter_dtype, /*slot_count=*/2);
      ASSERT_NE(pipeline, nullptr);
      auto rng_before = pipeline_rng_state();
      if (pending >= 1) {
        reference(
            first, BatchForwardType::PREFILL, first_params, parameter_dtype);
      }
      if (pending == 2) {
        reference(
            second, BatchForwardType::DECODE, second_params, parameter_dtype);
      }
      auto expected_rng = pipeline_rng_state();
      restore_pipeline_rng(rng_before);
      if (pending >= 1) {
        ASSERT_TRUE(
            pipeline
                ->submit(
                    {view(first), {BatchForwardType::PREFILL, 2}, first_params})
                .status.ok());
      }
      if (pending == 2) {
        ASSERT_TRUE(pipeline
                        ->submit({view(second),
                                  {BatchForwardType::DECODE, 2},
                                  second_params})
                        .status.ok());
      }
      pipeline.reset();
      EXPECT_TRUE(torch::equal(expected_rng, pipeline_rng_state()));
      compare_kv();
    }
    auto pipeline = make_pipeline(state, parameter_dtype, /*slot_count=*/2);
    ASSERT_NE(pipeline, nullptr);
    auto rng_before = pipeline_rng_state();
    auto expected_first = reference(
        first, BatchForwardType::PREFILL, first_params, parameter_dtype);
    auto expected_second = reference(
        second, BatchForwardType::DECODE, second_params, parameter_dtype);
    auto expected_rng = pipeline_rng_state();
    restore_pipeline_rng(rng_before);
    const auto a = pipeline->submit(
        {view(first), {BatchForwardType::PREFILL, 2}, first_params});
    const auto b = pipeline->submit(
        {view(second), {BatchForwardType::DECODE, 2}, second_params});
    ASSERT_TRUE(a.status.ok());
    ASSERT_TRUE(b.status.ok());
    auto future_a = pipeline->take_result_async(a.task_id);
    auto future_b = pipeline->take_result_async(b.task_id);
    auto duplicate = pipeline->take_result_async(a.task_id);
    pipeline.reset();
    compare_result(std::move(future_a).get(), expected_first);
    compare_result(std::move(future_b).get(), expected_second);
    EXPECT_EQ(std::move(duplicate).get().status.code(),
              StatusCode::INVALID_ARGUMENT);
    EXPECT_TRUE(torch::equal(expected_rng, pipeline_rng_state()));
    compare_kv();
  }
}

TEST_P(Qwen3SlotForwardTest,
       SequenceStateFeedsTwoSlotsAcrossContinuousRowPermutation) {
  for (torch::ScalarType parameter_dtype :
       {torch::kBFloat16, torch::kFloat32}) {
    SCOPED_TRACE(parameter_dtype);
    const auto reference = [&](const BatchData& batch,
                               BatchForwardType phase,
                               const SamplingParameters& params) {
      return this->reference(batch, phase, params, parameter_dtype);
    };

    ThreadPool state(/*num_threads=*/1);
    auto pipeline = make_pipeline(state, parameter_dtype, /*slot_count=*/2);
    ASSERT_NE(pipeline, nullptr);
    constexpr uint32_t kTasks = 9;
    std::vector<BatchData> batches;
    std::vector<SamplingParameters> parameters;
    std::vector<SampleOutput> expected;
    std::vector<std::array<SequenceStateKey, 2>> keys;
    batches.reserve(kTasks);
    parameters.reserve(kTasks);
    expected.reserve(kTasks);
    keys.reserve(kTasks);
    const auto rng_before = pipeline_rng_state();
    for (uint32_t step = 0; step < kTasks; ++step) {
      const std::array<uint32_t, 2> order = step % 2 == 0
                                                ? std::array<uint32_t, 2>{0, 1}
                                                : std::array<uint32_t, 2>{1, 0};
      const std::array<int32_t, 2> prompt{5, 3};
      BatchData batch =
          step == 0
              ? make_batch(/*q=*/{5, 3}, /*kv=*/{5, 3})
              : make_batch(
                    /*q=*/{1, 1},
                    /*kv=*/{prompt[order[0]] + static_cast<int32_t>(step),
                            prompt[order[1]] + static_cast<int32_t>(step)});
      if (step > 0) {
        for (uint32_t row = 0; row < 2; ++row) {
          batch.blocks[row] = static_cast<int32_t>(order[row]);
          batch.slots[row] =
              static_cast<int32_t>(128 * order[row]) + batch.positions[row];
          // Every decode flips row order relative to its predecessor.
          batch.tokens[row] = static_cast<int32_t>(
              expected.back().next_tokens[1 - row].item<int64_t>());
        }
      }
      auto params = sampling_for(batch, static_cast<int32_t>(step % 3));
      auto result = reference(
          batch,
          step == 0 ? BatchForwardType::PREFILL : BatchForwardType::DECODE,
          params);
      batches.emplace_back(std::move(batch));
      parameters.emplace_back(std::move(params));
      expected.emplace_back(std::move(result));
      keys.emplace_back(
          std::array<SequenceStateKey, 2>{SequenceStateKey{101 + order[0], 0},
                                          SequenceStateKey{101 + order[1], 0}});
    }
    const auto expected_rng = pipeline_rng_state();
    restore_pipeline_rng(rng_before);
    std::vector<uint64_t> tickets;
    std::vector<TokenResultTensors> retained;
    std::vector<torch::Tensor> snapshots;
    tickets.reserve(kTasks);
    retained.reserve(kTasks);
    snapshots.reserve(kTasks);
    const auto submit = [&](uint32_t step) {
      if (step > 0) {
        // These values carry no row index: identity and position authorize
        // reads.
        batches[step].tokens = {-7, -901};
      }
      const auto accepted = pipeline->submit(
          {view(batches[step]),
           {step == 0 ? BatchForwardType::PREFILL : BatchForwardType::DECODE,
            2},
           parameters[step],
           keys[step],
           {}});
      EXPECT_TRUE(accepted.status.ok()) << accepted.status.message();
      tickets.emplace_back(accepted.task_id);
      // PrepareAck released every caller borrow, including identity and
      // sampling.
      batches[step].tokens.assign(batches[step].tokens.size(), -999);
      parameters[step].top_p.zero_();
      keys[step] = {};
    };
    submit(/*step=*/0);
    submit(/*step=*/1);
    for (uint32_t step = 0; step < kTasks; ++step) {
      auto actual = pipeline->take_result_async(tickets[step]).get();
      compare_result(actual, expected[step]);
      snapshots.emplace_back(actual.output.tokens.tokens.clone());
      retained.emplace_back(std::move(actual.output.tokens));
      if (step + 2 < kTasks) {
        submit(step + 2);
      }
    }
    EXPECT_TRUE(torch::equal(expected_rng, pipeline_rng_state()));
    const std::array<SequenceStateKey, 2> retired{{{101, 0}, {102, 0}}};
    BatchData empty;
    const auto control = pipeline->submit(
        {view(empty), {BatchForwardType::EMPTY, 0}, {}, {}, retired});
    ASSERT_TRUE(control.status.ok());
    auto control_result = pipeline->take_result_async(control.task_id).get();
    EXPECT_TRUE(control_result.status.ok());
    EXPECT_FALSE(control_result.output.tokens.tokens.defined());
    pipeline.reset();
    for (uint32_t index = 0; index < retained.size(); ++index) {
      EXPECT_TRUE(torch::equal(retained[index].tokens, snapshots[index]));
    }
    compare_kv();
  }
}

TEST_P(Qwen3SlotForwardTest,
       SequenceStateHandlesUnsampledRowsResetAndUnclaimedDestruction) {
  for (torch::ScalarType parameter_dtype :
       {torch::kBFloat16, torch::kFloat32}) {
    SCOPED_TRACE(parameter_dtype);
    const auto reference = [&](const BatchData& batch,
                               BatchForwardType phase,
                               const SamplingParameters& params) {
      return this->reference(batch, phase, params, parameter_dtype);
    };

    ThreadPool state(/*num_threads=*/1);
    auto pipeline = make_pipeline(state, parameter_dtype, /*slot_count=*/2);
    ASSERT_NE(pipeline, nullptr);
    const std::array<SequenceStateKey, 2> first_keys{{{201, 0}, {202, 0}}};
    const std::array<SequenceStateKey, 2> second_keys{{{202, 0}, {201, 0}}};
    auto first = make_batch(/*q=*/{2, 1}, /*kv=*/{2, 1});
    auto one_row = make_batch(/*q=*/{1}, /*kv=*/{1});
    auto first_params = sampling_for(one_row, /*mode=*/0);
    first_params.selected_token_idxes = torch::tensor({2}, torch::kInt32);
    auto second = make_batch(/*q=*/{1, GetParam() ? 2 : 1},
                             /*kv=*/{2, GetParam() ? 4 : 3});
    second.blocks = {1, 0};
    for (uint32_t offset = 0; offset < second.slots.size(); ++offset) {
      second.slots[offset] = (offset == 0 ? 128 : 0) + second.positions[offset];
    }
    auto second_params = sampling_for(second, /*mode=*/2);
    const BatchForwardType second_phase =
        GetParam() ? BatchForwardType::MIXED : BatchForwardType::DECODE;
    auto rng_before = pipeline_rng_state();
    auto expected_first =
        reference(first, BatchForwardType::PREFILL, first_params);
    second.tokens[0] =
        static_cast<int32_t>(expected_first.next_tokens[0].item<int64_t>());
    auto expected_second = reference(second, second_phase, second_params);
    auto expected_rng = pipeline_rng_state();
    restore_pipeline_rng(rng_before);
    const auto a = pipeline->submit({view(first),
                                     {BatchForwardType::PREFILL, 2},
                                     first_params,
                                     first_keys,
                                     {}});
    ASSERT_TRUE(a.status.ok());
    second.tokens[0] = -7;
    const auto b = pipeline->submit(
        {view(second), {second_phase, 2}, second_params, second_keys, {}});
    ASSERT_TRUE(b.status.ok()) << b.status.message();
    auto result_a = pipeline->take_result_async(a.task_id).get();
    auto result_b = pipeline->take_result_async(b.task_id).get();
    compare_result(result_a, expected_first);
    compare_result(result_b, expected_second);
    EXPECT_TRUE(torch::equal(expected_rng, pipeline_rng_state()));
    compare_kv();

    const std::array<SequenceStateKey, 1> reset_key{{{201, 1}}};
    auto unknown = make_batch(/*q=*/{1}, /*kv=*/{1});
    unknown.tokens[0] = -1;
    auto unknown_params = sampling_for(unknown, /*mode=*/0);
    const auto rejected = pipeline->submit({view(unknown),
                                            {BatchForwardType::DECODE, 1},
                                            unknown_params,
                                            reset_key,
                                            first_keys});
    EXPECT_EQ(rejected.status.code(), StatusCode::INVALID_ARGUMENT);
    // Rejection cannot retire old identities or advance RNG before this retry.
    EXPECT_TRUE(torch::equal(expected_rng, pipeline_rng_state()));
    auto reset = make_batch(/*q=*/{2}, /*kv=*/{2});
    auto reset_params = sampling_for(reset, /*mode=*/1);
    auto next = make_batch(/*q=*/{1}, /*kv=*/{3});
    auto next_params = sampling_for(next, /*mode=*/0);
    rng_before = pipeline_rng_state();
    auto expected_reset =
        reference(reset, BatchForwardType::PREFILL, reset_params);
    next.tokens[0] =
        static_cast<int32_t>(expected_reset.next_tokens[0].item<int64_t>());
    reference(next, BatchForwardType::DECODE, next_params);
    expected_rng = pipeline_rng_state();
    restore_pipeline_rng(rng_before);
    const auto c = pipeline->submit({view(reset),
                                     {BatchForwardType::PREFILL, 1},
                                     reset_params,
                                     reset_key,
                                     first_keys});
    ASSERT_TRUE(c.status.ok()) << c.status.message();
    next.tokens[0] = -91;
    const auto d = pipeline->submit({view(next),
                                     {BatchForwardType::DECODE, 1},
                                     next_params,
                                     reset_key,
                                     {}});
    ASSERT_TRUE(d.status.ok()) << d.status.message();
    // Both accepted publications execute and retire even when no result is
    // taken.
    pipeline.reset();
    EXPECT_TRUE(torch::equal(expected_rng, pipeline_rng_state()));
    compare_kv();
    EXPECT_TRUE(torch::equal(result_a.output.tokens.tokens.squeeze(/*dim=*/1),
                             expected_first.next_tokens));
    EXPECT_TRUE(torch::equal(result_b.output.tokens.tokens.squeeze(/*dim=*/1),
                             expected_second.next_tokens));
  }
}

TEST_P(Qwen3SlotForwardTest,
       OldestResultKeepsSlotMetadataAcrossMixedTakeCalls) {
  for (torch::ScalarType parameter_dtype :
       {torch::kBFloat16, torch::kFloat32}) {
    SCOPED_TRACE(parameter_dtype);
    const auto reference = [&](const BatchData& batch,
                               BatchForwardType phase,
                               const SamplingParameters& params) {
      return this->reference(batch, phase, params, parameter_dtype);
    };

    ThreadPool state(/*num_threads=*/1);
    auto pipeline = make_pipeline(state, parameter_dtype, /*slot_count=*/2);
    ASSERT_NE(pipeline, nullptr);
    auto missing = pipeline->take_result_async().get();
    EXPECT_EQ(missing.status.code(), StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(missing.task_id, 0U);
    auto first = make_batch(/*q=*/{2, 1}, /*kv=*/{2, 1});
    auto second = make_batch(/*q=*/{1}, /*kv=*/{3});
    auto third = make_batch(/*q=*/{1}, /*kv=*/{4});
    auto first_params = sampling_for(first, /*mode=*/2);
    auto second_params = sampling_for(second, /*mode=*/0);
    auto third_params = sampling_for(third, /*mode=*/1);
    const auto first_mask = first_params.do_sample.clone();
    const auto second_mask = second_params.do_sample.clone();
    const auto third_mask = third_params.do_sample.clone();
    const auto rng_before = pipeline_rng_state();
    auto expected_first =
        reference(first, BatchForwardType::PREFILL, first_params);
    auto expected_second =
        reference(second, BatchForwardType::DECODE, second_params);
    auto expected_third =
        reference(third, BatchForwardType::DECODE, third_params);
    const auto expected_rng = pipeline_rng_state();
    restore_pipeline_rng(rng_before);

    LlmTaskInput input_a{view(first),
                         {BatchForwardType::PREFILL, 2},
                         first_params,
                         {},
                         {},
                         true};
    LlmTaskInput input_b{
        view(second), {BatchForwardType::DECODE, 1}, second_params};
    const auto a = pipeline->submit(input_a);
    const auto b = pipeline->submit(input_b);
    ASSERT_TRUE(a.status.ok());
    ASSERT_TRUE(b.status.ok());
    input_a.is_warmup = false;
    input_b.is_warmup = true;
    first_params.do_sample.zero_();
    second_params.do_sample.fill_(/*value=*/true);
    const auto wrong = pipeline->take_result_async(b.task_id).get();
    EXPECT_EQ(wrong.status.code(), StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(wrong.task_id, 0U);
    const auto zero = pipeline->take_result_async(/*task_id=*/0).get();
    EXPECT_EQ(zero.status.code(), StatusCode::INVALID_ARGUMENT);
    auto result_a = pipeline->take_result_async().get();
    ASSERT_EQ(result_a.task_id, a.task_id);
    compare_result(result_a, expected_first);
    EXPECT_TRUE(torch::equal(result_a.output.do_sample, first_mask));
    EXPECT_TRUE(result_a.output.is_warmup);
    EXPECT_FALSE(result_a.output.do_sample.is_pinned());

    const auto c = pipeline->submit(
        {view(third), {BatchForwardType::DECODE, 1}, third_params});
    ASSERT_TRUE(c.status.ok());
    auto result_b = pipeline->take_result_async(b.task_id).get();
    ASSERT_EQ(result_b.task_id, b.task_id);
    compare_result(result_b, expected_second);
    EXPECT_TRUE(torch::equal(result_b.output.do_sample, second_mask));
    EXPECT_FALSE(result_b.output.is_warmup);
    auto result_c = pipeline->take_result_async().get();
    ASSERT_EQ(result_c.task_id, c.task_id);
    compare_result(result_c, expected_third);
    EXPECT_TRUE(torch::equal(result_c.output.do_sample, third_mask));
    EXPECT_FALSE(result_c.output.is_warmup);
    EXPECT_EQ(pipeline->take_result_async(a.task_id).get().status.code(),
              StatusCode::INVALID_ARGUMENT);
    EXPECT_TRUE(torch::equal(expected_rng, pipeline_rng_state()));
    pipeline.reset();
    EXPECT_TRUE(torch::equal(result_a.output.do_sample, first_mask));
    EXPECT_TRUE(torch::equal(result_b.output.do_sample, second_mask));
    compare_kv();
  }
}

TEST_P(Qwen3SlotForwardTest,
       EmptyTasksRetainDistinctOutputMetadataWithoutSampling) {
  for (torch::ScalarType parameter_dtype :
       {torch::kBFloat16, torch::kFloat32}) {
    SCOPED_TRACE(parameter_dtype);

    ThreadPool state(/*num_threads=*/1);
    auto pipeline = make_pipeline(state, parameter_dtype, /*slot_count=*/2);
    ASSERT_NE(pipeline, nullptr);
    BatchData empty;
    SamplingParameters params;
    params.do_sample = torch::empty({0}, torch::kBool);
    const auto rng_before = pipeline_rng_state();
    const auto first = pipeline->submit(
        {view(empty), {BatchForwardType::EMPTY, 0}, params, {}, {}, true});
    const auto second =
        pipeline->submit({view(empty), {BatchForwardType::EMPTY, 0}, {}});
    ASSERT_TRUE(first.status.ok());
    ASSERT_TRUE(second.status.ok());
    const auto full =
        pipeline->submit({view(empty), {BatchForwardType::EMPTY, 0}, {}});
    EXPECT_EQ(full.status.code(), StatusCode::RESOURCE_EXHAUSTED);
    auto result_a = pipeline->take_result_async().get();
    auto result_b = pipeline->take_result_async().get();
    ASSERT_TRUE(result_a.status.ok());
    ASSERT_TRUE(result_b.status.ok());
    EXPECT_EQ(result_a.task_id, first.task_id);
    EXPECT_EQ(result_b.task_id, second.task_id);
    EXPECT_FALSE(result_a.output.tokens.tokens.defined());
    ASSERT_TRUE(result_a.output.do_sample.defined());
    EXPECT_EQ(result_a.output.do_sample.numel(), 0);
    EXPECT_FALSE(result_a.output.do_sample.is_pinned());
    EXPECT_TRUE(result_a.output.is_warmup);
    EXPECT_FALSE(result_b.output.tokens.tokens.defined());
    EXPECT_FALSE(result_b.output.do_sample.defined());
    EXPECT_FALSE(result_b.output.is_warmup);
    EXPECT_EQ(pipeline->take_result_async().get().status.code(),
              StatusCode::INVALID_ARGUMENT);
    EXPECT_TRUE(torch::equal(rng_before, pipeline_rng_state()));
  }
}

TEST_P(Qwen3SlotForwardTest, PipelineRejectsBeforeWritesAndReusesSlot) {
  ThreadPool state(/*num_threads=*/1);
  auto pipeline = make_pipeline(state);
  ASSERT_NE(pipeline, nullptr);
  auto batch = make_batch({3, 2}, {3, 2});
  auto params = sampling_for(batch, /*mode=*/0);
  const auto reject = [&](const LlmTaskInput& input) {
    const auto submitted = pipeline->submit(input);
    EXPECT_EQ(submitted.status.code(), StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(submitted.task_id, 0);
  };
  params.sample_idxes[0] = 20;
  reject({view(batch), {BatchForwardType::PREFILL, 2}, params});
  params.sample_idxes[0] = 0;
  batch.positions[0] = -1;
  reject({view(batch), {BatchForwardType::PREFILL, 2}, params});
  batch.positions[0] = 0;
  batch.blocks[1] = 999;
  reject({view(batch), {BatchForwardType::PREFILL, 2}, params});
  batch.blocks[1] = 1;
  params.return_probs = true;
  reject({view(batch), {BatchForwardType::PREFILL, 2}, params});
  params.return_probs = false;
  params.logprobs = false;
  reject({view(batch), {BatchForwardType::PREFILL, 2}, params});
  params.logprobs = true;
  // A valid task immediately after all rejections must behave as the first.
  auto expected = reference(batch, BatchForwardType::PREFILL, params);
  const auto submitted =
      pipeline->submit({view(batch), {BatchForwardType::PREFILL, 2}, params});
  ASSERT_TRUE(submitted.status.ok()) << submitted.status.message();
  EXPECT_EQ(submitted.task_id, 1);
  EXPECT_EQ(
      pipeline->take_result_async(submitted.task_id + 1).get().status.code(),
      StatusCode::INVALID_ARGUMENT);
  compare_result(pipeline->take_result_async(submitted.task_id).get(),
                 expected);
  compare_kv();
}

TEST_P(Qwen3SlotForwardTest,
       PipelineDestructionWithoutGetLastAndPendingFuture) {
  ThreadPool state(/*num_threads=*/1);
  auto pipeline = make_pipeline(state);
  ASSERT_NE(pipeline, nullptr);
  pipeline.reset();  // Zero submissions.
  pipeline = make_pipeline(state);
  auto batch = make_batch({3, 2}, {3, 2});
  auto params = sampling_for(batch, /*mode=*/0);
  auto expected = reference(batch, BatchForwardType::PREFILL, params);
  auto submitted =
      pipeline->submit({view(batch), {BatchForwardType::PREFILL, 2}, params});
  ASSERT_TRUE(submitted.status.ok()) << submitted.status.message();
  pipeline
      .reset();  // No GetLast; teardown must complete actual Device readers.
  compare_kv();
  pipeline = make_pipeline(state);
  batch = make_batch({1, 1}, {4, 3});
  params = sampling_for(batch, /*mode=*/0);
  expected = reference(batch, BatchForwardType::DECODE, params);
  submitted =
      pipeline->submit({view(batch), {BatchForwardType::DECODE, 2}, params});
  ASSERT_TRUE(submitted.status.ok());
  auto future = pipeline->take_result_async(submitted.task_id);
  auto duplicate = pipeline->take_result_async(submitted.task_id);
  pipeline.reset();  // Already requested Future must complete before release.
  compare_result(std::move(future).get(), expected);
  EXPECT_EQ(std::move(duplicate).get().status.code(),
            StatusCode::INVALID_ARGUMENT);
  compare_kv();
}

TEST_P(Qwen3SlotForwardTest, PipelineEmptyAndChunkedPrefillWithoutSampling) {
  ThreadPool state(/*num_threads=*/1);
  auto pipeline = make_pipeline(state);
  ASSERT_NE(pipeline, nullptr);
  BatchData empty;
  auto submitted =
      pipeline->submit({view(empty), {BatchForwardType::EMPTY, 0}, {}});
  ASSERT_TRUE(submitted.status.ok()) << submitted.status.message();
  auto result = pipeline->take_result_async(submitted.task_id).get();
  ASSERT_TRUE(result.status.ok());
  EXPECT_FALSE(result.output.tokens.tokens.defined());
  auto batch = make_batch({3, 2}, {3, 2});
  reference(batch, BatchForwardType::PREFILL, {});
  submitted =
      pipeline->submit({view(batch), {BatchForwardType::PREFILL, 2}, {}});
  ASSERT_TRUE(submitted.status.ok());
  result = pipeline->take_result_async(submitted.task_id).get();
  EXPECT_TRUE(result.status.ok());
  EXPECT_FALSE(result.output.tokens.tokens.defined());
  if (GetParam()) {
    batch = make_batch({2, 1}, {5, 3});
    auto params = sampling_for(batch, /*mode=*/0);
    auto expected = reference(batch, BatchForwardType::CHUNKED_PREFILL, params);
    submitted = pipeline->submit(
        {view(batch), {BatchForwardType::CHUNKED_PREFILL, 2}, params});
    ASSERT_TRUE(submitted.status.ok());
    compare_result(pipeline->take_result_async(submitted.task_id).get(),
                   expected);
  }
  compare_kv();
}

// Replay the service benchmark prompt with a controlled row schedule.
// Non-chunked uses four rows; chunked varies one to four rows with mixed
// prefill/decode. Reference page tables grow independently from Slot storage.
TEST_P(Qwen3SlotForwardTest, PipelineFixedFourRowServingSchedule) {
  SchedulerConfig::get_instance().max_tokens_per_batch(/*value=*/512);
  KVCacheCapacity kv_capacity;
  kv_capacity.n_blocks(/*value=*/8);
  kv_capacity.block_size(/*value=*/128);
  const KVCacheShape shape(kv_capacity, args_, /*world_size=*/1);
  KVCacheCreateOptions cache_options;
  cache_options.device(device_);
  cache_options.dtype(torch::kBFloat16);
  cache_options.num_layers(args_.n_layers());
  cache_options.model_type(args_.model_type());
  eager_kv_.clear();
  prepared_kv_.clear();
  for (int32_t layer = 0; layer < args_.n_layers(); ++layer) {
    eager_kv_.emplace_back(shape, cache_options, layer);
    prepared_kv_.emplace_back(shape, cache_options, layer);
    eager_kv_.back().get_k_cache().zero_();
    eager_kv_.back().get_v_cache().zero_();
    prepared_kv_.back().get_k_cache().zero_();
    prepared_kv_.back().get_v_cache().zero_();
  }
  ASSERT_EQ(aclrtSynchronizeDevice(), ACL_SUCCESS);
  LlmTaskCapacity capacity;
  capacity.model = {512, 4, 320};
  capacity.max_kv_seq_len = 256;
  capacity.max_positions = 256;
  capacity.block_size = 128;
  capacity.vocab_size = args_.vocab_size();
  capacity.hidden_size = args_.hidden_size();
  capacity.max_unique_tokens = 32;
  capacity.max_top_logprobs = 5;
  capacity.parameter_dtype = torch::kBFloat16;
  capacity.chunked_prefill = GetParam();
  std::unique_ptr<LlmTaskProgram> program;
  ASSERT_TRUE(LlmTaskProgram::create(
                  *model_, *slot_executor_, prepared_kv_, capacity, program)
                  .ok());
  ThreadPool state(/*num_threads=*/1);
  std::unique_ptr<TaskExecutionPipeline> pipeline;
  ASSERT_TRUE(
      TaskExecutionPipeline::create(state, std::move(program), pipeline).ok());
  std::string prompt;
  for (int32_t repeat = 0; repeat < 12; ++repeat) {
    prompt += "Explain how a CPU pipeline improves instruction throughput. ";
  }
  std::vector<int32_t> prompt_tokens;
  ASSERT_TRUE(tokenizer_->encode(prompt, &prompt_tokens));
  ASSERT_EQ(prompt_tokens.size(), 110);
  std::array<int32_t, 4> lengths{};
  std::array<int32_t, 4> previous{};
  for (int32_t step = 0; step < 64; ++step) {
    SCOPED_TRACE(step);
    const int32_t rows = GetParam() ? 1 + step % 4 : 4;
    std::vector<int32_t> q;
    std::vector<int32_t> kv;
    q.reserve(rows);
    kv.reserve(rows);
    for (int32_t row = 0; row < rows; ++row) {
      const int32_t query =
          lengths[row] == 0 ? static_cast<int32_t>(prompt_tokens.size()) : 1;
      q.emplace_back(query);
      lengths[row] += query;
      kv.emplace_back(lengths[row]);
    }
    auto batch = make_batch(std::move(q), std::move(kv));
    const int32_t columns =
        (*std::max_element(batch.kv.begin(), batch.kv.end()) + 127) / 128;
    batch.blocks.clear();
    batch.blocks.reserve(rows * columns);
    int32_t offset = 0;
    for (int32_t row = 0; row < rows; ++row) {
      for (int32_t column = 0; column < columns; ++column) {
        batch.blocks.emplace_back(row * 2 + column);
      }
      for (int32_t index = 0; index < batch.q[row]; ++index, ++offset) {
        batch.slots[offset] = row * 256 + batch.positions[offset];
        batch.tokens[offset] =
            batch.q[row] == 1 ? previous[row] : prompt_tokens[index];
      }
    }
    auto params = sampling_for(batch, /*mode=*/0);
    params.frequency_penalties.zero_();
    params.presence_penalties.zero_();
    params.repetition_penalties.fill_(/*value=*/1);
    const bool has_prefill =
        std::any_of(batch.q.begin(), batch.q.end(), [](int32_t query) {
          return query > 1;
        });
    const BatchForwardType phase =
        step == 0 ? BatchForwardType::PREFILL
                  : (has_prefill ? BatchForwardType::MIXED
                                 : BatchForwardType::DECODE);
    auto expected = reference(batch, phase, params);
    const auto task = pipeline->submit(
        {view(batch), {phase, static_cast<uint32_t>(rows)}, params});
    ASSERT_TRUE(task.status.ok()) << task.status.message();
    auto result = pipeline->take_result_async(task.task_id).get();
    compare_result(result, expected);
    for (int32_t row = 0; row < rows; ++row) {
      previous[row] = static_cast<int32_t>(
          result.output.tokens.tokens.const_data_ptr<int64_t>()[row]);
    }
  }
  pipeline.reset();
  compare_kv();
}

TEST_P(Qwen3SlotForwardTest, PipelinePreservesShmSamplingPrecision) {
  ThreadPool state(/*num_threads=*/1);
  auto pipeline = make_pipeline(state, torch::kFloat32);
  ASSERT_NE(pipeline, nullptr);
  for (int32_t step = 0; step < 12; ++step) {
    SCOPED_TRACE(step);
    auto batch = step == 0 ? make_batch({5, 3}, {5, 3})
                           : make_batch({1, 1}, {5 + step, 3 + step});
    auto params = sampling_for(batch, step % 3);
    const BatchForwardType phase =
        step == 0 ? BatchForwardType::PREFILL : BatchForwardType::DECODE;
    const auto rng_before = pipeline_rng_state();
    auto expected = reference(batch, phase, params, torch::kFloat32);
    const auto expected_rng = pipeline_rng_state();
    restore_pipeline_rng(rng_before);
    const auto task = pipeline->submit({view(batch), {phase, 2}, params});
    ASSERT_TRUE(task.status.ok()) << task.status.message();
    params.temperatures.fill_(/*value=*/0.5);
    params.frequency_penalties.zero_();
    params.repetition_penalties.fill_(/*value=*/1.0);
    compare_result(pipeline->take_result_async(task.task_id).get(), expected);
    EXPECT_TRUE(torch::equal(expected_rng, pipeline_rng_state()));
  }
  pipeline.reset();
  compare_kv();
}

TEST_P(Qwen3SlotForwardTest, PipelineClosedLoopPerformance) {
  if (std::getenv("XLLM_PIPELINE_PERF") == nullptr) {
    GTEST_SKIP() << "Enable XLLM_PIPELINE_PERF for isolated measurement.";
  }
  ThreadPool state(/*num_threads=*/1);
  auto pipeline = make_pipeline(state);
  ASSERT_NE(pipeline, nullptr);
  const auto now = [] { return std::chrono::steady_clock::now(); };
  for (int32_t pair = 0; pair < 3; ++pair) {
    for (int32_t order = 0; order < 2; ++order) {
      const bool prepared = (pair + order) % 2 == 0;
      auto batch = make_batch({5, 3}, {5, 3});
      auto params = sampling_for(batch, /*mode=*/0);
      if (prepared) {
        auto task = pipeline->submit(
            {view(batch), {BatchForwardType::PREFILL, 2}, params});
        ASSERT_TRUE(task.status.ok());
        ASSERT_TRUE(
            pipeline->take_result_async(task.task_id).get().status.ok());
      } else {
        reference(batch, BatchForwardType::PREFILL, params);
      }
      auto begin = now();
      // Warm both Decode paths before timing: their first Python execution
      // warms the model kernels, which otherwise favors the
      // second implementation in the first pair.
      for (int32_t step = -7; step <= 24; ++step) {
        if (step == 1) {
          begin = now();
        }
        batch = make_batch({1, 1}, {13 + step, 11 + step});
        params = sampling_for(batch, /*mode=*/0);
        if (prepared) {
          auto task = pipeline->submit(
              {view(batch), {BatchForwardType::DECODE, 2}, params});
          ASSERT_TRUE(task.status.ok());
          ASSERT_TRUE(
              pipeline->take_result_async(task.task_id).get().status.ok());
        } else {
          reference(batch, BatchForwardType::DECODE, params);
        }
      }
      const double us =
          std::chrono::duration<double, std::micro>(now() - begin).count() / 24;
      LOG(INFO) << "PIPELINE_PERF pair=" << pair << " prepared=" << prepared
                << " chunked=" << GetParam() << " us=" << us
                << " device_bytes=" << pipeline->device_bytes()
                << " pinned_bytes=" << pipeline->pinned_bytes();
    }
  }
  compare_kv();
}

INSTANTIATE_TEST_SUITE_P(PrefillModes, Qwen3SlotForwardTest, ::testing::Bool());

}  // namespace
}  // namespace xllm
