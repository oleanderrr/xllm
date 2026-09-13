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
          batch.q.empty() ? 0U : 1U};
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

  std::unique_ptr<TaskExecutionPipeline> make_pipeline(ThreadPool& state) {
    auto rng_before = pipeline_rng_state();
    LlmTaskCapacity capacity;
    capacity.model = {32, 3, 2};
    capacity.max_kv_seq_len = 128;
    capacity.max_positions = 128;
    capacity.block_size = 128;
    capacity.vocab_size = args_.vocab_size();
    capacity.hidden_size = args_.hidden_size();
    capacity.max_unique_tokens = 32;
    capacity.max_top_logprobs = 5;
    capacity.parameter_dtype = torch::kBFloat16;
    capacity.chunked_prefill = GetParam();
    std::unique_ptr<LlmTaskProgram> program;
    Status status = LlmTaskProgram::create(
        *model_, *slot_executor_, prepared_kv_, capacity, program);
    EXPECT_TRUE(status.ok()) << status.message();
    if (!status.ok()) {
      return nullptr;
    }
    std::unique_ptr<TaskExecutionPipeline> pipeline;
    status = TaskExecutionPipeline::create(state, std::move(program), pipeline);
    EXPECT_TRUE(status.ok()) << status.message();
    EXPECT_TRUE(torch::equal(rng_before, pipeline_rng_state()));
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
                         const SamplingParameters& params) {
    auto& binding = *bindings_[0];
    const Status status = binding.prepare(
        view(batch), {phase, static_cast<uint32_t>(batch.q.size())}, *prepare_);
    EXPECT_TRUE(status.ok()) << status.message();
    Stream original(c10_npu::getCurrentNPUStream(device_.index()));
    EXPECT_TRUE(original.wait_event(binding.ready_event()));
    ModelInputParams reference_params = binding.params();
    reference_params.attn_metadata.reset();
    reference_params.python_attention_metadata.reset();
    ModelOutput model_output = eager_executor_->forward(
        binding.tokens(), binding.positions(), eager_kv_, reference_params);
    if (!params.selected_token_idxes.defined()) {
      EXPECT_EQ(original.synchronize(), 0);
      return {};
    }
    auto device_params = params.to(device_, torch::kBFloat16);
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
    EXPECT_TRUE(torch::equal(actual.tokens.tokens.squeeze(/*dim=*/1),
                             expected.next_tokens));
    EXPECT_TRUE(torch::equal(actual.tokens.logprobs.squeeze(/*dim=*/1),
                             expected.logprobs));
    EXPECT_TRUE(torch::equal(actual.tokens.top_tokens.squeeze(/*dim=*/1),
                             expected.top_tokens));
    EXPECT_TRUE(torch::equal(actual.tokens.top_logprobs.squeeze(/*dim=*/1),
                             expected.top_logprobs));
    EXPECT_TRUE(torch::equal(actual.tokens.lengths,
                             torch::ones_like(actual.tokens.lengths)));
    EXPECT_TRUE(actual.tokens.tokens.device().is_cpu());
    EXPECT_FALSE(actual.tokens.tokens.is_pinned());
  }

  const torch::Device device_{torch::kPrivateUse1, 0};
  const torch::TensorOptions options_ =
      torch::dtype(torch::kBFloat16).device(device_);
  ModelArgs args_;
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
    snapshots.emplace_back(result.tokens.tokens.clone());
    retained.emplace_back(std::move(result.tokens));
  }
  pipeline.reset();
  for (uint32_t index = 0; index < retained.size(); ++index) {
    EXPECT_TRUE(torch::equal(retained[index].tokens, snapshots[index]));
  }
  compare_kv();
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
  EXPECT_FALSE(result.tokens.tokens.defined());
  auto batch = make_batch({3, 2}, {3, 2});
  reference(batch, BatchForwardType::PREFILL, {});
  submitted =
      pipeline->submit({view(batch), {BatchForwardType::PREFILL, 2}, {}});
  ASSERT_TRUE(submitted.status.ok());
  result = pipeline->take_result_async(submitted.task_id).get();
  EXPECT_TRUE(result.status.ok());
  EXPECT_FALSE(result.tokens.tokens.defined());
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
