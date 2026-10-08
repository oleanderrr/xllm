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

#include <chrono>
#include <functional>
#include <future>
#include <memory>
#include <vector>

#include "core/framework/config/model_config.h"
#include "core/layers/common/attention_metadata.h"
#include "core/runtime/executor_impl_factory.h"
#include "core/runtime/task_execution_pipeline.h"

namespace xllm {
namespace {

struct RecurrentObservation {
  torch::Tensor indices;
  torch::Tensor query_starts;
  torch::Tensor warm;
  torch::Tensor conv;
  torch::Tensor ssm;
  bool enable_graph = false;
};

// Advance real device state by a visible amount, retaining the state consumed
// by each forward. The first layer intentionally has no full-attention cache.
class RecurrentTestModel final : public CausalLM {
 public:
  explicit RecurrentTestModel(const torch::Device& device)
      : options_(torch::TensorOptions().device(device).dtype(torch::kFloat32)) {
    observations.reserve(8);
  }

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& /*positions*/,
                      std::vector<KVCache>& caches,
                      const ModelInputParams& params) override {
    if (before_forward) {
      before_forward(static_cast<uint32_t>(observations.size()));
    }
    const auto indices =
        params.embedding.linear_state_indices.to(torch::kInt64);
    auto conv = caches.front().get_conv_cache();
    auto ssm = caches.front().get_ssm_cache();
    const auto previous_conv = conv.index_select(/*dim=*/0, indices);
    const auto previous_ssm = ssm.index_select(/*dim=*/0, indices);
    observations.emplace_back(
        RecurrentObservation{params.embedding.linear_state_indices.clone(),
                             params.attn_metadata->q_cu_seq_lens.clone(),
                             params.attn_metadata->has_initial_states.clone(),
                             previous_conv.clone(),
                             previous_ssm.clone(),
                             params.enable_graph});
    const auto active = indices.gt(0).view({-1, 1, 1});
    const auto delta =
        params.attention.device.q_seq_lens.to(options_).view({-1, 1, 1});
    conv.index_copy_(
        /*dim=*/0,
        indices,
        torch::where(active, previous_conv + delta, previous_conv));
    ssm.index_copy_(
        /*dim=*/0,
        indices,
        torch::where(active, previous_ssm + delta * 10, previous_ssm));
    return ModelOutput(tokens.to(options_).view({-1, 1}));
  }

  torch::Tensor logits(const torch::Tensor& hidden,
                       const torch::Tensor& selected) override {
    auto result = torch::zeros({selected.numel(), 32}, hidden.options());
    result.select(/*dim=*/1, /*index=*/1).fill_(20);
    return result;
  }
  void load_model(std::unique_ptr<ModelLoader> /*loader*/) override {}
  torch::Device device() const override { return options_.device(); }
  const torch::TensorOptions& options() const override { return options_; }
  void prepare_expert_weight(int32_t /*layer_id*/,
                             const std::vector<int32_t>& /*experts*/) override {
  }
  void update_expert_weight(int32_t /*layer_id*/) override {}

  std::function<void(uint32_t)> before_forward;
  std::vector<RecurrentObservation> observations;

 private:
  torch::TensorOptions options_;
};

class RecurrentTestExecutor final : public ExecutorImpl {
 public:
  explicit RecurrentTestExecutor(CausalLM* model) : model_(model) {}
  bool supports_prepared_attention_metadata() const override { return true; }
  void prepare_attention_metadata(std::vector<KVCache>& /*caches*/,
                                  ModelInputParams& /*params*/) override {}
  void warmup_prepared_graph(const torch::Tensor& tokens,
                             const torch::Tensor& positions,
                             std::vector<KVCache>& caches,
                             const ModelInputParams& params) override {
    // Exercise the model reads/writes that real capture warmup performs;
    // graph storage and replay are covered by the Python runner tests.
    model_->forward(tokens, positions, caches, params);
  }
  ModelOutput run(const torch::Tensor& tokens,
                  const torch::Tensor& positions,
                  std::vector<KVCache>& caches,
                  const ModelInputParams& params) override {
    return model_->forward(tokens, positions, caches, params);
  }

 private:
  CausalLM* model_;
};

class RecurrentPipelineTest : public ::testing::Test {
 protected:
  void SetUp() override {
    static const bool registered =
        ExecutorImplFactory::get_instance().register_creator(
            "recurrent_pipeline_test",
            [](CausalLM* model,
               const ModelArgs& /*args*/,
               const torch::Device& /*device*/,
               const runtime::Options& /*options*/) {
              return std::make_unique<RecurrentTestExecutor>(model);
            });
    ASSERT_TRUE(registered);
    previous_model_impl_ = ModelConfig::get_instance().model_impl();
    ModelConfig::get_instance().model_impl("auto");
    model_ = std::make_unique<RecurrentTestModel>(device_);
    runtime::Options options;
    options.backend("recurrent_pipeline_test").enable_graph(false);
    executor_ =
        std::make_unique<Executor>(model_.get(), ModelArgs{}, device_, options);
    caches_.emplace_back(LinearAttentionKVCacheTensors{
        torch::full({8, 1, 1}, 7.0, model_->options()),
        torch::full({8, 1, 1}, 11.0, model_->options())});
    caches_.emplace_back(
        KVCacheTensors{torch::zeros({4, 16, 1, 1}, model_->options()),
                       torch::zeros({4, 16, 1, 1}, model_->options())});
    capacity_.slot_count = 2;
    capacity_.model = {16, 4, 2};
    capacity_.model.enable_linear_attention = true;
    capacity_.max_kv_seq_len = 32;
    capacity_.max_positions = 32;
    capacity_.logical_block_size = 16;
    capacity_.vocab_size = 32;
    capacity_.max_unique_tokens = 32;
    capacity_.hidden_size = 1;
    capacity_.chunked_prefill = true;
    ASSERT_EQ(aclrtSynchronizeDevice(), ACL_SUCCESS);
    const Status status = TaskExecutionPipeline::create(
        state_thread_, *model_, *executor_, caches_, capacity_, pipeline_);
    ASSERT_TRUE(status.ok()) << status.message();
  }

  void TearDown() override {
    pipeline_.reset();
    EXPECT_EQ(aclrtSynchronizeDevice(), ACL_SUCCESS);
    ModelConfig::get_instance().model_impl(previous_model_impl_);
  }

  LlmForwardInput input(int32_t slot,
                        int32_t query,
                        int32_t cached = 0,
                        bool decode = false) const {
    LlmForwardInput input;
    input.token_ids = torch::ones({query}, torch::kInt32);
    input.positions = torch::arange(cached, cached + query, torch::kInt32);
    auto& params = input.input_params;
    params.meta.batch_forward_type = decode ? BatchForwardType::DECODE
                                     : cached == 0
                                         ? BatchForwardType::PREFILL
                                         : BatchForwardType::CHUNKED_PREFILL;
    params.meta.num_sequences = 1;
    params.meta.actual_num_sequences = 1;
    params.attention.host.q_seq_lens = {query};
    params.attention.host.kv_seq_lens = {cached + query};
    params.attention.host.q_cu_seq_lens = {query};
    params.attention.host.block_tables = torch::tensor({{0, 1}}, torch::kInt32);
    params.attention.device.new_cache_slots = input.positions.clone();
    params.embedding.linear_state_ids = {slot};
    params.embedding.linear_state_indices =
        torch::tensor({slot}, torch::kInt32);
    LinearStateCacheOp op;
    op.linear_state_id = slot;
    op.reset_requested = cached == 0;
    params.linear_state_cache_ops.emplace_back(op);
    return input;
  }

  void finish(const TaskSubmission& submission) {
    ASSERT_TRUE(submission.status.ok()) << submission.status.message();
    const auto result = pipeline_->take_result_async(submission.task_id).get();
    ASSERT_TRUE(result.status.ok()) << result.status.message();
  }

  const torch::Device device_{torch::kPrivateUse1, 0};
  std::string previous_model_impl_;
  ThreadPool state_thread_{1};
  LlmTaskCapacity capacity_;
  std::unique_ptr<RecurrentTestModel> model_;
  std::unique_ptr<Executor> executor_;
  std::vector<KVCache> caches_;
  std::unique_ptr<TaskExecutionPipeline> pipeline_;
};

TEST_F(RecurrentPipelineTest, ColdChunkedDecodeAndReusedSlotPreserveState) {
  finish(pipeline_->submit(input(/*slot=*/1, /*query=*/2)));
  finish(pipeline_->submit(input(/*slot=*/1, /*query=*/2, /*cached=*/2)));
  finish(pipeline_->submit(
      input(/*slot=*/1, /*query=*/1, /*cached=*/4, /*decode=*/true)));
  finish(pipeline_->submit(input(/*slot=*/1, /*query=*/1)));
  ASSERT_EQ(model_->observations.size(), 4);
  const std::vector<float> conv{0, 2, 4, 0};
  const std::vector<float> ssm{0, 20, 40, 0};
  for (uint32_t index = 0; index < conv.size(); ++index) {
    const auto& observed = model_->observations[index];
    EXPECT_EQ(observed.conv.cpu().item<float>(), conv[index]);
    EXPECT_EQ(observed.ssm.cpu().item<float>(), ssm[index]);
    EXPECT_EQ(observed.warm.cpu().item<bool>(), index == 1 || index == 2);
    EXPECT_EQ(observed.query_starts.numel(), 2);
    EXPECT_EQ(observed.query_starts.cpu()[0].item<int32_t>(), 0);
  }
  EXPECT_EQ(caches_.front().get_conv_cache()[1].cpu().item<float>(), 1);
  EXPECT_EQ(caches_.front().get_conv_cache()[2].cpu().item<float>(), 7);
}

TEST_F(RecurrentPipelineTest,
       RestoreWaitsForEarlierForwardAndOwnsInputOperations) {
  std::promise<void> entered;
  auto entered_future = entered.get_future();
  std::promise<void> release;
  auto release_future = release.get_future();
  model_->before_forward = [&](uint32_t call) {
    if (call == 0) {
      entered.set_value();
      release_future.wait();
    }
  };
  const auto first = pipeline_->submit(input(/*slot=*/1, /*query=*/2));
  const auto entered_status = entered_future.wait_for(std::chrono::seconds(10));
  EXPECT_EQ(entered_status, std::future_status::ready);
  auto restore = input(/*slot=*/2, /*query=*/1, /*cached=*/2);
  auto& op = restore.input_params.linear_state_cache_ops.front();
  op.restore_requested = true;
  op.restore_src_slot_id = 1;
  const auto second = pipeline_->submit(restore);
  restore.input_params.linear_state_cache_ops.clear();
  restore.input_params.embedding.linear_state_ids.front() = 3;
  restore.input_params.embedding.linear_state_indices.fill_(3);
  release.set_value();
  finish(first);
  finish(second);
  ASSERT_EQ(model_->observations.size(), 2);
  const auto& observed = model_->observations[1];
  EXPECT_EQ(observed.indices.cpu().item<int32_t>(), 2);
  EXPECT_EQ(observed.conv.cpu().item<float>(), 2);
  EXPECT_EQ(observed.ssm.cpu().item<float>(), 20);
  EXPECT_TRUE(observed.warm.cpu().item<bool>());
  EXPECT_EQ(caches_.front().get_ssm_cache()[2].cpu().item<float>(), 30);
  EXPECT_EQ(caches_.front().get_ssm_cache()[3].cpu().item<float>(), 11);
}

TEST_F(RecurrentPipelineTest, InvalidRestoreDoesNotMutateRecurrentCache) {
  auto invalid = input(/*slot=*/2, /*query=*/1, /*cached=*/2);
  auto& op = invalid.input_params.linear_state_cache_ops.front();
  op.restore_requested = true;
  op.restore_src_slot_id = 8;
  EXPECT_FALSE(pipeline_->submit(invalid).status.ok());
  op.restore_src_slot_id = 1;
  op.reset_requested = true;
  EXPECT_FALSE(pipeline_->submit(invalid).status.ok());
  op.reset_requested = false;
  op.linear_state_id = 3;
  EXPECT_FALSE(pipeline_->submit(invalid).status.ok());
  EXPECT_TRUE(model_->observations.empty());
  EXPECT_EQ(caches_.front().get_conv_cache()[2].cpu().item<float>(), 7);
  EXPECT_EQ(caches_.front().get_ssm_cache()[2].cpu().item<float>(), 11);
  finish(pipeline_->submit(input(/*slot=*/2, /*query=*/1)));
  EXPECT_EQ(caches_.front().get_ssm_cache()[2].cpu().item<float>(), 10);
}

TEST_F(RecurrentPipelineTest, PaddedAndEmptySlotsUseReservedRecurrentState) {
  SlotBufferCapacity buffer_capacity;
  buffer_capacity.model = capacity_.model;
  buffer_capacity.max_unique_tokens = capacity_.max_unique_tokens;
  buffer_capacity.vocab_size = capacity_.vocab_size;
  std::unique_ptr<SlotBuffer> buffer;
  ASSERT_TRUE(SlotBuffer::create(buffer_capacity, device_, buffer).ok());
  Stream stream(device_);
  auto decode = input(/*slot=*/2, /*query=*/1, /*cached=*/2, /*decode=*/true);
  ASSERT_TRUE(buffer->validate(decode, /*previous_rows=*/0, stream).ok());
  buffer->prepare(decode, stream, /*padded_batch_size=*/4);
  ASSERT_EQ(stream.synchronize(), 0);
  const auto& params = buffer->model_params();
  const void* indices_storage =
      params.embedding.linear_state_indices.data_ptr();
  EXPECT_TRUE(torch::equal(params.embedding.linear_state_indices.cpu(),
                           torch::tensor({2, 0, 0, 0}, torch::kInt32)));
  EXPECT_TRUE(
      torch::equal(params.attn_metadata->has_initial_states.cpu(),
                   torch::tensor({true, false, false, false}, torch::kBool)));
  LlmForwardInput empty;
  empty.input_params.meta.batch_forward_type = BatchForwardType::DECODE;
  buffer->prepare(empty, stream, /*padded_batch_size=*/4);
  ASSERT_EQ(stream.synchronize(), 0);
  EXPECT_EQ(params.embedding.linear_state_indices.data_ptr(), indices_storage);
  EXPECT_EQ(params.embedding.linear_state_indices.cpu()
                .count_nonzero()
                .item<int64_t>(),
            0);
  EXPECT_EQ(params.attn_metadata->has_initial_states.cpu()
                .count_nonzero()
                .item<int64_t>(),
            0);
}

TEST_F(RecurrentPipelineTest,
       GraphWarmupPreservesLiveStateAndPaddedDecodeRestoresOnlyActualRows) {
  pipeline_.reset();
  capacity_.max_graph_batch_size = 4;
  const Status status = TaskExecutionPipeline::create(
      state_thread_, *model_, *executor_, caches_, capacity_, pipeline_);
  ASSERT_TRUE(status.ok()) << status.message();
  const auto initial_conv = caches_.front().get_conv_cache().cpu().clone();
  const auto initial_ssm = caches_.front().get_ssm_cache().cpu().clone();

  auto warmup = input(/*slot=*/1, /*query=*/1, /*cached=*/1, /*decode=*/true);
  warmup.token_ids = torch::ones({4}, torch::kInt32);
  warmup.positions = torch::ones({4}, torch::kInt32);
  auto& params = warmup.input_params;
  params.meta.is_graph_warmup = true;
  params.meta.num_sequences = 4;
  params.meta.actual_num_sequences = 4;
  params.attention.host.q_seq_lens.assign(4, 1);
  params.attention.host.kv_seq_lens.assign(4, 2);
  params.attention.host.q_cu_seq_lens = {1, 2, 3, 4};
  params.attention.host.block_tables =
      torch::tensor({{0, 1}}, torch::kInt32).repeat({4, 1});
  params.attention.device.new_cache_slots = warmup.positions.clone();
  params.embedding.linear_state_ids = {1, 2, 3, 4};
  params.embedding.linear_state_indices = torch::arange(1, 5, torch::kInt32);
  params.linear_state_cache_ops.resize(4);
  for (uint32_t row = 0; row < 4; ++row) {
    params.linear_state_cache_ops[row].linear_state_id =
        static_cast<int32_t>(row + 1);
    params.linear_state_cache_ops[row].reset_requested = true;
  }
  finish(pipeline_->submit(warmup));
  // Both Slots are captured and the admitted warmup Task then executes.
  ASSERT_EQ(model_->observations.size(), 3);
  for (const auto& observed : model_->observations) {
    EXPECT_TRUE(observed.enable_graph);
    EXPECT_EQ(observed.indices.cpu().count_nonzero().item<int64_t>(), 0);
    EXPECT_EQ(observed.warm.cpu().count_nonzero().item<int64_t>(), 0);
    EXPECT_TRUE(torch::equal(observed.query_starts.cpu(),
                             torch::tensor({0, 1, 2, 3, 4}, torch::kInt32)));
  }
  EXPECT_TRUE(
      torch::equal(caches_.front().get_conv_cache().cpu(), initial_conv));
  EXPECT_TRUE(torch::equal(caches_.front().get_ssm_cache().cpu(), initial_ssm));

  finish(pipeline_->submit(input(/*slot=*/1, /*query=*/2)));
  auto decode = input(/*slot=*/2, /*query=*/1, /*cached=*/2, /*decode=*/true);
  auto& restore = decode.input_params.linear_state_cache_ops.front();
  restore.restore_requested = true;
  restore.restore_src_slot_id = 1;
  finish(pipeline_->submit(decode));
  ASSERT_EQ(model_->observations.size(), 5);
  const auto& observed = model_->observations.back();
  EXPECT_TRUE(observed.enable_graph);
  EXPECT_TRUE(torch::equal(observed.indices.cpu(),
                           torch::tensor({2, 0, 0, 0}, torch::kInt32)));
  EXPECT_TRUE(
      torch::equal(observed.warm.cpu(),
                   torch::tensor({true, false, false, false}, torch::kBool)));
  EXPECT_EQ(observed.conv.cpu()[0].item<float>(), 2);
  EXPECT_EQ(observed.ssm.cpu()[0].item<float>(), 20);
  EXPECT_EQ(caches_.front().get_conv_cache()[2].cpu().item<float>(), 3);
  EXPECT_EQ(caches_.front().get_ssm_cache()[2].cpu().item<float>(), 30);
  EXPECT_EQ(caches_.front().get_conv_cache()[0].cpu().item<float>(), 7);
  EXPECT_EQ(caches_.front().get_ssm_cache()[0].cpu().item<float>(), 11);
}

}  // namespace
}  // namespace xllm
