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

#include <algorithm>
#include <array>
#include <cstdlib>
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
#include "core/runtime/executor.h"
#include "core/runtime/task_pipeline/model_input_binding.h"
#include "models/model_registry.h"

namespace xllm {
namespace {

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
    torch::set_num_threads(1);
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
    std::fill(batch.positions.begin(), batch.positions.end(), -99);
    const StreamEventPtr ready = prepare_->record_event();
    ASSERT_NE(ready, nullptr);
    // The reference uses the original Python Executor metadata construction
    // and an independent KV cache. Slot input storage is shared read-only.
    Stream original(c10_npu::getCurrentNPUStream(device_.index()));
    ASSERT_TRUE(original.wait_event(ready));
    ModelInputParams reference = binding.params();
    reference.attn_metadata.reset();
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
      EXPECT_TRUE(
          torch::equal(eager_logits.argmax(-1), prepared_logits.argmax(-1)));
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

INSTANTIATE_TEST_SUITE_P(PrefillModes, Qwen3SlotForwardTest, ::testing::Bool());

}  // namespace
}  // namespace xllm
