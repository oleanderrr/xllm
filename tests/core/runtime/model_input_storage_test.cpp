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

#include "core/runtime/task_pipeline/model_input_storage.h"

#include <gtest/gtest.h>

#include <array>
#include <limits>
#include <utility>

namespace xllm {
namespace {

std::unique_ptr<ModelInputStorage> make_storage() {
  std::unique_ptr<ModelInputStorage> storage;
  CHECK(ModelInputStorage::create(
            {5, 2, 3}, torch::Device(torch::kPrivateUse1, 0), storage)
            .ok());
  return storage;
}

std::array<torch::Tensor, 7> fields(const ModelInputTensors& tensors) {
  return {tensors.token_ids,
          tensors.positions,
          tensors.new_cache_slots,
          tensors.q_seq_lens,
          tensors.kv_seq_lens,
          tensors.q_cu_seq_lens,
          tensors.block_tables};
}

TEST(ModelInputStorageTest, AllocatesPinnedHostAndRequestedDevice) {
  auto storage = make_storage();
  EXPECT_TRUE(storage->host_buffer().device().is_cpu());
  EXPECT_TRUE(storage->host_buffer().is_pinned());
  EXPECT_EQ(storage->device_buffer().device(),
            torch::Device(torch::kPrivateUse1, 0));
  EXPECT_EQ(storage->host_buffer().nbytes(), 176);
  EXPECT_EQ(storage->device_buffer().nbytes(), 176);
}

TEST(ModelInputStorageTest, BindsAlignedFixedShapeViewsIntoBothBuffers) {
  auto storage = make_storage();
  const std::array<uint64_t, 7> offsets = {0, 32, 64, 96, 112, 128, 144};
  const std::array<int64_t, 7> sizes = {5, 5, 5, 2, 2, 2, 6};
  const auto host = fields(storage->host());
  const auto device = fields(storage->device());
  for (uint32_t i = 0; i < offsets.size(); ++i) {
    SCOPED_TRACE(i);
    EXPECT_EQ(host[i].scalar_type(), torch::kInt32);
    EXPECT_EQ(device[i].scalar_type(), torch::kInt32);
    EXPECT_EQ(host[i].numel(), sizes[i]);
    EXPECT_EQ(device[i].sizes(), host[i].sizes());
    EXPECT_EQ(reinterpret_cast<uintptr_t>(host[i].data_ptr()),
              reinterpret_cast<uintptr_t>(storage->host_buffer().data_ptr()) +
                  offsets[i]);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(device[i].data_ptr()),
              reinterpret_cast<uintptr_t>(storage->device_buffer().data_ptr()) +
                  offsets[i]);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(device[i].data_ptr()) %
                  kModelInputAlignment,
              0);
    EXPECT_TRUE(host[i].is_contiguous());
    EXPECT_TRUE(device[i].is_contiguous());
  }
  EXPECT_EQ(storage->device().block_tables.size(0), 2);
  EXPECT_EQ(storage->device().block_tables.size(1), 3);
  EXPECT_EQ(storage->device().block_tables.stride(0), 3);
  EXPECT_EQ(storage->device().block_tables.stride(1), 1);
}

TEST(ModelInputStorageTest, FieldWritesAliasBackingAndKeepOtherFieldsIntact) {
  auto storage = make_storage();
  storage->host_buffer().fill_(-1);
  storage->host().positions.fill_(23);
  const int32_t* host = storage->host_buffer().data_ptr<int32_t>();
  EXPECT_EQ(host[0], -1);
  EXPECT_EQ(host[8], 23);
  EXPECT_EQ(host[12], 23);
  EXPECT_EQ(host[13], -1);
  storage->device_buffer().fill_(-1);
  storage->device().block_tables.fill_(47);
  torch::Tensor copied = storage->device_buffer().cpu();
  const int32_t* values = copied.data_ptr<int32_t>();
  EXPECT_EQ(values[32], -1);
  EXPECT_EQ(values[36], 47);
  EXPECT_EQ(values[41], 47);
  EXPECT_EQ(values[42], -1);
}

TEST(ModelInputStorageTest, IndependentInstancesDoNotAlias) {
  auto first = make_storage();
  auto second = make_storage();
  EXPECT_NE(first->host_buffer().data_ptr(), second->host_buffer().data_ptr());
  EXPECT_NE(first->device_buffer().data_ptr(),
            second->device_buffer().data_ptr());
  first->device().token_ids.fill_(19);
  second->device().token_ids.fill_(31);
  EXPECT_TRUE(first->device().token_ids.cpu().eq(19).all().item<bool>());
  EXPECT_TRUE(second->device().token_ids.cpu().eq(31).all().item<bool>());
}

TEST(ModelInputStorageTest, OwnershipTransferPreservesAddresses) {
  auto source = make_storage();
  const void* host = source->host().token_ids.data_ptr();
  const void* device = source->device().token_ids.data_ptr();
  auto destination = std::move(source);
  EXPECT_EQ(source, nullptr);
  EXPECT_EQ(destination->host().token_ids.data_ptr(), host);
  EXPECT_EQ(destination->device().token_ids.data_ptr(), device);
}

TEST(ModelInputStorageTest, BorrowedViewsKeepBackingStorageAlive) {
  torch::Tensor host;
  torch::Tensor device;
  {
    auto storage = make_storage();
    host = storage->host().token_ids;
    device = storage->device().token_ids;
  }
  host.fill_(61);
  device.copy_(host);
  EXPECT_TRUE(device.cpu().eq(61).all().item<bool>());
  EXPECT_TRUE(host.is_pinned());
}

TEST(ModelInputStorageTest, InvalidCapacityPreservesExistingOwner) {
  auto storage = make_storage();
  const ModelInputStorage* original = storage.get();
  const uint32_t max = std::numeric_limits<uint32_t>::max();
  for (const ModelInputCapacity capacity :
       {ModelInputCapacity{0, 2, 3},
        ModelInputCapacity{5, 0, 3},
        ModelInputCapacity{5, 2, 0},
        ModelInputCapacity{max, max, max}}) {
    EXPECT_FALSE(ModelInputStorage::create(
                     capacity, torch::Device(torch::kPrivateUse1, 0), storage)
                     .ok());
    EXPECT_EQ(storage.get(), original);
  }
}

TEST(ModelInputStorageTest, UnsupportedOrUnindexedDevicePreservesOwner) {
  auto storage = make_storage();
  const ModelInputStorage* original = storage.get();
  for (const torch::Device& device :
       {torch::Device(torch::kCPU), torch::Device(torch::kPrivateUse1)}) {
    EXPECT_FALSE(ModelInputStorage::create({5, 2, 3}, device, storage).ok());
    EXPECT_EQ(storage.get(), original);
  }
}

}  // namespace
}  // namespace xllm
