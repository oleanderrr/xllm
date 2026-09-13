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

#include "core/runtime/task_pipeline/sequence_state_pool.h"

#include <c10/core/DeviceGuard.h>
#include <glog/logging.h>

#include <algorithm>
#include <limits>

namespace xllm {
namespace {

std::pair<uint64_t, uint64_t> map_key(const SequenceStateKey& key) {
  return {key.sequence_id, key.epoch};
}

Status invalid(const char* message) {
  return Status(StatusCode::INVALID_ARGUMENT, message);
}

}  // namespace

SequenceStateLease::SequenceStateLease(SequenceStatePool& pool,
                                       std::vector<SequenceStateHandle> handles)
    : pool_(&pool), handles_(std::move(handles)) {}

SequenceStateLease::~SequenceStateLease() { reset(); }

SequenceStateLease::SequenceStateLease(SequenceStateLease&& other) noexcept
    : pool_(std::exchange(other.pool_, nullptr)),
      handles_(std::move(other.handles_)) {
  other.handles_.clear();
}

SequenceStateLease& SequenceStateLease::operator=(
    SequenceStateLease&& other) noexcept {
  if (this != &other) {
    reset();
    pool_ = std::exchange(other.pool_, nullptr);
    handles_ = std::move(other.handles_);
    other.handles_.clear();
  }
  return *this;
}

void SequenceStateLease::reset() {
  if (pool_ != nullptr) {
    pool_->release(handles_);
  }
  pool_ = nullptr;
  handles_.clear();
}

Status SequenceStatePool::create(uint32_t capacity,
                                 const torch::Device& device,
                                 std::unique_ptr<SequenceStatePool>& output) {
  if (capacity == 0 || capacity > std::numeric_limits<int32_t>::max() ||
      device.type() != torch::kPrivateUse1 || !device.has_index()) {
    return invalid("Invalid sequence-state capacity or indexed NPU device.");
  }
  output = std::unique_ptr<SequenceStatePool>(
      new SequenceStatePool(capacity, device));
  return Status();
}

SequenceStatePool::SequenceStatePool(uint32_t capacity,
                                     const torch::Device& device)
    : entries_(capacity) {
  c10::DeviceGuard guard(device);
  tokens_ = torch::empty(
      {capacity}, torch::TensorOptions().dtype(torch::kInt64).device(device));
  free_rows_.reserve(capacity);
  key_scratch_.reserve(capacity);
  rows_by_key_.reserve(capacity);
  for (uint32_t row = capacity; row > 0; --row) {
    free_rows_.emplace_back(row - 1);
  }
}

SequenceStatePool::~SequenceStatePool() {
  for (const Entry& entry : entries_) {
    CHECK_EQ(entry.references, 0U) << "Sequence state still has Task leases.";
  }
}

Status SequenceStatePool::validate_keys(
    std::span<const SequenceStateKey> keys) {
  if (keys.size() > capacity()) {
    return Status(StatusCode::RESOURCE_EXHAUSTED,
                  "Sequence-state batch exceeds the fixed pool capacity.");
  }
  key_scratch_.assign(keys.begin(), keys.end());
  std::sort(key_scratch_.begin(),
            key_scratch_.end(),
            [](const SequenceStateKey& left, const SequenceStateKey& right) {
              return map_key(left) < map_key(right);
            });
  if (!key_scratch_.empty() && key_scratch_.front().sequence_id == 0) {
    return invalid("Sequence-state identity must be nonzero.");
  }
  const auto duplicate = std::adjacent_find(
      key_scratch_.begin(),
      key_scratch_.end(),
      [](const SequenceStateKey& left, const SequenceStateKey& right) {
        return map_key(left) == map_key(right);
      });
  if (duplicate != key_scratch_.end()) {
    return invalid("Duplicate sequence-state identity in one batch.");
  }
  return Status();
}

Status SequenceStatePool::acquire(std::span<const SequenceStateKey> keys,
                                  SequenceStateLease& output) {
  if (!output.empty()) {
    return Status(StatusCode::RESOURCE_EXHAUSTED,
                  "Retire the existing sequence-state lease before reuse.");
  }
  Status status = validate_keys(keys);
  if (!status.ok()) {
    return status;
  }
  uint32_t new_rows = 0;
  for (const SequenceStateKey& key : keys) {
    const auto it = rows_by_key_.find(map_key(key));
    if (it == rows_by_key_.end()) {
      ++new_rows;
      continue;
    }
    const Entry& entry = entries_[it->second];
    if (entry.retired) {
      return invalid("Sequence-state identity has been logically retired.");
    }
    if (entry.references == std::numeric_limits<uint32_t>::max()) {
      return Status(StatusCode::RESOURCE_EXHAUSTED,
                    "Sequence-state reference count exceeds capacity.");
    }
  }
  if (new_rows > free_rows_.size()) {
    return Status(StatusCode::RESOURCE_EXHAUSTED,
                  "No free sequence-state rows; retire live state first.");
  }
  if (keys.empty()) {
    return Status();
  }
  std::vector<SequenceStateHandle> handles;
  handles.reserve(keys.size());
  // Every expected rejection precedes row allocation or reference changes.
  for (const SequenceStateKey& key : keys) {
    auto it = rows_by_key_.find(map_key(key));
    if (it == rows_by_key_.end()) {
      const uint32_t row = free_rows_.back();
      free_rows_.pop_back();
      entries_[row].key = key;
      it = rows_by_key_.emplace(map_key(key), row).first;
    }
    Entry& entry = entries_[it->second];
    ++entry.references;
    handles.emplace_back(SequenceStateHandle{it->second, entry.generation});
  }
  output = SequenceStateLease(*this, std::move(handles));
  return Status();
}

Status SequenceStatePool::retire(std::span<const SequenceStateKey> keys) {
  Status status = validate_keys(keys);
  if (!status.ok()) {
    return status;
  }
  for (const SequenceStateKey& key : keys) {
    if (!rows_by_key_.contains(map_key(key))) {
      return invalid("Cannot retire an unknown sequence-state identity.");
    }
  }
  for (const SequenceStateKey& key : keys) {
    const uint32_t row = rows_by_key_.at(map_key(key));
    Entry& entry = entries_[row];
    entry.retired = true;
    if (entry.references == 0) {
      reclaim(row);
    }
  }
  return Status();
}

bool SequenceStatePool::is_current(const SequenceStateHandle& handle) const {
  return handle.row < capacity() && entries_[handle.row].key.sequence_id != 0 &&
         entries_[handle.row].generation == handle.generation;
}

void SequenceStatePool::release(std::span<const SequenceStateHandle> handles) {
  for (const SequenceStateHandle& handle : handles) {
    CHECK(is_current(handle)) << "Stale sequence-state lease.";
    CHECK_GT(entries_[handle.row].references, 0U);
  }
  for (const SequenceStateHandle& handle : handles) {
    Entry& entry = entries_[handle.row];
    --entry.references;
    if (entry.references == 0 && entry.retired) {
      reclaim(handle.row);
    }
  }
}

void SequenceStatePool::reclaim(uint32_t row) {
  Entry& entry = entries_[row];
  CHECK(entry.retired);
  CHECK_EQ(entry.references, 0U);
  CHECK_LT(entry.generation, std::numeric_limits<uint64_t>::max());
  CHECK_EQ(rows_by_key_.erase(map_key(entry.key)), 1U);
  entry.key = {};
  entry.retired = false;
  ++entry.generation;
  free_rows_.emplace_back(row);
}

}  // namespace xllm
