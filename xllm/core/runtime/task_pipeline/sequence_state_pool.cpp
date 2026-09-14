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

SequenceStateLease::SequenceStateLease(uint32_t capacity) {
  handles_.reserve(capacity);
}

SequenceStateLease::~SequenceStateLease() { reset(); }

SequenceStateLease::SequenceStateLease(SequenceStateLease&& other) noexcept
    : pool_(std::exchange(other.pool_, /*new_value=*/nullptr)),
      handles_(std::move(other.handles_)) {
  other.handles_.clear();
}

SequenceStateLease& SequenceStateLease::operator=(
    SequenceStateLease&& other) noexcept {
  if (this != &other) {
    reset();
    pool_ = std::exchange(other.pool_, /*new_value=*/nullptr);
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
  retirement_scratch_.reserve(capacity);
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

Status SequenceStatePool::validate_keys(std::vector<SequenceStateKey>& keys) {
  std::sort(keys.begin(),
            keys.end(),
            [](const SequenceStateKey& left, const SequenceStateKey& right) {
              return map_key(left) < map_key(right);
            });
  if (!keys.empty() && keys.front().sequence_id == 0) {
    return invalid("Sequence-state identity must be nonzero.");
  }
  const auto duplicate = std::adjacent_find(
      keys.begin(),
      keys.end(),
      [](const SequenceStateKey& left, const SequenceStateKey& right) {
        return map_key(left) == map_key(right);
      });
  if (duplicate != keys.end()) {
    return invalid("Duplicate sequence-state identity in one batch.");
  }
  return Status();
}

Status SequenceStatePool::admit(std::span<const SequenceStateAccess> accesses,
                                std::span<const SequenceStateKey> retired_keys,
                                SequenceStateLease& output) {
  if (!output.empty() || accesses.size() > output.handles_.capacity()) {
    return Status(StatusCode::RESOURCE_EXHAUSTED,
                  "Sequence-state lease is occupied or lacks fixed capacity.");
  }
  if (accesses.size() > capacity() || retired_keys.size() > capacity()) {
    return Status(StatusCode::RESOURCE_EXHAUSTED,
                  "Sequence-state input exceeds the fixed pool capacity.");
  }
  key_scratch_.clear();
  for (const SequenceStateAccess& access : accesses) {
    key_scratch_.emplace_back(access.key);
  }
  Status status = validate_keys(key_scratch_);
  if (!status.ok()) {
    return status;
  }
  retirement_scratch_.assign(retired_keys.begin(), retired_keys.end());
  status = validate_keys(retirement_scratch_);
  if (!status.ok()) {
    return status;
  }
  uint32_t reclaimable = 0;
  for (const SequenceStateKey& key : retired_keys) {
    const auto it = rows_by_key_.find(map_key(key));
    if (it == rows_by_key_.end()) {
      return invalid("Cannot retire an unknown sequence-state identity.");
    }
    reclaimable += entries_[it->second].references == 0 ? 1U : 0U;
  }
  uint32_t new_rows = 0;
  for (const SequenceStateAccess& access : accesses) {
    if ((access.read_position.has_value() &&
         *access.read_position > std::numeric_limits<int32_t>::max()) ||
        (access.publish_position.has_value() &&
         *access.publish_position > std::numeric_limits<int32_t>::max())) {
      return invalid("Sequence-state position exceeds the model index range.");
    }
    if (std::binary_search(
            retirement_scratch_.begin(),
            retirement_scratch_.end(),
            access.key,
            [](const SequenceStateKey& left, const SequenceStateKey& right) {
              return map_key(left) < map_key(right);
            })) {
      return invalid("Cannot access and retire the same sequence-state key.");
    }
    const auto it = rows_by_key_.find(map_key(access.key));
    if (it == rows_by_key_.end()) {
      if (access.read_position.has_value()) {
        return invalid("Sequence-state read has no accepted publication.");
      }
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
    if (access.read_position.has_value() &&
        access.read_position != entry.published_position) {
      return invalid("Sequence-state read does not match accepted position.");
    }
    if (access.publish_position.has_value() &&
        entry.published_position.has_value() &&
        *access.publish_position <= *entry.published_position) {
      return invalid("Sequence-state publication must advance its position.");
    }
  }
  if (new_rows > free_rows_.size() + reclaimable) {
    return Status(
        StatusCode::RESOURCE_EXHAUSTED,
        "Sequence-state rows remain live or leased by accepted Tasks.");
  }
  // No expected rejection follows this point. Apply the validated retirement
  // before allocation so one input can replace unleased state in a full pool.
  for (const SequenceStateKey& key : retired_keys) {
    const uint32_t row = rows_by_key_.at(map_key(key));
    Entry& entry = entries_[row];
    entry.retired = true;
    if (entry.references == 0) {
      reclaim(row);
    }
  }
  if (accesses.empty()) {
    return Status();
  }
  CHECK_EQ(output.pool_, nullptr);
  output.pool_ = this;
  for (const SequenceStateAccess& access : accesses) {
    auto it = rows_by_key_.find(map_key(access.key));
    if (it == rows_by_key_.end()) {
      const uint32_t row = free_rows_.back();
      free_rows_.pop_back();
      entries_[row].key = access.key;
      it = rows_by_key_.emplace(map_key(access.key), row).first;
    }
    Entry& entry = entries_[it->second];
    ++entry.references;
    if (access.publish_position.has_value()) {
      entry.published_position = access.publish_position;
    }
    output.handles_.emplace_back(
        SequenceStateHandle{it->second, entry.generation});
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
  entry.published_position.reset();
  ++entry.generation;
  free_rows_.emplace_back(row);
}

}  // namespace xllm
