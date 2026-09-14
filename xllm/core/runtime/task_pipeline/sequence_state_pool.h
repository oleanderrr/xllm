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

#pragma once

#include <absl/container/flat_hash_map.h>
#include <torch/types.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "core/common/types.h"
#include "core/framework/request/sequence_state_key.h"

namespace xllm {

struct SequenceStateHandle {
  uint32_t row = 0;
  uint64_t generation = 0;
};

// Positions describe accepted execution, not a CPU copy of the token value.
// A read requires an earlier accepted publication at exactly that position.
// Positions fit int32 model indices; each publication advances the position.
struct SequenceStateAccess {
  SequenceStateKey key;
  std::optional<uint32_t> read_position;
  std::optional<uint32_t> publish_position;
};

class SequenceStatePool;

// One Task's row references. The pool outlives the lease.
// Move/reset/destruction run on the state executor, after the Task's last
// Device reader or writer.
class SequenceStateLease final {
 public:
  // Capacity is allocated once; reset releases references but retains storage.
  explicit SequenceStateLease(uint32_t capacity = 0);
  ~SequenceStateLease();
  SequenceStateLease(SequenceStateLease&& other) noexcept;
  SequenceStateLease& operator=(SequenceStateLease&& other) noexcept;
  SequenceStateLease(const SequenceStateLease&) = delete;
  SequenceStateLease& operator=(const SequenceStateLease&) = delete;

  std::span<const SequenceStateHandle> handles() const { return handles_; }
  bool empty() const { return handles_.empty(); }
  void reset();

 private:
  friend class SequenceStatePool;

  SequenceStatePool* pool_ = nullptr;  // Non-owning; see lifetime contract.
  std::vector<SequenceStateHandle> handles_;
};

// Fixed cross-Task state, independent of Slot storage. All Host map/lease
// operations run on one state executor; Launch only accesses the Device tensor
// through indices prepared while holding a lease. Device reads/writes are
// ordered on the owner's single task stream. There is no CPU token shadow.
class SequenceStatePool final {
 public:
  static Status create(uint32_t capacity,
                       const torch::Device& device,
                       std::unique_ptr<SequenceStatePool>& output);
  ~SequenceStatePool();
  SequenceStatePool(const SequenceStatePool&) = delete;
  SequenceStatePool& operator=(const SequenceStatePool&) = delete;

  // Atomically retire old keys and acquire this Task's accesses in row order.
  // All expected rejection preserves mapping, generations, references,
  // publication positions and output. Access/retirement keys must be unique
  // and disjoint. Retired rows with outstanding leases cannot be reused.
  // Unknown retirements are invalid; repeats are accepted only while the
  // retired row still exists because a previous Task retains its lease.
  // The caller preallocates the output lease's capacity at Slot creation.
  // An accepted publication must run before any later Task's Device read;
  // cancellation discards its result, never its queued execution.
  Status admit(std::span<const SequenceStateAccess> accesses,
               std::span<const SequenceStateKey> retired_keys,
               SequenceStateLease& output);

  uint32_t capacity() const { return static_cast<uint32_t>(entries_.size()); }
  uint32_t available_rows() const {
    return static_cast<uint32_t>(free_rows_.size());
  }
  uint64_t device_bytes() const { return tokens_.nbytes(); }
  // Identity query only; this does not acquire a reference or authorize I/O.
  bool is_current(const SequenceStateHandle& handle) const;
  // int64 [capacity], initially uninitialized. Publish precedes every read;
  // retaining this tensor alone does not prevent row reuse. Keep the lease.
  const torch::Tensor& tokens() const { return tokens_; }

 private:
  friend class SequenceStateLease;
  struct Entry {
    SequenceStateKey key;
    uint64_t generation = 1;
    uint32_t references = 0;
    bool retired = false;
    std::optional<uint32_t> published_position;
  };

  SequenceStatePool(uint32_t capacity, const torch::Device& device);
  Status validate_keys(std::vector<SequenceStateKey>& keys);
  void release(std::span<const SequenceStateHandle> handles);
  void reclaim(uint32_t row);

  torch::Tensor tokens_;
  std::vector<Entry> entries_;
  std::vector<uint32_t> free_rows_;
  // Scratch is reserved once; validation does not modify logical pool state.
  std::vector<SequenceStateKey> key_scratch_;
  std::vector<SequenceStateKey> retirement_scratch_;
  absl::flat_hash_map<std::pair<uint64_t, uint64_t>, uint32_t> rows_by_key_;
};

}  // namespace xllm
