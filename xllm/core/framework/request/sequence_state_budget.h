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

#include <atomic>
#include <cstdint>
#include <memory>

namespace xllm {

class SequenceStateReservation;

// Request admission budget, independent of batch size and Device row leases.
// Each request reserves its maximum Sequence expansion for its whole lifetime.
class SequenceStateBudget final {
 public:
  explicit SequenceStateBudget(uint32_t capacity);
  ~SequenceStateBudget();
  SequenceStateBudget(const SequenceStateBudget&) = delete;
  SequenceStateBudget& operator=(const SequenceStateBudget&) = delete;

  uint32_t capacity() const { return capacity_; }
  uint32_t available() const {
    return available_.load(std::memory_order_acquire);
  }

 private:
  friend class SequenceStateReservation;
  bool try_reserve(uint64_t count);
  void release(uint32_t count);

  const uint32_t capacity_;
  std::atomic<uint32_t> available_;
};

// Unique responsibility for returning a request's reservation. The shared
// budget may outlive its Scheduler, but contains no Engine or Device reference.
class SequenceStateReservation final {
 public:
  SequenceStateReservation() = default;
  ~SequenceStateReservation();
  SequenceStateReservation(SequenceStateReservation&& other) noexcept;
  SequenceStateReservation& operator=(
      SequenceStateReservation&& other) noexcept;
  SequenceStateReservation(const SequenceStateReservation&) = delete;
  SequenceStateReservation& operator=(const SequenceStateReservation&) = delete;

  // Expected rejection leaves both this reservation and the budget unchanged.
  // Count must be nonzero; an already held reservation cannot be overwritten.
  bool acquire(std::shared_ptr<SequenceStateBudget> budget, uint64_t count);
  void reset();
  bool empty() const { return count_ == 0; }
  uint32_t count() const { return count_; }

 private:
  std::shared_ptr<SequenceStateBudget> budget_;
  uint32_t count_ = 0;
};

}  // namespace xllm
