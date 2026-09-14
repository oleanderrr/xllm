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

#include "core/framework/request/sequence_state_budget.h"

#include <glog/logging.h>

#include <utility>

namespace xllm {

SequenceStateBudget::SequenceStateBudget(uint32_t capacity)
    : capacity_(capacity), available_(capacity) {
  CHECK_GT(capacity, 0U);
}

SequenceStateBudget::~SequenceStateBudget() {
  CHECK_EQ(available(), capacity_);
}

bool SequenceStateBudget::try_reserve(uint64_t count) {
  if (count == 0 || count > capacity_) {
    return false;
  }
  const uint32_t amount = static_cast<uint32_t>(count);
  uint32_t available = available_.load(std::memory_order_relaxed);
  while (available >= amount) {
    if (available_.compare_exchange_weak(available,
                                         available - amount,
                                         std::memory_order_acquire,
                                         std::memory_order_relaxed)) {
      return true;
    }
  }
  return false;
}

void SequenceStateBudget::release(uint32_t count) {
  CHECK_GT(count, 0U);
  CHECK_LE(count, capacity_);
  // Sequence retirement notifications precede this release. Admission using
  // the returned quota must observe them before Engine drains its next input.
  const uint32_t previous =
      available_.fetch_add(count, std::memory_order_release);
  CHECK_LE(previous, capacity_ - count);
}

SequenceStateReservation::~SequenceStateReservation() { reset(); }

SequenceStateReservation::SequenceStateReservation(
    SequenceStateReservation&& other) noexcept
    : budget_(std::move(other.budget_)),
      count_(std::exchange(other.count_, /*new_value=*/0)) {}

SequenceStateReservation& SequenceStateReservation::operator=(
    SequenceStateReservation&& other) noexcept {
  if (this != &other) {
    reset();
    budget_ = std::move(other.budget_);
    count_ = std::exchange(other.count_, /*new_value=*/0);
  }
  return *this;
}

bool SequenceStateReservation::acquire(
    std::shared_ptr<SequenceStateBudget> budget,
    uint64_t count) {
  if (!empty() || budget == nullptr || !budget->try_reserve(count)) {
    return false;
  }
  budget_ = std::move(budget);
  count_ = static_cast<uint32_t>(count);
  return true;
}

void SequenceStateReservation::reset() {
  if (budget_ != nullptr) {
    budget_->release(count_);
    budget_.reset();
    count_ = 0;
  }
}

}  // namespace xllm
