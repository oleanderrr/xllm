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

#include "core/runtime/task_pipeline/task_execution_pipeline.h"

#include <limits>

namespace xllm {

Status TaskExecutionPipeline::create(
    ThreadPool& state_executor,
    std::unique_ptr<LlmTaskProgram> program,
    std::unique_ptr<TaskExecutionPipeline>& output) {
  if (state_executor.size() != 1 || program == nullptr) {
    return Status(
        StatusCode::INVALID_ARGUMENT,
        "Task pipeline requires one state thread and an LLM program.");
  }
  output = std::unique_ptr<TaskExecutionPipeline>(
      new TaskExecutionPipeline(state_executor, std::move(program)));
  return Status();
}

TaskExecutionPipeline::TaskExecutionPipeline(
    ThreadPool& state_executor,
    std::unique_ptr<LlmTaskProgram> program)
    : state_executor_(state_executor),
      program_(std::move(program)),
      execution_(program_->slot_count()),
      completed_(program_->slot_count()) {
  folly::Promise<std::thread::id> promise;
  auto future = promise.getFuture();
  state_executor_.schedule([promise = std::move(promise)]() mutable {
    promise.setValue(std::this_thread::get_id());
  });
  state_thread_id_ = std::move(future).get();
  launch_thread_ = std::thread([this] { launch_loop(); });
}

void TaskExecutionPipeline::check_external_thread() const {
  CHECK(std::this_thread::get_id() != state_thread_id_);
  CHECK(std::this_thread::get_id() != launch_thread_.get_id());
}

TaskSubmission TaskExecutionPipeline::submit(const LlmTaskInput& input) {
  check_external_thread();
  folly::Promise<TaskSubmission> promise;
  auto future = promise.getFuture();
  state_executor_.schedule(
      [this, &input, promise = std::move(promise)]() mutable {
        if (accepted_.size() == program_->slot_count()) {
          promise.setValue(TaskSubmission{
              Status(StatusCode::RESOURCE_EXHAUSTED,
                     "All task Slots still hold unconsumed results."),
              0});
          return;
        }
        // With at most two Slots, one remaining Task identifies the other
        // free Slot. The accepted FIFO is the only Host ownership record.
        const uint32_t slot_id =
            accepted_.empty() ? 0U : 1U - accepted_.front().slot_id;
        CHECK_LT(next_task_id_, std::numeric_limits<uint64_t>::max());
        Status status = program_->prepare(slot_id, input);
        if (!status.ok()) {
          promise.setValue(TaskSubmission{std::move(status), 0});
          return;
        }
        const SlotTicket ticket{slot_id, next_task_id_++};
        accepted_.emplace_back(ticket);
        execution_.push(ticket);
        promise.setValue(TaskSubmission{Status(), ticket.task_id});
      });
  return std::move(future).get();
}

folly::Future<TaskResult> TaskExecutionPipeline::take_result_async(
    uint64_t task_id) {
  check_external_thread();
  folly::Promise<TaskResult> promise;
  auto future = promise.getFuture();
  state_executor_.schedule(
      [this, task_id, promise = std::move(promise)]() mutable {
        if (task_id == 0 || accepted_.empty() ||
            task_id != accepted_.front().task_id) {
          promise.setValue(
              TaskResult{Status(StatusCode::INVALID_ARGUMENT,
                                "TaskId is not the oldest unconsumed task."),
                         {}});
          return;
        }
        const SlotTicket ticket = wait_completed_front();
        auto tokens = program_->consume(ticket.slot_id);
        accepted_.pop_front();
        promise.setValue(TaskResult{Status(), std::move(tokens)});
      });
  return future;
}

TaskExecutionPipeline::SlotTicket
TaskExecutionPipeline::wait_completed_front() {
  CHECK(!accepted_.empty());
  const SlotTicket ticket = completed_.pop();
  CHECK_EQ(ticket.task_id, accepted_.front().task_id);
  CHECK_EQ(ticket.slot_id, accepted_.front().slot_id);
  return ticket;
}

void TaskExecutionPipeline::launch_loop() {
  while (true) {
    const SlotTicket ticket = execution_.pop();
    if (ticket.task_id == 0) {
      return;
    }
    program_->launch(ticket.slot_id);
    // The completion queue covers every Slot, including unrequested results.
    completed_.push(ticket);
  }
}

TaskExecutionPipeline::~TaskExecutionPipeline() {
  check_external_thread();
  folly::Promise<folly::Unit> promise;
  auto future = promise.getFuture();
  state_executor_.schedule([promise = std::move(promise)]() mutable {
    promise.setValue(folly::unit);
  });
  std::move(future).get();
  execution_.push(SlotTicket{});
  launch_thread_.join();
  while (!accepted_.empty()) {
    const SlotTicket ticket = wait_completed_front();
    program_->discard(ticket.slot_id);
    accepted_.pop_front();
  }
  CHECK(completed_.empty());
}

}  // namespace xllm
