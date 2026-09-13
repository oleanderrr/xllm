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
    : state_executor_(state_executor), program_(std::move(program)) {
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
        if (active_task_id_ != 0) {
          promise.setValue(TaskSubmission{
              Status(StatusCode::RESOURCE_EXHAUSTED,
                     "Task Slot still holds an unconsumed result."),
              0});
          return;
        }
        Status status = program_->prepare(input);
        if (!status.ok()) {
          promise.setValue(TaskSubmission{std::move(status), 0});
          return;
        }
        CHECK_LT(next_task_id_, std::numeric_limits<uint64_t>::max());
        active_task_id_ = next_task_id_++;
        execution_.push(program_.get());
        promise.setValue(TaskSubmission{Status(), active_task_id_});
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
        if (task_id == 0 || task_id != active_task_id_) {
          promise.setValue(
              TaskResult{Status(StatusCode::INVALID_ARGUMENT,
                                "No unconsumed result for this TaskId."),
                         {}});
          return;
        }
        CHECK_EQ(completed_.pop(), program_.get());
        auto tokens = program_->consume();
        active_task_id_ = 0;
        promise.setValue(TaskResult{Status(), std::move(tokens)});
      });
  return future;
}

void TaskExecutionPipeline::launch_loop() {
  while (LlmTaskProgram* program = execution_.pop()) {
    program->launch();
    // At most one accepted Task: publishing never depends on GetLast.
    completed_.push(program);
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
  execution_.push(nullptr);
  launch_thread_.join();
  if (active_task_id_ != 0) {
    CHECK_EQ(completed_.pop(), program_.get());
    program_->discard();
    active_task_id_ = 0;
  }
  CHECK(completed_.empty());
}

}  // namespace xllm
