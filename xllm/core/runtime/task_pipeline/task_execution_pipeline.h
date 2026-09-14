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

#include <folly/futures/Future.h>

#include <deque>
#include <optional>

#include "core/runtime/task_pipeline/llm_task_program.h"
#include "core/util/concurrent_queue.h"
#include "core/util/threadpool.h"

namespace xllm {

struct TaskSubmission {
  Status status;
  uint64_t task_id = 0;
};

struct TaskResult {
  Status status;
  LlmTaskOutput output;
  uint64_t task_id = 0;
};

// One or two eager Slots. The owner prevents external calls racing with
// destruction. State executor, model and KV all outlive this pipeline.
// Public calls/destruction must occur outside its state/Launch threads.
class TaskExecutionPipeline final {
 public:
  static Status create(ThreadPool& state_executor,
                       std::unique_ptr<LlmTaskProgram> program,
                       std::unique_ptr<TaskExecutionPipeline>& output);
  ~TaskExecutionPipeline();
  TaskExecutionPipeline(const TaskExecutionPipeline&) = delete;
  TaskExecutionPipeline& operator=(const TaskExecutionPipeline&) = delete;

  // Waits for Prepare Ack only. On success no caller input storage is still
  // borrowed. Backpressure and validation return before accepting a Task.
  TaskSubmission submit(const LlmTaskInput& input);
  // Only the oldest accepted TaskId may be consumed. Other IDs are rejected
  // without removing a completion. Slot retirement precedes
  // Future completion. An unrequested result continues to occupy the Slot.
  folly::Future<TaskResult> take_result_async(uint64_t task_id);
  // The Step/GetLast adapter consumes the same FIFO without another ID queue.
  // Empty FIFO returns INVALID_ARGUMENT. Successful results carry their ID.
  folly::Future<TaskResult> take_result_async();
  uint64_t pinned_bytes() const { return program_->pinned_bytes(); }
  uint64_t device_bytes() const { return program_->device_bytes(); }

 private:
  struct SlotTicket {
    uint32_t slot_id = 0;
    uint64_t task_id = 0;
  };

  TaskExecutionPipeline(ThreadPool& state_executor,
                        std::unique_ptr<LlmTaskProgram> program);
  folly::Future<TaskResult> take_result_impl(
      std::optional<uint64_t> expected_task_id);
  void launch_loop();
  SlotTicket wait_completed_front();
  void check_external_thread() const;

  ThreadPool& state_executor_;
  std::unique_ptr<LlmTaskProgram> program_;
  ConcurrentQueue<SlotTicket> execution_;
  ConcurrentQueue<SlotTicket> completed_;
  std::thread launch_thread_;
  std::thread::id state_thread_id_;
  // Accessed exclusively on the state executor until the destructor barrier.
  uint64_t next_task_id_ = 1;
  std::deque<SlotTicket> accepted_;
};

}  // namespace xllm
