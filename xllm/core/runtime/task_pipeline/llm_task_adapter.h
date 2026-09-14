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

#include "core/runtime/forward_params.h"
#include "core/runtime/task_pipeline/llm_task_program.h"

namespace xllm {

// The source must remain alive through the pipeline's Prepare Ack. Packed
// transport must be unpacked on CPU before calling this function.
Status make_llm_task_input(const ForwardInput& source, LlmTaskInput& output);

// Tokens already own detached CPU storage. Only shape views are changed.
ForwardOutput make_llm_task_output(LlmTaskOutput result);

}  // namespace xllm
