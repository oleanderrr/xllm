# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prepared ACL graph address binding and admission contracts."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from xllm.python.model_executor.executor import ModelExecutor
from xllm.python.model_executor.runners.decode_acl_graph import DecodeAclGraphRunner


def _metadata(rows: int) -> SimpleNamespace:
    return SimpleNamespace(
        slot_mapping=torch.arange(rows, dtype=torch.int32),
        block_table=torch.zeros(rows, 2, dtype=torch.int32),
        q_seq_lens=torch.ones(rows, dtype=torch.int32),
        kv_seq_lens=torch.ones(rows, dtype=torch.int32),
        q_cu_seq_lens=torch.arange(1, rows + 1, dtype=torch.int32),
        kv_seq_lens_host_values=[1] * rows,
        paged_kv_indptr=None,
        paged_kv_indices=None,
        paged_kv_last_page_len=None,
        is_prefill=False,
        is_chunked_prefill=False,
        is_spec_verify=False,
        prepared_attention_state=object(),
    )


@pytest.fixture
def runner(monkeypatch: pytest.MonkeyPatch) -> DecodeAclGraphRunner:
    stream = Mock()
    npu = SimpleNamespace(
        Stream=Mock(return_value=stream),
        Event=Mock(return_value=Mock()),
        current_stream=Mock(return_value=stream),
        stream=lambda _: nullcontext(),
        memory_allocated=lambda _: 100,
        memory_reserved=lambda _: 200,
    )
    monkeypatch.setattr(torch, "npu", npu, raising=False)
    backend = SimpleNamespace(prepare=Mock())
    result = DecodeAclGraphRunner(torch.nn.Identity(), backend, torch.device("cpu"), 4, 8)
    result._capture = Mock(side_effect=lambda entry, _: setattr(entry, "graph", Mock()))
    result._allocate_entry = Mock(side_effect=AssertionError("owned input allocation"))
    result._fill_entry = Mock(side_effect=AssertionError("whole input copy"))
    return result


def test_capture_binds_each_slot_and_shape_without_input_copies(runner: DecodeAclGraphRunner) -> None:
    inputs = []
    for _ in range(2):
        tokens = torch.arange(4, dtype=torch.int32)
        positions = torch.zeros(4, dtype=torch.int32)
        metadata = _metadata(4)
        runner.warmup_prepared(tokens, positions, metadata)
        runner.warmup_prepared(tokens, positions, metadata)
        inputs.append((tokens, positions, metadata))
    runner.freeze_prepared()
    entries = []
    for tokens, positions, metadata in inputs:
        selection = runner.select_prepared(tokens, positions, metadata)
        entry = selection.entry
        assert entry.static_input_ids is tokens
        assert entry.static_positions is positions
        assert entry.static_metadata.slot_mapping is metadata.slot_mapping
        assert entry.static_metadata.block_table is metadata.block_table
        entries.append(entry)
    assert entries[0] is not entries[1]
    assert runner._capture.call_count == 2
    runner._allocate_entry.assert_not_called()
    runner._fill_entry.assert_not_called()


def test_frozen_misses_never_capture_or_mutate_backend(runner: DecodeAclGraphRunner) -> None:
    tokens = torch.arange(4, dtype=torch.int32)
    positions = torch.zeros_like(tokens)
    metadata = _metadata(4)
    assert runner.select_prepared(tokens, positions, metadata).miss_reason == "not_warmed"
    runner.warmup_prepared(tokens, positions, metadata)
    runner.freeze_prepared()
    runner.attention_backend.prepare.reset_mock()
    assert runner.select_prepared(tokens.clone(), positions, metadata).miss_reason == "binding_not_captured"
    assert runner.select_prepared(tokens[:3], positions[:3], _metadata(3)).miss_reason == "shape_not_captured"
    metadata.is_prefill = True
    assert runner.select_prepared(tokens, positions, metadata).miss_reason == "prefill"
    runner.attention_backend.prepare.assert_not_called()
    assert runner._capture.call_count == 1
    with pytest.raises(RuntimeError, match="frozen"):
        runner.warmup_prepared(tokens, positions, metadata)


def test_prepare_keeps_previous_graph_metadata_independent(runner: DecodeAclGraphRunner) -> None:
    tokens = torch.arange(2, dtype=torch.int32)
    positions = torch.zeros_like(tokens)
    metadata = _metadata(2)
    runner.warmup_prepared(tokens, positions, metadata)
    runner.freeze_prepared()
    selection = runner.select_prepared(tokens, positions, metadata)
    metadata.kv_seq_lens_host_values[:] = [7, 11]
    assert selection.entry.static_metadata.kv_seq_lens_host_values == [1, 1]
    assert runner.select_prepared(tokens, positions, metadata).entry is selection.entry
    selection.entry.static_output = torch.ones(2, 8)
    assert runner.execute_prepared(selection, tokens, positions, metadata) is selection.entry.static_output
    runner.attention_backend.prepare.assert_called_with(metadata, graph_mode=True)
    assert runner.prepared_replays == 1
    with pytest.raises(RuntimeError, match="binding changed"):
        runner.execute_prepared(selection, tokens, positions.clone(), metadata)


def test_prepared_executor_uses_selection_and_propagates_replay_failure(runner: DecodeAclGraphRunner) -> None:
    executor = object.__new__(ModelExecutor)
    executor._kv_bound = True
    executor.layerwise_split_size = 1
    executor.decode_graph_runner = runner
    executor.eager_runner = SimpleNamespace(execute=Mock(return_value="eager"))
    tokens = torch.arange(2, dtype=torch.int32)
    positions = torch.zeros_like(tokens)
    metadata = _metadata(2)
    runner.warmup_prepared(tokens, positions, metadata)
    runner.freeze_prepared()
    metadata.prepared_graph = runner.select_prepared(tokens, positions, metadata)
    metadata.prepared_graph.entry.graph.replay.side_effect = RuntimeError("device replay failed")
    with pytest.raises(RuntimeError, match="device replay failed"):
        executor.execute(tokens, positions, metadata)
    executor.eager_runner.execute.assert_not_called()
    metadata.prepared_graph = runner.select_prepared(tokens.clone(), positions, metadata)
    assert executor.execute(tokens, positions, metadata) == "eager"
    assert runner._capture.call_count == 1


def test_unplanned_capture_is_rejected_before_work(runner: DecodeAclGraphRunner) -> None:
    with pytest.raises(ValueError, match="unplanned"):
        runner.warmup_prepared(torch.zeros(3), torch.zeros(3), _metadata(3))
    runner._capture.assert_not_called()
    runner.attention_backend.prepare.assert_not_called()
