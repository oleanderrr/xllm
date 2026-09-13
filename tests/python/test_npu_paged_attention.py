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

"""Tests for the NPU paged-attention backend."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("torch_npu", reason="NPU paged-attention tests require torch_npu")

from xllm.python.attention.backend import LayerCache  # noqa: E402
from xllm.python.attention.npu_paged_attention import (  # noqa: E402
    NpuPagedAttentionBackend,
)


def test_uses_first_nonempty_key_cache() -> None:
    backend = NpuPagedAttentionBackend(
        num_heads=8,
        num_kv_heads=2,
        head_dim=64,
        scale=0.125,
        sliding_window=0,
        is_mla=False,
        device=torch.device("cpu"),
        dtype=torch.float16,
    )
    linear_cache = LayerCache(
        key=None,
        value=None,
        conv=torch.empty(8, 3, 64),
        ssm=torch.empty(8, 2, 4, 4),
    )
    key_cache = torch.empty(17, 128, 2, 64)
    value_cache = torch.empty_like(key_cache)

    backend.bind_kv_caches(
        [
            linear_cache,
            LayerCache(key=key_cache, value=value_cache),
        ]
    )

    assert backend.num_kv_blocks == 17
    assert backend.page_size == 128


@pytest.mark.parametrize("host_ends", [[3, 5], [0, 3, 5]])
def test_prepared_host_query_ends_avoid_device_readback(host_ends: list[int], monkeypatch: pytest.MonkeyPatch) -> None:
    backend = NpuPagedAttentionBackend(
        num_heads=8,
        num_kv_heads=2,
        head_dim=64,
        scale=0.125,
        sliding_window=0,
        is_mla=False,
        device=torch.device("cpu"),
        dtype=torch.float16,
    )
    cache = torch.empty(4, 128, 2, 64)
    backend.bind_kv_caches([LayerCache(key=cache, value=cache)])
    metadata = SimpleNamespace(
        q_cu_seq_lens=torch.tensor([3, 5], dtype=torch.int32),
        q_cu_seq_lens_host_values=host_ends,
        q_seq_lens=torch.tensor([3, 2], dtype=torch.int32),
        block_table=None,
        kv_seq_lens=None,
        is_prefill=True,
    )

    def reject_readback(self: torch.Tensor) -> torch.Tensor:
        raise AssertionError("prepared metadata must not copy Device lengths to Host")

    monkeypatch.setattr(torch.Tensor, "cpu", reject_readback)
    backend.prepare(metadata)
    assert backend._cumulative_seq_lens(metadata, 5) == [3, 5]


def _ordinary_metadata(*, paged: bool) -> SimpleNamespace:
    return SimpleNamespace(
        q_cu_seq_lens=torch.tensor([1, 2], dtype=torch.int32),
        q_cu_seq_lens_host_values=[0, 1, 2],
        q_seq_lens=torch.ones(2, dtype=torch.int32),
        block_table=torch.tensor([[0], [1]], dtype=torch.int32) if paged else None,
        kv_seq_lens=torch.tensor([6, 4], dtype=torch.int32),
        kv_seq_lens_host_values=[6, 4],
        is_prefill=not paged,
        is_spec_verify=False,
        has_kv_shard=False,
    )


def _ordinary_backend() -> NpuPagedAttentionBackend:
    backend = NpuPagedAttentionBackend(
        num_heads=8,
        num_kv_heads=2,
        head_dim=64,
        scale=0.125,
        sliding_window=0,
        is_mla=False,
        device=torch.device("cpu"),
        dtype=torch.float16,
    )
    cache = torch.empty(4, 128, 2, 64)
    backend.bind_kv_caches([LayerCache(key=cache, value=cache)])
    return backend


@pytest.mark.parametrize("paged", [False, True])
def test_private_metadata_preparation_preserves_active_slot(paged: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _ordinary_backend()
    active = _ordinary_metadata(paged=True)
    backend.prepare(active)
    old_query = backend._actual_seq_q
    old_kv = backend._actual_seq_kv
    old_table = backend._block_table_i32
    metadata = _ordinary_metadata(paged=paged)

    def reject_tensor_work(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("private metadata preparation and activation must use prepared Host values/views")

    monkeypatch.setattr(torch.Tensor, "cpu", reject_tensor_work)
    monkeypatch.setattr(torch.Tensor, "to", reject_tensor_work)
    monkeypatch.setattr(torch, "empty", reject_tensor_work)
    monkeypatch.setattr(torch, "arange", reject_tensor_work)
    state = backend.prepare_metadata(metadata)
    assert backend._metadata is active
    assert backend._actual_seq_q is old_query
    assert backend._actual_seq_kv is old_kv
    assert backend._block_table_i32 is old_table
    metadata.q_cu_seq_lens_host_values[:] = [-99]
    metadata.kv_seq_lens_host_values[:] = [-99]
    metadata.prepared_attention_state = state
    backend.prepare(metadata)
    assert backend._metadata is metadata
    assert backend._actual_seq_lens == [1, 2]
    assert backend._actual_seq_q == ([1, 2] if paged else [])
    assert backend._actual_seq_kv == ([6, 4] if paged else [])
    assert backend._block_table_i32 is metadata.block_table


@pytest.mark.parametrize("invalid", ["dtype", "query", "kv", "verify", "shard", "expanded"])
def test_private_metadata_rejection_preserves_active_slot(invalid: str) -> None:
    backend = _ordinary_backend()
    active = _ordinary_metadata(paged=True)
    backend.prepare(active)
    old_kv = backend._actual_seq_kv
    candidate = _ordinary_metadata(paged=True)
    if invalid == "dtype":
        candidate.block_table = candidate.block_table.to(torch.int64)
    elif invalid == "query":
        candidate.q_cu_seq_lens_host_values = [1]
    elif invalid == "kv":
        candidate.kv_seq_lens_host_values = [6]
    elif invalid == "verify":
        candidate.is_spec_verify = True
    elif invalid == "shard":
        candidate.has_kv_shard = True
    else:
        candidate.expanded_decode_metadata = SimpleNamespace(enabled=True)
    with pytest.raises(ValueError):
        backend.prepare_metadata(candidate)
    assert backend._metadata is active
    assert backend._actual_seq_kv is old_kv
