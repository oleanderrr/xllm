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

"""DeepSeek-V4 task ownership and compressed-cache addressing contracts."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from xllm.python.attention.backend import LayerCache
from xllm.python.attention.csa_attention import DsaAttentionBackend
from xllm.python.attention.dsa_metadata import DsaMetadata


def _backend() -> DsaAttentionBackend:
    backend = DsaAttentionBackend(
        compress_ratios=[1, 4, 128],
        window_size=128,
        n_layers=3,
        num_heads=2,
        attn_head_dim=8,
        index_topk=4,
        index_n_heads=2,
        index_head_dim=8,
        rope_head_dim=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    cache = torch.empty((16, 128, 1, 8))
    backend.bind_kv_caches(
        [
            LayerCache(key=None, value=None, swa=cache),
            LayerCache(
                key=cache,
                value=None,
                index=cache,
                swa=cache,
                compress_kv_state=cache,
                compress_score_state=cache,
                compress_index_kv_state=cache,
                compress_index_score_state=cache,
                indexer_scale=cache,
            ),
            LayerCache(key=cache, value=None, swa=cache, compress_kv_state=cache, compress_score_state=cache),
        ]
    )
    return backend


def _metadata(q_lengths: list[int], kv_lengths: list[int], *, prefill: bool = False) -> SimpleNamespace:
    batch_size = len(q_lengths)
    return SimpleNamespace(
        q_seq_lens_host=torch.tensor(q_lengths, dtype=torch.int32),
        kv_seq_lens_host=torch.tensor(kv_lengths, dtype=torch.int32),
        multi_block_tables=[
            torch.tensor([[3 + row * 4, 4 + row * 4] for row in range(batch_size)], dtype=torch.int32),
            torch.tensor([[9 + row] for row in range(batch_size)], dtype=torch.int32),
            torch.tensor([[11 + row] for row in range(batch_size)], dtype=torch.int32),
        ],
        is_prefill=prefill,
        is_chunked_prefill=not prefill and max(q_lengths, default=0) > 1,
        is_spec_verify=False,
        has_kv_shard=False,
        max_query_len=max(q_lengths, default=0),
        max_seq_len=max(kv_lengths, default=0),
        prepared_attention_state=None,
        dsa_metadata=None,
    )


@pytest.mark.parametrize(
    ("q_length", "kv_length", "prefill", "swa_slots", "c4_slots", "c128_slots"),
    [
        (1, 256, False, [639], [1215], [1409]),
        (8, 8, True, list(range(384, 392)), [1152, 1153], []),
        (5, 260, False, [639, 384, 385, 386, 387], [1215, 1216], [1409]),
    ],
)
def test_prepared_dsa_preserves_ring_and_compression_boundaries(
    q_length: int,
    kv_length: int,
    prefill: bool,
    swa_slots: list[int],
    c4_slots: list[int],
    c128_slots: list[int],
) -> None:
    backend = _backend()
    metadata = _metadata([q_length], [kv_length], prefill=prefill)

    prepared = backend.prepare_metadata(metadata)

    assert backend.supports_prepared_metadata
    assert backend._metadata is None
    assert metadata.dsa_metadata is None
    assert prepared.slot_mappings[0][0].tolist() == swa_slots
    assert prepared.slot_mappings[1][0].tolist() == c4_slots
    assert prepared.slot_mappings[2][0].tolist() == c128_slots
    assert prepared.block_tables[1][0].shape == (1, 1)
    assert prepared.block_tables[2][0].shape == (1, 1)
    assert prepared.input_positions.tolist() == list(range(kv_length - q_length, kv_length))


def test_preparing_next_dsa_task_keeps_active_forward_state(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _backend()
    active = _metadata([1, 1], [256, 9])
    active.prepared_attention_state = backend.prepare_metadata(active)
    backend.prepare(active)
    current_positions = torch.tensor([255, 8], dtype=torch.int64)
    rope = torch.arange(300 * 8, dtype=torch.float32).reshape(300, 8)
    backend.reset_forward(active)
    backend.attach_rope_tables(current_positions, rope, csa_cos_sin=rope, hca_cos_sin=rope, metadata=active)
    metadata_calls: list[DsaMetadata] = []
    monkeypatch.setattr(backend, "_build_precomputed_metadata", metadata_calls.append)
    backend.prepare_dsa_metadata_for_forward(active)
    current = active.dsa_metadata
    current.layer_id = 2
    current_cos = current.c4_cos.clone()

    following = _metadata([1, 1], [257, 12])
    following.prepared_attention_state = backend.prepare_metadata(following)

    assert backend._metadata is active
    assert active.dsa_metadata is current
    assert current.layer_id == 2
    assert current.input_positions is current_positions
    assert current.slot_mappings[0][0].tolist() == [639, 904]
    assert current.slot_mappings[1][0].tolist() == [1215]
    assert current.slot_mappings[2][0].tolist() == [1409]
    assert torch.equal(current.c4_cos, current_cos)
    assert following.dsa_metadata is None
    assert following.prepared_attention_state.slot_mappings[0][0].tolist() == [384, 907]
    assert following.prepared_attention_state.slot_mappings[1][0].tolist() == [1282]
    assert metadata_calls == [current]


@pytest.mark.parametrize("field", ["is_spec_verify", "has_kv_shard"])
def test_prepared_dsa_rejects_unsupported_execution_state(field: str) -> None:
    metadata = _metadata([1], [4])
    setattr(metadata, field, True)

    with pytest.raises(ValueError, match="ordinary unsharded"):
        _backend().prepare_metadata(metadata)


def test_prepared_dsa_requires_host_cache_groups() -> None:
    backend = _backend()
    metadata = _metadata([1], [4])
    metadata.multi_block_tables[1] = torch.empty((1, 1), dtype=torch.int32, device="meta")

    with pytest.raises(ValueError, match="Host int32 block tables"):
        backend.prepare_metadata(metadata)

    metadata.multi_block_tables.pop()
    with pytest.raises(ValueError, match="one block table per cache group"):
        backend.prepare_metadata(metadata)


@pytest.mark.parametrize("group_id", [0, 1, 2])
def test_prepared_dsa_checks_independent_cache_group_bounds(group_id: int) -> None:
    metadata = _metadata([1], [256])
    metadata.multi_block_tables[group_id][0, 0] = 16

    with pytest.raises(ValueError, match="outside its allocated cache group"):
        _backend().prepare_metadata(metadata)


def test_prepared_dsa_rejects_missing_compressed_write_block() -> None:
    metadata = _metadata([1], [256])
    metadata.multi_block_tables[2][0, 0] = -1

    with pytest.raises(ValueError, match="do not cover the current KV writes"):
        _backend().prepare_metadata(metadata)


@pytest.mark.parametrize("index_topk", [4, 256])
def test_prepared_dsa_empty_dp_peer_uses_reserved_cache_history(index_topk: int) -> None:
    backend = _backend()
    backend.index_topk = index_topk
    metadata = _metadata([1], [1], prefill=True)
    metadata.is_dummy = True

    prepared = backend.prepare_metadata(metadata)

    kv_length = max(index_topk, backend.window_size)
    assert prepared.seq_lens.tolist() == [kv_length]
    assert prepared.seq_lens_q.tolist() == [1]
    assert prepared.max_seq_len == kv_length
    assert prepared.max_query_len == 1
    assert prepared.slot_mappings[0][0].tolist() == [127]
    assert prepared.slot_mappings[1][0].tolist() == [kv_length // 4 - 1]
    assert prepared.slot_mappings[2][0].tolist() == [kv_length // 128 - 1]
    for layer_tables in prepared.block_tables:
        for table in layer_tables:
            assert torch.all(table[table >= 0] == 0)
    assert metadata.kv_seq_lens_host.tolist() == [1]
    assert metadata.multi_block_tables[0].tolist() == [[3, 4]]


def test_prepared_dsa_preserves_scheduler_resolved_swa_slots() -> None:
    backend = _backend()
    metadata = _metadata([5], [260])
    metadata.new_cache_slots_host_values = [387, 386, 385, 384, 639]

    prepared = backend.prepare_metadata(metadata)

    assert prepared.slot_mappings[0][0].tolist() == metadata.new_cache_slots_host_values
    assert prepared.slot_mappings[1][0].tolist() == [1215, 1216]
    assert prepared.slot_mappings[2][0].tolist() == [1409]


def test_prepared_dsa_launch_uses_staged_state(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _backend()
    metadata = _metadata([5], [260])
    metadata.prepared_attention_state = backend.prepare_metadata(metadata)
    positions = torch.arange(255, 260, dtype=torch.int64)
    rope = torch.arange(300 * 8, dtype=torch.float32).reshape(300, 8)
    backend.prepare(metadata)
    backend.reset_forward(metadata)
    backend.attach_rope_tables(positions, rope, csa_cos_sin=rope, hca_cos_sin=rope, metadata=metadata)

    def unexpected_build(**_kwargs: object) -> DsaMetadata:
        raise AssertionError("Launch must reuse Host metadata staged by Prepare")

    monkeypatch.setattr(backend._builder, "build", unexpected_build)
    monkeypatch.setattr(backend, "_build_precomputed_metadata", lambda _compressed: None)
    backend.prepare_dsa_metadata_for_forward(metadata)

    compressed = metadata.dsa_metadata
    assert compressed is metadata.prepared_attention_state
    assert compressed.input_positions is positions
    assert torch.equal(compressed.c4_cos, rope[[252, 256], :4])
    assert torch.equal(compressed.c128_cos, rope[[128], :4])
