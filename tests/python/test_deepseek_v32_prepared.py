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

"""DeepSeek-V3.2 BF16 indexer parity across private pipeline metadata."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("torch_npu", reason="NPU metadata tests require torch_npu")

from xllm.python import kernels  # noqa: E402
from xllm.python.attention.backend import LayerCache  # noqa: E402
from xllm.python.attention.npu_paged_attention import NpuPagedAttentionBackend  # noqa: E402
from xllm.python.model_executor.forward_context import ForwardContext, forward_context  # noqa: E402
from xllm.python.models.deepseek_v32 import DeepseekV3Config, DeepseekV3Indexer  # noqa: E402


@pytest.mark.parametrize("phase", ["prefill", "chunked", "decode", "graph"])
@torch.inference_mode()
def test_prepared_v32_preserves_bf16_indexer_inputs_and_slot_isolation(
    phase: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = DeepseekV3Config(
        hidden_size=8,
        q_lora_rank=4,
        qk_rope_head_dim=4,
        index_n_heads=2,
        index_head_dim=8,
        index_topk=2,
    )
    device = torch.device("cpu")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(29)
        indexer = DeepseekV3Indexer(cfg, torch.bfloat16, device)
        hidden = torch.randn(5, cfg.hidden_size, dtype=torch.bfloat16)
        queries = torch.randn(5, cfg.q_lora_rank, dtype=torch.bfloat16)
    assert not indexer.indexer_rope_interleave
    assert isinstance(indexer.wq_b, torch.nn.Linear)
    backend = NpuPagedAttentionBackend(2, 1, 16, 0.25, 0, True, device, torch.bfloat16)
    index_cache = torch.zeros(4, 4, 1, cfg.index_head_dim, dtype=torch.bfloat16)
    cache = LayerCache(
        key=torch.zeros(4, 4, 1, 8, dtype=torch.bfloat16),
        value=torch.zeros(4, 4, 1, 4, dtype=torch.bfloat16),
        index=index_cache,
    )
    backend.bind_kv_caches([cache])
    layer = SimpleNamespace(layer_id=0)
    decode = phase in ("decode", "graph")
    rows = 2 if decode else 5
    q_lengths = [1, 1] if decode else [3, 2]
    kv_lengths = [6, 3] if phase != "prefill" else q_lengths
    metadata = SimpleNamespace(
        q_seq_lens=torch.tensor(q_lengths, dtype=torch.int32),
        q_cu_seq_lens=torch.tensor([1, 2] if decode else [3, 5], dtype=torch.int32),
        q_cu_seq_lens_host_values=[1, 2] if decode else [3, 5],
        kv_seq_lens=torch.tensor(kv_lengths, dtype=torch.int32),
        kv_seq_lens_host_values=list(kv_lengths),
        block_table=torch.tensor([[2, 0], [1, 3]], dtype=torch.int32),
        slot_mapping=torch.tensor([1, 6] if decode else [8, 9, 10, 4, 5], dtype=torch.int32),
        is_prefill=phase == "prefill",
        is_chunked_prefill=phase == "chunked",
        is_spec_verify=False,
        has_kv_shard=False,
    )
    calls: list[tuple[torch.Tensor, ...]] = []

    def scatter(cache_view: torch.Tensor, indices: torch.Tensor, values: torch.Tensor) -> None:
        cache_view.index_copy_(0, indices.flatten().long(), values)

    def select(*args: object) -> torch.Tensor:
        # Preserve the BF16 LightningIndexer ABI while inspecting the actual
        # query projection, half-RoPE, cache writes and scheduler row views.
        calls.append(tuple(tensor.clone() for tensor in args[:6]))
        assert args[6:8] == ("TND", "PA_BSND")
        indices, values = args[-2:]
        assert indices.shape == (rows, 1, cfg.index_topk)
        assert indices.dtype == torch.int32 and values.dtype == torch.bfloat16
        indices.zero_()
        values.zero_()
        return indices

    monkeypatch.setattr(kernels, "scatter_nd_update", scatter)
    monkeypatch.setattr(kernels, "lightning_indexer_out", select)
    cosine = torch.ones(rows, cfg.qk_rope_head_dim // 2, dtype=torch.bfloat16)
    sine = torch.full_like(cosine, 0.25)

    def run() -> None:
        context = ForwardContext(backend, device, metadata, [cache])
        with forward_context(context):
            indexer.select_qli(hidden[:rows], queries[:rows], backend.mla_index_context(layer), cosine, sine)

    backend.prepare(metadata)
    run()
    expected = calls[-1]
    expected_cache = index_cache.clone()
    index_cache.zero_()

    metadata.prepared_attention_state = backend.prepare_metadata(metadata)
    backend.prepare(metadata, graph_mode=phase == "graph")
    active = backend._metadata
    following = SimpleNamespace(**vars(metadata))
    following.block_table = torch.tensor([[3, 1], [0, 2]], dtype=torch.int32)
    following.q_cu_seq_lens_host_values = list(metadata.q_cu_seq_lens_host_values)
    following.kv_seq_lens_host_values = [7, 4]
    following.kv_seq_lens = torch.tensor([7, 4], dtype=torch.int32)
    following.prepared_attention_state = backend.prepare_metadata(following)
    assert backend._metadata is active
    following.kv_seq_lens_host_values[:] = [99, 99]
    run()
    for actual, reference in zip(calls[-1], expected):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    torch.testing.assert_close(index_cache, expected_cache, rtol=0, atol=0)

    backend.prepare(following)
    assert backend._metadata is following.prepared_attention_state
    assert backend._metadata.kv_lengths == [7, 4]
    assert backend._mla_actual_seq_kv is following.kv_seq_lens
    assert backend._block_table_i32 is following.block_table
