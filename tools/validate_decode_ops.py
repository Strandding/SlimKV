"""GPU numerical regressions for fused decode RoPE and softmax merging."""
import argparse
import json
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--source_dir')
    p.add_argument('--quick', action='store_true')
    a = p.parse_args()
    if a.source_dir:
        sys.path.insert(0, a.source_dir)
    import torch
    from slimkv.llama3.decode_ops import fused_rope, fused_merge
    from slimkv.modeling_utils import RotaryEmbedding, rotate_half
    torch.manual_seed(42)
    torch.set_num_threads(1)
    rope = RotaryEmbedding(128, max_position_embeddings=2048).cuda()
    records = []
    batches = [1] if a.quick else [1, 2]
    lengths = [51] if a.quick else [2, 51, 128]
    dtypes = [torch.bfloat16] if a.quick else [torch.bfloat16, torch.float16]
    with torch.inference_mode():
        for batch in batches:
            for length in lengths:
                for queries in [1, 2]:
                    for sink in [0, 1]:
                        for dtype in dtypes:
                            history = 384
                            raw_indices = torch.arange(length - sink - 1, device='cuda') * 2
                            first_q = history + sink + 2 * raw_indices.numel()
                            positions = torch.arange(first_q, first_q + queries, device='cuda').repeat(batch, 1)
                            q = torch.randn(batch, queries, 32, 128, device='cuda', dtype=dtype).transpose(1, 2)
                            k = torch.randn(batch, 8, length, 128, device='cuda', dtype=dtype)
                            raw_positions = torch.cat([torch.arange(sink, device='cuda'),
                                                       raw_indices + sink + history,
                                                       positions[0, :1]]).repeat(batch, 1)
                            expected = []
                            for x, ids in [(q, positions), (k, raw_positions)]:
                                cos = rope.cos_cached[ids].to(dtype).unsqueeze(1)
                                sin = rope.sin_cached[ids].to(dtype).unsqueeze(1)
                                expected.append(x * cos + rotate_half(x) * sin)
                            actual = fused_rope(q, k, rope, positions, raw_indices, sink, history)
                            assert actual is not None
                            for x, y in zip(actual, expected):
                                assert torch.equal(x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8))
                            raw = torch.randn_like(q)
                            latent = torch.randn(batch, 32, queries, 128, device='cuda', dtype=dtype)
                            rl = torch.randn(batch, queries, 32, device='cuda').transpose(1, 2) * 4
                            ll = torch.randn_like(rl) * 4
                            if queries == 2:
                                ll[..., 0] = -torch.inf
                            total = torch.logaddexp(rl, ll)
                            reference = ((rl-total).exp()[..., None] * raw.float()
                                         + (ll-total).exp()[..., None] * latent.float()).to(dtype)
                            result = fused_merge(raw, rl, latent, ll)
                            assert result is not None and torch.isfinite(result).all()
                            torch.testing.assert_close(result, reference, atol=0.001, rtol=0.008)
                            if queries == 2:
                                assert torch.equal(result[:, :, 0], raw[:, :, 0])
                            records.append({'batch': batch, 'raw_length': length, 'query_length': queries,
                                            'sink': sink, 'dtype': str(dtype), 'rope_bitwise_equal': True,
                                            'merge_max_abs_error': (result-reference).abs().max().item(),
                                            'merge_changed_elements': (result!=reference).sum().item()})
    path = Path(a.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'status': 'PASS', 'cases': records}, indent=2))
    print('DECODE_OPS_VALIDATION_COMPLETE', len(records), flush=True)


if __name__ == '__main__':
    main()
