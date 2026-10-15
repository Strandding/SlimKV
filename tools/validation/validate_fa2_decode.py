"""Exercise SlimKV's FA2 extension against an independent FP32 reference on CUDA."""
import argparse
import importlib
import json
import math
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--extension_dir', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--quick', action='store_true')
    p.add_argument('--widths', type=int, nargs='+', default=[320],
                   help='Physical FA2 widths; use 128 200 208 216 224 232 240 248 256 320 for the full matrix.')
    p.add_argument('--batch_sizes', type=int, nargs='+', help='Focused width-320 batch scaling correctness cases.')
    args = p.parse_args()
    if any(width not in [128, *range(200, 257, 8), 320] for width in args.widths):
        p.error('Supported widths: 128, 200-256 in multiples of 8, and 320.')
    import torch
    sys.path.insert(0, args.extension_dir)
    ext = importlib.import_module('slimkv_fa2_decode_320')
    torch.manual_seed(42)
    rows = []
    cases = [(1, 1, 1, 32, 1, False, 1), (1, 2, 1, 32, 1, True, 1),
             (1, 2, 129, 32, 1, True, 3), (2, 1, 257, 8, 2, False, 3)]
    if args.batch_sizes:
        cases = [(b, q, n, 32, 1, q == 2, 0)
                 for b in args.batch_sizes for q in [1, 2] for n in [1, 129, 768]]
    elif not args.quick:
        cases += [(b, q, n, h, hk, causal, splits)
                  for b in [1, 2] for q in [1, 2] for n in [63, 128, 1027, 2048]
                  for h, hk in [(32, 1), (8, 2)]
                  for causal in ([False] if q == 1 else [False, True])
                  for splits in [1, 0, 16]]
    with torch.inference_mode():
        for width, (b, qlen, n, h, hk, causal, splits) in [(w, c) for w in args.widths for c in cases]:
            q = torch.randn(b, qlen, h, width, device='cuda', dtype=torch.bfloat16) * 0.3
            k = torch.randn(b, n, hk, width, device='cuda', dtype=torch.bfloat16) * 0.3
            v = torch.randn_like(k)
            k[..., width - 16:] = 0
            v[..., width - 8:] = 0
            q[..., width - 16:] = 0
            scale = 1 / math.sqrt(128)
            out, lse = ext.forward(q, k, v, scale, causal, splits, True)
            kf = k.float().repeat_interleave(h // hk, dim=2).transpose(1, 2)
            vf = v.float().repeat_interleave(h // hk, dim=2).transpose(1, 2)
            logits = q.float().transpose(1, 2) @ kf.transpose(-1, -2) * scale
            if causal:
                masked = torch.arange(n, device='cuda')[None] > torch.arange(qlen, device='cuda')[:, None] + n - qlen
                logits.masked_fill_(masked, -math.inf)
            expected_lse = logits.logsumexp(-1)
            # Native FA2's documented zero-output convention uses +inf LSE for
            # fully masked rows. The SlimKV wrapper converts it to log Z=-inf.
            native_lse = expected_lse.masked_fill(expected_lse.isneginf(), math.inf)
            expected = (logits.softmax(-1).nan_to_num() @ vf).transpose(1, 2)
            torch.testing.assert_close(out.float(), expected, atol=0.015, rtol=0.015)
            torch.testing.assert_close(lse, native_lse, atol=0.001, rtol=0.001)
            rows.append({'width': width, 'batch': b, 'query': qlen, 'key': n, 'heads': h, 'kv_heads': hk,
                         'causal': causal, 'splits': splits,
                         'max_output_error': (out.float() - expected).abs().max().item(),
                         'max_finite_lse_error': (lse - expected_lse)[expected_lse.isfinite()].abs().max().item()})
    torch.cuda.synchronize()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'status': 'PASS', 'torch': torch.__version__,
                                           'gpu': torch.cuda.get_device_name(),
                                           'compute_capability': torch.cuda.get_device_capability(),
                                           'extension': ext.__file__, 'cases': rows}, indent=2))
    print('FA2_DECODE_KERNEL_VALIDATION_COMPLETE', len(rows), flush=True)


if __name__ == '__main__':
    main()
