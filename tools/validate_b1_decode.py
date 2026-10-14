"""Exercise B1 GPU metadata, CUDA Graph replay, and window/chunk transitions."""
import argparse
import copy
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source_dir')
    parser.add_argument('--output', required=True)
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    root = Path(args.source_dir or Path(__file__).resolve().parents[1])
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / 'tests'))
    import torch
    from unittest.mock import patch
    from test_llama_reconstruction_cache import make_model
    from slimkv.llama3.decode_graph import DecodeGraphs

    def forward(q, k, v, scale, causal=False):
        k = k.repeat_interleave(q.shape[2] // k.shape[2], dim=2)
        v = v.repeat_interleave(q.shape[2] // v.shape[2], dim=2)
        scores = q.transpose(1, 2) @ k.transpose(1, 2).transpose(-1, -2) * scale
        if causal:
            mask = torch.arange(k.shape[1], device=q.device)[None] > torch.arange(q.shape[1], device=q.device)[:, None] + k.shape[1] - q.shape[1]
            scores = scores.masked_fill(mask, -float('inf'))
        # FA2 returns an FP32 log-sum-exp even when its Q/K/V are BF16/FP16.
        probabilities = torch.softmax(scores.float(), -1).nan_to_num()
        output = (probabilities @ v.transpose(1, 2).float()).to(q.dtype).transpose(1, 2)
        return output, scores.float().logsumexp(-1)

    torch.manual_seed(42)
    torch.set_num_threads(1)
    records = []
    for dtype in ([torch.bfloat16] if args.quick else [torch.float32, torch.bfloat16, torch.float16]):
        legacy = make_model(beacon_decode_backend='fa2', beacon_decode_fast=False).to(device='cuda', dtype=dtype)
        fast = copy.deepcopy(legacy)
        fast.config.beacon_decode_fast = True
        graphed = copy.deepcopy(fast)
        graphed.config.beacon_decode_graph = True
        calls = [torch.arange(17, device='cuda')[None]]
        calls += [torch.tensor([[i]], device='cuda') for i in range(18, 39)]
        calls += [torch.tensor([[40, 41, 42]], device='cuda')]
        calls += [torch.tensor([[i]], device='cuda') for i in range(43, 62)]
        with torch.inference_mode(), patch('slimkv.llama3.fa2_decode.flash_forward', side_effect=forward):
            for i, ids in enumerate(calls):
                expected = legacy(input_ids=ids).logits
                actual = fast(input_ids=ids).logits
                replay = graphed(input_ids=ids).logits
                assert torch.equal(expected, actual), (dtype, i, 'metadata')
                assert torch.equal(expected, replay), (dtype, i, 'graph')
                assert legacy.memory.get_memory_size() == graphed.memory.get_memory_size()
            assert fast.memory.all_input_ids.is_cuda
            graph_count = sum(len(x._decode_graphs.entries) for x in graphed.model.layers)
            assert graph_count == 8, graph_count
            # Diagnostics requesting all hidden states keep ordinary Tensor outputs.
            graphed(input_ids=torch.tensor([[62]], device='cuda'), output_hidden_states=True)
            graphed.to(dtype=dtype)
            assert all(x._decode_graphs is None for x in graphed.model.layers)
            for model in [legacy, fast, graphed]:
                model.memory.reset()
        records.append({'dtype': str(dtype), 'model_calls': len(calls), 'bitwise_equal': True, 'graphs': graph_count})
        del legacy, fast, graphed
    # Gradient paths remain ordinary PyTorch, even with an existing graph cache.
    runner = DecodeGraphs()
    x = torch.ones(1, 1, 8, device='cuda', requires_grad=True)
    y = runner('grad', lambda value: value.square(), x)
    y.sum().backward()
    assert torch.equal(x.grad, torch.full_like(x, 2)) and not runner.entries
    torch.cuda.synchronize()
    report = {'complete': True, 'cases': records, 'gradient_fallback': True}
    Path(args.output).write_text(json.dumps(report, indent=2))
    print('B1_GPU_VALIDATION_COMPLETE', report, flush=True)


if __name__ == '__main__':
    main()
