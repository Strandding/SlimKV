"""Measure aligned dense latent FA2 batches, using fixed LongBench decode inputs."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source_dir')
    p.add_argument('--model_name_or_path', required=True)
    p.add_argument('--samples_jsonl', required=True)
    p.add_argument('--forced_tokens_json', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--batch_sizes', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32])
    p.add_argument('--sample_indices', type=int, nargs='+', default=[0, 2])
    p.add_argument('--rounds', type=int, default=2)
    p.add_argument('--decode_tokens', type=int, default=64)
    p.add_argument('--validation_steps', type=int, default=16)
    a = p.parse_args()
    if not a.batch_sizes or min(a.batch_sizes) < 1 or a.rounds < 1 or a.decode_tokens < 2:
        p.error('Require positive batch sizes/rounds and at least two generated tokens.')
    if a.source_dir:
        sys.path.insert(0, a.source_dir)
    import torch
    from slimkv import ModelArgs, get_model_and_tokenizer
    from main.eval_longbench import process_longbench
    torch.manual_seed(42)
    torch.set_num_threads(8)
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    samples = [json.loads(x) for x in Path(a.samples_jsonl).read_text().splitlines()]
    supplied = json.loads(Path(a.forced_tokens_json).read_text())
    model, tokenizer = get_model_and_tokenizer(ModelArgs(
        model_name_or_path=a.model_name_or_path, enable_beacon=True,
        dtype='bf16', attn_impl='flash_attention_2', chat_template='llama-3',
        beacon_cache_mode='latent', beacon_decode_backend='fa2',
        beacon_decode_fast=True, beacon_decode_graph=True,
        beacon_ratio=[8], beacon_ratio_mix='sequence'), device='cuda:0')
    model.eval()
    order = a.batch_sizes + list(reversed(a.batch_sizes))
    metadata = {
        'args': vars(a), 'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0),
        'sm_count': torch.cuda.get_device_properties(0).multi_processor_count,
        'checkpoint_config_sha256': hashlib.sha256(Path(a.model_name_or_path, 'config.json').read_bytes()).hexdigest(),
        'samples_sha256': hashlib.sha256(Path(a.samples_jsonl).read_bytes()).hexdigest(),
        'forced_tokens_sha256': hashlib.sha256(Path(a.forced_tokens_json).read_bytes()).hexdigest(),
        'order_per_round': order,
        'scope': 'Replicated equal-length prompt rows and fixed decode tokens; each row is computed with its own KV storage. Independent distinct-row correctness is checked separately. This is dense batch scaling, not variable-length serving throughput.',
        'timing': 'Synchronized host wall time, forward plus argmax; no profiler/tokenization/IO/logit recording in timed region. All batch Graph keys prewarmed and retained.',
    }
    result = {'metadata': metadata, 'fixtures': [], 'complete': False}

    def save():
        (out / 'results.json').write_text(json.dumps(result, indent=2))

    def measure(ids, forced, batch):
        model.memory.reset()
        gc.collect()
        x = ids.repeat(batch, 1)
        tokens = forced.repeat(batch, 1)
        mask = torch.ones_like(x)
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        output = model(input_ids=x, attention_mask=mask)
        first = output.logits[:, -1].argmax(-1, keepdim=True)
        torch.cuda.synchronize()
        prefill = time.perf_counter() - start
        del output
        mask = torch.ones_like(tokens[:, :1])
        torch.cuda.synchronize()
        begin = time.perf_counter()
        generated = [first]
        for step in range(a.decode_tokens - 1):
            output = model(input_ids=tokens[:, step:step + 1], attention_mask=mask)
            generated.append(output.logits[:, -1].argmax(-1, keepdim=True))
            del output
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - begin
        peak = torch.cuda.max_memory_allocated()
        cached = sum(model.memory.cache_bytes().values())
        token_ids = torch.cat(generated, dim=1).cpu().tolist()
        return {'batch_size': batch, 'baseline_cuda_bytes': baseline,
                'prefill_ms': prefill * 1000, 'decode_ms': elapsed * 1000,
                'step_ms': elapsed * 1000 / (a.decode_tokens - 1),
                'aggregate_tokens_per_second': batch * (a.decode_tokens - 1) / elapsed,
                'per_sequence_tokens_per_second': (a.decode_tokens - 1) / elapsed,
                'peak_cuda_bytes': peak, 'kv_bytes_after_decode': cached,
                'generated_token_ids': token_ids}

    with torch.inference_mode():
        for index in a.sample_indices:
            sample = samples[index]
            data = process_longbench({'input': [sample['input']], 'context': [sample['context']]},
                                     [sample['source_index']], tokenizer, 'llama-3', sample['task'])
            ids = torch.tensor([data['input_ids'][0]], device='cuda')
            forced = torch.tensor([supplied[str(index)]], device='cuda')
            assert forced.shape[1] >= a.decode_tokens - 1
            warmups = [measure(ids, forced, b) for b in a.batch_sizes]
            runs = []
            for round_index in range(a.rounds):
                for position, b in enumerate(order):
                    row = measure(ids, forced, b)
                    row.update(round=round_index, position=position)
                    runs.append(row)
                    print('BATCH_PERF', index, b, round(row['step_ms'], 3), round(row['aggregate_tokens_per_second'], 2), flush=True)
            fields = ['prefill_ms', 'step_ms', 'aggregate_tokens_per_second', 'per_sequence_tokens_per_second', 'peak_cuda_bytes', 'kv_bytes_after_decode']
            medians = {str(b): {key: statistics.median(x[key] for x in runs if x['batch_size'] == b) for key in fields}
                       for b in a.batch_sizes}
            fixture = {'sample_index': index, 'input_tokens_per_row': ids.shape[1],
                       'input_ids_sha256': hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(),
                       'warmups': warmups, 'runs': runs, 'medians': medians, 'logit_validation': []}
            result['fixtures'].append(fixture)
            save()
            # Compare Graph replay to eager layer operations at the SAME batch;
            # batched GEMM can differ numerically from a B1 GEMM in BF16.
            for b in a.batch_sizes:
                vectors = {}
                for graph in [False, True]:
                    model.memory.reset()
                    model.config.beacon_decode_graph = graph
                    output = model(input_ids=ids.repeat(b, 1))
                    scores = [output.logits[:, -1].cpu()]
                    for step in range(a.validation_steps):
                        output = model(input_ids=forced[:, step:step + 1].repeat(b, 1))
                        scores.append(output.logits[:, -1].cpu())
                    vectors['graph' if graph else 'eager'] = torch.stack(scores)
                equal = torch.equal(vectors['graph'].contiguous().view(torch.uint8), vectors['eager'].contiguous().view(torch.uint8))
                assert equal, (index, b, 'Graph logits differ from same-batch eager')
                path = out / f'logits_{index}_b{b}.pt'
                torch.save(vectors, path)
                fixture['logit_validation'].append({'batch_size': b, 'shape': list(vectors['graph'].shape),
                    'bitwise_equal': equal, 'tensor_file': path.name, 'tensor_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
                save()
                del output, scores, vectors
            model.config.beacon_decode_graph = True
            model.memory.reset()
    result['complete'] = True
    save()
    print('BATCH_DECODE_BENCHMARK_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
