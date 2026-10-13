"""Measure ordinary reconstruction inference and a small LongBench-v1 subset."""
import argparse
from collections import defaultdict
import gc
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time


def install_compact_sink_control(memory, state):
    """Optional full-cache control: clone only the first sink views, preserving values."""
    update = memory.update_memory

    def controlled_update(past_key_values):
        first_sink = memory.sink_activations[0][0] is None
        update(past_key_values)
        if state['compact_sink'] and first_sink and not memory.latent_cache_active:
            memory.sink_activations = [
                tuple(None if x is None else x.clone() for x in pair)
                for pair in memory.sink_activations
            ]

    memory.update_memory = controlled_update


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_name_or_path', required=True)
    parser.add_argument('--samples_jsonl', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--source_dir', default=None)
    parser.add_argument('--attn_impl', choices=['eager', 'sdpa'], default='eager')
    parser.add_argument('--beacon_ratio', type=int, default=8)
    parser.add_argument('--decode_tokens', type=int, default=64)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--rounds', type=int, default=2,
                        help='Each symmetric round measures each cache mode twice.')
    parser.add_argument('--perf_sample_indices', type=int, nargs='+', default=[0, 2])
    parser.add_argument('--disable_optional_flash_import', action='store_true')
    args = parser.parse_args()
    if args.decode_tokens < 2 or args.warmup < 1 or args.rounds < 1:
        parser.error('Require at least 2 generated tokens, 1 warmup and 1 round.')
    if args.source_dir:
        import sys
        sys.path.insert(0, args.source_dir)

    import torch
    import transformers
    if args.disable_optional_flash_import:
        transformers.utils.is_flash_attn_2_available = lambda: False
        transformers.utils.import_utils.is_flash_attn_2_available = lambda: False
    from slimkv import ModelArgs, get_model_and_tokenizer
    from main.eval_longbench import process_longbench
    from main.longbench_utils import DATASET2MAXNEWTOKENS, scorer

    torch.manual_seed(42)
    torch.set_num_threads(8)
    samples = [json.loads(x) for x in Path(args.samples_jsonl).read_text().splitlines()]
    if any(i < 0 or i >= len(samples) for i in args.perf_sample_indices):
        parser.error('perf_sample_indices must refer to supplied samples.')
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = get_model_and_tokenizer(ModelArgs(
        model_name_or_path=args.model_name_or_path, enable_beacon=True,
        dtype='bf16', attn_impl=args.attn_impl, chat_template='llama-3',
        beacon_ratio=[args.beacon_ratio], beacon_ratio_mix='sequence'), device='cuda:0')
    model.eval()
    state = {'compact_sink': False}
    install_compact_sink_control(model.memory, state)
    encoded = []
    for sample in samples:
        x = process_longbench({'input': [sample['input']], 'context': [sample['context']]},
                             [sample['source_index']], tokenizer, 'llama-3', sample['task'])
        ids = x['input_ids'][0]
        assert 2 * model.config.beacon_window < len(ids) <= 31550
        encoded.append(ids)

    source = Path(args.source_dir or Path(__file__).resolve().parents[1])
    commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    metadata = {
        'source_commit': commit, 'checkpoint': args.model_name_or_path,
        'checkpoint_config_sha256': hashlib.sha256(Path(args.model_name_or_path, 'config.json').read_bytes()).hexdigest(),
        'samples_sha256': hashlib.sha256(Path(args.samples_jsonl).read_bytes()).hexdigest(),
        'benchmark_tool_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'torch': torch.__version__, 'transformers': transformers.__version__,
        'gpu': torch.cuda.get_device_name(0), 'args': vars(args),
        'model_parameter_bytes': sum(x.numel() * x.element_size() for x in model.parameters()),
        'timing': 'Synchronized host wall time; tokenization/file IO excluded; no logits recording.',
        'decode_steps': args.decode_tokens - 1,
        'measurement_order_per_round': ['full', 'full_compact', 'latent', 'latent', 'full_compact', 'full'],
        'memory': 'CUDA live allocated bytes, peak allocated bytes and unique persistent K/V storage; reserved allocator bytes excluded.',
        'full_compact': 'Full cache with first sink views cloned in the harness; same values and attention math.',
    }
    result = {'metadata': metadata, 'performance': [], 'quality': [], 'complete': False}

    def save():
        (outdir / 'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))

    def set_mode(mode):
        model.memory.reset()
        model.config.beacon_cache_mode = 'latent' if mode == 'latent' else 'full'
        state['compact_sink'] = mode == 'full_compact'
        gc.collect()
        torch.cuda.synchronize()

    def memory_snapshot():
        groups = [model.memory.sink_activations, model.memory.raw_activations,
                  model.memory.beacon_activations, model.memory.window_beacon_activations]
        storages = {}
        for group in groups:
            for pair in group:
                for tensor in pair:
                    if tensor is not None:
                        storage = tensor.untyped_storage()
                        storages[storage.data_ptr()] = storage.nbytes()
        return {'cuda_allocated_bytes': torch.cuda.memory_allocated(),
                'logical_kv_bytes': model.memory.cache_bytes(),
                'unique_kv_storage_bytes': sum(storages.values()),
                'memory_sizes': model.memory.get_memory_size()}

    def measure(ids, mask, mode):
        set_mode(mode)
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        output = model(input_ids=ids, attention_mask=mask)
        torch.cuda.synchronize()
        prefill_end = time.perf_counter()
        token = output.logits[:, -1].argmax(-1, keepdim=True)
        torch.cuda.synchronize()
        first_token_end = time.perf_counter()
        prefill_peak = torch.cuda.max_memory_allocated()
        del output
        torch.cuda.synchronize()
        after_prefill = memory_snapshot()
        if mode != 'full':
            assert after_prefill['unique_kv_storage_bytes'] == sum(after_prefill['logical_kv_bytes'].values())
        tokens = [token]
        token_mask = torch.ones_like(token)
        torch.cuda.reset_peak_memory_stats()
        decode_start = time.perf_counter()
        for _ in range(args.decode_tokens - 1):
            output = model(input_ids=token, attention_mask=token_mask)
            token = output.logits[:, -1].argmax(-1, keepdim=True)
            tokens.append(token)
            del output
        torch.cuda.synchronize()
        decode_end = time.perf_counter()
        decode_peak = torch.cuda.max_memory_allocated()
        after_decode = memory_snapshot()
        token_ids = torch.cat(tokens, dim=1).cpu().tolist()[0]
        model.memory.reset()
        return {
            'mode': mode, 'input_tokens': ids.shape[1], 'generated_tokens': len(token_ids),
            'generated_token_ids': token_ids, 'baseline_cuda_allocated_bytes': baseline,
            'prefill_ms': 1000 * (prefill_end - start),
            'ttft_ms': 1000 * (first_token_end - start),
            'decode_ms': 1000 * (decode_end - decode_start),
            'decode_ms_per_token': 1000 * (decode_end - decode_start) / (args.decode_tokens - 1),
            'decode_tokens_per_second': (args.decode_tokens - 1) / (decode_end - decode_start),
            'prefill_peak_cuda_allocated_bytes': prefill_peak,
            'decode_peak_cuda_allocated_bytes': decode_peak,
            'peak_cuda_allocated_bytes': max(prefill_peak, decode_peak),
            'after_prefill': after_prefill, 'after_decode': after_decode,
        }

    with torch.inference_mode():
        for sample_index in args.perf_sample_indices:
            sample = samples[sample_index]
            ids = torch.tensor([encoded[sample_index]], device='cuda:0')
            mask = torch.ones_like(ids)
            reference_tokens = None
            warmups = []
            for mode in ['full', 'full_compact', 'latent']:
                for _ in range(args.warmup):
                    warmup = measure(ids, mask, mode)
                    if reference_tokens is None:
                        reference_tokens = warmup['generated_token_ids']
                    assert warmup['generated_token_ids'] == reference_tokens
                    warmups.append(warmup)
            runs = []
            for round_index in range(args.rounds):
                for position, mode in enumerate(metadata['measurement_order_per_round']):
                    run = measure(ids, mask, mode)
                    assert run['generated_token_ids'] == reference_tokens
                    run.update(round=round_index, order_position=position)
                    runs.append(run)
                    print('PERF_RUN', sample_index, mode, round_index, round(run['prefill_ms'], 2),
                          round(run['decode_ms_per_token'], 3), flush=True)
            fields = ['prefill_ms', 'ttft_ms', 'decode_ms_per_token', 'decode_tokens_per_second',
                      'peak_cuda_allocated_bytes', 'prefill_peak_cuda_allocated_bytes',
                      'decode_peak_cuda_allocated_bytes']
            medians = {mode: {key: statistics.median(r[key] for r in runs if r['mode'] == mode)
                              for key in fields} for mode in ['full', 'full_compact', 'latent']}
            result['performance'].append({'sample_index': sample_index, 'task': sample['task'],
                                          'id': sample['_id'], 'input_tokens': ids.shape[1],
                                          'input_ids_sha256': hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(),
                                          'warmups': warmups, 'runs': runs, 'medians': medians})
            save()
            del ids, mask

        raw_eos = model.generation_config.eos_token_id
        raw_eos = [raw_eos] if isinstance(raw_eos, int) else list(raw_eos)
        qa_eos = list(raw_eos)
        newline = tokenizer.encode('\n', add_special_tokens=False)[-1]
        if newline not in qa_eos:
            qa_eos.append(newline)
        qa_tasks = {'2wikimqa', 'hotpotqa', 'musique', 'multifieldqa_en', 'qasper', 'narrativeqa', 'samsum'}
        pad = model.generation_config.pad_token_id
        if pad is None:
            pad = tokenizer.pad_token_id
        for index, sample in enumerate(samples):
            ids = torch.tensor([encoded[index]], device='cuda:0')
            mask = torch.ones_like(ids)
            row = {'task': sample['task'], 'id': sample['_id'], 'source_index': sample['source_index'],
                   'input_tokens': ids.shape[1], 'answers': sample['answers'],
                   'all_classes': sample['all_classes'],
                   'input_ids_sha256': hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest()}
            for mode in ['full', 'latent']:
                set_mode(mode)
                eos = qa_eos if sample['task'] in qa_tasks else raw_eos
                generated = model.generate(input_ids=ids, attention_mask=mask, do_sample=False,
                                           num_beams=1, temperature=1.0, top_p=1.0,
                                           max_new_tokens=DATASET2MAXNEWTOKENS[sample['task']],
                                           pad_token_id=pad, eos_token_id=eos,
                                           output_scores=False, return_dict_in_generate=False)
                token_ids = generated[0, ids.shape[1]:].tolist()
                prediction = tokenizer.decode(token_ids, skip_special_tokens=True)
                row[mode] = {'prediction': prediction, 'generated_token_ids': token_ids,
                             'score': scorer(sample['task'], [prediction], [sample['answers']], sample['all_classes']),
                             'max_new_tokens': DATASET2MAXNEWTOKENS[sample['task']], 'eos_token_ids': eos}
                print('QUALITY_SAMPLE', sample['task'], sample['source_index'], mode,
                      row[mode]['score'], repr(prediction[:180]), flush=True)
                del generated
            row['same_generated_tokens'] = row['full']['generated_token_ids'] == row['latent']['generated_token_ids']
            result['quality'].append(row)
            save()
            del ids, mask
        by_task = defaultdict(list)
        for row in result['quality']:
            by_task[row['task']].append(row)
        result['task_scores'] = {task: {mode: scorer(task, [r[mode]['prediction'] for r in rows],
                                                    [r['answers'] for r in rows], rows[0]['all_classes'])
                                      for mode in ['full', 'latent']} for task, rows in by_task.items()}
        result['complete'] = True
        save()
    print('RECONSTRUCTION_BENCHMARK_COMPLETE', len(result['quality']), flush=True)


if __name__ == '__main__':
    main()
