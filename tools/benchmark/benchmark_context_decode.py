"""Matched FA2 decoding at fixed context lengths: raw Full KV, reconstruction and latent."""
import argparse
import copy
import csv
import gc
import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess
import time
from contextlib import contextmanager
from unittest.mock import patch

import torch
from transformers.cache_utils import StaticCache

ATTENTION = 'slimkv_bench/attention'
RECONSTRUCTION = 'slimkv_bench/kv_reconstruction'
ROPE = 'slimkv_bench/rope'


class AppendCache(StaticCache):
    """Preallocated raw KV; rotate each new key once and retain every raw token."""
    def __init__(self, layers, capacity):
        self.capacity = capacity
        self.key_cache = [None] * layers
        self.value_cache = [None] * layers
        self.lengths = [0] * layers

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        start = self.lengths[layer_idx]
        end = start + key_states.shape[2]
        if end > self.capacity:
            raise ValueError('Full KV cache capacity exceeded; no tokens may be evicted.')
        if self.key_cache[layer_idx] is None:
            shape = (*key_states.shape[:2], self.capacity, key_states.shape[-1])
            self.key_cache[layer_idx] = key_states.new_empty(shape)
            self.value_cache[layer_idx] = value_states.new_empty(shape)
        self.key_cache[layer_idx][:, :, start:end].copy_(key_states)
        self.value_cache[layer_idx][:, :, start:end].copy_(value_states)
        self.lengths[layer_idx] = end
        return self.key_cache[layer_idx][:, :, :end], self.value_cache[layer_idx][:, :, :end]

    def get_seq_length(self, layer_idx=0):
        return self.lengths[layer_idx]

    def get_max_length(self):
        return self.capacity

    def to_legacy_cache(self):
        return tuple((k[:, :, :n], v[:, :, :n]) for k, v, n in
                     zip(self.key_cache, self.value_cache, self.lengths) if k is not None)

    def rewind(self, length):
        if not all(n >= length for n in self.lengths):
            raise ValueError('Cannot restore a prefix that has not been prefetched.')
        self.lengths = [length] * len(self.lengths)


class CanonicalRope(torch.nn.Module):
    """Expose the checkpoint's unchanged RoPE tables through the HF 4.40 API."""
    def __init__(self, rope):
        super().__init__()
        object.__setattr__(self, 'source', rope)

    def forward(self, x, position_ids):
        return (self.source.cos_cached[position_ids].to(x.dtype),
                self.source.sin_cached[position_ids].to(x.dtype))


def full_backbone(model, attn_impl='flash_attention_2'):
    from transformers.models.llama.configuration_llama import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaModel
    settings = model.config.to_dict()
    # HF 4.40 lacks Llama-3.1 scaling; use the checkpoint's canonical tables below.
    settings['rope_scaling'] = None
    config = LlamaConfig(**settings)
    config._attn_implementation = attn_impl
    with torch.device('meta'):
        raw = LlamaModel(config)
    state = {name: value for name, value in model.model.state_dict().items() if 'beacon' not in name}
    raw.load_state_dict(state, strict=True, assign=True)
    for target, source in zip(raw.layers, model.model.layers):
        target.self_attn.rotary_emb = CanonicalRope(source.self_attn.rotary_emb)
    raw.eval()
    if any(x.is_meta for x in raw.parameters()):
        raise RuntimeError('Full KV backbone has unloaded parameters.')
    return raw


def extend_rope_tables(model, required):
    """Append canonical CPU-computed positions without replacing existing tables."""
    for layer in model.model.layers:
        rope = layer.self_attn.rotary_emb
        if required <= rope.max_seq_len_cached:
            continue
        if type(rope).__name__ not in ('Llama3RotaryEmbedding', 'RotaryEmbedding'):
            raise ValueError('Extended benchmark positions require static or Llama3 RoPE.')
        positions = torch.arange(rope.max_seq_len_cached, required, dtype=torch.float32)
        frequencies = torch.outer(positions, rope.inv_freq.detach().cpu().float())
        embedding = torch.cat([frequencies, frequencies], dim=-1)
        rope.cos_cached = torch.cat([rope.cos_cached, embedding.cos().to(rope.cos_cached.device)])
        rope.sin_cached = torch.cat([rope.sin_cached, embedding.sin().to(rope.sin_cached.device)])
        rope.max_seq_len_cached = required


def synthetic_ids(tokenizer, length, seed, prompt_file=None):
    rng = random.Random(seed)
    paragraphs = [
        'The archive stores observations of rivers, forests and nearby towns. Researchers compare the dated reports before writing a summary.',
        'A laboratory team measures temperature and pressure twice each day. Every instrument is checked against the same reference.',
        'The railway timetable connects several coastal cities. Passengers compare departure times, transfer stations and ticket prices.',
        'A software service records requests, response times and error counts. Engineers inspect these records when testing a new release.',
        'The museum catalogue describes paintings, materials and restoration dates. Visitors can search the collection by artist or period.',
        'The market report lists quantities, delivery dates and warehouse locations. The coordinator verifies each order before dispatch.',
    ]
    if prompt_file:
        paragraphs = [Path(prompt_file).read_text(encoding='utf-8')]
    ids = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    while len(ids) < length:
        text = f'\nRecord {rng.randrange(1000000)}. {rng.choice(paragraphs)}\n'
        block = tokenizer.encode(text, add_special_tokens=False)
        if not block:
            raise ValueError('Synthetic input tokenized to an empty block.')
        ids.extend(block)
    return torch.tensor([ids[:length]], dtype=torch.long)


def cloned_memory(memory):
    # Keep model/config identity; clone every cache tensor and all mutable metadata.
    return copy.deepcopy(memory, memo={id(memory.config): memory.config})


@contextmanager
def attention_scopes(model, raw):
    from slimkv.llama3.memory import ReconstructionCache
    from slimkv.llama3 import decode_ops
    import transformers.models.llama.modeling_llama as hf_llama

    def wrapped(function, label):
        def call(*args, **kwargs):
            with torch.profiler.record_function(label):
                return function(*args, **kwargs)
        return call

    from contextlib import ExitStack
    with ExitStack() as stack:
        for layer in list(model.model.layers) + list(raw.layers):
            attn = layer.self_attn
            stack.enter_context(patch.object(attn, 'forward', wrapped(attn.forward, ATTENTION)))
        for layer in model.model.layers:
            attn = layer.self_attn
            stack.enter_context(patch.object(attn, '_apply_rotary_with_beacon',
                                             wrapped(attn._apply_rotary_with_beacon, ROPE)))
            if model.config.beacon_decode_backend == 'reconstruct':
                for projection in (attn.beacon_k_up_proj, attn.beacon_v_up_proj):
                    stack.enter_context(patch.object(projection, 'forward', wrapped(projection.forward, RECONSTRUCTION)))
        stack.enter_context(patch.object(ReconstructionCache, 'materialize', wrapped(ReconstructionCache.materialize, RECONSTRUCTION)))
        stack.enter_context(patch.object(decode_ops, 'fused_rope', wrapped(decode_ops.fused_rope, ROPE)))
        stack.enter_context(patch.object(hf_llama, 'apply_rotary_pos_emb', wrapped(hf_llama.apply_rotary_pos_emb, ROPE)))
        for layer in raw.layers:
            rope = layer.self_attn.rotary_emb
            stack.enter_context(patch.object(rope, 'forward', wrapped(rope.forward, ROPE)))
        yield


def scope_times(events, expected_calls):
    result = {}
    for label in (ATTENTION, RECONSTRUCTION, ROPE):
        selected = []
        for event in events:
            if event.name != label or event.device_type != torch.autograd.DeviceType.CPU:
                continue
            parent = event.cpu_parent
            while parent is not None and parent.name != label:
                parent = parent.cpu_parent
            if parent is None:
                selected.append(event)
        def kernel_us(event):
            # Torch 2.11 also emits CUDA user-annotation ranges. Their elapsed
            # spans are not kernels and must not be added to correlated kernels.
            own = sum(kernel.duration for kernel in event.kernels
                      if not kernel.name.startswith('slimkv_bench/'))
            return own + sum(kernel_us(child) for child in event.cpu_children)
        result[label] = sum(kernel_us(event) for event in selected) / 1000
        if label == ATTENTION and len(selected) != expected_calls:
            raise RuntimeError(f'Expected {expected_calls} attention calls, observed {len(selected)}.')
    if result[ATTENTION] <= 0:
        raise RuntimeError('Profiler did not capture attention GPU kernels.')
    flash = [event for event in events if event.device_type == torch.autograd.DeviceType.CUDA
             and 'flash' in event.name.lower() and ('fwd' in event.name.lower() or 'forward' in event.name.lower())]
    if not flash:
        raise RuntimeError('No FA2 forward kernels observed; refusing a non-FA2 result.')
    result['fa2_core'] = sum(event.device_time_total for event in flash) / 1000
    result['fa2_kernel_calls'] = len(flash)
    return result


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_name_or_path', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--context_lengths', type=int, nargs='+', default=[32768, 65536, 98304, 131072])
    parser.add_argument('--beacon_ratios', type=int, nargs='+', default=[4, 8])
    parser.add_argument('--decode_steps', type=int, default=128)
    parser.add_argument('--repetitions', type=int, default=4)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--prefill_chunk_size', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--prompt_file')
    parser.add_argument('--save_trace', action='store_true')
    args = parser.parse_args()
    if min(args.context_lengths + args.beacon_ratios + [args.decode_steps, args.repetitions, args.warmup, args.prefill_chunk_size]) < 1:
        parser.error('All lengths, ratios and repetition counts must be positive.')
    import transformers
    import flash_attn
    from slimkv import ModelArgs, get_model_and_tokenizer
    if not transformers.__version__.startswith('4.40.'):
        parser.error('This benchmark uses Transformers 4.40.x cache interfaces.')
    if not torch.cuda.is_available():
        parser.error('CUDA and FlashAttention 2 are required.')
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    model, tokenizer = get_model_and_tokenizer(ModelArgs(
        model_name_or_path=args.model_name_or_path, enable_beacon=True, dtype='bf16',
        attn_impl='flash_attention_2', beacon_ratio=[args.beacon_ratios[0]], beacon_ratio_mix='sequence',
        beacon_cache_mode='latent', beacon_decode_backend='latent', beacon_decode_fast=True,
        beacon_decode_graph=False, beacon_decode_graph_full=False, beacon_decode_compile=False,
        beacon_prefill_fast=True), device='cuda:0')
    model.eval()
    extend_rope_tables(model, max(args.context_lengths) + args.decode_steps)
    raw = full_backbone(model)
    if any(type(layer.self_attn).__name__ != 'LlamaFlashAttention2' for layer in list(raw.layers) + list(model.model.layers)):
        raise ValueError('This benchmark requires FA2 Llama models for every route.')
    ids_cpu = synthetic_ids(tokenizer, max(args.context_lengths), args.seed, args.prompt_file)
    torch.save(ids_cpu, outdir/'input_ids.pt')
    result = {'complete': False, 'metadata': {
        'args': vars(args), 'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
        'transformers': transformers.__version__, 'flash_attn': flash_attn.__version__,
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'tool_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'checkpoint_config_sha256': hashlib.sha256(Path(args.model_name_or_path, 'config.json').read_bytes()).hexdigest(),
        'input_sha256': hashlib.sha256(ids_cpu.numpy().tobytes()).hexdigest(),
        'full_kv': 'Uncompressed raw backbone from the supplied checkpoint, shared weights, native HF Llama, incremental K RoPE and preallocated append cache; no Beacons.',
        'protocol': 'B1 BF16 FA2 for all routes; Graph and compile disabled; exact 128 by default single-token forwards after prefill; identical teacher inputs from Full KV warmup; EOS ignored.',
        'attention_kernel_scope': 'Sum of GPU kernel durations inside all attention modules, including QKV/O, reconstruction, RoPE, cache operations and latent mapping/merge; excludes norm, MLP and LM head.',
        'end_to_end_scope': 'Synchronized host wall time for all decode forwards and argmax; excludes prefill, first prefill prediction, cache restore, profiler, tokenization and file IO.',
        'context_unit': 'Exact prompt token count; K=1024. At 128K, append 128 canonical RoPE entries without changing existing frequencies/tables.',
        'prefill': 'Original-dimensional FA2 for all methods; Full KV uses chunked causal prefill and retains all raw tokens.',
    }, 'contexts': []}
    save = lambda: atomic_json(outdir/'results.json', result)
    save()

    with torch.inference_mode():
        for length in args.context_lengths:
            ids = ids_cpu[:, :length].cuda()
            positions = torch.arange(length + args.decode_steps, device='cuda')[None]
            cache = AppendCache(len(raw.layers), length + args.decode_steps)
            for start in range(0, length, args.prefill_chunk_size):
                end = min(start + args.prefill_chunk_size, length)
                output = raw(input_ids=ids[:, start:end], past_key_values=cache,
                             position_ids=positions[:, start:end], cache_position=positions[0, start:end], use_cache=True)
            first = model.lm_head(output.last_hidden_state[:, -1:]).float().argmax(-1)
            del output
            snapshots = {}
            modes = ['full_kv'] + [f'{backend}_t{ratio}' for ratio in args.beacon_ratios for backend in ['reconstruct', 'latent']]
            for ratio in args.beacon_ratios:
                for backend in ['reconstruct', 'latent']:
                    model.config.beacon_ratio = [ratio]
                    model.config.beacon_decode_backend = backend
                    model.memory.reset()
                    output = model(input_ids=ids, num_logits_to_keep=1)
                    del output
                    snapshots[f'{backend}_t{ratio}'] = cloned_memory(model.memory)
                    print('PREFILL_READY', length, backend, ratio, flush=True)
            context = {'input_tokens': length, 'input_sha256': hashlib.sha256(ids_cpu[:, :length].numpy().tobytes()).hexdigest(),
                       'runs': [], 'kernel_profiles': {}, 'summaries': {}}
            result['contexts'].append(context)
            save()

            def restore(mode):
                if mode == 'full_kv':
                    cache.rewind(length)
                else:
                    backend, ratio = mode.split('_t')
                    model.config.beacon_ratio = [int(ratio)]
                    model.config.beacon_decode_backend = backend
                    model.memory = cloned_memory(snapshots[mode])
                gc.collect()
                torch.cuda.synchronize()

            def step(mode, token, index):
                if mode == 'full_kv':
                    hidden = raw(input_ids=token, past_key_values=cache,
                                 position_ids=positions[:, length+index:length+index+1],
                                 cache_position=positions[0, length+index:length+index+1], use_cache=True).last_hidden_state
                    logits = model.lm_head(hidden).float()
                else:
                    logits = model(input_ids=token, num_logits_to_keep=1).logits
                return logits[:, -1].argmax(-1, keepdim=True)

            # Full KV supplies exactly the same 128 input tokens to every timed route.
            restore('full_kv')
            token = first
            forced = []
            for index in range(args.decode_steps):
                forced.append(token)
                token = step('full_kv', token, index)
            forced = torch.cat(forced, dim=1)
            torch.save(forced.cpu(), outdir/f'decode_inputs_{length}.pt')
            context['decode_input_sha256'] = hashlib.sha256(forced.cpu().numpy().tobytes()).hexdigest()

            def decode(mode):
                predictions = []
                for index in range(args.decode_steps):
                    predictions.append(step(mode, forced[:, index:index+1], index))
                return predictions

            references = {}
            for mode in modes:
                for _ in range(args.warmup):
                    restore(mode)
                    reference = torch.cat(decode(mode), dim=1).cpu().tolist()[0]
                references[mode] = reference
            for repetition in range(args.repetitions):
                order = modes if repetition % 2 == 0 else list(reversed(modes))
                for mode in order:
                    restore(mode)
                    torch.cuda.reset_peak_memory_stats()
                    begin = time.perf_counter()
                    predictions = decode(mode)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - begin
                    tokens = torch.cat(predictions, dim=1).cpu().tolist()[0]
                    if tokens != references[mode]:
                        raise RuntimeError(f'{mode} changed predictions after cache restoration.')
                    row = {'mode': mode, 'repetition': repetition, 'decode_steps': args.decode_steps,
                           'end_to_end_total_ms': elapsed*1000, 'end_to_end_ms_per_token': elapsed*1000/args.decode_steps,
                           'tokens_per_second': args.decode_steps/elapsed, 'peak_allocated_bytes': torch.cuda.max_memory_allocated()}
                    context['runs'].append(row)
                    print('DECODE_TIMING', length, mode, repetition, row['tokens_per_second'], flush=True)
                    save()
            for mode in modes:
                restore(mode)
                with attention_scopes(model, raw), torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profiler:
                    predictions = decode(mode)
                    torch.cuda.synchronize()
                if torch.cat(predictions, dim=1).cpu().tolist()[0] != references[mode]:
                    raise RuntimeError('Kernel instrumentation changed predictions.')
                metrics = scope_times(profiler.events(), len(raw.layers)*args.decode_steps)
                context['kernel_profiles'][mode] = metrics
                if args.save_trace:
                    profiler.export_chrome_trace(str(outdir/f'kernels_{length}_{mode}.json'))
                runs = [r for r in context['runs'] if r['mode'] == mode]
                context['summaries'][mode] = {
                    'end_to_end_total_ms': statistics.median(r['end_to_end_total_ms'] for r in runs),
                    'end_to_end_ms_per_token': statistics.median(r['end_to_end_ms_per_token'] for r in runs),
                    'tokens_per_second': statistics.median(r['tokens_per_second'] for r in runs),
                    'attention_kernel_total_ms': metrics[ATTENTION],
                    'attention_kernel_ms_per_token': metrics[ATTENTION]/args.decode_steps,
                    'reconstruction_ms_per_token': metrics[RECONSTRUCTION]/args.decode_steps,
                    'rope_ms_per_token': metrics[ROPE]/args.decode_steps,
                    'fa2_core_ms_per_token': metrics['fa2_core']/args.decode_steps}
                print('KERNEL_TIMING', length, mode, context['summaries'][mode], flush=True)
                save()
            baseline = context['summaries']['full_kv']
            for row in context['summaries'].values():
                row['attention_speedup'] = baseline['attention_kernel_ms_per_token']/row['attention_kernel_ms_per_token']
                row['end_to_end_speedup'] = baseline['end_to_end_ms_per_token']/row['end_to_end_ms_per_token']
            save()
            model.memory.reset()
            del snapshots, cache, ids, positions, predictions, forced
            gc.collect()
            torch.cuda.empty_cache()
    rows = [{'input_tokens': context['input_tokens'], 'mode': mode, **row}
            for context in result['contexts'] for mode, row in context['summaries'].items()]
    with (outdir/'summary.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result['complete'] = True
    save()
    print('CONTEXT_DECODE_BENCHMARK_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
