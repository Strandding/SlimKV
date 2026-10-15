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
from contextlib import contextmanager, nullcontext
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
    from bisect import bisect_right
    gpu = [event for event in events if event.device_type == torch.autograd.DeviceType.CUDA
           and not getattr(event, 'is_user_annotation', False)
           and not event.name.startswith('slimkv_bench/')]
    annotation_names = {event.name for event in events if event.device_type == torch.autograd.DeviceType.CUDA
                        and getattr(event, 'is_user_annotation', False)}
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
        if label == ATTENTION and len(selected) != expected_calls:
            raise RuntimeError(f'Expected {expected_calls} attention calls, observed {len(selected)}.')
        intervals = sorted((event.time_range.start, event.time_range.end) for event in events
                           if event.name == label and event.device_type == torch.autograd.DeviceType.CUDA
                           and getattr(event, 'is_user_annotation', False))
        if intervals:
            merged = []
            for start, end in intervals:
                if merged and start <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
                else:
                    merged.append((start, end))
            starts = [start for start, end in merged]
            duration = 0.0
            for event in gpu:
                index = bisect_right(starts, event.time_range.start + 0.1)-1
                if index >= 0 and event.time_range.end <= merged[index][1]+0.1:
                    duration += event.device_time_total
            result[label] = duration/1000
        else:
            def kernel_us(event):
                own = sum(kernel.duration for kernel in event.kernels
                          if not kernel.name.startswith('slimkv_bench/')
                          and kernel.name not in annotation_names)
                return own+sum(kernel_us(child) for child in event.cpu_children)
            result[label] = sum(kernel_us(event) for event in selected)/1000
    if result[ATTENTION] <= 0:
        raise RuntimeError('Profiler did not capture attention GPU kernels.')
    flash = [event for event in gpu if 'flash' in event.name.lower()
             and ('fwd' in event.name.lower() or 'forward' in event.name.lower())]
    if not flash:
        raise RuntimeError('No FA2 forward kernels observed; refusing a non-FA2 result.')
    result['fa2_core'] = sum(event.device_time_total for event in flash)/1000
    result['fa2_kernel_calls'] = len(flash)
    if result['fa2_core'] > result[ATTENTION]*1.001:
        raise RuntimeError('FA2 kernels exceed attention total; profiler attribution is incomplete.')
    if result[RECONSTRUCTION] > result[ATTENTION]*1.001 or result[ROPE] > result[ATTENTION]*1.001:
        raise RuntimeError('Attention breakdown exceeds its enclosing total.')
    return result


def validate_runtime_cache(runtime, reference):
    maximum = 0.0
    bitwise = True
    def compare(left, right):
        nonlocal maximum, bitwise
        if not torch.equal(left, right):
            bitwise = False
            maximum = max(maximum, float((left.float()-right.float()).abs().max()))
    if runtime.mode == 'full_kv':
        length = runtime.prefix+runtime.step_index
        for state, key, value in zip(runtime.layers, reference.key_cache, reference.value_cache):
            compare(state['k'][:, :, :length], key[:, :, :length])
            compare(state['v'][:, :, :length], value[:, :, :length])
    else:
        for i, state in enumerate(runtime.layers):
            for name, j in [('k', 0), ('v', 1)]:
                sink = reference.sink_activations[i][j]
                raw = reference.raw_activations[i][j]
                history = reference.beacon_activations[i][j]
                window = reference.window_beacon_activations[i][j]
                if sink is not None:
                    compare(state['raw_'+name][:, :, :runtime.sink], sink)
                if raw is not None:
                    compare(state['raw_'+name][:, :, runtime.sink:runtime.sink+raw.shape[2]], raw)
                total = torch.cat([x.transpose(1, 2).reshape(1, x.shape[2], -1)
                                   for x in [history, window] if x is not None], dim=1).unsqueeze(1)
                compare(state['latent_'+name][:, :, :total.shape[2]], total)
    return {'bitwise': bitwise, 'max_error': maximum}


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
    parser.add_argument('--execution_mode', choices=['matched', 'optimized'], default='matched')
    parser.add_argument('--compile_tail', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--beacon_decode_compress', action=argparse.BooleanOptionalAction, default=True,
                        help='Continue token compression after prefill; False keeps all new raw KV.')
    parser.add_argument('--reference_dir', type=Path)
    args = parser.parse_args()
    if min(args.context_lengths + args.beacon_ratios + [args.decode_steps, args.repetitions, args.warmup, args.prefill_chunk_size]) < 1:
        parser.error('All lengths, ratios and repetition counts must be positive.')
    import transformers
    import flash_attn
    from slimkv import ModelArgs, get_model_and_tokenizer
    if not transformers.__version__.startswith('4.40.'):
        parser.error('This benchmark uses Transformers 4.40.x cache interfaces.')
    if args.execution_mode == 'optimized' and args.compile_tail and not torch.__version__.startswith('2.11.'):
        parser.error('Exact decoder-tail compilation requires Torch 2.11; otherwise use --no-compile_tail.')
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
        beacon_decode_compress=args.beacon_decode_compress,
        beacon_decode_graph=False, beacon_decode_graph_full=False, beacon_decode_compile=False,
        beacon_prefill_fast=True), device='cuda:0')
    model.eval()
    extend_rope_tables(model, max(args.context_lengths) + args.decode_steps)
    raw = full_backbone(model)
    if any(type(layer.self_attn).__name__ != 'LlamaFlashAttention2' for layer in list(raw.layers) + list(model.model.layers)):
        raise ValueError('This benchmark requires FA2 Llama models for every route.')
    if args.reference_dir:
        ids_cpu = torch.load(args.reference_dir/'input_ids.pt', map_location='cpu', weights_only=True)
        if ids_cpu.ndim != 2 or ids_cpu.shape[0] != 1 or ids_cpu.shape[1] < max(args.context_lengths):
            raise ValueError('Reference input shape differs from requested context matrix.')
        ids_cpu = ids_cpu[:, :max(args.context_lengths)]
    else:
        ids_cpu = synthetic_ids(tokenizer, max(args.context_lengths), args.seed, args.prompt_file)
    torch.save(ids_cpu, outdir/'input_ids.pt')
    result = {'complete': False, 'metadata': {
        'args': {key: str(value) if isinstance(value, Path) else value
                 for key, value in vars(args).items()},
        'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
        'transformers': transformers.__version__, 'flash_attn': flash_attn.__version__,
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'tool_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'runtime_sources': {str(path.relative_to(Path.cwd())): hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in [Path(__file__), Path('tools/benchmark/context_decode_runtime.py').resolve(),
                                         Path('tools/benchmark/context_decode_kernels.py').resolve(),
                                         Path('slimkv/llama3/paired_decode.py').resolve(),
                                         Path('slimkv/llama3/static_decode.py').resolve()]},
        'source_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip()),
        'checkpoint_config_sha256': hashlib.sha256(Path(args.model_name_or_path, 'config.json').read_bytes()).hexdigest(),
        'input_sha256': hashlib.sha256(ids_cpu.numpy().tobytes()).hexdigest(),
        'full_kv': 'Uncompressed raw backbone from the supplied checkpoint, shared weights, native HF Llama, incremental K RoPE and preallocated append cache; no Beacons.',
        'protocol': 'B1 BF16 FA2 for all routes; Graph and compile disabled; exact 128 by default single-token forwards after prefill; identical teacher inputs from Full KV warmup; EOS ignored.',
        'attention_kernel_scope': 'Sum of GPU kernel durations inside all attention modules, including QKV/O, reconstruction, RoPE, cache operations and latent mapping/merge; excludes norm, MLP and LM head.',
        'end_to_end_scope': 'Synchronized host wall time for all decode forwards and argmax; excludes prefill, first prefill prediction, cache restore, profiler, tokenization and file IO.',
        'context_unit': 'Exact prompt token count; K=1024. At 128K, append 128 canonical RoPE entries without changing existing frequencies/tables.',
        'prefill': 'Original-dimensional FA2 for all methods; Full KV uses chunked causal prefill and retains all raw tokens.',
    }, 'contexts': []}
    if args.execution_mode == 'optimized':
        result['metadata']['protocol'] = 'B1 BF16 FA2; preallocated cache and complete-model CUDA Graphs for every route; identical optional exact decoder-tail compilation; 128 single-token forwards after prefill.'
        result['metadata']['kernel_execution'] = 'Same optimized operations in an independent uncaptured profiler pass; Graph replay is used for end-to-end timing. Real GPU kernels are summed inside GPU annotation ranges when available.'
        result['metadata']['optimized_scope'] = 'Completed Beacon window, fixed continuation no longer than one window; benchmark runtime, not a serving scheduler.'
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

            runtimes = {}
            profiling_mode = False

            def restore(mode):
                if mode == 'full_kv':
                    cache.rewind(length)
                else:
                    backend, ratio = mode.split('_t')
                    model.config.beacon_ratio = [int(ratio)]
                    model.config.beacon_decode_backend = backend
                    model.memory = cloned_memory(snapshots[mode])
                if mode in runtimes:
                    runtimes[mode].reset()
                gc.collect()
                torch.cuda.synchronize()

            def reference_logits(mode, token, index):
                if mode == 'full_kv':
                    hidden = raw(input_ids=token, past_key_values=cache,
                                 position_ids=positions[:, length+index:length+index+1],
                                 cache_position=positions[0, length+index:length+index+1], use_cache=True).last_hidden_state
                    logits = model.lm_head(hidden).float()
                else:
                    logits = model(input_ids=token, num_logits_to_keep=1).logits
                return logits[:, -1:]

            def step(mode, token, index):
                if mode in runtimes:
                    logits = runtimes[mode].run(token, graph=not profiling_mode)
                else:
                    logits = reference_logits(mode, token, index)
                return logits[:, -1].argmax(-1, keepdim=True)

            # Full KV supplies exactly the same 128 input tokens to every timed route.
            restore('full_kv')
            token = first
            forced = []
            for index in range(args.decode_steps):
                forced.append(token)
                token = step('full_kv', token, index)
            forced = torch.cat(forced, dim=1)
            if args.reference_dir:
                saved = torch.load(args.reference_dir/f'decode_inputs_{length}.pt', map_location='cpu', weights_only=True)
                saved = saved[:, :args.decode_steps]
                if not torch.equal(forced.cpu(), saved):
                    raise RuntimeError('Full KV continuation differs from the first experiment.')
                forced = saved.cuda()
            torch.save(forced.cpu(), outdir/f'decode_inputs_{length}.pt')
            context['decode_input_sha256'] = hashlib.sha256(forced.cpu().numpy().tobytes()).hexdigest()

            def decode(mode):
                predictions = []
                for index in range(args.decode_steps):
                    predictions.append(step(mode, forced[:, index:index+1], index))
                return predictions

            if args.execution_mode == 'optimized':
                from tools.benchmark.context_decode_runtime import OptimizedDecodeRuntime
                context['numerical_validation'] = {}
                for mode in modes:
                    restore(mode)
                    # Always compare to the original per-route arithmetic, including
                    # reconstruction and Beacon insertion, before enabling graphs.
                    model.config.beacon_decode_fast = False
                    expected = torch.cat([reference_logits(mode, forced[:, index:index+1], index).cpu()
                                          for index in range(args.decode_steps)], dim=1)
                    if mode == 'full_kv':
                        cache.rewind(length)
                    runtime = OptimizedDecodeRuntime(model, 'full_kv' if mode == 'full_kv' else mode.split('_t')[0],
                                                     args.decode_steps, memory=snapshots.get(mode),
                                                     raw_cache=cache if mode == 'full_kv' else None,
                                                     compile_tail=args.compile_tail)
                    checks = {}
                    for use_graph in [False, True]:
                        runtime.reset()
                        actual = torch.cat([runtime.run(forced[:, index:index+1], graph=use_graph).cpu()
                                            for index in range(args.decode_steps)], dim=1)
                        check = {'logits_bitwise': torch.equal(actual, expected),
                                 'max_logit_error': float((actual-expected).abs().max()),
                                 'argmax_equal': torch.equal(actual.argmax(-1), expected.argmax(-1))}
                        checks['graph' if use_graph else 'uncaptured'] = check
                        context['numerical_validation'][mode] = checks
                        save()
                        if not check['logits_bitwise']:
                            check['mismatch_steps'] = (actual != expected).any(-1).flatten().nonzero().flatten().tolist()
                            check['max_error_by_step'] = (actual-expected).abs().amax(-1).flatten().tolist()
                            torch.save({'expected': expected, 'actual': actual}, outdir/f'failed_logits_{length}_{mode}.pt')
                            save()
                            raise RuntimeError(f'Optimized {mode} failed exact logits validation: {check}')
                        check['kv'] = validate_runtime_cache(runtime, cache if mode == 'full_kv' else model.memory)
                        save()
                        if not check['kv']['bitwise']:
                            raise RuntimeError(f'Optimized {mode} changed persistent KV: {check}')
                    runtimes[mode] = runtime
                    print('OPTIMIZED_VALIDATED', length, mode, checks, flush=True)
                model.config.beacon_decode_fast = True

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
                    graphs_before = len(runtimes[mode].graphs) if runtimes else 0
                    begin = time.perf_counter()
                    predictions = decode(mode)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - begin
                    if runtimes and len(runtimes[mode].graphs) != graphs_before:
                        raise RuntimeError('Graph capture occurred inside the timed interval.')
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
                profiling_mode = bool(runtimes)
                if mode in runtimes:
                    runtimes[mode].scope = True
                with (nullcontext() if runtimes else attention_scopes(model, raw)), torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profiler:
                    predictions = decode(mode)
                    torch.cuda.synchronize()
                profiling_mode = False
                if mode in runtimes:
                    runtimes[mode].scope = False
                if torch.cat(predictions, dim=1).cpu().tolist()[0] != references[mode]:
                    raise RuntimeError('Kernel instrumentation changed predictions.')
                events = profiler.events()
                if runtimes and length == args.context_lengths[0]:
                    audit = {}
                    for event in events:
                        if event.device_type != torch.autograd.DeviceType.CUDA:
                            continue
                        key = (event.name, bool(getattr(event, 'is_user_annotation', False)))
                        row = audit.setdefault(key, {'name': event.name, 'annotation': key[1], 'count': 0, 'total_us': 0.0})
                        row['count'] += 1
                        row['total_us'] += event.device_time_total
                    atomic_json(outdir/f'kernel_event_audit_{mode}.json', list(audit.values()))
                metrics = scope_times(events, len(raw.layers)*args.decode_steps)
                if runtimes and metrics[ROPE] <= 0:
                    raise RuntimeError('Optimized RoPE kernels were not attributed to their scope.')
                if runtimes and mode.startswith('reconstruct') and metrics[RECONSTRUCTION] <= 0:
                    raise RuntimeError('Reconstruction kernels were not attributed to their scope.')
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
            del snapshots, cache, ids, positions, predictions, forced, runtimes
            if args.execution_mode == 'optimized':
                del runtime, expected, actual
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
