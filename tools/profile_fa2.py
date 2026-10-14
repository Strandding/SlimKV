"""NVTX model profiling and replay of actual raw/latent FA2 inputs."""
import argparse
from collections import defaultdict
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['model', 'capture', 'kernel'], required=True)
    p.add_argument('--source_dir')
    p.add_argument('--output_dir', required=True)
    p.add_argument('--model_name_or_path')
    p.add_argument('--samples_jsonl')
    p.add_argument('--sample_index', type=int, default=0)
    p.add_argument('--attn_impl', default='flash_attention_2')
    p.add_argument('--phase', choices=['prefill', 'decode'], default='decode')
    p.add_argument('--decode_steps', type=int, default=16)
    p.add_argument('--decode_variant', choices=['legacy', 'fast', 'graph'], default='fast')
    p.add_argument('--case_file')
    a = p.parse_args()
    if a.source_dir:
        sys.path.insert(0, a.source_dir)
    import torch
    from slimkv.llama3 import fa2_decode as backend
    torch.manual_seed(42)
    torch.set_num_threads(8)
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    counts = defaultdict(int)

    @contextmanager
    def nvtx(name):
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()

    if a.mode == 'kernel':
        case = torch.load(a.case_file, map_location='cuda', weights_only=True)
        q, k, v = (case[name] for name in ('q', 'k', 'v'))
        args = q, k, v, case['scale'], case['causal']
        with torch.inference_mode():
            for _ in range(10):
                actual, lse = backend.flash_forward(*args)
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            with nvtx('kernel_probe'):
                actual, lse = backend.flash_forward(*args)
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
        error = (actual.float() - case['output'].float()).abs().max().item()
        finite = case['lse'].isfinite()
        lse_error = (lse[finite] - case['lse'][finite]).abs().max().item()
        assert error <= 0.015 and lse_error <= 0.001
        record = {'complete': True, 'case_sha256': hashlib.sha256(Path(a.case_file).read_bytes()).hexdigest(),
                  'shape_qkv': [list(x.shape) for x in (q, k, v)], 'causal': case['causal'],
                  'output_max_error': error, 'lse_max_error': lse_error,
                  'args': vars(a), 'torch': torch.__version__}
        (out/'kernel_probe.json').write_text(json.dumps(record, indent=2))
        print('FA2_KERNEL_PROBE_COMPLETE', flush=True)
        return

    from slimkv import ModelArgs, get_model_and_tokenizer
    from main.eval_longbench import process_longbench
    model, tokenizer = get_model_and_tokenizer(ModelArgs(
        model_name_or_path=a.model_name_or_path, enable_beacon=True,
        dtype='bf16', attn_impl=a.attn_impl, chat_template='llama-3',
        beacon_cache_mode='latent', beacon_decode_backend='fa2',
        beacon_decode_fast=a.decode_variant != 'legacy',
        beacon_decode_graph=a.decode_variant == 'graph',
        beacon_ratio=[8], beacon_ratio_mix='sequence'), device='cuda:0')
    model.eval()
    sample = [json.loads(x) for x in Path(a.samples_jsonl).read_text().splitlines()][a.sample_index]
    data = process_longbench({'input': [sample['input']], 'context': [sample['context']]},
                             [sample['source_index']], tokenizer, 'llama-3', sample['task'])
    ids = torch.tensor([data['input_ids'][0]], device='cuda')
    mask = torch.ones_like(ids)
    active = {'layer': -1, 'profiling': False}
    saved = {}

    def wrap_function(name, function):
        def wrapped(*args, **kwargs):
            if not active['profiling']:
                return function(*args, **kwargs)
            label = f'L{active["layer"]}/{name}'
            if name == 'flash':
                q, k, v = args[:3]
                label += f'/Q{q.shape[1]}_N{k.shape[1]}_R{q.shape[-1]}_{v.shape[-1]}'
            with nvtx(label):
                result = function(*args, **kwargs)
            counts[name] += 1
            if (a.mode == 'capture' and name == 'flash' and active['layer'] in (0, 2, 28)
                    and len(saved) < 12):
                q, k, v, scale = args[:4]
                causal = args[4] if len(args) > 4 else kwargs.get('causal', False)
                key = f'L{active["layer"]}_Q{q.shape[1]}_R{q.shape[-1]}_{v.shape[-1]}'
                if key not in saved:
                    saved[key] = {'q': q.detach().cpu().clone(), 'k': k.detach().cpu().clone(),
                                  'v': v.detach().cpu().clone(), 'scale': scale, 'causal': causal,
                                  'output': result[0].detach().cpu().clone(), 'lse': result[1].detach().cpu().clone()}
            return result
        return wrapped

    for name in ('project_query', 'project_value', 'merge_attention', 'flash_forward'):
        setattr(backend, name, wrap_function('flash' if name == 'flash_forward' else name,
                                             getattr(backend, name)))
    for i, layer in enumerate(model.model.layers):
        for module, name in [(layer.self_attn, 'attention'), (layer.mlp, 'mlp')]:
            original = module.forward

            def wrapped(*args, _original=original, _index=i, _name=name, **kwargs):
                previous = active['layer']
                active['layer'] = _index
                try:
                    if active['profiling']:
                        with nvtx(f'L{_index}/{_name}'):
                            return _original(*args, **kwargs)
                    return _original(*args, **kwargs)
                finally:
                    active['layer'] = previous
            module.forward = wrapped

    with torch.inference_mode():
        # Warm all used projection and FA2 paths before the captured interval.
        output = model(input_ids=ids, attention_mask=mask)
        token = output.logits[:, -1].argmax(-1, keepdim=True)
        for _ in range(8):
            output = model(input_ids=token, attention_mask=torch.ones_like(token))
            token = output.logits[:, -1].argmax(-1, keepdim=True)
        model.memory.reset()
        if a.phase == 'decode':
            output = model(input_ids=ids, attention_mask=mask)
            token = output.logits[:, -1].argmax(-1, keepdim=True)
        torch.cuda.synchronize()
        active['profiling'] = True
        torch.cuda.cudart().cudaProfilerStart()
        with nvtx(f'model_{a.phase}'):
            if a.phase == 'prefill':
                output = model(input_ids=ids, attention_mask=mask)
            else:
                for step in range(a.decode_steps):
                    with nvtx(f'decode_step_{step}'):
                        output = model(input_ids=token, attention_mask=torch.ones_like(token))
                        token = output.logits[:, -1].argmax(-1, keepdim=True)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        active['profiling'] = False
    cases = []
    for name, case in saved.items():
        path = out/f'{name}.pt'
        torch.save(case, path)
        cases.append({'name': name, 'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                      'shape_qkv': [list(case[n].shape) for n in ('q', 'k', 'v')], 'causal': case['causal']})
    record = {'complete': True, 'args': vars(a), 'input_tokens': ids.shape[1], 'counts': dict(counts),
              'cases': cases, 'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0)}
    (out/'profile.json').write_text(json.dumps(record, indent=2))
    print('FA2_MODEL_PROFILE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
