"""Generate a single response from a local SlimKV checkpoint."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_name_or_path', required=True)
    parser.add_argument('--prompt', default='Explain why a KV cache helps language model inference.')
    parser.add_argument('--prompt_file', type=Path, help='UTF-8 text; overrides --prompt.')
    parser.add_argument('--max_new_tokens', type=int, default=32)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--dtype', choices=['bf16', 'fp16', 'fp32'], default='bf16')
    parser.add_argument('--chat_template', default='llama-3')
    parser.add_argument('--attn_impl', choices=['eager', 'flash_attention_2'], default=None,
                        help='Defaults to FA2 on CUDA and eager on CPU.')
    parser.add_argument('--beacon_cache_mode', choices=['auto', 'full', 'latent'], default=None)
    parser.add_argument('--beacon_decode_backend', choices=['reconstruct', 'latent', 'fa2'], default=None)
    parser.add_argument('--beacon_decode_graph', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--beacon_decode_graph_full', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--beacon_decode_compile', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--beacon_prefill_fast', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error('max_new_tokens must be positive.')
    checkpoint = Path(args.model_name_or_path)
    if not (checkpoint / 'config.json').is_file():
        parser.error('Provide a local SlimKV checkpoint containing config.json, weights and tokenizer files.')

    if args.attn_impl is None:
        args.attn_impl = 'eager' if args.device == 'cpu' else 'flash_attention_2'
    if args.beacon_decode_backend is None:
        args.beacon_decode_backend = 'latent' if args.attn_impl == 'flash_attention_2' else 'reconstruct'
    if args.beacon_cache_mode is None:
        args.beacon_cache_mode = 'latent' if args.beacon_decode_backend in ('latent', 'fa2') else 'auto'
    fast = (args.beacon_decode_backend == 'fa2' or
            (args.beacon_decode_backend == 'latent' and args.attn_impl == 'flash_attention_2'))
    if args.beacon_decode_graph is None:
        args.beacon_decode_graph = fast
    if args.beacon_decode_compile is None:
        args.beacon_decode_compile = fast

    import torch
    from slimkv import ModelArgs, apply_chat_template, get_model_and_tokenizer
    model, tokenizer = get_model_and_tokenizer(ModelArgs(
        model_name_or_path=str(checkpoint), enable_beacon=True,
        dtype=args.dtype, attn_impl=args.attn_impl, chat_template=args.chat_template,
        beacon_cache_mode=args.beacon_cache_mode, beacon_decode_backend=args.beacon_decode_backend,
        beacon_decode_fast=fast, beacon_decode_graph=args.beacon_decode_graph,
        beacon_decode_graph_full=args.beacon_decode_graph_full,
        beacon_prefill_fast=args.beacon_prefill_fast,
        beacon_decode_compile=args.beacon_decode_compile), device=args.device)
    prompt = args.prompt_file.read_text(encoding='utf-8') if args.prompt_file else args.prompt
    encoded = apply_chat_template(
        args.chat_template, messages=[{'role': 'user', 'content': prompt}],
        tokenizer=tokenizer, add_generation_prompt=True).encoded
    ids = torch.tensor([encoded['input_ids']], device=args.device)
    mask = torch.tensor([encoded['attention_mask']], device=args.device)
    model.memory.reset()
    try:
        with torch.inference_mode():
            output = model.generate(input_ids=ids, attention_mask=mask,
                                    max_new_tokens=args.max_new_tokens, do_sample=False, num_logits_to_keep=1)
        print(tokenizer.decode(output[0, ids.shape[1]:], skip_special_tokens=True))
    finally:
        model.memory.reset()


if __name__ == '__main__':
    main()
