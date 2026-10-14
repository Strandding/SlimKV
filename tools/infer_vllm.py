"""Run independent prompts with vLLM-owned SlimKV pages and scheduling."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='Converted SlimKV Llama checkpoint.')
    prompts = parser.add_mutually_exclusive_group(required=True)
    prompts.add_argument('--prompt', action='append', help='Repeat for independent requests.')
    prompts.add_argument('--prompts-json', type=Path, help='JSON list of plain-text prompts.')
    parser.add_argument('--token-ratio', type=int, default=8)
    parser.add_argument('--max-new-tokens', type=int, default=64)
    parser.add_argument('--max-model-len', type=int, default=16384)
    parser.add_argument('--max-num-seqs', type=int, default=4)
    parser.add_argument('--gpu-memory-utilization', type=float, default=.55)
    args = parser.parse_args()
    texts = args.prompt if args.prompt is not None else json.loads(args.prompts_json.read_text())
    if not isinstance(texts, list) or not texts or any(not isinstance(x, str) for x in texts):
        parser.error('Prompts must be a nonempty JSON list of strings.')
    if min(args.token_ratio, args.max_new_tokens, args.max_model_len, args.max_num_seqs) <= 0:
        parser.error('Token ratio, lengths and concurrency must be positive.')
    from vllm import LLM, SamplingParams
    from slimkv.vllm_plugin import register
    register()
    llm = LLM(model=args.model, dtype='bfloat16', enforce_eager=True,
              tensor_parallel_size=1, max_model_len=args.max_model_len,
              max_num_seqs=args.max_num_seqs, max_num_batched_tokens=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization,
              enable_chunked_prefill=False, enable_prefix_caching=False,
              async_scheduling=False, generation_config='auto',
              disable_hybrid_kv_cache_manager=False,
              hf_overrides={'architectures': ['SlimKVLlamaForCausalLM'],
                            'slimkv_native_paged': True, 'beacon_ratio': [args.token_ratio],
                            'beacon_ratio_mix': 'sequence'})
    tokenizer = llm.get_tokenizer()
    requests = [{'prompt_token_ids': tokenizer.apply_chat_template(
        [{'role': 'user', 'content': text}], tokenize=True, add_generation_prompt=True,
        return_dict=False)}
        for text in texts]
    for result in llm.generate(requests, SamplingParams(temperature=0, max_tokens=args.max_new_tokens)):
        completion = result.outputs[0]
        print(json.dumps({'request_id': result.request_id, 'text': completion.text,
                          'token_ids': completion.token_ids}, ensure_ascii=False))


if __name__ == '__main__':
    main()
