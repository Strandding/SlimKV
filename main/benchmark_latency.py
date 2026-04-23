import argparse
import json
import time
from statistics import mean
from typing import Dict, List

import torch

from src.latency.loader_latency import (
    MODE_BEACON_DENSE,
    MODE_BEACON_LOWRANK_HYBRID,
    MODE_BEACON_LOWRANK_RECON,
    MODE_RAW,
    load_model_for_latency_mode,
)


ALL_MODES = [
    MODE_RAW,
    MODE_BEACON_DENSE,
    MODE_BEACON_LOWRANK_RECON,
    MODE_BEACON_LOWRANK_HYBRID,
]


def _resolve_model_or_ckpt_for_mode(mode: str, args) -> str:
    if mode in (MODE_RAW, MODE_BEACON_DENSE):
        ckpt = args.dense_model_or_ckpt or args.model_or_ckpt
    elif mode in (MODE_BEACON_LOWRANK_RECON, MODE_BEACON_LOWRANK_HYBRID):
        ckpt = args.lowrank_model_or_ckpt or args.model_or_ckpt
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if not ckpt:
        raise ValueError(
            f"No checkpoint provided for mode={mode}. "
            "Please set --model_or_ckpt as fallback, or set "
            "--dense_model_or_ckpt / --lowrank_model_or_ckpt explicitly."
        )
    return ckpt


def _sync_if_cuda(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _peak_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024 ** 2)


def _make_prompt_ids(prompt_len: int, vocab_size: int, device: torch.device, seed: int) -> torch.LongTensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    high = max(11, vocab_size - 1)
    ids = torch.randint(low=10, high=high, size=(1, prompt_len), generator=g, dtype=torch.long)
    return ids.to(device)


@torch.no_grad()
def _benchmark_once(model, prompt_ids: torch.LongTensor, gen_len: int, device: torch.device) -> Dict[str, float]:
    bsz = prompt_ids.shape[0]
    attention_mask = prompt_ids.new_ones((bsz, prompt_ids.shape[1]))

    def _build_position_ids(mask: torch.Tensor, last_token_only: bool = False) -> torch.Tensor:
        position_ids = mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(mask == 0, 1)
        if last_token_only:
            position_ids = position_ids[:, -1:]
        return position_ids

    def _build_raw_native_prefill_mask(seq_len: int) -> torch.Tensor:
        dtype = next(model.parameters()).dtype
        min_value = torch.finfo(dtype).min
        causal = torch.full((seq_len, seq_len), min_value, device=device, dtype=dtype)
        causal = torch.triu(causal, diagonal=1)
        return causal.unsqueeze(0).unsqueeze(0).expand(bsz, 1, seq_len, seq_len)

    def _build_raw_native_decode_mask(kv_len: int) -> torch.Tensor:
        dtype = next(model.parameters()).dtype
        return torch.zeros((bsz, 1, 1, kv_len), device=device, dtype=dtype)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        use_native_raw_path = hasattr(model, "_enable_beacon") and model._enable_beacon is False
        _sync_if_cuda(device)
        prefill_start = time.perf_counter()
        prefill_attention_mask = attention_mask
        if use_native_raw_path:
            prefill_attention_mask = _build_raw_native_prefill_mask(attention_mask.shape[1])
        outputs = model(
            input_ids=prompt_ids,
            attention_mask=prefill_attention_mask,
            position_ids=_build_position_ids(attention_mask, last_token_only=False),
            use_cache=True,
        )
        _sync_if_cuda(device)
        prefill_sec = time.perf_counter() - prefill_start

        past_key_values = outputs.past_key_values
        next_token = outputs.logits[:, -1:].argmax(dim=-1)

        decode_sec = 0.0
        if gen_len > 0:
            _sync_if_cuda(device)
            decode_start = time.perf_counter()
            for _ in range(gen_len):
                attention_mask = torch.cat([attention_mask, attention_mask.new_ones((bsz, 1))], dim=1)
                decode_attention_mask = attention_mask
                if use_native_raw_path:
                    decode_attention_mask = _build_raw_native_decode_mask(attention_mask.shape[1])
                model_inputs = {
                    "input_ids": next_token,
                    "attention_mask": decode_attention_mask,
                    "position_ids": _build_position_ids(
                        attention_mask,
                        last_token_only=(not use_native_raw_path),
                    ),
                    "past_key_values": past_key_values,
                    "use_cache": True,
                }
                outputs = model(**model_inputs)
                past_key_values = outputs.past_key_values
                next_token = outputs.logits[:, -1:].argmax(dim=-1)
            _sync_if_cuda(device)
            decode_sec = time.perf_counter() - decode_start

    total_sec = prefill_sec + decode_sec
    decode_toks_per_sec = (gen_len / decode_sec) if decode_sec > 0 else 0.0

    return {
        "prefill_ms": prefill_sec * 1000.0,
        "decode_ms": decode_sec * 1000.0,
        "total_ms": total_sec * 1000.0,
        "decode_toks_per_sec": decode_toks_per_sec,
        "peak_gpu_memory_mb": _peak_memory_mb(device),
    }


def _summarize(records: List[Dict[str, float]]) -> Dict[str, float]:
    keys = ["prefill_ms", "decode_ms", "total_ms", "decode_toks_per_sec", "peak_gpu_memory_mb"]
    return {k: mean([r[k] for r in records]) for k in keys}


def main():
    parser = argparse.ArgumentParser(description="Latency benchmark for Llama3.1 beacon modes.")
    parser.add_argument("--model_or_ckpt", type=str, default=None)
    parser.add_argument("--dense_model_or_ckpt", type=str, default=None)
    parser.add_argument("--lowrank_model_or_ckpt", type=str, default=None)
    parser.add_argument("--mode", type=str, default="all", choices=["all"] + ALL_MODES)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--attn_impl", type=str, default="eager")
    parser.add_argument("--device_map", type=str, default=None)
    parser.add_argument("--cpu", action="store_true")

    parser.add_argument("--prompt_len", type=int, default=2048)
    parser.add_argument("--gen_len", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--beacon_kv_latent_compress_ratio", type=int, default=2)
    parser.add_argument("--beacon_ratio", type=int, default=8)
    parser.add_argument("--beacon_k_latent_head_dim_list", type=int, nargs="*", default=None)
    parser.add_argument("--beacon_v_latent_head_dim_list", type=int, nargs="*", default=None)

    parser.add_argument("--output_json", type=str, default=None)

    args = parser.parse_args()

    if args.prompt_len <= 0:
        raise ValueError("prompt_len must be positive")
    if args.gen_len < 0:
        raise ValueError("gen_len must be non-negative")
    if args.warmup < 0 or args.repeat <= 0:
        raise ValueError("warmup must be >=0 and repeat must be >0")
    if args.beacon_ratio <= 0:
        raise ValueError("beacon_ratio must be > 0")

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    modes = ALL_MODES if args.mode == "all" else [args.mode]
    all_results = []

    for mode in modes:
        model_or_ckpt = _resolve_model_or_ckpt_for_mode(mode, args)
        model, tokenizer = load_model_for_latency_mode(
            mode=mode,
            model_or_ckpt=model_or_ckpt,
            dtype=args.dtype,
            attn_impl=args.attn_impl,
            device=str(device),
            device_map=args.device_map,
            latent_compress_ratio=args.beacon_kv_latent_compress_ratio,
            k_latent_head_dim_list=args.beacon_k_latent_head_dim_list,
            v_latent_head_dim_list=args.beacon_v_latent_head_dim_list,
            beacon_ratio=args.beacon_ratio,
        )

        if args.device_map is None:
            model = model.to(device)

        model.eval()
        prompt_ids = _make_prompt_ids(
            prompt_len=args.prompt_len,
            vocab_size=model.config.vocab_size,
            device=device,
            seed=args.seed,
        )

        for _ in range(args.warmup):
            _benchmark_once(model, prompt_ids, args.gen_len, device)

        records = []
        for trial in range(args.repeat):
            trial_seed = args.seed + trial + 1
            trial_prompt = _make_prompt_ids(
                prompt_len=args.prompt_len,
                vocab_size=model.config.vocab_size,
                device=device,
                seed=trial_seed,
            )
            records.append(_benchmark_once(model, trial_prompt, args.gen_len, device))

        summary = _summarize(records)
        result = {
            "mode": mode,
            "model_or_ckpt": model_or_ckpt,
            "dtype": args.dtype,
            "attn_impl": args.attn_impl,
            "prompt_len": args.prompt_len,
            "gen_len": args.gen_len,
            "repeat": args.repeat,
            "warmup": args.warmup,
            "summary": summary,
            "trials": records,
        }
        all_results.append(result)

        print(json.dumps(result, ensure_ascii=False))

        # Free memory before next mode.
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if args.output_json is not None:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
