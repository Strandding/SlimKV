"""SlimKV – LongBench evaluation.

Supports two modes:
  1. SlimKV mode (default): windowed prefill with anchor-token KV compression,
     followed by autoregressive decode.
  2. Original model mode (--no_slimkv): standard HuggingFace generate without
     any patching – useful as a baseline.

Multi-GPU evaluation is handled via HuggingFace Accelerate.
If the LongBench dataset is not found locally, it will be automatically
downloaded from HuggingFace Hub (THUDM/LongBench).
"""

import os
import sys
import json
import time
import torch
import datasets
from tqdm import tqdm
from typing import Optional, List
from functools import partial
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from accelerate import Accelerator
from transformers import HfArgumentParser, AutoTokenizer, AutoModelForCausalLM
from transformers.utils import logging
from torch.utils.data import DataLoader

from slimkv import (
    ModelArgs,
    DefaultDataCollator,
    FileLogger,
    get_model_and_tokenizer,
    makedirs,
    apply_chat_template,
)
from slimkv.memory import _cat_kv
from .longbench_utils import (
    DATASET2PROMPT,
    DATASET2MAXNEWTOKENS,
    DATASET2CATEGORY,
    DATASET2METRIC_NAME,
    scorer,
)

logger = logging.get_logger(__name__)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

@dataclass
class Args(ModelArgs):
    eval_data: str = field(
        default="",
        metadata={"help": "Path to LongBench JSONL directory. Auto-downloaded if empty."},
    )
    output_dir: str = field(
        default="data/results/longbench/",
        metadata={"help": "Base directory for saving results and logs."},
    )
    result_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Sub-directory (relative to output_dir) for this run."},
    )

    tasks: List[str] = field(
        default_factory=lambda: [
            "narrativeqa", "qasper", "multifieldqa_en",
            "hotpotqa", "2wikimqa", "musique",
            "gov_report", "qmsum", "multi_news",
            "trec", "triviaqa", "samsum",
            "lcc", "repobench-p",
        ],
        metadata={"help": "LongBench sub-tasks to evaluate."},
    )
    newline_as_eos: bool = field(
        default=True,
        metadata={"help": "Use newline as additional EOS for tasks that require it (default: samsum)."},
    )
    max_length: int = field(
        default=31500,
        metadata={"help": "Max input length (tokens)."},
    )
    truncate_from_middle: bool = field(
        default=True,
        metadata={"help": "Truncate long inputs from the middle."},
    )
    load_result: bool = field(
        default=False,
        metadata={"help": "Load results from saved files instead of re-running."},
    )
    context_mode: str = field(
        default="full",
        metadata={"help": "Input context mode: full | close_book"},
    )

    # SlimKV toggle
    no_slimkv: bool = field(
        default=False,
        metadata={"help": "Disable SlimKV patching – use the original model."},
    )

    do_sample: bool = False
    cpu: bool = False


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

LONGBENCH_HF_DATASET = "THUDM/LongBench"

# Standard local search paths (before downloading)
_LOCAL_SEARCH_PATHS = [
    "/dataset/common/tzh/dataset/long-llm/longbench",
    "/dataset/common/tzh/datasets/long-llm/longbench",
    "/dataset/common/tzh/.dataset/long-llm/longbench",
]

# Keep newline-as-EOS limited to SAMSum for comparability with
# the official LongBench evaluation script.
TASKS_WITH_NEWLINE_EOS = {"samsum"}


def _find_or_download_longbench(eval_data: str, tasks: List[str], cache_dir: Optional[str]) -> str:
    """Return the path to a directory containing ``{task}.jsonl`` files.

    Resolution order:
      1. *eval_data* if it already contains the task files.
      2. Well-known local paths (cluster shared storage).
      3. Download from HuggingFace Hub and convert to JSONL.
    """
    # 1. Explicit path
    if eval_data and os.path.isfile(os.path.join(eval_data, f"{tasks[0]}.jsonl")):
        return eval_data

    # 2. Well-known local paths
    for p in _LOCAL_SEARCH_PATHS:
        if os.path.isfile(os.path.join(p, f"{tasks[0]}.jsonl")):
            return p

    # 3. Download from HuggingFace Hub
    dest_dir = os.path.join(cache_dir or "data", "longbench")
    if os.path.isfile(os.path.join(dest_dir, f"{tasks[0]}.jsonl")):
        return dest_dir

    print(f"[data] LongBench not found locally – downloading from {LONGBENCH_HF_DATASET} ...", flush=True)
    os.makedirs(dest_dir, exist_ok=True)

    for task in list(DATASET2PROMPT.keys()):
        out_path = os.path.join(dest_dir, f"{task}.jsonl")
        if os.path.isfile(out_path):
            continue
        try:
            ds = datasets.load_dataset(LONGBENCH_HF_DATASET, task, split="test", cache_dir=cache_dir)
        except Exception as e:
            print(f"[data] skipping {task}: {e}", flush=True)
            continue
        with open(out_path, "w", encoding="utf-8") as f:
            for sample in ds:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        print(f"[data] saved {task} ({len(ds)} samples)", flush=True)

    return dest_dir


# ---------------------------------------------------------------------------
# Data processing
# ---------------------------------------------------------------------------

def process_longbench(
    data,
    indices,
    tokenizer,
    chat_template,
    task,
    max_length=31500,
    truncate_from_middle=True,
    context_mode="full",
):
    outputs = {"input_ids": [], "attention_mask": [], "index": []}

    for input_text, context, index in zip(data["input"], data["context"], indices):
        prompt_template = DATASET2PROMPT[task]
        prompt_context = context
        if context_mode == "close_book":
            prompt_context = ""
        prompt = prompt_template.format(input=input_text, context=prompt_context)

        if truncate_from_middle:
            tokenized_prompt = tokenizer.encode(prompt)
            if len(tokenized_prompt) > max_length:
                half = int(max_length / 2)
                prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        else:
            tokenized_prompt = tokenizer.encode(prompt)
            prompt = tokenizer.decode(tokenized_prompt[-max_length:], skip_special_tokens=True)

        # Apply chat template for most tasks, except Few-Shot Learning and Code Completion
        if not any(x in DATASET2CATEGORY[task] for x in ["Few-Shot Learning", "Code Completion"]):
            encoded = apply_chat_template(
                chat_template,
                messages=[{"role": "user", "content": prompt}],
                tokenizer=tokenizer,
                add_generation_prompt=True,
            ).encoded
        else:
            encoded = tokenizer(prompt)

        outputs["input_ids"].append(encoded["input_ids"])
        outputs["attention_mask"].append(encoded["attention_mask"])
        outputs["index"].append(index)

    return outputs


# ---------------------------------------------------------------------------
# Window length distribution (diagnostic)
# ---------------------------------------------------------------------------

def build_window_length_distribution(input_ids_list, window_size: Optional[int]):
    if window_size is None or window_size <= 0:
        return None

    total = len(input_ids_list)
    if total == 0:
        return {"window_size": window_size, "num_samples": 0, "max_input_tokens": 0, "buckets": {}}

    bucket_counts = defaultdict(int)
    max_input_tokens = 0
    for token_ids in input_ids_list:
        token_len = len(token_ids)
        max_input_tokens = max(max_input_tokens, token_len)
        if token_len <= 0:
            continue
        bucket_idx = (token_len - 1) // window_size
        bucket_counts[bucket_idx] += 1

    if not bucket_counts:
        return {"window_size": window_size, "num_samples": total, "max_input_tokens": max_input_tokens, "buckets": {}}

    buckets = {}
    max_bucket_idx = max(bucket_counts.keys())
    for i in range(max_bucket_idx + 1):
        low = i * window_size + 1
        high = (i + 1) * window_size
        name = "within_1_window" if i == 0 else f"{i}_to_{i+1}_windows"
        count = bucket_counts.get(i, 0)
        buckets[name] = {"token_range": [low, high], "count": count, "ratio": round(count / total, 4)}

    return {"window_size": window_size, "num_samples": total, "max_input_tokens": max_input_tokens, "buckets": buckets}


# ---------------------------------------------------------------------------
# SlimKV generation (windowed prefill + autoregressive decode)
# ---------------------------------------------------------------------------

@torch.no_grad()
def slimkv_generate(model, input_ids, attention_mask, max_new_tokens, eos_token_id=None):
    """Generate tokens with SlimKV: windowed prefill then greedy autoregressive decode."""
    memory = model.memory
    memory.reset()
    bsz = input_ids.shape[0]
    device = input_ids.device
    dtype = model.config.torch_dtype if hasattr(model.config, "torch_dtype") and model.config.torch_dtype is not None else torch.bfloat16

    # Normalise eos_token_id to a list
    if eos_token_id is None:
        _eos_list = []
    elif isinstance(eos_token_id, int):
        _eos_list = [eos_token_id]
    else:
        _eos_list = list(eos_token_id)

    # === Phase 1: Windowed prefill ===
    memory.prepare(input_ids, attention_mask, labels=None)

    last_logits = None
    while not memory.finish:
        chunk_ids, chunk_mask, chunk_pos, past_kv, _ = memory.step()
        outputs = model._patched_forward(
            input_ids=chunk_ids,
            attention_mask=chunk_mask,
            position_ids=chunk_pos,
            past_key_values=past_kv,
            labels=None,
            use_cache=True,
        )
        memory.update_memory(outputs.past_key_values)
        last_logits = outputs.logits

    if last_logits is None:
        # Edge case: empty input
        return input_ids

    # First generated token
    next_token = last_logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [bsz, 1]
    generated = [next_token]

    if max_new_tokens <= 1:
        return torch.cat([input_ids, torch.cat(generated, dim=1)], dim=1)

    # Check immediate EOS
    if _eos_list and all(next_token[b, 0].item() in _eos_list for b in range(bsz)):
        return torch.cat([input_ids, torch.cat(generated, dim=1)], dim=1)

    # === Phase 2: Build decode cache from memory state ===
    decode_cache = []
    for layer_idx in range(memory.num_layers):
        ak, av = memory.anchor_kv[layer_idx]
        rk, rv = memory.raw_kv[layer_idx]
        k = _cat_kv(ak, rk, dim=2)
        v = _cat_kv(av, rv, dim=2)
        decode_cache.append((k, v))

    base_cache_len = decode_cache[0][0].shape[2] if decode_cache[0][0] is not None else 0

    # === Phase 3: Autoregressive decode ===
    for step in range(1, max_new_tokens):
        cur_cache_len = base_cache_len + step  # already includes previously generated tokens
        total_len = cur_cache_len + 1

        # Position ids for all KV + new token
        position_ids = torch.arange(total_len, dtype=torch.long, device=device).unsqueeze(0).expand(bsz, -1)

        # Attention mask: new token can attend to everything (no masking needed)
        attn_mask = torch.zeros(1, 1, 1, total_len, device=device, dtype=dtype)

        # Build past_key_values
        past_key_values = []
        dummy_anchor_indices = torch.zeros(1, device=device, dtype=torch.long)
        for k, v in decode_cache:
            past_key_values.append((k, v, 0, dummy_anchor_indices))

        outputs = model._patched_forward(
            input_ids=next_token,
            attention_mask=attn_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            labels=None,
            use_cache=True,
        )

        # Update decode cache with the new token's KV
        new_decode_cache = []
        for layer_idx, (new_k, new_v, _, _) in enumerate(outputs.past_key_values):
            old_k, old_v = decode_cache[layer_idx]
            new_decode_cache.append((
                _cat_kv(old_k, new_k, dim=2),
                _cat_kv(old_v, new_v, dim=2),
            ))
        decode_cache = new_decode_cache

        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_token)

        if _eos_list and all(next_token[b, 0].item() in _eos_list for b in range(bsz)):
            break

    return torch.cat([input_ids, torch.cat(generated, dim=1)], dim=1)


# ---------------------------------------------------------------------------
# Load original (un-patched) model
# ---------------------------------------------------------------------------

def get_original_model_and_tokenizer(args: Args, device="cpu"):
    """Load a vanilla HuggingFace model without SlimKV patches."""
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(args.dtype, torch.float32)

    tokenizer = load_tokenizer(args)

    attn_kwargs = {}
    if args.attn_impl is not None:
        attn_kwargs["attn_implementation"] = args.attn_impl

    device_map = args.device_map
    if device_map is None:
        device_map = {"": device}

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.model_cache_dir,
        torch_dtype=dtype,
        device_map=device_map,
        token=args.access_token,
        trust_remote_code=True,
        **attn_kwargs,
    )
    model.eval()
    return model, tokenizer


def load_tokenizer(args: Args):
    """Load tokenizer and ensure pad token is available."""
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.model_cache_dir,
        padding_side=args.padding_side,
        token=args.access_token,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def generate_task_with_hf_or_slimkv(
    dataset,
    task: str,
    model,
    tokenizer,
    batch_size: int,
    max_new_tokens: int,
    task_eos: List[int],
    use_slimkv: bool,
    cpu: bool,
    accelerator: Accelerator,
):
    """Run one task with HF/SlimKV backend and return predictions + efficiency stats."""
    data_collator = DefaultDataCollator(tokenizer=tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
        pin_memory=not cpu,
    )

    if accelerator.process_index == 0:
        try:
            dl_len = len(dataloader)
        except Exception:
            dl_len = None
        print(
            f"[gen] preparing task={task} batch_size={batch_size} "
            f"dataloader_len={dl_len}",
            flush=True,
        )

    dataloader = accelerator.prepare(dataloader)
    generating_iter = dataloader
    if accelerator.process_index == 0:
        generating_iter = tqdm(
            dataloader,
            desc=f"Generating ({task})",
            dynamic_ncols=True,
            mininterval=1.0,
            file=sys.stdout,
        )

    preds = []
    indices = []
    total_input_tokens = 0
    total_output_tokens = 0

    for batch_i, x in enumerate(generating_iter):
        if accelerator.process_index == 0 and (batch_i % 10 == 0):
            print(f"[gen] task={task} batch={batch_i}", flush=True)

        batch_indices = x.pop("index").tolist()
        input_length = x["input_ids"].shape[1]
        batch_size_actual = x["input_ids"].shape[0]

        # Force deterministic decoding for benchmark comparability.
        # Some models (e.g., Qwen3) ship with sampling enabled in
        # generation_config, so we override it explicitly.
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "num_beams": 1,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 50,
        }
        if task in TASKS_WITH_NEWLINE_EOS:
            gen_kwargs["eos_token_id"] = task_eos

        if use_slimkv:
            # Reset memory before each sample
            model.memory.reset()
            output = slimkv_generate(
                model,
                x["input_ids"],
                x["attention_mask"],
                max_new_tokens=max_new_tokens,
                eos_token_id=gen_kwargs.get("eos_token_id"),
            )
        else:
            output = model.generate(**x, **gen_kwargs)

        if isinstance(output, torch.Tensor):
            total_input_tokens += input_length * batch_size_actual
            total_output_tokens += (output.shape[1] - input_length) * batch_size_actual
            output = output[:, input_length:]
            output = tokenizer.batch_decode(output, skip_special_tokens=True)

        if accelerator.num_processes > 1:
            output = accelerator.gather_for_metrics(output)
            batch_indices = accelerator.gather_for_metrics(batch_indices)

        if accelerator.process_index == 0:
            preds.extend(output)
            indices.extend(batch_indices)

    return preds, indices, total_input_tokens, total_output_tokens


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@torch.no_grad()
def main():
    parser = HfArgumentParser([Args])
    args: Args = parser.parse_args_into_dataclasses()[0]

    accelerator = Accelerator(cpu=args.cpu)
    use_slimkv = not args.no_slimkv
    if args.context_mode not in {"full", "close_book"}:
        raise ValueError(f"Unsupported context_mode={args.context_mode}, expected one of: full | close_book")

    # --- Load model ---
    if use_slimkv:
        model, tokenizer = get_model_and_tokenizer(args, device=accelerator.device)
    else:
        model, tokenizer = get_original_model_and_tokenizer(args, device=accelerator.device)

    # --- EOS token setup ---
    if hasattr(model, "generation_config") and model.generation_config is not None:
        eos_token_id = model.generation_config.eos_token_id
    else:
        eos_token_id = tokenizer.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    elif eos_token_id is None:
        eos_token_id = [tokenizer.eos_token_id]
    eos_token_id = [x for x in eos_token_id if x is not None]
    if not eos_token_id and tokenizer.eos_token_id is not None:
        eos_token_id = [tokenizer.eos_token_id]
    if args.newline_as_eos:
        eos_token_id.append(tokenizer.encode("\n", add_special_tokens=False)[-1])

    # --- Resolve tasks ---
    if args.tasks == ["all"]:
        tasks = list(DATASET2PROMPT.keys())
    else:
        tasks = args.tasks

    # --- Find or download dataset ---
    with accelerator.main_process_first():
        eval_data = _find_or_download_longbench(args.eval_data, tasks, args.dataset_cache_dir)
        if accelerator.process_index == 0:
            print(f"[data] eval_data={eval_data}", flush=True)

    # --- Process datasets ---
    with accelerator.main_process_first():
        all_datasets = {}
        for task in tasks:
            process_fn = partial(
                process_longbench,
                tokenizer=tokenizer,
                chat_template=args.chat_template,
                task=task,
                max_length=args.max_length,
                truncate_from_middle=args.truncate_from_middle,
                context_mode=args.context_mode,
            )

            path = os.path.join(eval_data, f"{task}.jsonl")
            if not os.path.isfile(path):
                if accelerator.process_index == 0:
                    print(f"[warn] {path} not found, skipping {task}", flush=True)
                continue

            raw_dataset = datasets.load_dataset("json", data_files=path, cache_dir=args.dataset_cache_dir, split="train")
            map_num_proc = int(os.environ.get("MAP_NUM_PROC", "32"))
            map_batch_size = int(os.environ.get("MAP_BATCH_SIZE", "10"))
            map_load_from_cache_file_env = os.environ.get("MAP_LOAD_FROM_CACHE_FILE", "")
            map_load_from_cache_file: Optional[bool] = None
            if map_load_from_cache_file_env != "":
                map_load_from_cache_file = bool(int(map_load_from_cache_file_env))

            if accelerator.process_index == 0:
                print(f"[map] start task={task} num_proc={map_num_proc} batch_size={map_batch_size}", flush=True)

            dataset = raw_dataset.map(
                process_fn,
                batched=True,
                num_proc=map_num_proc,
                batch_size=map_batch_size,
                with_indices=True,
                remove_columns=raw_dataset.column_names,
                desc=f"LongBench map ({task})",
                load_from_cache_file=map_load_from_cache_file,
            )
            if accelerator.process_index == 0:
                print(f"[map] done task={task}", flush=True)

            all_datasets[task] = (raw_dataset, dataset)

    result_dir = os.path.join(args.output_dir, args.result_dir) if args.result_dir else args.output_dir

    metrics = {}
    length_distributions = {}
    task_metric_names = {}

    stat_window_size = args.window_size if use_slimkv else None

    for i, task in enumerate(all_datasets.keys()):
        if accelerator.process_index == 0:
            logger.info(f"Evaluating {task} ({i + 1} / {len(all_datasets)})...")

        result_path = os.path.join(result_dir, f"{task}.json")
        raw_dataset, dataset = all_datasets[task]

        # Length distribution
        if accelerator.process_index == 0:
            dist = build_window_length_distribution(dataset["input_ids"], stat_window_size)
            if dist is not None:
                length_distributions[task] = dist
                metrics[f"{task}_length_distribution"] = dist
                print(f"[length_dist] {task}: {json.dumps(dist, ensure_ascii=False)}", flush=True)

        if not (args.load_result and os.path.exists(result_path)):
            max_new_tokens = DATASET2MAXNEWTOKENS[task]

            # Efficiency tracking
            task_start_time = time.time()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            # Per-task EOS tokens
            task_eos = list(eos_token_id)  # copy

            preds, indices, total_input_tokens, total_output_tokens = generate_task_with_hf_or_slimkv(
                dataset=dataset,
                task=task,
                model=model,
                tokenizer=tokenizer,
                batch_size=args.batch_size,
                max_new_tokens=max_new_tokens,
                task_eos=task_eos,
                use_slimkv=use_slimkv,
                cpu=args.cpu,
                accelerator=accelerator,
            )

            task_elapsed = time.time() - task_start_time
            peak_memory_mb = 0.0
            if torch.cuda.is_available():
                peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        else:
            if accelerator.process_index == 0:
                preds = []
                indices = []
                with open(result_path, "r", encoding="utf-8") as f:
                    f.readline()  # first line is result header: {"score": ..., "metric": ...}
                    for line in f:
                        item = json.loads(line)
                        preds.append(item["pred"])
                        indices.append(len(indices))

        if accelerator.process_index == 0:
            answers = raw_dataset["answers"]
            all_classes = raw_dataset["all_classes"][0]
            score = scorer(task, preds, answers, all_classes)
            metric_name = DATASET2METRIC_NAME.get(task, "Score")
            task_metric_names[task] = metric_name

            logger.info(f"{task} [{metric_name}]: {score}")
            metrics[task] = score

            # Efficiency log
            if not (args.load_result and os.path.exists(result_path)):
                num_samples = len(preds)
                throughput = total_output_tokens / task_elapsed if task_elapsed > 0 else 0
                eff = {
                    "time_sec": round(task_elapsed, 2),
                    "num_samples": num_samples,
                    "sec_per_sample": round(task_elapsed / num_samples, 3) if num_samples > 0 else 0,
                    "input_tokens": total_input_tokens,
                    "output_tokens": total_output_tokens,
                    "output_tokens_per_sec": round(throughput, 1),
                    "peak_gpu_memory_mb": round(peak_memory_mb, 1),
                }
                metrics[f"{task}_efficiency"] = eff
                print(f"[efficiency] {task}: {json.dumps(eff)}", flush=True)

            with open(makedirs(result_path), "w", encoding="utf-8") as f:
                f.write(json.dumps({"score": score, "metric": metric_name}, ensure_ascii=False) + "\n")
                for idx, pred in zip(indices, preds):
                    sample = raw_dataset[idx]
                    for drop_key in ("all_classes", "context", "language", "_id"):
                        sample.pop(drop_key, None)
                    sample["pred"] = pred
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    # --- Aggregate metrics ---
    if accelerator.process_index == 0:
        args.save(os.path.join(result_dir, "config.json"))

        category_metrics = defaultdict(list)
        for dataset_name in tasks:
            if dataset_name not in metrics:
                continue
            category = DATASET2CATEGORY[dataset_name]
            category_metrics[category].append(metrics[dataset_name])
        for k, v in category_metrics.items():
            if isinstance(v[0], dict):
                cat_m = {}
                for kk in v[0].keys():
                    vv = [vj[kk] for vj in v]
                    cat_m[kk] = round(sum(vv) / len(vv), 2)
                category_metrics[k] = cat_m
            else:
                category_metrics[k] = round(sum(v) / len(v), 2)

        task_metrics = {k: metrics[k] for k in tasks if k in metrics}
        if task_metrics:
            vals = list(task_metrics.values())
            if isinstance(vals[0], dict):
                avg = defaultdict(list)
                for v in vals:
                    for kk, vv in v.items():
                        avg[kk].append(vv)
                avg = {k: round(sum(v) / len(v), 2) for k, v in avg.items()}
            else:
                avg = round(sum(vals) / len(vals), 2)
            metrics["avg"] = avg

        file_logger = FileLogger(makedirs(os.path.join(args.output_dir, "metrics.log")))
        file_logger.log(
            metrics,
            Args=asdict(args),
            Category_Metrics=dict(category_metrics),
            Task_Metrics=task_metric_names,
            Length_Distributions=length_distributions,
        )

        print(f"\n{'='*60}", flush=True)
        print(f"  Results saved to: {result_dir}", flush=True)
        for t in tasks:
            if t in metrics:
                mname = task_metric_names.get(t, "Score")
                print(f"  {t} [{mname}]: {metrics[t]}", flush=True)
        if "avg" in metrics:
            print(f"  avg: {metrics['avg']}", flush=True)
        print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
