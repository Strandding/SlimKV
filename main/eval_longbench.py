import os
import datasets
import json
import time
import torch
import sys
from tqdm import tqdm
from typing import Optional, Dict, List
from functools import partial
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from accelerate import Accelerator
from transformers import HfArgumentParser
from transformers.utils import logging
from torch.utils.data import DataLoader

from slimkv import ModelArgs, DefaultDataCollator, FileLogger, get_model_and_tokenizer, makedirs, apply_chat_template
from .longbench_utils import DATASET2PROMPT, DATASET2MAXNEWTOKENS, DATASET2CATEGORY, scorer

logger = logging.get_logger(__name__)


def _format_metric(metric):
    if isinstance(metric, dict):
        return json.dumps(metric, ensure_ascii=False, sort_keys=True)
    if isinstance(metric, float):
        return f"{metric:.2f}"
    return str(metric)


def print_task_score_table(tasks: List[str], metrics: Dict):
    rows = []
    for task in tasks:
        if task in metrics:
            rows.append((task, _format_metric(metrics[task])))
    if "avg" in metrics:
        rows.append(("avg", _format_metric(metrics["avg"])))
    if not rows:
        return

    task_w = max(len("task"), max(len(task) for task, _ in rows))
    score_w = max(len("score"), max(len(score) for _, score in rows))
    border = f"+-{'-' * task_w}-+-{'-' * score_w}-+"

    print("\n[score_table] LongBench task scores", flush=True)
    print(border, flush=True)
    print(f"| {'task'.ljust(task_w)} | {'score'.ljust(score_w)} |", flush=True)
    print(border, flush=True)
    for task, score in rows:
        print(f"| {task.ljust(task_w)} | {score.ljust(score_w)} |", flush=True)
    print(border, flush=True)


def build_window_length_distribution(input_ids_list, window_size: Optional[int]):
    if window_size is None or window_size <= 0:
        return None

    total = len(input_ids_list)
    if total == 0:
        return {
            "window_size": window_size,
            "num_samples": 0,
            "max_input_tokens": 0,
            "buckets": {},
        }

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
        return {
            "window_size": window_size,
            "num_samples": total,
            "max_input_tokens": max_input_tokens,
            "buckets": {},
        }

    buckets = {}
    max_bucket_idx = max(bucket_counts.keys())
    for i in range(max_bucket_idx + 1):
        low_tokens = i * window_size + 1
        high_tokens = (i + 1) * window_size
        if i == 0:
            bucket_name = "within_1_window"
        else:
            bucket_name = f"{i}_to_{i+1}_windows"
        count = bucket_counts.get(i, 0)
        buckets[bucket_name] = {
            "token_range": [low_tokens, high_tokens],
            "count": count,
            "ratio": round(count / total, 4),
        }

    return {
        "window_size": window_size,
        "num_samples": total,
        "max_input_tokens": max_input_tokens,
        "buckets": buckets,
    }


@dataclass
class Args(ModelArgs):
    eval_data: str = field(
        default="long-llm:longbench/",
        metadata={'help': 'The evaluation json data path.'}
    )
    output_dir: str = field(
        default="data/results/longbench/",
        metadata={'help': 'The base directory for saving results and logs.'}
    )
    result_dir: Optional[str] = field(
        default=None,
        metadata={'help': 'The directory relative to output_dir for saving results.'}
    )

    tasks: List[str] = field(
        default_factory=lambda: ['narrativeqa', 'qasper', 'multifieldqa_en', 'hotpotqa', '2wikimqa', 'musique', 'gov_report', 'qmsum', 'multi_news', 'trec', 'triviaqa', 'samsum', 'lcc', 'repobench-p'],
        metadata={'help': 'Which dataset to evaluate?'}
    )
    newline_as_eos: bool = field(
        default=True,
        metadata={'help': 'Whether to use new line as eos (for QA tasks only) or not.'}
    )

    max_length: int = field(
        default=31500,
        metadata={'help': 'Max input length.'}
    )
    truncate_from_middle: bool = field(
        default=True,
        metadata={'help': 'Truncate inputs from the middle.'}
    )
    load_result: bool = field(
        default=False,
        metadata={'help': 'Load result from saved files?'}
    )
    question_only: bool = field(
        default=False,
        metadata={'help': 'Drop context and keep only question input in the prompt.'}
    )
    max_samples_per_task: Optional[int] = field(
        default=None,
        metadata={'help': 'Maximum number of evaluation samples per LongBench task.'}
    )
    sample_seed: int = field(
        default=42,
        metadata={'help': 'Random seed used when sub-sampling evaluation data.'}
    )
    sample_shuffle: bool = field(
        default=True,
        metadata={'help': 'Shuffle before applying max_samples_per_task.'}
    )

    do_sample: bool = False


def process_longbench(data, indices, tokenizer, chat_template, task, max_length=31500, truncate_from_middle=True, question_only=False):
    outputs = {'input_ids': [], 'attention_mask': [], "index": []}

    for input, context, index in zip(data['input'], data['context'], indices):
        if question_only:
            context = ""
        prompt_template = DATASET2PROMPT[task]
        prompt = prompt_template.format(input=input, context=context)

        if truncate_from_middle:
            tokenized_prompt = tokenizer.encode(prompt)
            if len(tokenized_prompt) > max_length:
                half = int(max_length / 2)
                prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        else:
            tokenized_prompt = tokenizer.encode(prompt)
            prompt = tokenizer.decode(tokenized_prompt[-max_length:], skip_special_tokens=True)

        # in fewshot learning and code completion we do not need chat template
        if not any(x in DATASET2CATEGORY[task] for x in ["Few-Shot Learning", "Code Completion"]):
            encoded = apply_chat_template(
                chat_template, 
                messages=[{'role': 'user', 'content': prompt}],
                tokenizer=tokenizer,
                add_generation_prompt=True,
            ).encoded
        else:
            encoded = tokenizer(prompt)

        outputs["input_ids"].append(encoded["input_ids"])
        outputs["attention_mask"].append(encoded["attention_mask"])
        outputs["index"].append(index)

    return outputs


@torch.no_grad()
def main():
    parser = HfArgumentParser([Args])
    args = parser.parse_args_into_dataclasses()[0]

    accelerator = Accelerator(cpu=args.cpu)
    model, tokenizer = get_model_and_tokenizer(args, device=accelerator.device)

    if hasattr(model, "generation_config"):
        eos_token_id = model.generation_config.eos_token_id
    else:
        eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        eos_token_id = []
    elif isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    elif not isinstance(eos_token_id, list):
        eos_token_id = list(eos_token_id)
    # stop generation for QA tasks when \n appears
    if args.newline_as_eos:
        newline_eos = tokenizer.encode("\n", add_special_tokens=False)[-1]
        if newline_eos not in eos_token_id:
            eos_token_id.append(newline_eos)

    if hasattr(model, "generation_config"):
        pad_token_id = model.generation_config.pad_token_id
    else:
        pad_token_id = None
    if pad_token_id is None:
        pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = eos_token_id[0]
    if hasattr(model, "generation_config") and model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = pad_token_id

    if args.tasks == ["all"]:
        tasks = list(DATASET2PROMPT.keys())
    else:
        tasks = args.tasks

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
                question_only=args.question_only,
            )

            path = os.path.join(args.eval_data, f"{task}.jsonl")
            raw_dataset = datasets.load_dataset("json", data_files=path, cache_dir=args.dataset_cache_dir, split="train")
            if args.max_samples_per_task is not None and len(raw_dataset) > args.max_samples_per_task:
                if args.sample_shuffle:
                    raw_dataset = raw_dataset.shuffle(seed=args.sample_seed)
                raw_dataset = raw_dataset.select(range(args.max_samples_per_task))
            map_num_proc = int(os.environ.get("MAP_NUM_PROC", "32"))
            map_batch_size = int(os.environ.get("MAP_BATCH_SIZE", "10"))
            map_load_from_cache_file_env = os.environ.get("MAP_LOAD_FROM_CACHE_FILE", "")
            map_load_from_cache_file: Optional[bool]
            if map_load_from_cache_file_env != "":
                map_load_from_cache_file = bool(int(map_load_from_cache_file_env))
            else:
                map_load_from_cache_file = None
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

    result_dir = os.path.join(args.output_dir, args.result_dir)

    if accelerator.process_index == 0:
        from tools.record_run import record_run
        record_run(os.path.join(result_dir, 'run_manifest.json'), asdict(args),
                   [os.path.join(args.model_name_or_path, 'config.json')]
                   + [os.path.join(args.eval_data, task + '.jsonl') for task in tasks])

    metrics = {}
    length_distributions = {}

    stat_window_size = args.beacon_window
    if stat_window_size is None and hasattr(model, "config"):
        stat_window_size = getattr(model.config, "beacon_window", None)

    for i, task in enumerate(all_datasets.keys()):
        if accelerator.process_index == 0:
            logger.info(f"Evaluating {task} ({i + 1} / {len(all_datasets)})...")

        result_path = os.path.join(result_dir, f"{task}.json")

        raw_dataset, dataset = all_datasets[task]

        if accelerator.process_index == 0:
            token_length_dist = build_window_length_distribution(dataset["input_ids"], stat_window_size)
            if token_length_dist is not None:
                length_distributions[task] = token_length_dist
                metrics[f"{task}_length_distribution"] = token_length_dist
                print(f"[length_dist] {task}: {json.dumps(token_length_dist, ensure_ascii=False)}", flush=True)
            else:
                print("[length_dist] skip (window_size is unset or invalid).", flush=True)

        if not (args.load_result and os.path.exists(result_path)):
            data_collator = DefaultDataCollator(tokenizer=tokenizer)
            dataloader = DataLoader(
                dataset, 
                batch_size=args.batch_size, 
                collate_fn=data_collator,
                # only pin memory when no gpu
                pin_memory=not args.cpu,
            )

            if accelerator.process_index == 0:
                try:
                    dl_len = len(dataloader)
                except Exception:
                    dl_len = None
                print(f"[gen] preparing task={task} batch_size={args.batch_size} dataloader_len={dl_len}", flush=True)

            dataloader = accelerator.prepare(dataloader)

            indices = []
            preds = []
            max_new_tokens = DATASET2MAXNEWTOKENS[task]

            # --- Efficiency metrics collection ---
            task_start_time = time.time()
            total_input_tokens = 0
            total_output_tokens = 0
            peak_memory_mb = 0.0
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            if accelerator.process_index == 0:
                try:
                    total = len(dataloader)
                except Exception:
                    total = None
                print(f"[gen] start task={task} total_batches={total}", flush=True)

            generating_iter = dataloader
            if accelerator.process_index == 0:
                generating_iter = tqdm(
                    dataloader,
                    desc=f"Generating ({task})",
                    dynamic_ncols=True,
                    mininterval=1.0,
                    file=sys.stdout,
                )

            for i, x in enumerate(generating_iter):
                if accelerator.process_index == 0 and (i % 10 == 0):
                    # Print lightweight progress even if tqdm output is suppressed.
                    print(f"[gen] task={task} batch={i}", flush=True)
                index = x.pop("index").tolist()
                input_length = x["input_ids"].shape[1]
                batch_size_actual = x["input_ids"].shape[0]

                # NOTE: important to reset memory for every batch
                if hasattr(model, "memory"):
                    model.memory.reset()

                kwargs = {"max_new_tokens": max_new_tokens, "pad_token_id": pad_token_id}
                if task in ["2wikimqa", "hotpotqa", "musique", "multifieldqa_en", "qasper", "narrativeqa", "samsum"]:
                    kwargs["eos_token_id"] = eos_token_id

                # NOTE: very important to include \n as an eos token for QA tasks, otherwise the F1 score is devastating
                output = model.generate(
                    **x,
                    **kwargs
                )
                if isinstance(output, torch.Tensor):
                    # 1, max_new_tokens
                    total_input_tokens += input_length * batch_size_actual
                    total_output_tokens += (output.shape[1] - input_length) * batch_size_actual
                    output = output[:, input_length:]
                    output = tokenizer.batch_decode(output, skip_special_tokens=True)
                elif isinstance(output, list):
                    pass

                if accelerator.num_processes > 1:
                    output = accelerator.gather_for_metrics(output)
                    index = accelerator.gather_for_metrics(index)

                if accelerator.process_index == 0:
                    preds.extend(output)
                    indices.extend(index)

            # --- Record efficiency metrics for this task ---
            task_elapsed = time.time() - task_start_time
            if torch.cuda.is_available():
                peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        else:
            if accelerator.process_index == 0:
                preds = []
                indices = []

                with open(result_path, "r", encoding="utf-8") as f:
                    # the first line is the metric score
                    f.readline()

                    for line in f:
                        item = json.loads(line)
                        preds.append(item["pred"])
                        indices.append(len(indices))

        if accelerator.process_index == 0:
            answers = raw_dataset["answers"]
            lengths = raw_dataset["length"]
            all_classes = raw_dataset["all_classes"][0]
            score = scorer(task, preds, answers, all_classes)

            logger.info(f"{task}: {score}")
            metrics[task] = score

            # Log efficiency metrics (only when we actually ran inference)
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
                f.write(json.dumps(score, ensure_ascii=False) + "\n")
                for index, pred in zip(indices, preds):
                    sample = raw_dataset[index]
                    del sample["all_classes"]
                    del sample["context"]
                    del sample["language"]
                    del sample["_id"]
                    sample["pred"] = pred
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    if accelerator.process_index == 0:
        # save config
        args.save(os.path.join(result_dir, "config.json"))

        # compute category score
        category_metrics = defaultdict(list)
        for dataset in tasks:
            if dataset not in metrics:
                continue
            metric = metrics[dataset]
            category = DATASET2CATEGORY[dataset]
            category_metrics[category].append(metric)
        for k, v in category_metrics.items():
            # when evaluating on longbench_e, each metric is a dict of float
            if isinstance(v[0], dict):
                category_metric = {}
                for kk in v[0].keys():
                    vv = [v[j][kk] for j in range(len(v))]
                    category_metric[kk] = round(sum(vv) / len(vv), 2)
                category_metrics[k] = category_metric
            else:
                category_metrics[k] = round(sum(v) / len(v), 2)
        
        # compute average score (exclude efficiency entries)
        task_metrics = {k: metrics[k] for k in tasks if k in metrics}
        if task_metrics:
            if isinstance(next(iter(task_metrics.values())), dict):
                avg = defaultdict(list)
                for k, v in task_metrics.items():
                    for kk, vv in v.items():
                        avg[kk].append(vv)
                for k, v in avg.items():
                    avg[k] = round(sum(v) / len(v), 2)
            else:
                avg = round(sum(task_metrics.values()) / len(task_metrics), 2)
            metrics["avg"] = avg

        print_task_score_table(tasks, metrics)

        file_logger = FileLogger(makedirs(os.path.join(args.output_dir, "metrics.log")))
        file_logger.log(metrics, Args=asdict(args), Category_Metrics=category_metrics, Length_Distributions=length_distributions)


if __name__ == "__main__":
    main()
