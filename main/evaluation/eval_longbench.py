import os
import datasets
import json
import time
import csv
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
from .longbench_utils import DATASET2PROMPT, DATASET2MAXNEWTOKENS, DATASET2CATEGORY, ENGLISH_TASKS, scorer

logger = logging.get_logger(__name__)


def save_json(path, value):
    """Replace reports atomically so readers never see a partial JSON file."""
    temporary = makedirs(path + '.tmp')
    with open(temporary, 'w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def save_task_timings(result_dir, rows):
    path = os.path.join(result_dir, 'task_timings.csv')
    with open(path + '.tmp', 'w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            'task', 'score', 'num_samples', 'time_sec', 'sec_per_sample',
            'input_tokens', 'output_tokens', 'output_tokens_per_sec',
            'peak_gpu_memory_mb', 'preprocess_sec', 'model_load_sec',
        ], extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    os.replace(path + '.tmp', path)


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


class EqualLengthBatchSampler:
    """Batch equal-length inputs without introducing padding into Beacon memory."""

    def __init__(self, input_ids, batch_size, max_batch_tokens=None):
        if batch_size < 1:
            raise ValueError('batch_size must be positive')
        if max_batch_tokens is not None and max_batch_tokens < 1:
            raise ValueError('max_batch_tokens must be positive')
        groups = defaultdict(list)
        for index, tokens in enumerate(input_ids):
            groups[len(tokens)].append(index)
        self.batches = []
        for length, indices in sorted(groups.items(), reverse=True):
            limit = batch_size if max_batch_tokens is None else min(
                batch_size, max(1, max_batch_tokens // max(length, 1)))
            self.batches.extend(indices[start:start + limit] for start in range(0, len(indices), limit))

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


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
        default_factory=lambda: list(ENGLISH_TASKS),
        metadata={'help': 'Task names, english (14 tasks), or all (including Chinese and synthetic tasks).'}
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
    truncate_after_chat: bool = field(
        default=False,
        metadata={'help': 'Apply the input limit to final chat-formatted tokens, without decode/re-encode.'}
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
    group_by_input_length: bool = field(
        default=False,
        metadata={'help': 'Batch exactly equal-length inputs without padding; preserves each input.'}
    )
    num_logits_to_keep: Optional[int] = field(
        default=None,
        metadata={'help': 'Optional inference logit limit, supported by the SlimKV Llama3 model.'}
    )
    max_batch_tokens: Optional[int] = field(
        default=None,
        metadata={'help': 'Limit total input tokens in each equal-length batch; a single input is always retained.'}
    )

    do_sample: bool = False


def process_longbench(data, indices, tokenizer, chat_template, task, max_length=31500, truncate_from_middle=True, question_only=False, truncate_after_chat=False):
    outputs = {'input_ids': [], 'attention_mask': [], "index": []}

    for input, context, index in zip(data['input'], data['context'], indices):
        if question_only:
            context = ""
        prompt_template = DATASET2PROMPT[task]
        prompt = prompt_template.format(input=input, context=context)

        if truncate_after_chat:
            pass
        elif truncate_from_middle:
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

        if truncate_after_chat and len(encoded['input_ids']) > max_length:
            if truncate_from_middle:
                head = max_length // 2
                tail = max_length - head
                positions = list(range(head)) + list(range(len(encoded['input_ids']) - tail, len(encoded['input_ids'])))
            else:
                positions = list(range(len(encoded['input_ids']) - max_length, len(encoded['input_ids'])))
            encoded = {key: [values[position] for position in positions]
                       for key, values in encoded.items()}

        outputs["input_ids"].append(encoded["input_ids"])
        outputs["attention_mask"].append(encoded["attention_mask"])
        outputs["index"].append(index)

    return outputs


@torch.no_grad()
def main():
    parser = HfArgumentParser([Args])
    args = parser.parse_args_into_dataclasses()[0]
    if args.max_length < 1 or (args.max_samples_per_task is not None and args.max_samples_per_task < 1):
        raise ValueError('Input length and sample limits must be positive.')
    if args.tasks == ['all']:
        tasks = list(DATASET2PROMPT.keys())
    elif args.tasks == ['english']:
        tasks = list(ENGLISH_TASKS)
    else:
        tasks = args.tasks
    if not tasks or len(tasks) != len(set(tasks)) or any(task not in DATASET2PROMPT for task in tasks):
        raise ValueError('Specify unique, supported LongBench-v1 tasks.')

    result_dir = os.path.join(args.output_dir, args.result_dir)
    configuration = asdict(args)
    config_path = os.path.join(result_dir, 'config.json')
    if args.load_result and os.path.exists(config_path):
        with open(config_path, encoding='utf-8') as stream:
            previous = json.load(stream)
        changed = [key for key, value in configuration.items()
                   if key != 'load_result' and previous.get(key) != value]
        if changed:
            raise ValueError(f'Saved result configuration differs: {", ".join(changed)}. Use a new result_dir.')

    accelerator = Accelerator(cpu=args.cpu)
    load_start = time.monotonic()
    model, tokenizer = get_model_and_tokenizer(args, device=accelerator.device)
    if accelerator.device.type == 'cuda':
        torch.cuda.synchronize(accelerator.device)
    model_load_sec = time.monotonic() - load_start
    if accelerator.process_index == 0:
        save_json(config_path, configuration)

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
    # QA-only stops must not mutate the shared generation configuration.
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

    preprocess_times = {}
    with accelerator.main_process_first():
        all_datasets = {}

        for task in tasks:
            preprocess_start = time.monotonic()
            process_fn = partial(
                process_longbench,
                tokenizer=tokenizer,
                chat_template=args.chat_template,
                task=task,
                max_length=args.max_length,
                truncate_from_middle=args.truncate_from_middle,
                question_only=args.question_only,
                truncate_after_chat=args.truncate_after_chat,
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
            preprocess_times[task] = time.monotonic() - preprocess_start

    if accelerator.process_index == 0:
        from tools.record_run import record_run
        record_run(os.path.join(result_dir, 'run_manifest.json'), asdict(args),
                   [os.path.join(args.model_name_or_path, 'config.json')]
                   + [os.path.join(args.eval_data, task + '.jsonl') for task in tasks])

    metrics = {}
    length_distributions = {}
    timing_rows = []

    stat_window_size = args.beacon_window
    if stat_window_size is None and hasattr(model, "config"):
        stat_window_size = getattr(model.config, "beacon_window", None)

    for i, task in enumerate(all_datasets.keys()):
        task_number = i
        # Release captured shapes from earlier tasks while keeping the loaded model.
        for layer in getattr(getattr(model, 'model', None), 'layers', []):
            if hasattr(layer, '_decode_graphs'):
                layer._decode_graphs = None
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
            batching = ({'batch_sampler': EqualLengthBatchSampler(dataset['input_ids'], args.batch_size, args.max_batch_tokens)}
                        if args.group_by_input_length else {'batch_size': args.batch_size})
            dataloader = DataLoader(
                dataset, 
                **batching,
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
            batch_size_counts = defaultdict(int)
            total_input_tokens = 0
            total_output_tokens = 0
            peak_memory_mb = 0.0
            if accelerator.device.type == 'cuda':
                torch.cuda.synchronize(accelerator.device)
                torch.cuda.reset_peak_memory_stats(accelerator.device)
            task_start_time = time.monotonic()
            if accelerator.process_index == 0:
                save_json(os.path.join(result_dir, 'progress.json'), {
                    'status': 'generating', 'pid': os.getpid(), 'task': task,
                    'generated_samples': 0, 'total_samples': len(raw_dataset),
                    'completed_tasks': [name for name in tasks if name in metrics],
                })

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
                batch_size_counts[str(batch_size_actual)] += 1

                # NOTE: important to reset memory for every batch
                if hasattr(model, "memory"):
                    model.memory.reset()

                kwargs = {"max_new_tokens": max_new_tokens, "pad_token_id": pad_token_id}
                if args.num_logits_to_keep is not None:
                    kwargs['num_logits_to_keep'] = args.num_logits_to_keep
                if task in ["2wikimqa", "hotpotqa", "musique", "multifieldqa_en", "qasper", "narrativeqa", "samsum"]:
                    kwargs["eos_token_id"] = eos_token_id

                # NOTE: very important to include \n as an eos token for QA tasks, otherwise the F1 score is devastating
                output = model.generate(
                    **x,
                    **kwargs
                )
                if isinstance(output, torch.Tensor):
                    # 1, max_new_tokens
                    total_input_tokens += int(x['attention_mask'].sum())
                    total_output_tokens += int((output[:, input_length:] != pad_token_id).sum())
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
                    save_json(os.path.join(result_dir, 'progress.json'), {
                        'status': 'generating', 'pid': os.getpid(), 'task': task,
                        'generated_samples': len(preds), 'total_samples': len(raw_dataset),
                        'generation_sec': round(time.monotonic() - task_start_time, 2),
                        'batch_size_counts': dict(batch_size_counts),
                        'completed_tasks': [name for name in tasks if name in metrics],
                    })

            # --- Record efficiency metrics for this task ---
            if accelerator.device.type == 'cuda':
                torch.cuda.synchronize(accelerator.device)
                peak_memory_mb = torch.cuda.max_memory_allocated(accelerator.device) / (1024 ** 2)
            task_elapsed = time.monotonic() - task_start_time
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
                        indices.append(item.get('index', len(indices)))

        if accelerator.process_index == 0:
            if sorted(indices) != list(range(len(raw_dataset))):
                raise ValueError(f'{task}: predictions must cover every selected sample exactly once.')
            answers = raw_dataset["answers"]
            answers = [answers[index] for index in indices]
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
                    'preprocess_sec': round(preprocess_times[task], 2),
                    'model_load_sec': round(model_load_sec, 2) if task_number == 0 else 0.0,
                    'batch_size_counts': dict(batch_size_counts),
                    'timing_scope': 'Generation wall time includes prefill, decode, data transfer, text decoding and progress writes; excludes model loading and dataset preprocessing.',
                    'world_size': accelerator.num_processes,
                }
                metrics[f"{task}_efficiency"] = eff
                print(f"[efficiency] {task}: {json.dumps(eff)}", flush=True)
                timing = dict(eff, task=task, score=score)
                save_json(os.path.join(result_dir, f'{task}.timing.json'), timing)
            else:
                timing_path = os.path.join(result_dir, f'{task}.timing.json')
                timing = None
                if os.path.exists(timing_path):
                    with open(timing_path, encoding='utf-8') as stream:
                        timing = json.load(stream)
                    metrics[f'{task}_efficiency'] = {key: value for key, value in timing.items() if key not in ['task', 'score']}
            if timing is not None:
                timing_rows.append(dict(timing, score=score))
                save_task_timings(result_dir, timing_rows)

            with open(makedirs(result_path + '.tmp'), "w", encoding="utf-8") as f:
                f.write(json.dumps(score, ensure_ascii=False) + "\n")
                for index, pred in zip(indices, preds):
                    sample = raw_dataset[index]
                    del sample["all_classes"]
                    del sample["context"]
                    del sample["language"]
                    del sample["_id"]
                    sample["pred"] = pred
                    sample['index'] = index
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            os.replace(result_path + '.tmp', result_path)

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
        save_json(os.path.join(result_dir, 'scores.json'), dict(task_metrics, avg=metrics.get('avg')))
        save_json(os.path.join(result_dir, 'progress.json'), {
            'status': 'complete', 'pid': os.getpid(), 'completed_tasks': tasks,
            'num_samples': sum(len(raw) for raw, _ in all_datasets.values()),
        })

        file_logger = FileLogger(makedirs(os.path.join(args.output_dir, "metrics.log")))
        file_logger.log(metrics, Args=asdict(args), Category_Metrics=category_metrics, Length_Distributions=length_distributions)


if __name__ == "__main__":
    main()
