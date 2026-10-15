import os
import json
import torch
from tqdm import tqdm
from typing import Optional, List
from functools import partial
from dataclasses import dataclass, field, asdict

import datasets
from accelerate import Accelerator
from transformers import HfArgumentParser
from transformers.utils import logging
from torch.utils.data import DataLoader

from slimkv import ModelArgs, DefaultDataCollator, get_model_and_tokenizer, makedirs, apply_chat_template
from ..evaluation.longbench_utils import DATASET2PROMPT, DATASET2MAXNEWTOKENS, DATASET2CATEGORY


logger = logging.get_logger(__name__)


@dataclass
class Args(ModelArgs):
    eval_data: str = field(
        default="long-llm:longbench/",
        metadata={"help": "The evaluation json data path."},
    )
    output_dir: str = field(
        default="data/results/longbench_probe/",
        metadata={"help": "The base directory for saving probe artifacts."},
    )
    result_dir: Optional[str] = field(
        default="beacon_probe",
        metadata={"help": "The directory relative to output_dir for saving results."},
    )
    tasks: List[str] = field(
        default_factory=lambda: ["narrativeqa"],
        metadata={"help": "Which LongBench tasks to run for probing."},
    )
    max_length: int = field(
        default=31500,
        metadata={"help": "Max input length."},
    )
    truncate_from_middle: bool = field(
        default=True,
        metadata={"help": "Truncate inputs from the middle."},
    )
    question_only: bool = field(
        default=False,
        metadata={"help": "Drop context and keep only question input in the prompt."},
    )
    max_samples_per_task: int = field(
        default=10,
        metadata={"help": "Probe runs only this many samples per task."},
    )
    sample_seed: int = field(
        default=42,
        metadata={"help": "Random seed used when sub-sampling evaluation data."},
    )
    sample_shuffle: bool = field(
        default=True,
        metadata={"help": "Shuffle before applying max_samples_per_task."},
    )
    probe_max_new_tokens: int = field(
        default=1,
        metadata={"help": "Generation length for probe run."},
    )
    probe_output_name: str = field(
        default="beacon_prev_raw8_attn",
        metadata={"help": "Base name of probe output jsonl."},
    )

    do_sample: bool = False


def process_longbench(data, indices, tokenizer, chat_template, task, max_length=31500, truncate_from_middle=True, question_only=False):
    outputs = {"input_ids": [], "attention_mask": [], "index": []}

    for input_text, context, index in zip(data["input"], data["context"], indices):
        if question_only:
            context = ""
        prompt_template = DATASET2PROMPT[task]
        prompt = prompt_template.format(input=input_text, context=context)

        if truncate_from_middle:
            tokenized_prompt = tokenizer.encode(prompt)
            if len(tokenized_prompt) > max_length:
                half = int(max_length / 2)
                prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + tokenizer.decode(
                    tokenized_prompt[-half:], skip_special_tokens=True
                )
        else:
            tokenized_prompt = tokenizer.encode(prompt)
            prompt = tokenizer.decode(tokenized_prompt[-max_length:], skip_special_tokens=True)

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


@torch.no_grad()
def main():
    parser = HfArgumentParser([Args])
    args = parser.parse_args_into_dataclasses()[0]

    if args.beacon_probe_prev_raw8_attn is None:
        args.beacon_probe_prev_raw8_attn = True
    if args.beacon_probe_prev_raw_tokens is None:
        args.beacon_probe_prev_raw_tokens = 8

    if args.attn_impl != "eager":
        logger.warning(
            "Probe mode requires eager attention to access per-query softmax probabilities. "
            "Overriding attn_impl=%s -> eager.",
            args.attn_impl,
        )
        args.attn_impl = "eager"

    if not args.enable_beacon:
        raise ValueError("Probe script requires --enable_beacon so beacon queries exist.")

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

    if hasattr(model, "generation_config"):
        pad_token_id = model.generation_config.pad_token_id
    else:
        pad_token_id = None
    if pad_token_id is None:
        pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None and eos_token_id:
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
            if len(raw_dataset) > args.max_samples_per_task:
                if args.sample_shuffle:
                    raw_dataset = raw_dataset.shuffle(seed=args.sample_seed)
                raw_dataset = raw_dataset.select(range(args.max_samples_per_task))

            map_num_proc = int(os.environ.get("MAP_NUM_PROC", "8"))
            map_batch_size = int(os.environ.get("MAP_BATCH_SIZE", "10"))
            if accelerator.process_index == 0:
                print(f"[probe-map] start task={task} num_proc={map_num_proc} batch_size={map_batch_size}", flush=True)

            dataset = raw_dataset.map(
                process_fn,
                batched=True,
                num_proc=map_num_proc,
                batch_size=map_batch_size,
                with_indices=True,
                remove_columns=raw_dataset.column_names,
                desc=f"LongBench probe map ({task})",
            )

            if accelerator.process_index == 0:
                print(f"[probe-map] done task={task}", flush=True)

            all_datasets[task] = dataset

    result_dir = os.path.join(args.output_dir, args.result_dir)
    rank_output_path = os.path.join(
        result_dir,
        f"{args.probe_output_name}.rank{accelerator.process_index}.jsonl",
    )
    makedirs(rank_output_path)

    if accelerator.process_index == 0:
        print("[probe] writing probe records to:", flush=True)
    print(f"[probe] rank={accelerator.process_index} -> {rank_output_path}", flush=True)

    unwrapped_model = accelerator.unwrap_model(model)

    with open(rank_output_path, "w", encoding="utf-8") as f_out:
        for task_idx, task in enumerate(tasks):
            dataset = all_datasets[task]
            dataloader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                collate_fn=DefaultDataCollator(tokenizer=tokenizer),
                pin_memory=not args.cpu,
            )
            dataloader = accelerator.prepare(dataloader)

            if accelerator.process_index == 0:
                print(f"[probe-gen] task={task} ({task_idx + 1}/{len(tasks)})", flush=True)

            iter_loader = dataloader
            if accelerator.process_index == 0:
                iter_loader = tqdm(
                    dataloader,
                    desc=f"Probe ({task})",
                    dynamic_ncols=True,
                )

            for step_idx, batch in enumerate(iter_loader):
                sample_indices = batch.pop("index").tolist()

                if hasattr(unwrapped_model, "clear_beacon_probe_records"):
                    unwrapped_model.clear_beacon_probe_records()

                max_new_tokens = min(DATASET2MAXNEWTOKENS.get(task, args.probe_max_new_tokens), args.probe_max_new_tokens)
                _ = model.generate(
                    **batch,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=pad_token_id,
                )

                probe_records = []
                if hasattr(unwrapped_model, "pop_beacon_probe_records"):
                    probe_records = unwrapped_model.pop_beacon_probe_records()

                for rec in probe_records:
                    batch_idx = rec.get("batch_idx", 0)
                    if batch_idx < len(sample_indices):
                        sample_index = int(sample_indices[batch_idx])
                    else:
                        sample_index = -1
                    rec["task"] = task
                    rec["sample_index"] = sample_index
                    rec["step_idx"] = int(step_idx)
                    f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

                if accelerator.process_index == 0 and (step_idx % 5 == 0):
                    print(f"[probe-gen] task={task} step={step_idx} records={len(probe_records)}", flush=True)

    accelerator.wait_for_everyone()

    if accelerator.process_index == 0:
        args.save(os.path.join(result_dir, "config_probe.json"))
        print("[probe] done.", flush=True)


if __name__ == "__main__":
    main()
