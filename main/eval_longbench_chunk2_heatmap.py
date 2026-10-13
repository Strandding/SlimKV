import json
import math
import os
from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional

import datasets
import matplotlib
import numpy as np
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import HfArgumentParser

from slimkv import DefaultDataCollator, ModelArgs, apply_chat_template, get_model_and_tokenizer, makedirs

from .longbench_utils import DATASET2CATEGORY, DATASET2MAXNEWTOKENS, DATASET2PROMPT


matplotlib.use("Agg")
from matplotlib import pyplot as plt


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
        default="beacon_chunk_heatmap",
        metadata={"help": "The directory relative to output_dir for saving probe outputs."},
    )
    figure_dir: str = field(
        default="data/fig",
        metadata={"help": "Directory for saving heatmap figures."},
    )
    figure_name: str = field(
        default="beacon_chunk_attn_heatmap",
        metadata={"help": "Figure base name."},
    )
    heatmap_vmax: float = field(
        default=0.1,
        metadata={"help": "Heatmap colorbar max. <=0 means auto by data max."},
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
        metadata={"help": "Probe this many samples per task."},
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
    stop_after_first_heatmap: bool = field(
        default=True,
        metadata={"help": "Stop once the first heatmap figure is generated."},
    )

    do_sample: bool = False


def process_longbench(data, indices, tokenizer, chat_template, task, max_length=31500, truncate_from_middle=True, question_only=False):
    outputs = {"input_ids": [], "attention_mask": [], "index": []}

    for input_text, context, index in zip(data["input"], data["context"], indices):
        if question_only:
            context = ""
        prompt = DATASET2PROMPT[task].format(input=input_text, context=context)

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


def plot_chunk2_heatmap_grid(records, out_png, title, heatmap_vmax):
    if not records:
        raise ValueError("No heatmap records found.")

    records = sorted(records, key=lambda x: x.get("layer_idx", -1))
    heatmaps = [np.asarray(r["heatmap"], dtype=np.float32) for r in records]
    num_layers = len(heatmaps)

    ncols = min(8, max(1, int(math.ceil(math.sqrt(num_layers)))))
    nrows = int(math.ceil(num_layers / ncols))
    fig_w = max(10.0, ncols * 2.3)
    fig_h = max(6.0, nrows * 2.1)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fig_w, fig_h),
        dpi=220,
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)

    if heatmap_vmax is not None and float(heatmap_vmax) > 0:
        vmax = float(heatmap_vmax)
    else:
        vmax = max(float(np.max(h)) for h in heatmaps)
        if vmax <= 0:
            vmax = 1.0

    img = None
    for i, (record, heatmap) in enumerate(zip(records, heatmaps)):
        ax = axes[i // ncols, i % ncols]
        img = ax.imshow(
            heatmap,
            cmap="viridis",
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            vmin=0.0,
            vmax=vmax,
        )
        ax.set_title(f"Layer {record['layer_idx']}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    for i in range(num_layers, nrows * ncols):
        axes[i // ncols, i % ncols].axis("off")

    cbar = fig.colorbar(img, ax=axes.ravel().tolist(), shrink=0.96, pad=0.01)
    cbar.set_label("Attention Probability (Head Mean)", fontsize=9)
    fig.suptitle(title, fontsize=12, y=0.995)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def build_query_subset_records(records, positions_key):
    subset_records = []
    for record in records:
        heatmap = np.asarray(record["heatmap"], dtype=np.float32)
        positions = record.get(positions_key, [])
        valid_pos = [int(p) for p in positions if 0 <= int(p) < heatmap.shape[0]]
        if not valid_pos:
            continue

        sub_heatmap = heatmap[valid_pos, :]
        sub_record = dict(record)
        sub_record["heatmap"] = sub_heatmap.tolist()
        sub_record["q_len"] = int(sub_heatmap.shape[0])
        subset_records.append(sub_record)
    return subset_records


def build_query_key_subset_records(records, query_positions_key, key_positions_key):
    subset_records = []
    for record in records:
        heatmap = np.asarray(record["heatmap"], dtype=np.float32)
        query_positions = record.get(query_positions_key, [])
        if key_positions_key == "raw_key_positions":
            beacon_keys = record.get("beacon_key_positions", [])
            beacon_key_set = {
                int(p)
                for p in beacon_keys
                if 0 <= int(p) < heatmap.shape[1]
            }
            key_positions = [p for p in range(int(heatmap.shape[1])) if p not in beacon_key_set]
        else:
            key_positions = record.get(key_positions_key, [])
        valid_q = [int(p) for p in query_positions if 0 <= int(p) < heatmap.shape[0]]
        valid_k = [int(p) for p in key_positions if 0 <= int(p) < heatmap.shape[1]]
        if not valid_q or not valid_k:
            continue

        sub_heatmap = heatmap[np.ix_(valid_q, valid_k)]
        sub_record = dict(record)
        sub_record["heatmap"] = sub_heatmap.tolist()
        sub_record["q_len"] = int(sub_heatmap.shape[0])
        sub_record["kv_seq_len"] = int(sub_heatmap.shape[1])
        subset_records.append(sub_record)
    return subset_records


@torch.no_grad()
def main():
    parser = HfArgumentParser([Args])
    args = parser.parse_args_into_dataclasses()[0]

    if args.attn_impl != "eager":
        raise ValueError("Chunk heatmap probe requires attn_impl=eager.")

    raw_probe_mode = not bool(args.enable_beacon)
    if not raw_probe_mode:
        if args.beacon_ratio is None:
            args.beacon_ratio = [8]
        if args.beacon_ratio != [8]:
            raise ValueError(f"Chunk heatmap probe is fixed to beacon_ratio=[8], got {args.beacon_ratio}.")
        if args.beacon_ratio_mix is None:
            args.beacon_ratio_mix = "sequence"
    args.beacon_probe_chunk2_heatmap = True
    if args.beacon_probe_chunk_idx is None:
        args.beacon_probe_chunk_idx = 2
    if int(args.beacon_probe_chunk_idx) <= 0:
        raise ValueError(f"beacon_probe_chunk_idx must be >=1, got {args.beacon_probe_chunk_idx}.")

    accelerator = Accelerator(cpu=args.cpu)
    model, tokenizer = get_model_and_tokenizer(args, device=accelerator.device)
    # Model is not wrapped by accelerator.prepare in this script, so directly
    # using model avoids accelerate->deepspeed unwrap import path.
    unwrapped_model = model

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

    tasks = list(DATASET2PROMPT.keys()) if args.tasks == ["all"] else args.tasks

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

            dataset = raw_dataset.map(
                process_fn,
                batched=True,
                num_proc=int(os.environ.get("MAP_NUM_PROC", "8")),
                batch_size=int(os.environ.get("MAP_BATCH_SIZE", "10")),
                with_indices=True,
                remove_columns=raw_dataset.column_names,
                desc=f"LongBench heatmap map ({task})",
            )
            all_datasets[task] = dataset

    result_dir = os.path.join(args.output_dir, args.result_dir)
    makedirs(os.path.join(result_dir, "dummy.txt"))
    makedirs(os.path.join(args.figure_dir, "dummy.txt"))

    found_heatmap = False
    for task in tasks:
        dataset = all_datasets[task]
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            collate_fn=DefaultDataCollator(tokenizer=tokenizer),
            pin_memory=not args.cpu,
        )
        dataloader = accelerator.prepare(dataloader)

        iter_loader = dataloader
        if accelerator.process_index == 0:
            iter_loader = tqdm(dataloader, desc=f"Heatmap probe ({task})", dynamic_ncols=True)

        for step_idx, batch in enumerate(iter_loader):
            sample_indices = batch.pop("index").tolist()
            if hasattr(unwrapped_model, "clear_beacon_chunk2_heatmap_records"):
                unwrapped_model.clear_beacon_chunk2_heatmap_records()

            max_new_tokens = min(DATASET2MAXNEWTOKENS.get(task, args.probe_max_new_tokens), args.probe_max_new_tokens)
            _ = model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                pad_token_id=pad_token_id,
            )

            records = []
            if hasattr(unwrapped_model, "pop_beacon_chunk2_heatmap_records"):
                records = unwrapped_model.pop_beacon_chunk2_heatmap_records()

            if accelerator.process_index == 0 and records:
                sample_index = int(sample_indices[0]) if sample_indices else -1
                chunk_tag = f"chunk{int(args.beacon_probe_chunk_idx)}"
                base_name = f"{args.figure_name}.{task}.sample{sample_index}.{chunk_tag}"
                out_png = os.path.join(args.figure_dir, f"{base_name}.png")
                plot_chunk2_heatmap_grid(
                    records=records,
                    out_png=out_png,
                    title=f"Chunk-{int(args.beacon_probe_chunk_idx)} Attention Heatmap ({task}, head-mean, ratio=8)",
                    heatmap_vmax=args.heatmap_vmax,
                )

                beacon_query_records = build_query_subset_records(records, "beacon_query_positions")
                out_beacon_q_png = os.path.join(args.figure_dir, f"{base_name}.beacon_query.png")
                if beacon_query_records:
                    plot_chunk2_heatmap_grid(
                        records=beacon_query_records,
                        out_png=out_beacon_q_png,
                        title=(
                            f"Chunk-{int(args.beacon_probe_chunk_idx)} Beacon-Query Attention Heatmap "
                            f"({task}, head-mean, ratio=8)"
                        ),
                        heatmap_vmax=args.heatmap_vmax,
                    )

                raw_query_records = build_query_subset_records(records, "raw_query_positions")
                out_raw_q_png = os.path.join(args.figure_dir, f"{base_name}.raw_query.png")
                if raw_query_records:
                    plot_chunk2_heatmap_grid(
                        records=raw_query_records,
                        out_png=out_raw_q_png,
                        title=(
                            f"Chunk-{int(args.beacon_probe_chunk_idx)} Raw-Query Attention Heatmap "
                            f"({task}, head-mean, ratio=8)"
                        ),
                        heatmap_vmax=args.heatmap_vmax,
                    )

                beacon_to_beacon_records = build_query_key_subset_records(
                    records,
                    query_positions_key="beacon_query_positions",
                    key_positions_key="beacon_key_positions",
                )
                out_beacon_to_beacon_png = os.path.join(args.figure_dir, f"{base_name}.beacon_query_beacon_key.png")
                if beacon_to_beacon_records:
                    plot_chunk2_heatmap_grid(
                        records=beacon_to_beacon_records,
                        out_png=out_beacon_to_beacon_png,
                        title=(
                            f"Chunk-{int(args.beacon_probe_chunk_idx)} Beacon-Query to Beacon-Key Heatmap "
                            f"({task}, head-mean, ratio=8)"
                        ),
                        heatmap_vmax=args.heatmap_vmax,
                    )

                raw_to_beacon_records = build_query_key_subset_records(
                    records,
                    query_positions_key="raw_query_positions",
                    key_positions_key="beacon_key_positions",
                )
                out_raw_to_beacon_png = os.path.join(args.figure_dir, f"{base_name}.raw_query_beacon_key.png")
                if raw_to_beacon_records:
                    plot_chunk2_heatmap_grid(
                        records=raw_to_beacon_records,
                        out_png=out_raw_to_beacon_png,
                        title=(
                            f"Chunk-{int(args.beacon_probe_chunk_idx)} Raw-Query to Beacon-Key Heatmap "
                            f"({task}, head-mean, ratio=8)"
                        ),
                        heatmap_vmax=args.heatmap_vmax,
                    )

                beacon_to_raw_records = build_query_key_subset_records(
                    records,
                    query_positions_key="beacon_query_positions",
                    key_positions_key="raw_key_positions",
                )
                out_beacon_to_raw_png = os.path.join(args.figure_dir, f"{base_name}.beacon_query_raw_key.png")
                if beacon_to_raw_records:
                    plot_chunk2_heatmap_grid(
                        records=beacon_to_raw_records,
                        out_png=out_beacon_to_raw_png,
                        title=(
                            f"Chunk-{int(args.beacon_probe_chunk_idx)} Beacon-Query to Raw-Key Heatmap "
                            f"({task}, head-mean, ratio=8)"
                        ),
                        heatmap_vmax=args.heatmap_vmax,
                    )

                raw_to_raw_records = build_query_key_subset_records(
                    records,
                    query_positions_key="raw_query_positions",
                    key_positions_key="raw_key_positions",
                )
                out_raw_to_raw_png = os.path.join(args.figure_dir, f"{base_name}.raw_query_raw_key.png")
                if raw_to_raw_records:
                    plot_chunk2_heatmap_grid(
                        records=raw_to_raw_records,
                        out_png=out_raw_to_raw_png,
                        title=(
                            f"Chunk-{int(args.beacon_probe_chunk_idx)} Raw-Query to Raw-Key Heatmap "
                            f"({task}, head-mean, ratio=8)"
                        ),
                        heatmap_vmax=args.heatmap_vmax,
                    )

                json_path = os.path.join(result_dir, f"{base_name}.json")
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(records, f, ensure_ascii=False)

                print(f"[heatmap] saved {out_png}", flush=True)
                if beacon_query_records:
                    print(f"[heatmap] saved {out_beacon_q_png}", flush=True)
                if raw_query_records:
                    print(f"[heatmap] saved {out_raw_q_png}", flush=True)
                if beacon_to_beacon_records:
                    print(f"[heatmap] saved {out_beacon_to_beacon_png}", flush=True)
                if raw_to_beacon_records:
                    print(f"[heatmap] saved {out_raw_to_beacon_png}", flush=True)
                if beacon_to_raw_records:
                    print(f"[heatmap] saved {out_beacon_to_raw_png}", flush=True)
                if raw_to_raw_records:
                    print(f"[heatmap] saved {out_raw_to_raw_png}", flush=True)
                print(f"[heatmap] saved {json_path}", flush=True)
                found_heatmap = True

            if found_heatmap and args.stop_after_first_heatmap:
                break

        if found_heatmap and args.stop_after_first_heatmap:
            break

    accelerator.wait_for_everyone()
    if accelerator.process_index == 0:
        args.save(os.path.join(result_dir, "config_probe_chunk_heatmap.json"))
        if not found_heatmap:
            print(f"[heatmap] no chunk-{int(args.beacon_probe_chunk_idx)} heatmap was captured.", flush=True)


if __name__ == "__main__":
    main()
