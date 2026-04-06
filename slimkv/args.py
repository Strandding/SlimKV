"""SlimKV – Command-line arguments."""

import os
import json
from dataclasses import dataclass, field, asdict
from transformers.training_args import TrainingArguments
from typing import Optional, List


@dataclass
class ModelArgs:
    model_name_or_path: str = field(
        default="/dataset/common/tzh/.model/Qwen2-7B-Instruct",
    )
    model_cache_dir: str = field(default=None)
    dataset_cache_dir: str = field(default=None)
    data_root: str = field(default="/dataset/common/tzh/.dataset/long-llm")
    train_data: Optional[List[str]] = field(default=None)
    eval_data: Optional[str] = field(default=None)

    padding_side: str = field(default="left")
    access_token: Optional[str] = field(default=None)
    attn_impl: Optional[str] = field(default="flash_attention_2")

    max_length: int = field(default=4096)
    chat_template: str = field(default="hf")

    # SlimKV specific
    anchor_kv_type: str = field(default="full", metadata={"help": "Anchor KV projection type: 'full' or 'lowrank'."})
    latent_dim: int = field(default=64, metadata={"help": "Low-rank latent dimension for anchor K/V (only used when anchor_kv_type='lowrank')."})
    skip_anchor_rope_k: bool = field(default=False, metadata={"help": "Skip RoPE for anchor token keys."})
    shared_kv_down: bool = field(default=False, metadata={"help": "Share the down projection between anchor K and V (only used when anchor_kv_type='lowrank')."})

    dtype: str = field(default="bf16")
    device_map: Optional[str] = field(default=None)
    batch_size: int = field(default=1)

    def save(self, path):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)


@dataclass
class TrainingArgs(TrainingArguments):
    min_length: int = field(default=0)
    only_train_anchor: bool = field(
        default=True,
        metadata={"help": "Freeze all parameters except anchor-related ones."},
    )
    group_by_stride: Optional[str] = field(default=None)
    sort_by_stride: Optional[str] = field(default=None)
    length_column_name: str = field(default="length")
