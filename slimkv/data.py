"""SlimKV training data and sampler utilities.

This module vendors the minimal training-time components that previously
depended on external lora_beacon/activation_beacon repositories.
"""

import os
import re
import math
import random
from functools import partial
from typing import List, Optional

import datasets
import torch
from torch.utils.data import Sampler, Dataset
from transformers.tokenization_utils_base import BatchEncoding
from transformers.trainer import is_datasets_available
from transformers.utils import logging

from .utils import apply_chat_template

logger = logging.get_logger(__name__)


def _add_eos(encoded, eos_token_id: int):
    """Append EOS for list-backed tokenizer outputs if it is missing."""
    if not isinstance(encoded.get("input_ids"), list):
        return encoded
    if len(encoded["input_ids"]) and encoded["input_ids"][-1] == eos_token_id:
        return encoded

    for key, values in encoded.items():
        if not isinstance(values, list):
            continue
        if key in {"input_ids", "labels"}:
            encoded[key] = values + [eos_token_id]
        elif key == "attention_mask":
            encoded[key] = values + [1]
        elif key == "position_ids":
            encoded[key] = values + [values[-1] + 1]
        elif key == "token_type_ids":
            encoded[key] = values + values[-1:]
    return encoded


class Data:
    @staticmethod
    def _process_pretrain_data(data, indices):
        outputs = {"labels": [], "index": [], "length": []}
        for input_ids, index in zip(data["input_ids"], indices):
            outputs["index"].append(index)
            outputs["length"].append(len(input_ids))
            # labels are generated on-the-fly in Trainer.compute_loss
            outputs["labels"].append(None)
        return outputs

    @staticmethod
    def _process_language_modeling(data, indices, tokenizer, min_length, max_length):
        outputs = {"input_ids": [], "labels": [], "length": [], "index": []}

        for i, text in enumerate(data["text"]):
            # Truncate at tokenization time to avoid processing extremely long
            # documents (and suppress oversized-sequence warnings).
            encoded = tokenizer(text, truncation=True, max_length=max_length)
            if len(encoded["input_ids"]) < min_length:
                continue
            if len(encoded["input_ids"]) < max_length:
                encoded = _add_eos(encoded, tokenizer.eos_token_id)

            encoded["labels"] = None

            for key, values in encoded.items():
                if key in outputs:
                    outputs[key].append(values)
            outputs["length"].append(len(encoded["input_ids"]))
            outputs["index"].append(indices[i])

        return outputs

    @staticmethod
    def _process_instruction_tuning(
        data,
        indices,
        tokenizer,
        chat_template,
        min_length,
        max_length,
        eval_mode=False,
    ):
        outputs = {"input_ids": [], "labels": [], "length": [], "index": []}

        for i, source in enumerate(data["conversations"]):
            if source and source[0]["role"] != "user":
                source = source[1:]

            if eval_mode:
                labels = source[1]["content"] if len(source) > 1 else None
                source = source[:1]
            else:
                labels = None

            encoded = apply_chat_template(
                chat_template,
                source,
                tokenizer=tokenizer,
                add_generation_prompt=eval_mode,
                return_labels=not eval_mode,
            ).encoded

            if min_length is not None and len(encoded["input_ids"]) < min_length:
                continue
            if max_length is not None and len(encoded["input_ids"]) > max_length:
                continue

            if eval_mode:
                encoded["labels"] = labels

            for key, values in encoded.items():
                if key in outputs:
                    outputs[key].append(values)
            outputs["length"].append(len(encoded["input_ids"]))
            outputs["index"].append(indices[i])

        return outputs

    @staticmethod
    def prepare_train_data(
        data_files=None,
        tokenizer=None,
        max_length=4096,
        min_length=512,
        chat_template="hf",
        seed=42,
        cache_dir=None,
        load_from_cache_file=None,
        ignore_index=False,
        ignore_length=False,
    ):
        if data_files is None:
            return None

        if isinstance(data_files, str):
            data_files = [data_files]
        elif not isinstance(data_files, list):
            raise ValueError(f"Invalid training data: {data_files}")

        data_to_num_sample = {}
        for data_file in data_files:
            match = re.search(r"\[(\d*)\]", data_file)
            if match:
                max_sample_num = int(match.group(1))
                data_file = re.sub(r"\[(\d*)\]", "", data_file)
            else:
                max_sample_num = None
            data_to_num_sample[data_file] = max_sample_num

        random.seed(seed)
        train_datasets = []
        for data_file, max_sample_num in data_to_num_sample.items():
            if os.path.isdir(data_file) and os.path.exists(os.path.join(data_file, "dataset_info.json")):
                dataset = datasets.load_from_disk(data_file)
                dataset = dataset.map(
                    Data._process_pretrain_data,
                    batched=True,
                    num_proc=32,
                    batch_size=32,
                    with_indices=True,
                )
            else:
                dataset = datasets.load_dataset("json", data_files=data_file, split="train", cache_dir=cache_dir)
                columns = dataset.column_names
                if "text" in columns:
                    process_fn = partial(
                        Data._process_language_modeling,
                        tokenizer=tokenizer,
                        min_length=min_length,
                        max_length=max_length,
                    )
                elif "conversations" in columns:
                    process_fn = partial(
                        Data._process_instruction_tuning,
                        tokenizer=tokenizer,
                        chat_template=chat_template,
                        min_length=min_length,
                        max_length=max_length,
                    )
                else:
                    raise ValueError("Found neither 'text' nor 'conversations' in training data.")

                dataset = dataset.map(
                    process_fn,
                    batched=True,
                    num_proc=32,
                    batch_size=32,
                    with_indices=True,
                    remove_columns=dataset.column_names,
                    load_from_cache_file=load_from_cache_file,
                )

            if max_sample_num is not None and len(dataset) > max_sample_num:
                dataset = dataset.train_test_split(max_sample_num, seed=seed)["test"]

            if "index" in dataset.column_names and ignore_index:
                dataset = dataset.remove_columns(["index"])
            if "length" in dataset.column_names and ignore_length:
                dataset = dataset.remove_columns(["length"])

            train_datasets.append(dataset)

        return datasets.concatenate_datasets(train_datasets)

    @staticmethod
    def prepare_eval_data(
        data_files=None,
        tokenizer=None,
        max_length=4096,
        min_length=512,
        chat_template="hf",
        max_eval_num=None,
        cache_dir=None,
        seed=42,
        load_from_cache_file=None,
        ignore_index=False,
        ignore_length=False,
    ):
        if data_files is None:
            return None

        random.seed(seed)
        if max_eval_num is not None:
            dataset = datasets.load_dataset("json", data_files=data_files, split=f"train[:{max_eval_num}]", cache_dir=cache_dir)
        else:
            dataset = datasets.load_dataset("json", data_files=data_files, split="train", cache_dir=cache_dir)

        columns = dataset.column_names
        if "text" in columns:
            process_fn = partial(
                Data._process_language_modeling,
                tokenizer=tokenizer,
                min_length=min_length,
                max_length=max_length,
            )
        elif "conversations" in columns:
            process_fn = partial(
                Data._process_instruction_tuning,
                tokenizer=tokenizer,
                chat_template=chat_template,
                min_length=min_length,
                max_length=max_length,
                eval_mode=True,
            )
        else:
            raise ValueError("Found neither 'text' nor 'conversations' in eval data.")

        dataset = dataset.map(
            process_fn,
            batched=True,
            num_proc=32,
            with_indices=True,
            remove_columns=dataset.column_names,
            load_from_cache_file=load_from_cache_file,
        )
        if "index" in dataset.column_names and ignore_index:
            dataset = dataset.remove_columns(["index"])
        if "length" in dataset.column_names and ignore_length:
            dataset = dataset.remove_columns(["length"])
        return dataset


class StrideGroupedSampler(Sampler):
    """Group samples with similar stride counts to stabilize memory usage."""

    def __init__(
        self,
        batch_size: int,
        window: int,
        stride: int,
        group: str,
        sort: Optional[str] = None,
        dataset: Optional[Dataset] = None,
        lengths: Optional[List[int]] = None,
        model_input_name: Optional[str] = None,
    ):
        if dataset is None and lengths is None:
            raise ValueError("One of dataset and lengths must be provided.")
        if group is None:
            raise ValueError("group cannot be None.")

        if lengths is None:
            model_input_name = model_input_name or "input_ids"
            if (
                not (isinstance(dataset[0], dict) or isinstance(dataset[0], BatchEncoding))
                or model_input_name not in dataset[0]
            ):
                raise ValueError(
                    f"Cannot infer lengths automatically: missing '{model_input_name}' in dataset items."
                )
            lengths = [len(feature[model_input_name]) for feature in dataset]
        elif isinstance(lengths, torch.Tensor):
            lengths = lengths.tolist()

        indices = list(range(len(lengths)))
        num_strides = [math.ceil((length - window) / stride) + 1 for length in lengths]
        index_stride_pairs = list(zip(indices, num_strides))
        random.shuffle(index_stride_pairs)
        index_stride_pairs = sorted(index_stride_pairs, key=lambda x: x[1])

        batches = []
        batch = []
        prev_stride = None
        for index, num_stride in index_stride_pairs:
            if num_stride != prev_stride:
                if group == "strict":
                    batch.clear()
                elif group != "relaxed":
                    raise ValueError(f"Unknown group mode: {group} (expected strict|relaxed)")

            batch.append(index)
            prev_stride = num_stride
            if len(batch) == batch_size:
                batches.append((batch.copy(), num_stride))
                batch.clear()

        if batch and group == "relaxed":
            batches.append((batch.copy(), prev_stride if prev_stride is not None else 0))

        if sort is None:
            random.shuffle(batches)
        elif sort == "ascend":
            batches = sorted(batches, key=lambda x: x[1])
        elif sort == "descend":
            batches = sorted(batches, key=lambda x: x[1], reverse=True)
        else:
            raise ValueError(f"Unknown sort mode: {sort} (expected None|ascend|descend)")

        self.indices = sum((x[0] for x in batches), [])

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        return iter(self.indices)
