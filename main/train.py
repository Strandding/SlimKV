"""SlimKV – Training entry point."""

import logging
import time
import torch
from transformers import HfArgumentParser, Trainer, TrainerCallback

from slimkv import (
    Data,
    DefaultDataCollator,
    ModelArgs,
    FileLogger,
    get_model_and_tokenizer,
    makedirs,
    format_numel_str,
)
from slimkv.args import TrainingArgs

logger = logging.getLogger(__name__)


class SlimKVTrainer(Trainer):
    """Trainer that resets Memory before each forward and produces labels on the fly."""

    def __init__(self, *args, model_args=None, file_logger=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_args = model_args
        self.file_logger = file_logger

    def _get_train_sampler(self):
        """Group samples by stride count so all ranks in a step iterate the same
        number of chunks, matching activation_beacon's behaviour."""
        from src.trainer import StrideGroupedSampler
        if self.args.group_by_stride is not None:
            lengths = self.train_dataset["length"] if "length" in self.train_dataset.column_names else None
            return StrideGroupedSampler(
                batch_size=self.args.train_batch_size * self.args.world_size,
                window=self.model.memory.window,
                stride=self.model.memory.stride,
                group=self.args.group_by_stride,
                dataset=self.train_dataset,
                lengths=lengths,
            )
        return super()._get_train_sampler()

    def compute_loss(self, model, inputs, return_outputs=False):
        inputs.pop("length", None)
        inputs.pop("index", None)

        if inputs["labels"][0] is None:
            inputs["labels"] = inputs["input_ids"].clone()

        if hasattr(model, "memory"):
            model.memory.reset()

        return super().compute_loss(model, inputs, return_outputs)


def main():
    parser = HfArgumentParser([ModelArgs, TrainingArgs])
    model_args, training_args = parser.parse_args_into_dataclasses()

    model, tokenizer = get_model_and_tokenizer(model_args, device="cuda", evaluation_mode=False)

    # Freeze all non-anchor parameters
    if training_args.only_train_anchor:
        for name, param in model.named_parameters():
            if "anchor" not in name:
                param.requires_grad_(False)

    anchor_total = sum(p.numel() for n, p in model.named_parameters() if "anchor" in n)
    anchor_trainable = sum(p.numel() for n, p in model.named_parameters() if "anchor" in n and p.requires_grad)
    trainable_all = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Anchor params (total): {format_numel_str(anchor_total)}")
    logger.info(f"Anchor params (trainable): {format_numel_str(anchor_trainable)}")
    logger.info(f"All trainable params: {format_numel_str(trainable_all)}")

    with training_args.main_process_first():
        train_dataset = Data.prepare_train_data(
            model_args.train_data,
            tokenizer=tokenizer,
            max_length=model_args.max_length,
            min_length=training_args.min_length,
            chat_template=model_args.chat_template,
            seed=training_args.seed,
            cache_dir=model_args.dataset_cache_dir,
        )

    log_path = training_args.log_path or training_args.output_dir
    trainer = SlimKVTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        model_args=model_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=DefaultDataCollator(tokenizer),
        file_logger=FileLogger(makedirs(log_path)),
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()
