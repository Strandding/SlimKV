"""SlimKV – Training entry point."""

import logging
import os
import inspect
import torch
from transformers import HfArgumentParser, Trainer

from slimkv import (
    DefaultDataCollator,
    Data,
    ModelArgs,
    StrideGroupedSampler,
    get_model_and_tokenizer,
    format_numel_str,
)
from slimkv.args import TrainingArgs

logger = logging.getLogger(__name__)


class SlimKVTrainer(Trainer):
    """Trainer that resets Memory before each forward and produces labels on the fly."""

    @staticmethod
    def _call_super_get_train_sampler(super_obj, dataset):
        """Compatibility wrapper for old/new transformers sampler signatures."""
        try:
            return super_obj._get_train_sampler(dataset)
        except TypeError:
            return super_obj._get_train_sampler()

    @staticmethod
    def _call_super_compute_loss(super_obj, model, inputs, return_outputs, num_items_in_batch):
        """Compatibility wrapper for old/new transformers compute_loss signatures."""
        try:
            return super_obj.compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )
        except TypeError:
            return super_obj.compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
            )

    def _get_train_sampler(self, train_dataset=None):
        """Group samples by stride count so all ranks in a step iterate the same
        number of chunks, matching activation_beacon's behaviour."""
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        if self.args.group_by_stride is not None:
            try:
                ds_len = len(dataset)
            except Exception:
                ds_len = "unknown"
            logger.info(
                f"rank={self.args.process_index} building StrideGroupedSampler "
                f"(group={self.args.group_by_stride}, dataset_size={ds_len})..."
            )
            lengths = dataset["length"] if "length" in dataset.column_names else None
            sampler = StrideGroupedSampler(
                batch_size=self.args.train_batch_size * self.args.world_size,
                stride=self.model.memory.stride,
                group=self.args.group_by_stride,
                sort=self.args.sort_by_stride,
                dataset=dataset,
                lengths=lengths,
            )
            logger.info(f"rank={self.args.process_index} StrideGroupedSampler ready.")
            return sampler
        return self._call_super_get_train_sampler(super(), dataset)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs.pop("length", None)
        inputs.pop("index", None)

        if inputs["labels"][0] is None:
            inputs["labels"] = inputs["input_ids"].clone()

        if hasattr(model, "memory"):
            model.memory.reset()

        return self._call_super_compute_loss(
            super(),
            model=model,
            inputs=inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )


def main():
    parser = HfArgumentParser([ModelArgs, TrainingArgs])
    model_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.per_device_train_batch_size != 1:
        raise ValueError(
            "SlimKV currently assumes per_device_train_batch_size=1 "
            "to avoid unnecessary padding logic."
        )

    # torchrun starts one process per GPU. Explicitly bind each process to its
    # local GPU to avoid all ranks loading the model on cuda:0.
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", training_args.local_rank))
        if local_rank >= 0:
            torch.cuda.set_device(local_rank)
            load_device = f"cuda:{local_rank}"
        else:
            load_device = "cuda"
    else:
        load_device = "cpu"
        local_rank = -1

    logger.info(f"rank={training_args.process_index} local_rank={local_rank} load_device={load_device}")
    model, tokenizer = get_model_and_tokenizer(model_args, device=load_device, evaluation_mode=False)

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

    logger.info(f"rank={training_args.process_index} preparing train dataset...")
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
    logger.info(f"rank={training_args.process_index} train dataset ready, size={len(train_dataset)}")

    trainer_processing_kwargs = {}
    trainer_init_params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_init_params:
        trainer_processing_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_init_params:
        trainer_processing_kwargs["tokenizer"] = tokenizer

    trainer = SlimKVTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=DefaultDataCollator(tokenizer),
        **trainer_processing_kwargs,
    )

    logger.info(f"rank={training_args.process_index} starting trainer.train()")
    trainer.train()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()
