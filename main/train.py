import logging
import time
import torch
from transformers import HfArgumentParser, TrainerCallback
from transformers.integrations import is_deepspeed_zero3_enabled
from slimkv import (
    Data,
    DefaultDataCollator,
    ModelArgs,
    FileLogger,
    get_model_and_tokenizer,
    makedirs,
    format_numel_str
)
from slimkv.args import TrainingArgs
from slimkv.metrics import Metric
from slimkv.trainer import SlimKVTrainer


class EfficiencyCallback(TrainerCallback):
    """Log tokens/sec and peak GPU memory at each logging step."""

    def on_train_begin(self, args, state, control, **kwargs):
        self._step_tokens = 0
        self._step_start = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_step_end(self, args, state, control, model=None, **kwargs):
        # Accumulate tokens processed this step (batch_size * seq_len * gradient_accum * n_gpu)
        # HF Trainer tracks state.num_input_tokens_seen since v4.36
        pass

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or state.global_step == 0:
            return
        elapsed = time.time() - self._step_start
        # num_input_tokens_seen is tracked by HF Trainer (since v4.36)
        if hasattr(state, "num_input_tokens_seen") and state.num_input_tokens_seen > 0:
            tokens_since = state.num_input_tokens_seen - self._step_tokens
            if elapsed > 0:
                logs["tokens_per_sec"] = round(tokens_since / elapsed, 1)
            self._step_tokens = state.num_input_tokens_seen
        if torch.cuda.is_available():
            logs["peak_gpu_memory_mb"] = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1)
        self._step_start = time.time()

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser([ModelArgs, TrainingArgs])
    model_args, training_args = parser.parse_args_into_dataclasses()

    model, tokenizer = get_model_and_tokenizer(model_args, device="cuda", evaluation_mode=False)

    if model_args.enable_beacon and training_args.only_train_beacon:
        for name, param in model.named_parameters():
            if "beacon" not in name:
                param.requires_grad_(False)

    if training_args.lora_tune:
        from peft import (
            LoraConfig,
            get_peft_model,
        )
        # copied from LongLoRA
        config = LoraConfig(
            r=training_args.lora_rank,
            lora_alpha=training_args.lora_alpha,
            target_modules=training_args.lora_targets,
            modules_to_save=training_args.lora_extra_params,
            lora_dropout=training_args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)

    beacon_total_params = sum(p.numel() for name, p in model.named_parameters() if "beacon" in name)
    beacon_trainable_params = sum(p.numel() for name, p in model.named_parameters() if "beacon" in name and p.requires_grad)
    logger.info(f"Beacon Added Params (total): {format_numel_str(beacon_total_params)}")
    logger.info(f"Beacon Added Params (trainable): {format_numel_str(beacon_trainable_params)}")
    logger.info(f"Trainable Model params: {format_numel_str(sum(p.numel() for p in model.parameters() if p.requires_grad))}")

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

    with training_args.main_process_first():
        if is_deepspeed_zero3_enabled() and training_args.eval_method != "perplexity":
            logger.warning(f"In deepspeed zero3, evaluation with generation is may lead to hang because of the unequal number of forward passes across different devices.")
        eval_dataset = Data.prepare_eval_data(
            model_args.eval_data, 
            tokenizer=tokenizer,
            max_length=training_args.eval_max_length,
            min_length=training_args.eval_min_length,
            chat_template=model_args.chat_template,
            seed=training_args.seed,
            cache_dir=model_args.dataset_cache_dir,
        )

    trainer = SlimKVTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        model_args=model_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DefaultDataCollator(tokenizer),
        file_logger=FileLogger(makedirs(training_args.log_path)),
        compute_metrics=Metric.get_metric_fn(
            metrics=training_args.metrics,
            save_path=Metric.get_save_path(
                model_args.eval_data,
                training_args.output_dir
            ) if model_args.eval_data is not None else None
        ),
        callbacks=[EfficiencyCallback()],
    )
    if train_dataset is not None:
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    elif eval_dataset is not None:
        trainer.evaluate()

if __name__ == "__main__":
    main()
