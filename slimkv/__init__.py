"""SlimKV – Load model and apply monkey-patch for anchor-token KV compression."""

import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .args import ModelArgs
from .data import Data, StrideGroupedSampler
from .memory import Memory
from .patch import patch_model

# Standalone utilities (no external dependency)
from .utils import (
    DefaultDataCollator,
    FileLogger,
    makedirs,
    format_numel_str,
    apply_chat_template,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)

def get_model_and_tokenizer(model_args: ModelArgs, device="cpu", evaluation_mode=True):
    """Load a pretrained model and apply SlimKV patches."""
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(model_args.dtype, torch.float32)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.model_cache_dir,
        padding_side=model_args.padding_side,
        token=model_args.access_token,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    attn_kwargs = {}
    if model_args.attn_impl is not None:
        attn_kwargs["attn_implementation"] = model_args.attn_impl

    device_map = model_args.device_map
    if device_map is None:
        device_map = {"": device}

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.model_cache_dir,
        torch_dtype=dtype,
        device_map=device_map,
        token=model_args.access_token,
        trust_remote_code=True,
        **attn_kwargs,
    )

    # --- Add anchor token ---
    anchor_token_id = len(tokenizer)
    tokenizer.add_tokens(["<anchor>"])
    model.resize_token_embeddings(len(tokenizer))
    # Initialise anchor embedding from EOS
    with torch.no_grad():
        model.model.embed_tokens.weight.data[anchor_token_id] = (
            model.model.embed_tokens.weight.data[tokenizer.eos_token_id].clone()
        )

    # --- Create Memory ---
    config = model.config
    memory = Memory(
        config=config,
        anchor_token_id=anchor_token_id,
        num_layers=config.num_hidden_layers,
    )

    # --- Apply monkey-patch ---
    patch_model(
        model, memory, anchor_token_id,
        anchor_kv_type=model_args.anchor_kv_type,
        latent_dim=model_args.latent_dim,
        skip_anchor_rope_k=model_args.skip_anchor_rope_k,
        shared_kv_down=model_args.shared_kv_down,
    )

    if evaluation_mode:
        model.eval()

    return model, tokenizer
