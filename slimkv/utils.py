"""SlimKV – Standalone utilities (no external dependency on lora_beacon)."""

import os
import json
import time
import torch
import dataclasses
from typing import List, Dict, Any, Optional
from transformers.tokenization_utils import PreTrainedTokenizer


# ---------------------------------------------------------------------------
# makedirs / format_numel_str
# ---------------------------------------------------------------------------

def makedirs(path):
    """Create parent directories for *path* and return *path* unchanged."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    return path


def format_numel_str(n):
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.2f}K"
    return str(n)


# ---------------------------------------------------------------------------
# FileLogger  (writes JSON lines with timestamp)
# ---------------------------------------------------------------------------

class FileLogger:
    def __init__(self, path):
        self.path = path

    def log(self, metrics, **kwargs):
        entry = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "metrics": metrics}
        entry.update(kwargs)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# DefaultDataCollator  (micro-batch size = 1, no dynamic padding)
# ---------------------------------------------------------------------------

class DefaultDataCollator:
    """Collate a single sample into batched tensors.

    SlimKV training is configured with micro-batch size 1, so dynamic padding
    is unnecessary here.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        assert len(batch) == 1, (
            "DefaultDataCollator expects micro-batch size 1. "
            "Please set per_device_train_batch_size=1 (and eval batch_size=1 if reused)."
        )

        sample = batch[0]
        result = {}

        for key, value in sample.items():
            if isinstance(value, list):
                result[key] = torch.tensor(value, dtype=torch.long).unsqueeze(0)
            elif torch.is_tensor(value):
                result[key] = value.unsqueeze(0)
            elif isinstance(value, (int, float)):
                result[key] = torch.tensor([value])
            else:
                result[key] = [value]

        return result


# ---------------------------------------------------------------------------
# apply_chat_template  (lightweight, supports "hf" and "no")
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ChatTemplateOutput:
    raw: str = None
    encoded: Any = None


def apply_chat_template(
    template: str,
    messages: List[Dict[str, str]],
    system_message: Optional[str] = None,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    add_generation_prompt: bool = False,
    return_labels: bool = False,
    **tokenization_kwargs,
) -> ChatTemplateOutput:
    """Apply a chat template to messages and return tokenized output.

    Supported templates:
      - "hf": uses the tokenizer's built-in apply_chat_template
      - "no": plain concatenation (no special formatting)
      - Any other string supported by the tokenizer's Jinja template
    """
    if template == "no":
        assert tokenizer is not None
        conversation = ""
        assistant_segments = []
        for message in messages:
            content = message["content"]
            if message.get("role") == "assistant":
                assistant_segments.append(content)
                conversation += " " + content + (tokenizer.eos_token or "")
            else:
                conversation += content
        encoded = tokenizer(conversation, **tokenization_kwargs)
        if return_labels:
            labels = encoded["input_ids"].copy()
            if assistant_segments:
                assistant_text = " ".join(assistant_segments) + (tokenizer.eos_token or "")
                assistant_len = len(tokenizer.encode(assistant_text.lstrip(), add_special_tokens=False))
                labels[:-assistant_len] = [-100 for _ in labels[:-assistant_len]]
            else:
                labels = [-100 for _ in labels]
            encoded["labels"] = labels
        return ChatTemplateOutput(raw=conversation, encoded=encoded)

    # Default: use HF tokenizer's built-in chat template
    assert tokenizer is not None
    tokenization_kwargs["return_dict"] = True

    raw = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=add_generation_prompt,
        tokenize=False,
    )
    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=add_generation_prompt,
        **tokenization_kwargs,
    )
    # Some tokenizers wrap in a list
    if isinstance(encoded["input_ids"][0], list):
        for k, v in encoded.items():
            encoded[k] = v[0]

    if return_labels:
        # Build loss mask over assistant message spans. This is template-agnostic and
        # works for HF chat templates by measuring token span growth per assistant turn.
        labels = [-100 for _ in encoded["input_ids"]]

        def _to_flat_ids(x):
            if isinstance(x, dict):
                x = x["input_ids"]
            if len(x) and isinstance(x[0], list):
                return x[0]
            return x

        for i, msg in enumerate(messages):
            if msg.get("role") != "assistant":
                continue
            prefix_ids = tokenizer.apply_chat_template(
                messages[:i],
                add_generation_prompt=True,
                tokenize=True,
            )
            upto_ids = tokenizer.apply_chat_template(
                messages[: i + 1],
                add_generation_prompt=False,
                tokenize=True,
            )
            prefix_ids = _to_flat_ids(prefix_ids)
            upto_ids = _to_flat_ids(upto_ids)
            start = min(len(prefix_ids), len(labels))
            end = min(len(upto_ids), len(labels))
            for j in range(start, end):
                labels[j] = encoded["input_ids"][j]

        encoded["labels"] = labels

    return ChatTemplateOutput(raw=raw, encoded=encoded)
