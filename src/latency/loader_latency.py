import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

import torch
from transformers import AutoTokenizer

from src import ModelArgs, get_model_and_tokenizer
from src.llama3.configuration_llama import LlamaConfig
from src.latency.modeling_llama_latency import LlamaForCausalLM as HybridLatencyLlamaForCausalLM


MODE_RAW = "raw"
MODE_BEACON_DENSE = "beacon_dense"
MODE_BEACON_LOWRANK_RECON = "beacon_lowrank_reconstruct"
MODE_BEACON_LOWRANK_HYBRID = "beacon_lowrank_hybrid_latent"


@dataclass
class CheckpointLayout:
    has_beacon: bool
    has_dense_k: bool
    has_dense_v: bool
    has_lowrank_k: bool
    has_lowrank_v: bool
    has_shared_down: bool


def resolve_torch_dtype(dtype: str) -> torch.dtype:
    d = dtype.lower()
    if d == "bf16":
        return torch.bfloat16
    if d == "fp16":
        return torch.float16
    if d == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def _checkpoint_dir(model_or_ckpt: str) -> str:
    if os.path.isdir(model_or_ckpt):
        return model_or_ckpt
    raise ValueError(f"Expected checkpoint directory, found: {model_or_ckpt}")


def _load_weight_map_from_index(ckpt_dir: str) -> Optional[Dict[str, str]]:
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        idx_path = os.path.join(ckpt_dir, name)
        if os.path.exists(idx_path):
            with open(idx_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            weight_map = data.get("weight_map", None)
            if isinstance(weight_map, dict):
                return weight_map
    return None


def collect_checkpoint_keys(model_or_ckpt: str) -> Set[str]:
    ckpt_dir = _checkpoint_dir(model_or_ckpt)
    weight_map = _load_weight_map_from_index(ckpt_dir)
    if weight_map is not None:
        return set(weight_map.keys())

    keys: Set[str] = set()

    st_files = [
        os.path.join(ckpt_dir, x)
        for x in os.listdir(ckpt_dir)
        if x.endswith(".safetensors")
    ]
    if st_files:
        from safetensors import safe_open

        for fp in sorted(st_files):
            with safe_open(fp, framework="pt", device="cpu") as f:
                for k in f.keys():
                    keys.add(k)
        return keys

    bin_path = os.path.join(ckpt_dir, "pytorch_model.bin")
    if os.path.exists(bin_path):
        sd = torch.load(bin_path, map_location="cpu")
        return set(sd.keys())

    raise FileNotFoundError(
        f"No model weight files found under {ckpt_dir}. Expected safetensors or pytorch_model.bin."
    )


def inspect_checkpoint_layout(keys: Set[str]) -> CheckpointLayout:
    has_dense_k = any(".beacon_k_proj.weight" in k for k in keys)
    has_dense_v = any(".beacon_v_proj.weight" in k for k in keys)

    has_k_up = any(".beacon_k_up_proj.weight" in k for k in keys)
    has_v_up = any(".beacon_v_up_proj.weight" in k for k in keys)
    has_k_down = any(".beacon_k_down_proj.weight" in k for k in keys)
    has_v_down = any(".beacon_v_down_proj.weight" in k for k in keys)
    has_shared_down = any(".beacon_kv_down_proj.weight" in k for k in keys)

    has_lowrank_k = has_k_up and (has_k_down or has_shared_down)
    has_lowrank_v = has_v_up and (has_v_down or has_shared_down)

    has_beacon = any("beacon_" in k for k in keys)
    return CheckpointLayout(
        has_beacon=has_beacon,
        has_dense_k=has_dense_k,
        has_dense_v=has_dense_v,
        has_lowrank_k=has_lowrank_k,
        has_lowrank_v=has_lowrank_v,
        has_shared_down=has_shared_down,
    )


def validate_mode_contract(mode: str, layout: CheckpointLayout) -> None:
    if mode == MODE_RAW:
        return

    if mode == MODE_BEACON_DENSE:
        if not (layout.has_dense_k and layout.has_dense_v):
            raise ValueError(
                "`beacon_dense` requires dense beacon K/V weights (beacon_k_proj/beacon_v_proj). "
                "Input checkpoint does not match dense contract."
            )
        return

    if mode in (MODE_BEACON_LOWRANK_RECON, MODE_BEACON_LOWRANK_HYBRID):
        if not (layout.has_lowrank_k and layout.has_lowrank_v):
            raise ValueError(
                f"`{mode}` requires low-rank beacon K/V weights (down/up projections). "
                "Input checkpoint does not match low-rank contract."
            )
        return

    raise ValueError(f"Unknown mode: {mode}")


def _make_model_args(
    model_or_ckpt: str,
    dtype: str,
    attn_impl: str,
    enable_beacon: bool,
    device_map: Optional[str],
) -> ModelArgs:
    args = ModelArgs(
        model_name_or_path=model_or_ckpt,
        dtype=dtype,
        attn_impl=attn_impl,
        enable_beacon=enable_beacon,
    )
    args.device_map = device_map
    return args


def _load_tokenizer(model_or_ckpt: str):
    tokenizer = AutoTokenizer.from_pretrained(model_or_ckpt, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def _load_raw_model(
    model_or_ckpt: str,
    dtype: str,
    attn_impl: str,
    device: str,
    device_map: Optional[str],
):
    args = _make_model_args(
        model_or_ckpt=model_or_ckpt,
        dtype=dtype,
        attn_impl=attn_impl,
        enable_beacon=False,
        device_map=device_map,
    )
    model, tokenizer = get_model_and_tokenizer(
        args,
        device=device,
        evaluation_mode=True,
        beacon_headwise_up_proj=False,
        beacon_up_proj_group_size=1,
    )
    return model, tokenizer


def _load_beacon_dense_model(
    model_or_ckpt: str,
    dtype: str,
    attn_impl: str,
    device: str,
    device_map: Optional[str],
    beacon_ratio: int,
):
    args = _make_model_args(
        model_or_ckpt=model_or_ckpt,
        dtype=dtype,
        attn_impl=attn_impl,
        enable_beacon=True,
        device_map=device_map,
    )
    model, tokenizer = get_model_and_tokenizer(
        args,
        device=device,
        evaluation_mode=True,
        beacon_headwise_up_proj=False,
        beacon_up_proj_group_size=1,
        beacon_ratio=[beacon_ratio],
        beacon_ratio_mix="sequence",
        beacon_k_compression="none",
        beacon_v_compression="none",
    )
    return model, tokenizer


def _load_beacon_lowrank_reconstruct_model(
    model_or_ckpt: str,
    dtype: str,
    attn_impl: str,
    device: str,
    device_map: Optional[str],
    beacon_ratio: int,
):
    args = _make_model_args(
        model_or_ckpt=model_or_ckpt,
        dtype=dtype,
        attn_impl=attn_impl,
        enable_beacon=True,
        device_map=device_map,
    )
    model, tokenizer = get_model_and_tokenizer(
        args,
        device=device,
        evaluation_mode=True,
        beacon_headwise_up_proj=False,
        beacon_up_proj_group_size=1,
        beacon_ratio=[beacon_ratio],
        beacon_ratio_mix="sequence",
        beacon_k_compression="lowrank_svd",
        beacon_v_compression="lowrank_svd",
        beacon_skip_rope_for_beacon_k=False,
    )
    return model, tokenizer


def _resolve_latent_dims(
    config: LlamaConfig,
    latent_compress_ratio: int,
    k_latent_head_dim_list: Optional[List[int]],
    v_latent_head_dim_list: Optional[List[int]],
) -> Tuple[List[int], List[int]]:
    num_layers = config.num_hidden_layers
    head_dim = config.head_dim if getattr(config, "head_dim", None) is not None else config.hidden_size // config.num_attention_heads

    if k_latent_head_dim_list is not None:
        if len(k_latent_head_dim_list) != num_layers:
            raise ValueError(
                f"k_latent_head_dim_list length must be {num_layers}, found {len(k_latent_head_dim_list)}"
            )
        k_list = list(k_latent_head_dim_list)
    elif getattr(config, "beacon_k_latent_head_dim_list", None):
        k_list = list(config.beacon_k_latent_head_dim_list)
    else:
        k_dim = getattr(config, "beacon_k_latent_head_dim", None)
        if k_dim is None:
            k_dim = head_dim // latent_compress_ratio
        k_list = [int(k_dim)] * num_layers

    if v_latent_head_dim_list is not None:
        if len(v_latent_head_dim_list) != num_layers:
            raise ValueError(
                f"v_latent_head_dim_list length must be {num_layers}, found {len(v_latent_head_dim_list)}"
            )
        v_list = list(v_latent_head_dim_list)
    elif getattr(config, "beacon_v_latent_head_dim_list", None):
        v_list = list(config.beacon_v_latent_head_dim_list)
    else:
        v_dim = getattr(config, "beacon_v_latent_head_dim", None)
        if v_dim is None:
            v_dim = head_dim // latent_compress_ratio
        v_list = [int(v_dim)] * num_layers

    for idx, d in enumerate(k_list):
        if not isinstance(d, int) or d <= 0 or d > head_dim:
            raise ValueError(f"Invalid K latent dim at layer {idx}: {d}, head_dim={head_dim}")
    for idx, d in enumerate(v_list):
        if not isinstance(d, int) or d <= 0 or d > head_dim:
            raise ValueError(f"Invalid V latent dim at layer {idx}: {d}, head_dim={head_dim}")

    return k_list, v_list


def _load_named_tensors_from_safetensors(
    ckpt_dir: str,
    tensor_names: Iterable[str],
) -> Dict[str, torch.Tensor]:
    names = set(tensor_names)
    tensors: Dict[str, torch.Tensor] = {}

    idx_path = os.path.join(ckpt_dir, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        with open(idx_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})

        by_file: Dict[str, List[str]] = {}
        for name in names:
            shard = weight_map.get(name, None)
            if shard is None:
                continue
            by_file.setdefault(shard, []).append(name)

        from safetensors import safe_open

        for shard, shard_names in by_file.items():
            fp = os.path.join(ckpt_dir, shard)
            with safe_open(fp, framework="pt", device="cpu") as f:
                for n in shard_names:
                    tensors[n] = f.get_tensor(n)
        return tensors

    st_files = [
        os.path.join(ckpt_dir, x)
        for x in os.listdir(ckpt_dir)
        if x.endswith(".safetensors")
    ]
    if not st_files:
        raise ValueError(
            "Hybrid mode tensor remapping currently requires safetensors checkpoints. "
            f"No safetensors file found under {ckpt_dir}."
        )

    from safetensors import safe_open

    remaining = set(names)
    for fp in sorted(st_files):
        if not remaining:
            break
        with safe_open(fp, framework="pt", device="cpu") as f:
            existing = [n for n in remaining if n in f.keys()]
            for n in existing:
                tensors[n] = f.get_tensor(n)
                remaining.remove(n)

    return tensors


def _set_param_by_name(model: torch.nn.Module, name: str, value: torch.Tensor) -> None:
    module = model
    parts = name.split(".")
    for p in parts[:-1]:
        module = module[int(p)] if p.isdigit() else getattr(module, p)
    param_name = parts[-1]
    target = getattr(module, param_name)
    if target.shape != value.shape:
        raise ValueError(f"Shape mismatch for {name}: target={tuple(target.shape)} source={tuple(value.shape)}")
    target.data.copy_(value.to(dtype=target.dtype, device=target.device))


def _reshape_up_weight_for_shared_if_needed(
    weight: torch.Tensor,
    num_kv_heads: int,
    head_dim: int,
    global_latent_dim: int,
    key_name: str,
) -> torch.Tensor:
    if weight.dim() == 2:
        expected_2d = (num_kv_heads * head_dim, global_latent_dim)
        if tuple(weight.shape) != expected_2d:
            raise ValueError(
                f"Unsupported 2D shape for {key_name}: {tuple(weight.shape)}, expected {expected_2d}. "
                "This up-proj cannot be mapped to hybrid shared-latent mode."
            )
        return weight

    if weight.dim() == 3:
        expected = (num_kv_heads, head_dim, global_latent_dim)
        if tuple(weight.shape) != expected:
            raise ValueError(f"Unexpected shape for {key_name}: {tuple(weight.shape)}, expected {expected}")
        return weight.reshape(num_kv_heads * head_dim, global_latent_dim)

    raise ValueError(f"Unsupported tensor rank for {key_name}: dim={weight.dim()}")


def _remap_lowrank_weights_for_hybrid(
    model: torch.nn.Module,
    ckpt_dir: str,
    config: LlamaConfig,
) -> None:
    if getattr(config, "beacon_up_proj_group_size", 1) != 1:
        raise ValueError(
            "Hybrid latent mode currently requires beacon_up_proj_group_size=1 to map K/V up-proj weights."
        )

    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim if getattr(config, "head_dim", None) is not None else config.hidden_size // config.num_attention_heads

    source_names: List[str] = []
    for i in range(num_layers):
        prefix = f"model.layers.{i}.self_attn"
        source_names.extend(
            [
                f"{prefix}.beacon_k_down_proj.weight",
                f"{prefix}.beacon_kv_down_proj.weight",
                f"{prefix}.beacon_k_up_proj.weight",
                f"{prefix}.beacon_v_down_proj.weight",
                f"{prefix}.beacon_v_up_proj.weight",
            ]
        )

    loaded = _load_named_tensors_from_safetensors(ckpt_dir, source_names)

    for i in range(num_layers):
        prefix = f"model.layers.{i}.self_attn"
        k_lat_dim = int(config.beacon_k_latent_head_dim_list[i])
        v_lat_dim = int(config.beacon_v_latent_head_dim_list[i])

        k_down_src = f"{prefix}.beacon_k_down_proj.weight"
        kv_down_src = f"{prefix}.beacon_kv_down_proj.weight"
        v_down_src = f"{prefix}.beacon_v_down_proj.weight"

        k_up_src = f"{prefix}.beacon_k_up_proj.weight"
        v_up_src = f"{prefix}.beacon_v_up_proj.weight"

        k_down_tensor = loaded.get(k_down_src, None)
        shared_down_tensor = loaded.get(kv_down_src, None)
        v_down_tensor = loaded.get(v_down_src, None)

        if k_down_tensor is None:
            if shared_down_tensor is None:
                raise ValueError(f"Missing {k_down_src} (or shared {kv_down_src}) for hybrid remap.")
            k_down_tensor = shared_down_tensor

        if v_down_tensor is None:
            if shared_down_tensor is None:
                raise ValueError(f"Missing {v_down_src} (or shared {kv_down_src}) for hybrid remap.")
            if k_lat_dim != v_lat_dim:
                raise ValueError(
                    f"Layer {i}: shared down-proj exists but K/V latent dims differ ({k_lat_dim} vs {v_lat_dim})."
                )
            v_down_tensor = shared_down_tensor

        k_up_tensor = loaded.get(k_up_src, None)
        if k_up_tensor is None:
            raise ValueError(f"Missing {k_up_src} for hybrid remap.")
        v_up_tensor = loaded.get(v_up_src, None)
        if v_up_tensor is None:
            raise ValueError(f"Missing {v_up_src} for hybrid remap.")

        k_up_tensor = _reshape_up_weight_for_shared_if_needed(
            k_up_tensor,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            global_latent_dim=num_kv_heads * k_lat_dim,
            key_name=k_up_src,
        )
        v_up_tensor = _reshape_up_weight_for_shared_if_needed(
            v_up_tensor,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            global_latent_dim=num_kv_heads * v_lat_dim,
            key_name=v_up_src,
        )

        _set_param_by_name(model, f"{prefix}.beacon_k_latent_down_proj.weight", k_down_tensor)
        _set_param_by_name(model, f"{prefix}.beacon_k_latent_up_proj.weight", k_up_tensor)
        _set_param_by_name(model, f"{prefix}.beacon_v_latent_proj.weight", v_down_tensor)
        _set_param_by_name(model, f"{prefix}.beacon_v_latent_up_proj.weight", v_up_tensor)


def _load_beacon_lowrank_hybrid_model(
    model_or_ckpt: str,
    dtype: str,
    attn_impl: str,
    device_map: Optional[str],
    latent_compress_ratio: int,
    k_latent_head_dim_list: Optional[List[int]],
    v_latent_head_dim_list: Optional[List[int]],
    beacon_ratio: int,
):
    torch_dtype = resolve_torch_dtype(dtype)
    tokenizer = _load_tokenizer(model_or_ckpt)

    config = LlamaConfig.from_pretrained(
        model_or_ckpt,
        torch_dtype=torch_dtype,
        attn_implementation=attn_impl,
        beacon_ratio=[beacon_ratio],
        beacon_ratio_mix="sequence",
        beacon_hybrid_latent_attn=True,
        beacon_headwise_up_proj=False,
        beacon_up_proj_group_size=1,
        beacon_skip_rope_for_beacon_k=True,
    )

    k_list, v_list = _resolve_latent_dims(
        config=config,
        latent_compress_ratio=latent_compress_ratio,
        k_latent_head_dim_list=k_latent_head_dim_list,
        v_latent_head_dim_list=v_latent_head_dim_list,
    )

    config.beacon_hybrid_latent_attn = True
    config.beacon_skip_rope_for_beacon_k = True
    config.beacon_k_latent_head_dim_list = k_list
    config.beacon_v_latent_head_dim_list = v_list

    kwargs = {
        "config": config,
        "torch_dtype": torch_dtype,
    }
    if device_map is not None:
        kwargs["device_map"] = device_map

    model = HybridLatencyLlamaForCausalLM.from_pretrained(model_or_ckpt, **kwargs)
    _remap_lowrank_weights_for_hybrid(model, _checkpoint_dir(model_or_ckpt), config)

    model.eval()
    model.requires_grad_(False)
    return model, tokenizer


def load_model_for_latency_mode(
    mode: str,
    model_or_ckpt: str,
    dtype: str,
    attn_impl: str,
    device: str,
    device_map: Optional[str],
    latent_compress_ratio: int = 2,
    k_latent_head_dim_list: Optional[List[int]] = None,
    v_latent_head_dim_list: Optional[List[int]] = None,
    beacon_ratio: int = 8,
):
    keys = collect_checkpoint_keys(model_or_ckpt)
    layout = inspect_checkpoint_layout(keys)
    validate_mode_contract(mode, layout)

    if mode == MODE_RAW:
        return _load_raw_model(
            model_or_ckpt=model_or_ckpt,
            dtype=dtype,
            attn_impl=attn_impl,
            device=device,
            device_map=device_map,
        )

    if mode == MODE_BEACON_DENSE:
        return _load_beacon_dense_model(
            model_or_ckpt=model_or_ckpt,
            dtype=dtype,
            attn_impl=attn_impl,
            device=device,
            device_map=device_map,
            beacon_ratio=beacon_ratio,
        )

    if mode == MODE_BEACON_LOWRANK_RECON:
        return _load_beacon_lowrank_reconstruct_model(
            model_or_ckpt=model_or_ckpt,
            dtype=dtype,
            attn_impl=attn_impl,
            device=device,
            device_map=device_map,
            beacon_ratio=beacon_ratio,
        )

    if mode == MODE_BEACON_LOWRANK_HYBRID:
        return _load_beacon_lowrank_hybrid_model(
            model_or_ckpt=model_or_ckpt,
            dtype=dtype,
            attn_impl=attn_impl,
            device_map=device_map,
            latent_compress_ratio=latent_compress_ratio,
            k_latent_head_dim_list=k_latent_head_dim_list,
            v_latent_head_dim_list=v_latent_head_dim_list,
            beacon_ratio=beacon_ratio,
        )

    raise ValueError(f"Unknown mode: {mode}")
