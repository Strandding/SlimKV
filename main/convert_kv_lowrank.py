import logging
import math
import time
from dataclasses import dataclass, field

import torch
from transformers import HfArgumentParser
from tqdm.auto import tqdm

from slimkv import ModelArgs, get_model_and_tokenizer, makedirs

logger = logging.getLogger(__name__)


@dataclass
class ConversionArgs:
    output_dir: str = field(
        metadata={"help": "Directory to save the converted low-rank checkpoint."}
    )
    save_tokenizer: bool = field(
        default=True,
        metadata={"help": "Save tokenizer files alongside the converted checkpoint."}
    )
    safe_serialization: bool = field(
        default=True,
        metadata={"help": "Save the converted checkpoint in safetensors format when possible."}
    )
    save_dtype: str = field(
        default="fp16",
        metadata={"help": "Checkpoint save dtype. One of {fp16, bf16, fp32, inherit}."}
    )
    svd_device: str = field(
        default="cpu",
        metadata={"help": "Device used for SVD decomposition. Usually `cpu` or `cuda`."}
    )
    svd_partition: str = field(
        default="global",
        metadata={"help": "How to partition dense projection before SVD. One of {global, per_head}."}
    )
    save_svd_spectra: bool = field(
        default=True,
        metadata={"help": "Save a summary figure of singular value spectra for all converted layers."}
    )
    save_svd_spectra_data: bool = field(
        default=True,
        metadata={"help": "Save raw singular values for each converted layer to a .pt file."}
    )


def _save_singular_value_overview(output_dir, singular_values_by_layer, global_rank, proj_name="v"):
    if len(singular_values_by_layer) == 0:
        return None

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        logger.warning("Skip singular value plotting because matplotlib is unavailable: %s", exc)
        return None

    num_layers = len(singular_values_by_layer)
    ncols = 4
    nrows = math.ceil(num_layers / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), squeeze=False)

    for layer_idx, singular_values in enumerate(singular_values_by_layer):
        ax = axes[layer_idx // ncols][layer_idx % ncols]
        singular_values = singular_values.float()
        if singular_values.numel() == 0:
            ax.set_visible(False)
            continue

        normalized = singular_values / singular_values[0].clamp_min(1e-12)
        normalized = normalized.clamp_min(1e-12)
        x = torch.arange(1, normalized.numel() + 1)
        ax.plot(x.cpu().numpy(), normalized.cpu().numpy(), linewidth=1.5)
        layer_rank = global_rank[layer_idx] if isinstance(global_rank, (list, tuple)) else global_rank
        ax.axvline(layer_rank, color="tab:red", linestyle="--", linewidth=1.0)
        ax.set_yscale("log")
        ax.set_title(f"Layer {layer_idx}", fontsize=10)
        ax.set_xlabel("Singular value index")
        ax.set_ylabel("Normalized value")
        ax.grid(alpha=0.25, linewidth=0.5)

    for empty_idx in range(num_layers, nrows * ncols):
        axes[empty_idx // ncols][empty_idx % ncols].set_visible(False)

    label = proj_name.upper()
    rank_text = "variable" if isinstance(global_rank, (list, tuple)) else str(global_rank)
    fig.suptitle(
        f"Beacon-{label} singular spectra (normalized, log scale, rank cutoff={rank_text})",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    plot_path = f"{output_dir}/beacon_{proj_name}_singular_values_overview.png"
    fig.savefig(plot_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def _save_cumulative_energy_overview(output_dir, singular_values_by_layer, global_rank, proj_name="v"):
    if len(singular_values_by_layer) == 0:
        return None

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        logger.warning("Skip cumulative energy plotting because matplotlib is unavailable: %s", exc)
        return None

    num_layers = len(singular_values_by_layer)
    ncols = 4
    nrows = math.ceil(num_layers / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), squeeze=False)

    for layer_idx, singular_values in enumerate(singular_values_by_layer):
        ax = axes[layer_idx // ncols][layer_idx % ncols]
        singular_values = singular_values.float()
        if singular_values.numel() == 0:
            ax.set_visible(False)
            continue

        energy = singular_values.pow(2)
        cumulative_energy = energy.cumsum(0) / energy.sum().clamp_min(1e-12)
        x = torch.arange(1, cumulative_energy.numel() + 1)
        ax.plot(x.cpu().numpy(), cumulative_energy.cpu().numpy(), linewidth=1.5)
        layer_rank = global_rank[layer_idx] if isinstance(global_rank, (list, tuple)) else global_rank
        ax.axvline(layer_rank, color="tab:red", linestyle="--", linewidth=1.0)
        ax.set_ylim(0.0, 1.02)
        ax.set_title(f"Layer {layer_idx}", fontsize=10)
        ax.set_xlabel("Singular value index")
        ax.set_ylabel("Cumulative energy")
        ax.grid(alpha=0.25, linewidth=0.5)

    for empty_idx in range(num_layers, nrows * ncols):
        axes[empty_idx // ncols][empty_idx % ncols].set_visible(False)

    label = proj_name.upper()
    rank_text = "variable" if isinstance(global_rank, (list, tuple)) else str(global_rank)
    fig.suptitle(
        f"Beacon-{label} cumulative spectral energy (rank cutoff={rank_text})",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    plot_path = f"{output_dir}/beacon_{proj_name}_cumulative_energy_overview.png"
    fig.savefig(plot_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def _resolve_layer_latent_head_dims(
    latent_head_dim,
    latent_head_dim_list,
    num_layers,
    head_dim,
    proj_name,
    list_field_name,
):
    if latent_head_dim_list is not None:
        if not isinstance(latent_head_dim_list, (list, tuple)):
            raise ValueError(
                f"{list_field_name} must be a list/tuple, got {type(latent_head_dim_list).__name__}."
            )
        if len(latent_head_dim_list) != num_layers:
            raise ValueError(
                f"{list_field_name} length must equal num_hidden_layers "
                f"({num_layers}), found {len(latent_head_dim_list)}."
            )
        resolved = []
        for idx, val in enumerate(latent_head_dim_list):
            if not isinstance(val, int):
                raise ValueError(
                    f"{list_field_name}[{idx}] must be int, got {type(val).__name__}."
                )
            if val <= 0 or val > head_dim:
                raise ValueError(
                    f"Invalid {list_field_name}[{idx}]={val}. Must be in [1, {head_dim}]."
                )
            resolved.append(val)
        return resolved

    if latent_head_dim is None:
        raise ValueError(
            f"Missing latent head dim for beacon {proj_name.upper()}. "
            f"Set {list_field_name} or beacon_{proj_name}_latent_head_dim."
        )
    if latent_head_dim <= 0 or latent_head_dim > head_dim:
        raise ValueError(
            f"Invalid latent_head_dim={latent_head_dim} for beacon {proj_name.upper()}. Must be in [1, {head_dim}]."
        )
    return [latent_head_dim for _ in range(num_layers)]


def _convert_proj(layers, proj_name, latent_head_dims, svd_device, svd_partition, return_sv):
    """Convert a single projection (k or v) across all layers. Returns (rank, sv_list)."""
    dense_attr = f"beacon_{proj_name}_proj"
    dense_layers = [layer for layer in layers if hasattr(layer, dense_attr)]
    if len(dense_layers) != len(layers):
        raise ValueError(
            f"The source checkpoint does not contain dense {dense_attr} weights for every layer. "
            "Load a dense beacon checkpoint or pass beacon construction flags for a dense beacon model."
        )

    convert_fn_name = f"convert_beacon_{proj_name}_proj_to_lowrank"
    singular_values_by_layer = []
    converted_ranks = []
    progress = tqdm(dense_layers, desc=f"Converting {dense_attr}", unit="layer")
    for idx, (layer, layer_latent_head_dim) in enumerate(zip(progress, latent_head_dims), start=1):
        layer_start = time.time()
        conversion_info = getattr(layer, convert_fn_name)(
            layer_latent_head_dim,
            svd_device=svd_device,
            svd_partition=svd_partition,
            return_singular_values=return_sv,
        )
        if isinstance(conversion_info, dict):
            converted_rank = conversion_info["global_rank"]
            singular_values_by_layer.append(conversion_info["singular_values"])
        else:
            converted_rank = conversion_info
        converted_ranks.append(converted_rank)
        progress.set_postfix(
            rank=converted_rank,
            latent_head_dim=layer_latent_head_dim,
            layer=f"{idx}/{len(dense_layers)}",
            sec=f"{time.time() - layer_start:.1f}",
        )
    if len(converted_ranks) > 0 and all(rank == converted_ranks[0] for rank in converted_ranks):
        return converted_ranks[0], singular_values_by_layer
    return converted_ranks, singular_values_by_layer


def _save_spectra(output_dir, proj_name, singular_values_by_layer, global_rank, save_data, save_plots):
    if save_data and len(singular_values_by_layer) > 0:
        spectra_data_path = f"{output_dir}/beacon_{proj_name}_singular_values.pt"
        torch.save(
            {
                "global_rank": global_rank,
                "layer_indices": list(range(len(singular_values_by_layer))),
                "singular_values": singular_values_by_layer,
            },
            spectra_data_path,
        )
        logger.info("Saved raw %s singular values to %s", proj_name.upper(), spectra_data_path)
    if save_plots and len(singular_values_by_layer) > 0:
        plot_path = _save_singular_value_overview(output_dir, singular_values_by_layer, global_rank, proj_name)
        if plot_path is not None:
            logger.info("Saved %s singular value overview plot to %s", proj_name.upper(), plot_path)
        energy_plot_path = _save_cumulative_energy_overview(output_dir, singular_values_by_layer, global_rank, proj_name)
        if energy_plot_path is not None:
            logger.info("Saved %s cumulative energy overview plot to %s", proj_name.upper(), energy_plot_path)


def main():
    parser = HfArgumentParser([ModelArgs, ConversionArgs])
    model_args, conversion_args = parser.parse_args_into_dataclasses()

    model_args.enable_beacon = True
    # Load with dense projections so SVD can decompose them
    model_args.beacon_kv_compression = "none"
    model_args.beacon_k_compression = "none"
    model_args.beacon_v_compression = "none"
    # Offline SVD conversion does not run attention kernels, so force a CPU-safe implementation.
    model_args.attn_impl = "eager"

    model, tokenizer = get_model_and_tokenizer(
        model_args,
        device="cpu",
        evaluation_mode=True,
    )

    attn_layers = [layer.self_attn for layer in model.model.layers]
    supports_lowrank_conversion = all(
        hasattr(layer, "convert_beacon_k_proj_to_lowrank") and hasattr(layer, "convert_beacon_v_proj_to_lowrank")
        for layer in attn_layers
    )
    if not supports_lowrank_conversion:
        raise NotImplementedError(
            "This model does not expose beacon KV low-rank conversion methods. "
            "Currently supported by Qwen2 and Llama3 beacon models in this repository."
        )

    beacon_param = getattr(model.config, "beacon_param", [])
    if conversion_args.svd_device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(
            f"Requested svd_device={conversion_args.svd_device}, but CUDA is not available."
        )
    if conversion_args.svd_partition not in {"global", "per_head"}:
        raise ValueError(
            f"Unsupported svd_partition={conversion_args.svd_partition}. "
            "Expected one of {global, per_head}."
        )

    return_sv = conversion_args.save_svd_spectra or conversion_args.save_svd_spectra_data

    start_time = time.time()

    num_layers = len(attn_layers)
    head_dim = attn_layers[0].head_dim
    kv_latent_head_dim_list = model_args.beacon_kv_latent_head_dim_list
    k_latent_head_dim_list = model_args.beacon_k_latent_head_dim_list
    v_latent_head_dim_list = model_args.beacon_v_latent_head_dim_list
    if kv_latent_head_dim_list is not None:
        raise ValueError(
            "beacon_kv_latent_head_dim_list is deprecated and not supported. "
            "Use beacon_k_latent_head_dim_list and beacon_v_latent_head_dim_list."
        )
    if k_latent_head_dim_list is not None:
        logger.info("Using per-layer latent dims from beacon_k_latent_head_dim_list.")
    if v_latent_head_dim_list is not None:
        logger.info("Using per-layer latent dims from beacon_v_latent_head_dim_list.")

    # Convert K
    k_rank = None
    if "k" in beacon_param:
        k_latent_head_dim = model_args.beacon_k_latent_head_dim
        if k_latent_head_dim is None:
            k_latent_head_dim = getattr(model.config, "beacon_k_latent_head_dim", 64)
        k_layerwise_list = k_latent_head_dim_list
        k_latent_head_dims = _resolve_layer_latent_head_dims(
            k_latent_head_dim,
            k_layerwise_list,
            num_layers,
            head_dim,
            proj_name="k",
            list_field_name="beacon_k_latent_head_dim_list",
        )
        logger.info("Converting beacon K projections (latent_head_dims=%s)...", k_latent_head_dims)
        with torch.no_grad():
            k_rank, k_sv = _convert_proj(
                attn_layers,
                "k",
                k_latent_head_dims,
                conversion_args.svd_device,
                conversion_args.svd_partition,
                return_sv,
            )
        model.config.beacon_k_latent_head_dim = k_latent_head_dims[0]

    # Convert V
    v_rank = None
    if "v" in beacon_param:
        v_latent_head_dim = model_args.beacon_v_latent_head_dim
        if v_latent_head_dim is None:
            v_latent_head_dim = getattr(model.config, "beacon_v_latent_head_dim", 64)
        v_layerwise_list = v_latent_head_dim_list
        v_latent_head_dims = _resolve_layer_latent_head_dims(
            v_latent_head_dim,
            v_layerwise_list,
            num_layers,
            head_dim,
            proj_name="v",
            list_field_name="beacon_v_latent_head_dim_list",
        )
        logger.info("Converting beacon V projections (latent_head_dims=%s)...", v_latent_head_dims)
        with torch.no_grad():
            v_rank, v_sv = _convert_proj(
                attn_layers,
                "v",
                v_latent_head_dims,
                conversion_args.svd_device,
                conversion_args.svd_partition,
                return_sv,
            )
        model.config.beacon_v_latent_head_dim = v_latent_head_dims[0]

    if "k" in beacon_param and k_rank is not None:
        model.config.beacon_k_latent_head_dim_list = list(k_latent_head_dims)
    if "v" in beacon_param and v_rank is not None:
        model.config.beacon_v_latent_head_dim_list = list(v_latent_head_dims)
    model.config.beacon_kv_latent_head_dim_list = None

    output_dir = makedirs(conversion_args.output_dir)
    save_dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
        "inherit": None,
    }
    if conversion_args.save_dtype not in save_dtype_map:
        raise ValueError(
            f"Unsupported save_dtype={conversion_args.save_dtype}. "
            "Expected one of {fp16, bf16, fp32, inherit}."
        )
    target_save_dtype = save_dtype_map[conversion_args.save_dtype]
    if target_save_dtype is not None:
        model = model.to(target_save_dtype)
        model.config.torch_dtype = target_save_dtype

    logger.info(
        "Saving low-rank beacon-KV checkpoint to %s (k_rank=%s, v_rank=%s, svd_partition=%s, save_dtype=%s, elapsed=%.1fs)...",
        output_dir, k_rank, v_rank, conversion_args.svd_partition, conversion_args.save_dtype, time.time() - start_time,
    )
    model.save_pretrained(output_dir, safe_serialization=conversion_args.safe_serialization)
    if conversion_args.save_tokenizer:
        tokenizer.save_pretrained(output_dir)

    if "k" in beacon_param:
        _save_spectra(output_dir, "k", k_sv, k_rank,
                      conversion_args.save_svd_spectra_data, conversion_args.save_svd_spectra)
    if "v" in beacon_param:
        _save_spectra(output_dir, "v", v_sv, v_rank,
                      conversion_args.save_svd_spectra_data, conversion_args.save_svd_spectra)


if __name__ == "__main__":
    main()
