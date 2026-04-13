import logging
import math
import time
from dataclasses import dataclass, field

import torch
from transformers import HfArgumentParser
from tqdm.auto import tqdm

from src import ModelArgs, get_model_and_tokenizer, makedirs

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
    svd_device: str = field(
        default="cpu",
        metadata={"help": "Device used for SVD decomposition. Usually `cpu` or `cuda`."}
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
        ax.axvline(global_rank, color="tab:red", linestyle="--", linewidth=1.0)
        ax.set_yscale("log")
        ax.set_title(f"Layer {layer_idx}", fontsize=10)
        ax.set_xlabel("Singular value index")
        ax.set_ylabel("Normalized value")
        ax.grid(alpha=0.25, linewidth=0.5)

    for empty_idx in range(num_layers, nrows * ncols):
        axes[empty_idx // ncols][empty_idx % ncols].set_visible(False)

    label = proj_name.upper()
    fig.suptitle(
        f"Beacon-{label} singular spectra (normalized, log scale, rank cutoff={global_rank})",
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
        ax.axvline(global_rank, color="tab:red", linestyle="--", linewidth=1.0)
        ax.set_ylim(0.0, 1.02)
        ax.set_title(f"Layer {layer_idx}", fontsize=10)
        ax.set_xlabel("Singular value index")
        ax.set_ylabel("Cumulative energy")
        ax.grid(alpha=0.25, linewidth=0.5)

    for empty_idx in range(num_layers, nrows * ncols):
        axes[empty_idx // ncols][empty_idx % ncols].set_visible(False)

    label = proj_name.upper()
    fig.suptitle(
        f"Beacon-{label} cumulative spectral energy (rank cutoff={global_rank})",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    plot_path = f"{output_dir}/beacon_{proj_name}_cumulative_energy_overview.png"
    fig.savefig(plot_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def _convert_proj(layers, proj_name, latent_head_dim, svd_device, return_sv):
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
    converted_rank = None
    progress = tqdm(dense_layers, desc=f"Converting {dense_attr}", unit="layer")
    for idx, layer in enumerate(progress, start=1):
        layer_start = time.time()
        conversion_info = getattr(layer, convert_fn_name)(
            latent_head_dim,
            svd_device=svd_device,
            return_singular_values=return_sv,
        )
        if isinstance(conversion_info, dict):
            converted_rank = conversion_info["global_rank"]
            singular_values_by_layer.append(conversion_info["singular_values"])
        else:
            converted_rank = conversion_info
        progress.set_postfix(
            rank=converted_rank,
            layer=f"{idx}/{len(dense_layers)}",
            sec=f"{time.time() - layer_start:.1f}",
        )
    return converted_rank, singular_values_by_layer


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

    return_sv = conversion_args.save_svd_spectra or conversion_args.save_svd_spectra_data

    start_time = time.time()

    # Convert K
    k_rank = None
    if "k" in beacon_param:
        k_latent_head_dim = model_args.beacon_k_latent_head_dim
        if k_latent_head_dim is None:
            k_latent_head_dim = getattr(model.config, "beacon_k_latent_head_dim", 64)
        logger.info("Converting beacon K projections (latent_head_dim=%s)...", k_latent_head_dim)
        with torch.no_grad():
            k_rank, k_sv = _convert_proj(attn_layers, "k", k_latent_head_dim, conversion_args.svd_device, return_sv)
        model.config.beacon_k_latent_head_dim = k_latent_head_dim

    # Convert V
    v_rank = None
    if "v" in beacon_param:
        v_latent_head_dim = model_args.beacon_v_latent_head_dim
        if v_latent_head_dim is None:
            v_latent_head_dim = getattr(model.config, "beacon_v_latent_head_dim", 64)
        logger.info("Converting beacon V projections (latent_head_dim=%s)...", v_latent_head_dim)
        with torch.no_grad():
            v_rank, v_sv = _convert_proj(attn_layers, "v", v_latent_head_dim, conversion_args.svd_device, return_sv)
        model.config.beacon_v_latent_head_dim = v_latent_head_dim

    output_dir = makedirs(conversion_args.output_dir)
    logger.info(
        "Saving low-rank beacon-KV checkpoint to %s (k_rank=%s, v_rank=%s, elapsed=%.1fs)...",
        output_dir, k_rank, v_rank, time.time() - start_time,
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
