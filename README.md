<h1 align="center"><img src="docs/assets/slimkv-logo.svg" width="370" alt="SlimKV"></h1>

<p align="center"><strong>SlimKV: Joint Token-Feature KV Cache Compression with Reconstruction-Free Beacon Attention</strong></p>
<p align="center"><em>EMNLP 2026 Main Conference (Oral)</em></p>

<p align="center">
  <a href="https://arxiv.org/abs/2610.02953"><img src="https://img.shields.io/badge/EMNLP%202026-Oral-8a2be2.svg" alt="EMNLP 2026 Oral"></a>
  <a href="https://arxiv.org/abs/2610.02953"><img src="https://img.shields.io/badge/arXiv-2610.02953-b31b1b.svg" alt="arXiv:2610.02953"></a>
  <a href="https://huggingface.co/collections/Zihant0513/slimkv"><img src="https://img.shields.io/badge/Hugging%20Face-Models-FFD21E?logo=huggingface" alt="Model checkpoints on Hugging Face"></a>
  <a href="https://github.com/Strandding/SlimKV/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/Strandding/SlimKV/ci.yml?branch=master&amp;label=CI" alt="GitHub Actions CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="Apache License 2.0"></a>
</p>

<p align="center">
  <a href="#news">News</a> ·
  <a href="#overview">Overview</a> ·
  <a href="#method">Method</a> ·
  <a href="#results">Results</a> ·
  <a href="#models">Models</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#longbench-v1">LongBench-v1</a> ·
  <a href="#decode-speed">Speed</a> ·
  <a href="#citation">Citation</a>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2610.02953">Paper</a> ·
  <a href="https://github.com/Strandding/SlimKV">Code</a> ·
  <a href="https://huggingface.co/collections/Zihant0513/slimkv">🤗 Models</a> ·
  <a href="docs/zh/README.md">Chinese</a>
</p>

## News

- **2026-10-15:** Llama-3.1-8B and Qwen2.5-14B checkpoints with 2×, 4× and 8× feature compression are available in the [🤗 Hugging Face collection](https://huggingface.co/collections/Zihant0513/slimkv).
- **2026-10-02:** The paper is available on [arXiv](https://arxiv.org/abs/2610.02953).

## Overview

SlimKV compresses both the number of cached tokens and their feature dimensions for long-context LLM inference. It learns compact Beacon memories with layer-adaptive low-rank K/V representations, improving the accuracy–memory trade-off under a fixed cache budget.

On LongBench, SlimKV outperforms the evaluated baselines at 16× and 32× compression and retains over 96% of the full-cache score at 4× and 8×.

## Method

Conventional feature compression reconstructs full-dimensional Keys before applying RoPE. Beacon Keys are less sensitive to removing key-side RoPE than raw-token Keys. SlimKV uses this positional asymmetry to train low-rank Beacon projections without Beacon K-RoPE, while retaining RoPE for raw-token Keys and all Queries.

<p align="center">
  <img src="docs/assets/positional_asymmetry.svg" width="450" alt="LongBench comparison under Key RoPE ablations">
</p>

<p align="center"><em>Removing K-RoPE from raw-token Keys causes substantial degradation. Beacon Keys are less sensitive, and SlimKV learns low-rank memories without Beacon K-RoPE (Figure 2).</em></p>

<p align="center">
  <img src="docs/assets/method.svg" width="1000" alt="SlimKV method overview">
</p>

Prefill computes attention in the original feature dimensions and stores completed windows as compressed Beacon K/V. Decode can use either reconstruction followed by attention, or direct latent-space attention without reconstructing historical Beacon K/V. Raw-token tails and sinks retain full-dimensional K/V. See the [paper](https://arxiv.org/abs/2610.02953) for experiments and results.

## Results

LongBench accuracy is reported from the paper. Decoding measurements below use NVIDIA H20 and this repository's FA2 implementation.

### LongBench average scores

`f2`, `f4` and `f8` denote 2×, 4× and 8× feature compression. Their token compression ratios are 8×, 4× and 2× at 16× total compression, and 16×, 8× and 4× at 32×.

#### 16× compression

<div align="center">

| Method | Llama-3.1-8B | Qwen2.5-14B |
| :---: | :---: | :---: |
| **Full KV (1×)** | **49.50** | **51.52** |
| KVZip | 28.96 | 13.18 |
| PALU | 18.92 | 27.70 |
| SnapKV | 45.60 | 45.81 |
| Activation Beacon | 45.30 | 44.15 |
| SlimKV-f2 | 46.19 | **46.48** |
| SlimKV-f4 | **46.36** | 46.23 |
| SlimKV-f8 | 45.05 | 44.33 |

</div>

#### 32× compression

<div align="center">

| Method | Llama-3.1-8B | Qwen2.5-14B |
| :---: | :---: | :---: |
| **Full KV (1×)** | **49.50** | **51.52** |
| KVZip | 9.40 | 5.12 |
| PALU | 4.79 | 9.25 |
| SnapKV | 43.24 | 42.28 |
| Activation Beacon | 43.67 | 42.76 |
| SlimKV-f2 | 42.91 | 44.66 |
| SlimKV-f4 | **44.75** | **45.09** |
| SlimKV-f8 | 43.93 | 44.69 |

</div>

Averages from Table 2. Full KV is the uncompressed reference. Bold compressed scores mark the best result for each backbone and ratio.

<details>
<summary>Additional results at 4× / 8× compression</summary>

At lower compression ratios, SlimKV-f2 retains over 96% of the Full KV LongBench score. Averages from Table 4:

<div align="center">

| Backbone | Compression | KVZip | PALU | Activation Beacon | SlimKV-f2 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Llama-3.1-8B | 4× | 36.39 | 46.66 | 48.21 | 48.10 |
| Llama-3.1-8B | 8× | 37.25 | 46.42 | 46.93 | 47.77 |
| Qwen2.5-14B | 4× | 47.72 | 45.95 | 48.86 | 49.92 |
| Qwen2.5-14B | 8× | 46.74 | 45.15 | 45.50 | 49.94 |

</div>

</details>

### H20 decoding with Flash Attention 2

Llama-3.1-8B-r4 on one **NVIDIA H20**, with **B1, BF16 and Flash Attention 2 (FA2)**. Every method uses the same preallocation policy, whole-model CUDA Graphs and `torch.compile` decoder-tail compilation, with PyTorch 2.11.0+cu129 and FlashAttention 2.8.3. Full KV uses the same checkpoint's raw backbone, Recon rebuilds Beacon K/V at every step, and SlimKV uses direct latent attention.

**End-to-end decode time (ms/token, lower is better)**

<div align="center">

| Context | Full KV | Recon t4f4 | SlimKV t4f4 | Recon t8f4 | SlimKV t8f4 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 32K | 10.81 | 15.32 | 9.86 | 12.11 | 8.63 |
| 64K | 14.40 | 21.95 | 11.58 | 15.03 | 9.18 |
| 96K | 17.97 | 28.61 | 13.17 | 18.22 | 9.81 |
| 128K | 21.53 | 35.55 | 14.71 | 21.54 | 10.35 |

</div>

<p align="center"><em>NVIDIA H20 · BF16 · B1 · Flash Attention 2 · whole-model CUDA Graphs · torch.compile (decoder tail). Each run measures 128 decode steps after prefill; values are medians of four runs after one warmup. Prefill, compilation and Graph capture are excluded.</em></p>

**Attention GPU kernel time (ms/token, lower is better)**

<div align="center">

| Context | Full KV | Recon t4f4 | SlimKV t4f4 | Recon t8f4 | SlimKV t8f4 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 32K | 5.83 | 10.38 | 4.84 | 6.95 | 3.61 |
| 64K | 9.44 | 17.24 | 6.57 | 10.10 | 4.17 |
| 96K | 12.97 | 24.10 | 8.14 | 13.44 | 4.80 |
| 128K | 16.50 | 31.19 | 9.69 | 16.84 | 5.32 |

</div>

<p align="center"><em>NVIDIA H20 · BF16 · B1 · Flash Attention 2. Kernel times come from a separate uncaptured profiling pass, including QKV/output projections, cache operations, per-step KV reconstruction and RoPE, and excluding Norm, MLP and the LM head.</em></p>

<details>
<summary>End-to-end throughput (tokens/s, higher is better)</summary>

<div align="center">

| Context | Full KV | Recon t4f4 | SlimKV t4f4 | Recon t8f4 | SlimKV t8f4 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 32K | 92.50 | 65.28 | 101.38 | 82.60 | 115.89 |
| 64K | 69.43 | 45.57 | 86.35 | 66.55 | 108.89 |
| 96K | 55.64 | 34.96 | 75.94 | 54.89 | 101.89 |
| 128K | 46.45 | 28.13 | 67.97 | 46.42 | 96.62 |

</div>

</details>

All methods receive identical synthetic inputs and Full KV continuation tokens. Each optimized path must match its reference logits and final KV bitwise before timing. See [Decode speed](#decode-speed) for run commands, [installation](#1-install) for compilation, and [benchmarking](docs/benchmark.md) for metric definitions.

## Models

<div align="center">

| Backbone | 2× features | 4× features | 8× features |
| :---: | :---: | :---: | :---: |
| Llama-3.1-8B | [r2](https://huggingface.co/Zihant0513/SlimKV-Llama3.1-8B-r2) | [r4](https://huggingface.co/Zihant0513/SlimKV-Llama3.1-8B-r4) | [r8](https://huggingface.co/Zihant0513/SlimKV-Llama3.1-8B-r8) |
| Qwen2.5-14B | [r2](https://huggingface.co/Zihant0513/SlimKV-Qwen2.5-14B-r2) | [r4](https://huggingface.co/Zihant0513/SlimKV-Qwen2.5-14B-r4) | [r8](https://huggingface.co/Zihant0513/SlimKV-Qwen2.5-14B-r8) |

</div>

`r2`, `r4` and `r8` denote feature compression, not token compression. Token compression is chosen at inference: the r4 checkpoint with token ratio 4 is t4f4, nominally 16× joint compression; token ratio 8 gives t8f4, nominally 32×. These ratios describe KV compression, not total GPU memory reduction.

The walkthrough below uses **Llama-3.1-8B-r4**. It includes the backbone, trained Beacon projections, configuration and tokenizer; no separate base-model download or weight merge is needed. Qwen uses the reconstruction path; see [backend support](docs/implementation_status.md).

## Quick start

Run all commands from the repository root in the same activated shell. The FA2 setup below targets a single **H20/H100-class SM90 GPU**, Python 3.11 or 3.12, a compatible NVIDIA driver and the **CUDA 12.9 development toolkit (`nvcc`)**. The r4 latent FA2 path needs the included width-320 extension. For other hardware or eager attention, see [installation](docs/installation.md).

### 1. Install

```bash
git clone https://github.com/Strandding/SlimKV.git
cd SlimKV
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu129
python -m pip install -e . "huggingface_hub[cli]>=0.34,<1" ninja packaging
MAX_JOBS=2 python -m pip install flash-attn==2.8.3 --no-build-isolation

git clone --depth 1 --branch v2.8.3 \
  https://github.com/Dao-AILab/flash-attention.git .cache/flash-attention-v2.8.3
git -C .cache/flash-attention-v2.8.3 submodule update --init csrc/cutlass
python -m tools.build_fa2_decode \
  --fa2_source .cache/flash-attention-v2.8.3 \
  --output_dir .cache/fa2_decode_320 --jobs 2
export SLIMKV_FA2_DECODE_EXTENSION_DIR="$PWD/.cache/fa2_decode_320"
```

The `tools.build_fa2_decode` command above compiles the SM90 BF16 CUDA extension. The benchmark command uses `DECODE_COMPILE=True` to enable `torch.compile`; the decoder tail compiles automatically on its first call, with no separate compilation script. Set `DECODE_COMPILE=False` to disable it. Standalone scripts use Transformers 4.40.2; keep vLLM in a separate environment.

### 2. Download the checkpoint

```bash
export MODEL_NAME_OR_PATH="$PWD/models/SlimKV-Llama3.1-8B-r4"
hf download Zihant0513/SlimKV-Llama3.1-8B-r4 \
  --local-dir "$MODEL_NAME_OR_PATH"
```

Keep all downloaded weight shards, the shard index, config and tokenizer together. `hf download` reuses files already present when rerun. In a new shell, reactivate `.venv` and export `MODEL_NAME_OR_PATH` and `SLIMKV_FA2_DECODE_EXTENSION_DIR` again.

### 3. Generate a response

```bash
python -m main.inference.infer \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --prompt 'Explain why a KV cache helps language model inference.' \
  --max_new_tokens 32
```

Use `--prompt_file /path/to/context.txt` for a UTF-8 context. CUDA defaults to BF16, original-dimension FA2 prefill and direct latent decode, with Graphs and optional decoder-tail compilation for eligible B1 calls. See [quick start](docs/quickstart.md) for the Python API and eager/reconstruction options.

## LongBench-v1

### Download the data

```bash
export EVAL_DATA="$PWD/data/datasets/longbench"
python -m tools.data.download_longbench --output_dir "$EVAL_DATA"
```

The downloader fetches the official v1 archive and extracts the selected JSONL task files. No dataset files are bundled in this repository.

### Check the workflow with four samples

```bash
TASKS="qasper hotpotqa" MAX_SAMPLES_PER_TASK=2 BATCH_SIZE=1 \
RESULT_DIR=quickstart-t4f4 \
  bash scripts/evaluation/eval_longbench_slimkv.sh
```

This evaluates the first two samples of each selected task. It checks the full evaluation workflow; use the next commands for the complete English/code preset.

### Run t4f4 or t8f4

```bash
BEACON_RATIO=4 RESULT_DIR=longbench-t4f4 \
  bash scripts/evaluation/eval_longbench_slimkv.sh

BEACON_RATIO=8 RESULT_DIR=longbench-t8f4 \
  bash scripts/evaluation/eval_longbench_slimkv.sh
```

The preset uses 14 English/code tasks, BF16 greedy decoding, a final chat-formatted input limit of **31,500 tokens**, and head/tail truncation. The training corpus is Chinese-only, so Chinese subsets are filtered; `passage_count` and `passage_retrieval_en` are excluded separately. Batch size is capped at 32, with equal-length grouping and no padding. Adjust `BATCH_SIZE` and `MAX_BATCH_TOKENS` for your GPU.

Files are written under `data/results/longbench/<RESULT_DIR>/`: `scores.json` contains scores, `task_timings.csv` contains each subset's elapsed time, and `progress.json` tracks progress. Set `LOAD_RESULT=True` with the same configuration to resume completed subsets. See [evaluation](docs/evaluation.md) for task selection, output definitions and all backend options.

## Decode speed

No LongBench download is required: the runner constructs and saves synthetic input tokens. Use the checkpoint and extension exported above.

Start with one 32K context and t4f4:

```bash
CONTEXT_LENGTHS=32768 BEACON_RATIOS=4 REPETITIONS=1 \
OUTPUT_DIR=data/results/speed-smoke \
  bash scripts/benchmark/benchmark_context_decode.sh
```

Then run the default comparison at **32K, 64K, 96K and 128K**, with both **t4f4 and t8f4**:

```bash
DECODE_COMPILE=True OUTPUT_DIR=data/results/speed-32k-128k \
  bash scripts/benchmark/benchmark_context_decode.sh
```

Each context compares Full KV, reconstruction and direct latent attention at **B1, BF16 and FA2**, with the same preallocation, whole-model Graph and optional compilation policies. Every run measures **128 decode forwards after prefill**, using the same inputs and continuation tokens. The full preset uses one warmup and four measurements; prefill, compilation and Graph capture are excluded from decode timing. Optimized logits and final KV are checked against each method's reference path before timing.

Two metrics are saved: **attention GPU kernel time**, including per-step KV reconstruction and RoPE for the reconstruction path, and **end-to-end decode time**. `results.json` contains timings, tokens/s and numerical checks; `summary.csv` contains per-mode medians. Use a new `OUTPUT_DIR` for each benchmark invocation. Full KV, reference snapshots and Graph workspaces need additional memory; reduce `CONTEXT_LENGTHS` if needed. See [benchmarking](docs/benchmark.md) for metric boundaries, runtime switches and profiling.

Raw generated outputs stay under ignored `data/results/` directories. Paper accuracy and H20 timing summaries appear above; see the [paper](https://arxiv.org/abs/2610.02953) for its full experiments.

## Training and other tools

<div align="center">

| Task | Entry point | Guide |
| :---: | :---: | :---: |
| Low-rank initialization | `python -m main.conversion.convert_kv_lowrank --help` | [Training](docs/training.md) |
| Pretraining | `bash scripts/training/train_llama_8B.sh` | [Training](docs/training.md) |
| Finetuning | `bash scripts/training/finetune_llama_8B.sh` | [Training](docs/training.md) |
| Needle-in-a-Haystack | `python -m main.evaluation.eval_needle --help` | [Evaluation](docs/evaluation.md) |
| vLLM inference | `python -m main.inference.infer_vllm --help` | [vLLM](docs/vllm.md) |

</div>

See [reproduction](docs/reproducibility.md), the [source map](docs/structure.md) and [backend support](docs/implementation_status.md) for further details.

## Citation

The formal ACL Anthology record is not yet available. Use this provisional conference citation with the arXiv link.

```bibtex
@inproceedings{teng2026slimkv,
  title = {SlimKV: Joint Token-Feature KV Cache Compression with Reconstruction-Free Beacon Attention},
  author = {Teng, Zihan and Zhao, Jiayu and Ren, Wentao and Fan, Minhao and Ma, Tianrui and Chen, Song and Liu, Weichen},
  booktitle = {2026 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year = {2026},
  note = {Accepted to the Main Conference (Oral). Provisional conference citation.},
  eprint = {2610.02953},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  url = {https://arxiv.org/abs/2610.02953}
}
```

## Acknowledgements and license

SlimKV builds on [Activation Beacon](https://github.com/FlagOpen/FlagEmbedding/tree/master/research/Long_LLM/activation_beacon), Hugging Face Transformers and [FlashAttention](https://github.com/Dao-AILab/flash-attention). See [third-party notices](THIRD_PARTY_NOTICES.md) for attribution and licenses. Original SlimKV code is licensed under [Apache License 2.0](LICENSE). Third-party components retain their own copyrights and licenses. Model weights and datasets remain subject to their respective terms.
