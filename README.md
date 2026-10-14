<h1 align="center">SlimKV</h1>

<p align="center"><strong>Joint Token-Feature KV Cache Compression with Reconstruction-Free Beacon Attention</strong></p>
<p align="center"><em>EMNLP 2026 Main Conference (Oral)</em></p>

<p align="center">
  <a href="https://arxiv.org/abs/2610.02953"><img src="https://img.shields.io/badge/EMNLP%202026-Oral-8a2be2.svg" alt="EMNLP 2026 Oral"></a>
  <a href="https://arxiv.org/abs/2610.02953"><img src="https://img.shields.io/badge/arXiv-2610.02953-b31b1b.svg" alt="arXiv:2610.02953"></a>
  <a href="https://github.com/Strandding/SlimKV/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/Strandding/SlimKV/ci.yml?branch=master&amp;label=CI" alt="GitHub Actions CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="Apache License 2.0"></a>
</p>

<p align="center">
  <a href="#news">News</a> ·
  <a href="#overview">Overview</a> ·
  <a href="#method">Method</a> ·
  <a href="#results">Results</a> ·
  <a href="#installation">Installation</a> ·
  <a href="#usage">Usage</a> ·
  <a href="#reproduction">Reproduction</a> ·
  <a href="#citation">Citation</a>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2610.02953">Paper</a> ·
  <a href="https://github.com/Strandding/SlimKV">Code</a> ·
  <a href="docs/zh/README.md">中文</a>
</p>

## News

- **2026-10-14:** [Inference](docs/quickstart.md), [training](docs/training.md), [evaluation](docs/evaluation.md) and [benchmarking](docs/benchmark.md) guides are available.
- **2026-10-02:** The paper is available on [arXiv](https://arxiv.org/abs/2610.02953).

## Overview

SlimKV compresses both the number of cached tokens and their feature dimensions for long-context LLM inference. It learns compact Beacon memories with layer-adaptive low-rank K/V representations, improving the accuracy–memory trade-off under a fixed cache budget.

On LongBench, SlimKV outperforms the evaluated baselines at 16× and 32× compression and retains over 96% of the full-cache score at 4× and 8×. At 128K context length, the 4× token / 4× feature configuration achieves 7.34× attention kernel speedup and 3.38× end-to-end decoding speedup over Full KV.

## Method

Conventional feature compression often reconstructs full-dimensional Keys before applying RoPE. We find that Beacon Keys are much less sensitive to removing key-side RoPE than raw-token Keys. SlimKV therefore trains low-rank Beacon projections without Beacon K-RoPE, while retaining RoPE for raw-token Keys and all Queries. This **positional asymmetry** enables reconstruction-free latent-space attention during decoding.

<p align="center">
  <img src="docs/assets/positional_asymmetry.svg" width="650" alt="LongBench comparison under Key RoPE ablations">
</p>

<p align="center"><em>Removing K-RoPE from raw-token Keys causes substantial degradation. Beacon Keys are less sensitive, and SlimKV learns low-rank memories without Beacon K-RoPE (Figure 2).</em></p>

<p align="center">
  <img src="docs/assets/method.svg" width="1000" alt="SlimKV method overview">
</p>

SlimKV progressively compresses completed context windows into Beacon memories. Its K/V projections are trained directly at the target rank, with the backbone frozen. Layer-adaptive ranks account for differences in low-rank structure across layers rather than assigning every layer the same capacity.

During decoding, attention scores and value aggregation operate on cached latent Beacon states. Queries are projected into the latent Key space, and attention outputs are projected back through the Value basis. This avoids reconstructing the historical Beacon K/V. Raw-token tails and sinks retain full-dimensional K/V.

## Results

<table align="center">
  <thead>
    <tr>
      <th align="center"><strong>16× / 32×</strong></th>
      <th align="center"><strong>7.34×</strong></th>
      <th align="center"><strong>3.38×</strong></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">Leading LongBench results under high compression</td>
      <td align="center">Attention kernel speedup at 128K</td>
      <td align="center">End-to-end decoding speedup at 128K</td>
    </tr>
  </tbody>
</table>

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
<summary>Decoding latency speedups across 32K–128K contexts</summary>

Llama-3.1-8B on an A100 80GB, using BF16 eager execution. The two metrics measure attention kernel latency and end-to-end decoding latency per token. Every entry is **Full KV latency / method latency**, so higher is faster. Recon reconstructs cached K/V; SlimKV uses reconstruction-free latent attention (Table 6).

#### 16× compression: 4× tokens, 4× features

<div align="center">

| Context | Kernel · Recon | Kernel · SlimKV | End-to-end · Recon | End-to-end · SlimKV |
| :---: | :---: | :---: | :---: | :---: |
| 32K | 1.76× | **1.94×** | 1.13× | **1.22×** |
| 64K | 2.81× | **3.81×** | 1.64× | **2.04×** |
| 96K | 3.63× | **5.08×** | 2.13× | **2.63×** |
| 128K | 5.34× | **7.34×** | 2.86× | **3.38×** |

</div>

#### 32× compression: 8× tokens, 4× features

<div align="center">

| Context | Kernel · Recon | Kernel · SlimKV | End-to-end · Recon | End-to-end · SlimKV |
| :---: | :---: | :---: | :---: | :---: |
| 32K | 1.52× | **1.62×** | 1.15× | **1.23×** |
| 64K | 2.35× | **2.54×** | 1.67× | **1.75×** |
| 96K | 3.84× | **4.36×** | 2.60× | **2.82×** |
| 128K | 4.92× | **6.09×** | 3.06× | **3.49×** |

</div>

</details>

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

See the [paper](https://arxiv.org/abs/2610.02953) for the full protocol and [benchmarking](docs/benchmark.md) for runnable latency and memory comparisons.

## Installation

Use Python 3.11 and install a PyTorch build that matches your hardware, then:

```bash
git clone https://github.com/Strandding/SlimKV.git
cd SlimKV
python -m pip install -e .
```

The model code uses Transformers 4.40.x, tested with 4.40.2. Eager inference does not require FlashAttention. See [installation](docs/installation.md) for training dependencies and the optional FA2 extension.

## Usage

Model weights are not distributed with this repository. Use an existing SlimKV checkpoint or [train one](docs/training.md), including its weights, configuration and tokenizer.

```bash
python -m main.infer \
  --model_name_or_path /path/to/slimkv-checkpoint \
  --prompt 'Explain why a KV cache helps language model inference.' \
  --max_new_tokens 32
```

Use `--prompt_file /path/to/context.txt` for a long context. The example reads ranks and Beacon settings from the checkpoint and runs greedy BF16 inference on `cuda:0`.

The default backend stores latent Beacon K/V and reconstructs them for attention. Llama3 also supports reconstruction-free eager and FA2 decode. Training and prefill use the original attention dimension. See [quick start](docs/quickstart.md) for the Python API and backend options.

## Reproduction

Run the following entry points from the repository root. Model, data and output paths can be set through the environment variables in [.env.example](.env.example).

<div align="center">

| Stage | Entry point | Guide |
| :---: | :---: | :---: |
| Low-rank initialization | `python -m main.convert_kv_lowrank --help` | [Training](docs/training.md) |
| Pretraining | `bash scripts/training/train_llama_8B.sh` | [Training](docs/training.md) |
| Finetuning | `bash scripts/training/finetune_llama_8B.sh` | [Training](docs/training.md) |
| LongBench-v1 | `bash scripts/evaluation/eval_longbench_slimkv.sh` | [Evaluation](docs/evaluation.md) |
| Needle-in-a-Haystack | `python -m main.eval_needle --help` | [Evaluation](docs/evaluation.md) |
| Latency and memory | `python -m tools.benchmark_fa2_decode --help` | [Benchmarking](docs/benchmark.md) |

</div>

See the [reproduction guide](docs/reproducibility.md) for experiment protocols and required resources, the [source map](docs/structure.md) for code navigation, and [backend support](docs/implementation_status.md) for cache and attention paths.

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
