# SlimKV

<p align="center"><em>*联合 Token–Feature KV Cache 压缩与无需重建的 Beacon 注意力*</em></p>

<p align="center"><em>*EMNLP 2026 Main Conference（Oral）*</em></p>

[论文](https://arxiv.org/abs/2610.02953) · [代码](https://github.com/Strandding/SlimKV) · [English](README.md)

## 概览

SlimKV 同时压缩 KV Cache 的 token 数量和特征维度，用于长上下文大模型推理。它通过训练紧凑的 Beacon 记忆和逐层自适应低秩 K/V 表示，在固定缓存预算下改善精度与显存之间的权衡。

传统特征压缩通常需要先重建完整维度的 Key，再应用 RoPE。我们发现，去掉 key-side RoPE 对 Beacon Keys 的影响显著小于对 raw-token Keys 的影响。SlimKV 利用这种位置不对称性，在关闭 Beacon K-RoPE 的约束下训练低秩投影，同时保留 raw K 和所有 Q 的 RoPE，从而在 decode 中直接计算 latent-space attention，避免历史 Beacon K/V 的完整维度重建。

在 LongBench 上，SlimKV 在4×和8×压缩下保留超过96%的 Full KV 得分，在16×和32×压缩下优于所评测的基线。128K上下文下，t4f4配置实现7.34× attention加速和3.38×端到端decode加速。

## 方法

<p align="center">
  <img src="docs/assets/method.svg" width="1000" alt="SlimKV 方法概览">
</p>

SlimKV 将已完成的上下文窗口逐步压缩为 Beacon 记忆。Backbone 保持冻结，Beacon K/V 投影直接在目标低秩约束下训练。逐层 rank 根据各层低秩结构的差异分配容量，而不是给所有层相同维度。

Decode 时，attention scores 和 Value 聚合在缓存的 latent Beacon 状态上完成。Query 投影到 latent Key 空间，attention 输出通过 Value basis 映射回原始空间，因此无需重建历史 Beacon K/V。Raw tail 和 sink 仍保留完整维度缓存。

<p align="center">
  <img src="docs/assets/positional_asymmetry.svg" width="400" alt="K-RoPE 消融下的 LongBench 比较">
</p>

<p align="center"><em>去掉 raw-token K-RoPE 会造成显著退化，Beacon Keys 对此更不敏感。SlimKV 在无 Beacon K-RoPE 的约束下学习低秩记忆（Figure 2）。</em></p>

## 结果

<table align="center">
  <thead>
    <tr>
      <th align="center"><strong>96%+</strong></th>
      <th align="center"><strong>7.34×</strong></th>
      <th align="center"><strong>3.38×</strong></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">4× / 8× 压缩下保留的 Full KV LongBench 得分</td>
      <td align="center">128K attention 加速</td>
      <td align="center">128K 端到端 decode 加速</td>
    </tr>
  </tbody>
</table>

上述速度来自 Llama-3.1-8B 的 t4f4 配置，使用 A100 80GB、BF16 和 eager。完整结果与评测协议见[论文](https://arxiv.org/abs/2610.02953)。

<div align="center">

| 模型 | 压缩比 | KVZip | PALU | Activation Beacon | SlimKV-f2 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Llama-3.1-8B | 4× | 36.39 | 46.66 | 48.21 | 48.10 |
| Llama-3.1-8B | 8× | 37.25 | 46.42 | 46.93 | 47.77 |
| Qwen2.5-14B | 4× | 47.72 | 45.95 | 48.86 | 49.92 |
| Qwen2.5-14B | 8× | 46.74 | 45.15 | 45.50 | 49.94 |

</div>

<p align="center"><em>4×和8×压缩下的 LongBench 平均分（Table 4）。SlimKV-f2 使用2×特征压缩。</em></p>

<div align="center">

| 压缩比 | 指标 | 方法 | 32K | 64K | 96K | 128K |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 16× (t4f4) | Attention | Recon | 1.76× | 2.81× | 3.63× | 5.34× |
| 16× (t4f4) | Attention | SlimKV | 1.94× | 3.81× | 5.08× | 7.34× |
| 16× (t4f4) | Decode | Recon | 1.13× | 1.64× | 2.13× | 2.86× |
| 16× (t4f4) | Decode | SlimKV | 1.22× | 2.04× | 2.63× | 3.38× |
| 32× (t8f4) | Attention | Recon | 1.52× | 2.35× | 3.84× | 4.92× |
| 32× (t8f4) | Attention | SlimKV | 1.62× | 2.54× | 4.36× | 6.09× |
| 32× (t8f4) | Decode | Recon | 1.15× | 1.67× | 2.60× | 3.06× |
| 32× (t8f4) | Decode | SlimKV | 1.23× | 1.75× | 2.82× | 3.49× |

</div>

<p align="center"><em>相对 Full KV 的 attention 与每 token decode 加速（Table 6）。Recon 表示重建版本。</em></p>

Benchmark 脚本和 H20 FA2 局部测量见[性能评测](docs/zh/benchmark.md)。

## 安装

使用 Python 3.11，先安装匹配硬件的 PyTorch，再执行：

```bash
git clone https://github.com/Strandding/SlimKV.git
cd SlimKV
python -m pip install -e .
```

模型代码使用 Transformers 4.40.x，已验证4.40.2。Eager推理不需要 FlashAttention。训练依赖和可选 FA2 扩展见[安装说明](docs/zh/installation.md)。

## 使用

需要训练后的 SlimKV checkpoint，包含权重、配置和 tokenizer。目前尚未公开托管 checkpoint，可按[训练说明](docs/zh/training.md)初始化和训练。

```bash
python -m main.infer \
  --model_name_or_path /path/to/slimkv-checkpoint \
  --prompt 'Explain why a KV cache helps language model inference.' \
  --max_new_tokens 32
```

长上下文使用 `--prompt_file /path/to/context.txt`。示例从 checkpoint 读取 rank 和 Beacon 设置，在 cuda:0 上进行 BF16 greedy 推理。

默认后端存储低维 Beacon K/V，在注意力计算前重建。Llama3 另支持可选的免重建 FA2 decode；训练和 prefill 在原始注意力维度计算。[快速开始](docs/zh/quickstart.md)提供 Python API 和后端选项。

## 复现

从仓库根目录启动，模型、数据和输出路径由 [.env.example](.env.example) 中的环境变量配置。

<table align="center">
  <thead>
    <tr>
      <th align="center">阶段</th>
      <th align="center">入口</th>
      <th align="center">说明</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">低秩初始化</td>
      <td align="center"><code>python -m main.convert_kv_lowrank --help</code></td>
      <td align="center"><a href="docs/zh/training.md">训练</a></td>
    </tr>
    <tr>
      <td align="center">预训练</td>
      <td align="center"><code>bash scripts/training/train_llama_8B.sh</code></td>
      <td align="center"><a href="docs/zh/training.md">训练</a></td>
    </tr>
    <tr>
      <td align="center">微调</td>
      <td align="center"><code>bash scripts/training/finetune_llama_8B.sh</code></td>
      <td align="center"><a href="docs/zh/training.md">训练</a></td>
    </tr>
    <tr>
      <td align="center">LongBench</td>
      <td align="center"><code>python -m main.eval_longbench --help</code></td>
      <td align="center"><a href="docs/zh/evaluation.md">评测</a></td>
    </tr>
    <tr>
      <td align="center">NIAH</td>
      <td align="center"><code>python -m main.eval_needle --help</code></td>
      <td align="center"><a href="docs/zh/evaluation.md">评测</a></td>
    </tr>
    <tr>
      <td align="center">耗时与显存</td>
      <td align="center"><code>python -m tools.benchmark_fa2_decode --help</code></td>
      <td align="center"><a href="docs/zh/benchmark.md">性能评测</a></td>
    </tr>
  </tbody>
</table>

最终 checkpoint 和处理后的训练数据 manifests 仍需发布。[复现指南](docs/zh/reproducibility.md)列出所需资源与实验协议，[模块说明](docs/zh/structure.md)介绍源码组织，[实现状态](docs/zh/implementation_status.md)记录待完成的论文组件。

## 引用

```bibtex
@misc{teng2026slimkv,
  title = {SlimKV: Joint Token-Feature KV Cache Compression with Reconstruction-Free Beacon Attention},
  author = {Teng, Zihan and Zhao, Jiayu and Ren, Wentao and Fan, Minhao and Ma, Tianrui and Chen, Song and Liu, Weichen},
  year = {2026},
  eprint = {2610.02953},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  url = {https://arxiv.org/abs/2610.02953}
}
```

## 致谢与许可

SlimKV 基于 [Activation Beacon](https://github.com/FlagOpen/FlagEmbedding/tree/master/research/Long_LLM/activation_beacon)、Hugging Face Transformers 和 [FlashAttention](https://github.com/Dao-AILab/flash-attention)。归属和许可见[第三方声明](THIRD_PARTY_NOTICES.md)。SlimKV 原创贡献的许可证尚待确定。
