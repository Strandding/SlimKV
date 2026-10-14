# 精度评测

[English](../evaluation.md)

使用训练后的 checkpoint 和本地 [LongBench-v1](https://huggingface.co/datasets/THUDM/LongBench)，每个 task 放在 `<dataset_dir>/<task>.jsonl`。记录数据 revision 和 hash，不要与 v2 混用。

```bash
python -m main.eval_longbench \
  --model_name_or_path /path/to/slimkv-checkpoint \
  --eval_data /path/to/longbench \
  --output_dir data/results/longbench --result_dir smoke \
  --enable_beacon --attn_impl eager --dtype bf16 --chat_template llama-3 \
  --beacon_ratio 8 --beacon_ratio_mix sequence \
  --tasks qasper hotpotqa --max_samples_per_task 2 \
  --sample_seed 42 --sample_shuffle False --batch_size 1
```

这是每任务原始前两行，不是历史 length 2500–6000 的选样。历史 fixtures 用 [benchmark 工具](benchmark.md) 选。评测入口保留现有 max_length/模板/截断规则，比较时要一致。

完整预设：设置 MODEL_NAME_OR_PATH、EVAL_DATA、CUDA_VISIBLE_DEVICES=0、NPROC_PER_NODE=1、BEACON_RATIO=8、ATTN_IMPL=eager，再启动 `scripts/evaluation/eval_longbench_llama31_8B.sh`。预设排除了中文子集，不能称完整双语 LongBench。BEACON_RATIO 是 token 压缩；feature ratio 来自 checkpoint，整体显存不直接按两者乘积降低。

对照使用相同权重分别传 `--beacon_cache_mode full` / `latent`。启用 latent FA2 需相应后端和扩展；QA 建议 B1，因为 variable-length padding 不符合 FA2 条件。独立 batch 会 reset Memory；生成模板、EOS、换行停止、code 补全和 greedy 设置保持一致，浮点后端差异不保证逐 token 一致。

结果目录含每任务 score 首行加预测记录、config.json、run_manifest.json；后者记录配置、源码 SHA、checkpoint config/task 文件 SHA、软件与 GPU。整体 metrics 日志在 output_dir。Config SHA 不等于权重 SHA，需保留 exact shards；用环境变量认证，避免 CLI 凭据进入日志。

其他入口：NIAH 为 `main.eval_needle`（需 haystack、可选 judge/API，目前非论文三 seed 协议）；`main.eval_passkey` 是 synthetic retrieval；`main.eval_lm` 是 perplexity；generation/MMLU/topic 入口需各自 schema。probe/heatmap 用于诊断，不是正式 aggregate quality。
