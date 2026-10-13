# SlimKV 本机与 H20 同步

GitHub 仓库为 `https://github.com/Strandding/SlimKV.git`，默认分支为 `master`。本机工作目录为 `/data/workspace/SlimKV`；H20 工作目录为 `/root/SlimKV`。两端通过相同 Git 提交同步源码。

## 日常同步

在修改代码的一端提交并推送：

```bash
git status -sb
git diff --check
python tools/check_repository.py
git add <需要同步的文件>
git commit -m "描述变更"
git push origin master
```

另一端确认没有未提交修改后执行：

```bash
git pull --ff-only origin master
git status -sb
git rev-parse HEAD
```

若存在本地修改或分支分叉，应先保留并协调修改，再合并。不要使用强制覆盖命令同步工作目录。认证由各端本地的 SSH key 或 credential helper 提供，不把令牌写入 remote URL 或源码。

## 本地产物

`model/`、论文 PDF、`data/outputs/`、`data/results/`、`data/fig/`、数据集和 Python 缓存不参与源码同步。模型内部 `beacon_*` 参数键及现有 checkpoint 名称继续兼容。

首次整理前，本机历史保存在工作区外的 Git bundle 与源码 archive；H20 原代码副本保存在 `/root/SlimKV_migration_backup_20261013`。已有结果和图恢复到新根目录后由 `.gitignore` 排除。

## 结构迁移

| 旧位置或入口 | 当前位置或入口 |
|---|---|
| Python `src` 包 | `slimkv` |
| `src/modeling_beacon.py` | `slimkv/memory.py` |
| `src/latency/loader_latency.py` | `slimkv/latency/loader.py` |
| `main.convert_beacon_kv_lowrank` | `main.convert_kv_lowrank` |
| 根目录训练和评测 shell 脚本 | `scripts/training/`、`scripts/evaluation/` |
| `data/deepspeed/`、`data/config/` | `config/deepspeed/`、`config/data/` |
| `vLLM/` | `scripts/vllm/` |
| 根目录 `train.py` 的 GPU 监控程序 | `tools/watch_gpus.py` |

算法行为与 checkpoint 参数结构保持原有版本；已知实现问题单独记录在 `implementation_status.md`。
