# Source synchronization

[中文](zh/synchronization.md)
The source remote is `https://github.com/Strandding/SlimKV.git`, branch `master`. Clone/pull the same commit on each machine; configure SSH or credential helpers locally and never put credentials in remote URLs.

```bash
git status -sb
git diff --check
python tools/check_repository.py
git add <reviewed-files>
git commit -m 'Describe the verified change'
git push origin master
```

On the other machine, preserve any uncommitted work before:

```bash
git pull --ff-only origin master
git rev-parse HEAD
git status -sb
```

Compare source manifests as well as HEAD if a worktree is dirty. Never force-overwrite a working directory to synchronize it. Checkpoints, datasets, paper drafts and generated results remain local; transfer them separately with manifests if needed. Existing `beacon_*` keys and checkpoint directory names remain compatible.

## Historical path changes

| Earlier entry | Current entry |
|---|---|
| Python `src` package | `slimkv` |
| `src/modeling_beacon.py` | `slimkv/memory.py` |
| `src/latency/loader_latency.py` | `slimkv/latency/loader.py` |
| `main.convert_beacon_kv_lowrank` | `main.convert_kv_lowrank` |
| Root training/evaluation shell scripts | `scripts/training`, `scripts/evaluation` |
| `data/deepspeed`, `data/config` | `config/deepspeed`, `config/data` |
| `vLLM` | `scripts/vllm` |
| Root GPU monitoring script | `tools/watch_gpus.py` |

The documentation refresh adds guides and maintenance checks without further broad path migrations.
