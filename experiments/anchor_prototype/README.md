# SlimKV 早期 Anchor 原型

此目录保留 GitHub 整理前 `98e98b4` 的 Python 源码、脚本与配置。它使用 monkey-patch、`anchor_*` 字段和 Qwen2 实现，与主目录中兼容已有逐层 Beacon checkpoint 的实现独立。

主项目安装仅包含根目录的 `slimkv`、`main`、`tools`，不安装此原型。需要查看原型时从本目录使用其自身入口和环境，不将其 `anchor_*` checkpoint 与主实现的 `beacon_*` checkpoint 混用。

该目录用于保留历史实现，不代表完整论文实现或通过最新模型验收。
