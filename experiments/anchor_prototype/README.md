# Historical Anchor prototype

[中文](README.zh-CN.md)

This directory preserves the Python source, launchers and configuration from commit `98e98b4`, before the main repository cleanup. It uses monkey-patching, `anchor_*` fields and an independent Qwen2 implementation.

Root package discovery excludes this directory. Its checkpoints are not interchangeable with the main implementation's `beacon_*` checkpoints. Run its own entry points from this directory in a separate environment only when studying the prototype. `MODEL_NAME_OR_PATH` overrides the example local model location in its scripts.

This archive preserves useful research code; it is not the current paper reproduction path and has not passed the latest main-model validation. See [implementation status](../../docs/implementation_status.md).
