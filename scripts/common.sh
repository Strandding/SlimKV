#!/usr/bin/env bash
# Shared paths for SlimKV launchers. Source this file; it does not launch jobs.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SLIMKV_ROOT="${PROJECT_ROOT}"
export SLIMKV_DATA_ROOT="${SLIMKV_DATA_ROOT:-${PROJECT_ROOT}/data/datasets}"
export SLIMKV_BASE_MODEL_DIR="${SLIMKV_BASE_MODEL_DIR:-${PROJECT_ROOT}/model}"
export SLIMKV_OUTPUT_DIR="${SLIMKV_OUTPUT_DIR:-${PROJECT_ROOT}/data/outputs}"
cd "${PROJECT_ROOT}" || return 1
