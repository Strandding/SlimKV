"""Merge separate beacon_k_down_proj and beacon_v_down_proj into shared beacon_kv_down_proj.

Usage:
    python -m main.merge_kv_down_proj --input_dir <existing_lowrank_checkpoint> --output_dir <new_checkpoint>
"""
import argparse
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Existing lowrank checkpoint with separate K/V down_proj")
    parser.add_argument("--output_dir", required=True, help="Output directory for merged checkpoint")
    args = parser.parse_args()

    if os.path.abspath(args.input_dir) == os.path.abspath(args.output_dir):
        raise ValueError("input_dir and output_dir must be different")

    # Copy everything first
    if os.path.exists(args.output_dir):
        raise ValueError(f"output_dir already exists: {args.output_dir}")
    shutil.copytree(args.input_dir, args.output_dir)

    # Find all safetensors files
    st_files = sorted(f for f in os.listdir(args.output_dir) if f.endswith(".safetensors"))
    if not st_files:
        raise ValueError("No .safetensors files found")

    for st_file in st_files:
        path = os.path.join(args.output_dir, st_file)
        state_dict = load_file(path)

        new_state_dict = {}
        k_down_keys = [k for k in state_dict if "beacon_k_down_proj" in k]
        v_down_keys = [k for k in state_dict if "beacon_v_down_proj" in k]

        # Build mapping: beacon_v_down_proj.xxx.weight -> beacon_kv_down_proj.xxx.weight
        v_to_shared = {}
        for vk in v_down_keys:
            shared_key = vk.replace("beacon_v_down_proj", "beacon_kv_down_proj")
            v_to_shared[vk] = shared_key

        k_to_v = {}
        for kk in k_down_keys:
            vk = kk.replace("beacon_k_down_proj", "beacon_v_down_proj")
            k_to_v[kk] = vk

        for key, tensor in state_dict.items():
            if "beacon_k_down_proj" in key:
                # Skip — will be merged into shared
                continue
            elif "beacon_v_down_proj" in key:
                # Average with K's down_proj, rename to shared
                k_key = key.replace("beacon_v_down_proj", "beacon_k_down_proj")
                if k_key in state_dict:
                    avg = (state_dict[k_key] + tensor) / 2
                else:
                    avg = tensor
                shared_key = key.replace("beacon_v_down_proj", "beacon_kv_down_proj")
                new_state_dict[shared_key] = avg
                print(f"  {key} + {k_key} -> {shared_key}")
            else:
                new_state_dict[key] = tensor

        if k_down_keys or v_down_keys:
            save_file(new_state_dict, path)
            print(f"Saved {st_file} ({len(k_down_keys)} K down_proj merged)")

    # Update config.json
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    config["beacon_shared_kv_down_proj"] = True
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"\nDone. Shared checkpoint saved to {args.output_dir}")


if __name__ == "__main__":
    main()
