#!/usr/bin/env python3
import argparse
import datetime as dt
import os
import socket
import time

import torch
import torch.distributed as dist


def parse_args():
    parser = argparse.ArgumentParser("NCCL communication smoke/perf test")
    parser.add_argument("--rounds", type=int, default=20, help="Number of all_reduce rounds for perf test.")
    parser.add_argument(
        "--tensor-mb",
        type=int,
        default=64,
        help="Tensor size per rank in MB for perf test (float32).",
    )
    parser.add_argument("--warmup-rounds", type=int, default=5, help="Warmup rounds before timing.")
    return parser.parse_args()


def main():
    args = parse_args()

    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ or "LOCAL_RANK" not in os.environ:
        raise RuntimeError("Please launch with torchrun so RANK/WORLD_SIZE/LOCAL_RANK are set.")

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    host = socket.gethostname()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    timeout = dt.timedelta(minutes=10)
    dist.init_process_group(backend="nccl", timeout=timeout)

    # 1) Basic correctness check.
    x = torch.tensor([rank + 1.0], device=device)
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    expected = world_size * (world_size + 1) / 2.0
    ok = abs(x.item() - expected) < 1e-5

    # Gather host/device info.
    local_name = torch.cuda.get_device_name(local_rank)
    infos = [None for _ in range(world_size)]
    dist.all_gather_object(
        infos,
        {
            "rank": rank,
            "host": host,
            "local_rank": local_rank,
            "device": local_name,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        },
    )

    # 2) Simple throughput test using repeated all_reduce.
    numel = args.tensor_mb * 1024 * 1024 // 4  # float32 -> 4 bytes
    buf = torch.ones(numel, dtype=torch.float32, device=device)

    for _ in range(max(args.warmup_rounds, 0)):
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize(device)
    dist.barrier()

    start = time.time()
    for _ in range(args.rounds):
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize(device)
    dist.barrier()
    elapsed = time.time() - start

    bytes_per_round = args.tensor_mb * 1024 * 1024
    # Approximate algorithmic per-rank BW (GiB/s)
    bw_gib_s = (bytes_per_round * args.rounds) / max(elapsed, 1e-9) / (1024**3)

    if rank == 0:
        print("========== NCCL Comm Test ==========")
        print(f"WORLD_SIZE={world_size}")
        print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}")
        print("Rank mapping:")
        for item in sorted(infos, key=lambda z: z["rank"]):
            print(
                f"  rank={item['rank']} host={item['host']} local_rank={item['local_rank']} "
                f"device={item['device']}"
            )
        print(f"Correctness all_reduce: {'PASS' if ok else 'FAIL'} (value={x.item():.3f}, expected={expected:.3f})")
        print(
            f"Perf all_reduce: rounds={args.rounds}, tensor_mb={args.tensor_mb}, "
            f"elapsed={elapsed:.3f}s, approx_per_rank_bw={bw_gib_s:.3f} GiB/s"
        )
        print("====================================")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
