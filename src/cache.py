import torch
import os
import sys
import time
import argparse
import threading


def _get_gpu_ids():
    """Read GPU ids from CUDA_VISIBLE_DEVICES or fall back to all available."""
    env = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if env:
        return list(range(len(env.split(","))))
    return list(range(torch.cuda.device_count()))


def _allocate(device, target_bytes):
    """Allocate *target_bytes* on *device* in chunks. Return list of tensors."""
    chunk_elements = 64 * 1024 * 1024  # 256 MB per chunk (float32)
    chunk_bytes = chunk_elements * 4
    tensors = []
    allocated = 0
    while allocated + chunk_bytes <= target_bytes:
        tensors.append(torch.zeros(chunk_elements, device=device, dtype=torch.float32))
        allocated += chunk_bytes
    remaining = target_bytes - allocated
    if remaining > 4:
        n = int(remaining // 4)
        tensors.append(torch.zeros(n, device=device, dtype=torch.float32))
        allocated += n * 4
    return tensors, allocated


def _occupy_one_gpu(gpu_id, reserve_mb=512, mem_gb=None, compute=True):
    """Occupy one GPU.

    Args:
        reserve_mb: MB to leave free (used when mem_gb is None).
        mem_gb: If set, allocate exactly this many GB instead of filling up.
        compute: Whether to spin matmul to consume compute.
    """
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)

    free_bytes, total_bytes = torch.cuda.mem_get_info(device)

    if mem_gb is not None:
        target_bytes = int(mem_gb * 1024 ** 3)
        if target_bytes > free_bytes:
            print(f"[GPU {gpu_id}] Requested {mem_gb:.1f} GB but only "
                  f"{free_bytes / 1024**3:.1f} GB free, allocating as much as possible.")
            target_bytes = free_bytes - 256 * 1024 ** 2  # keep 256 MB safety margin
    else:
        target_bytes = free_bytes - reserve_mb * 1024 ** 2

    if target_bytes <= 0:
        print(f"[GPU {gpu_id}] Only {free_bytes / 1024**2:.0f} MB free, skip.")
        return

    tensors, allocated = _allocate(device, target_bytes)

    alloc_gb = allocated / 1024 ** 3
    total_gb = total_bytes / 1024 ** 3
    print(f"[GPU {gpu_id}] Occupied {alloc_gb:.2f} / {total_gb:.2f} GB")

    if not compute:
        # Hold memory only — just sleep forever.
        while True:
            time.sleep(60)

    # Compute loop: keep GPU busy with matmul.
    size = 2048
    a = torch.randn(size, size, device=device, dtype=torch.float16)
    b = torch.randn(size, size, device=device, dtype=torch.float16)
    while True:
        torch.mm(a, b)


def occupy_gpu():
    parser = argparse.ArgumentParser(description="GPU memory & compute occupation tool")
    parser.add_argument("--gpus", type=str, default=None,
                        help="GPU indices (0-based within CUDA_VISIBLE_DEVICES). "
                             "Default: all visible GPUs.")
    parser.add_argument("--reserve-mb", type=int, default=512,
                        help="MB to leave free per GPU (default: 512, ignored when --mem is set)")
    parser.add_argument("--mem", type=float, default=None,
                        help="Exact GB to occupy per GPU (e.g. --mem 20). "
                             "Default: fill up as much as possible.")
    parser.add_argument("--no-compute", action="store_true",
                        help="Only occupy memory, do NOT consume compute.")
    args = parser.parse_args()

    # Initialize CUDA in the main thread first to avoid thread init errors.
    if not torch.cuda.is_available():
        print("CUDA not available.")
        return
    _ = torch.cuda.device_count()

    if args.gpus is not None:
        gpu_ids = [int(x.strip()) for x in args.gpus.split(",")]
    else:
        gpu_ids = _get_gpu_ids()

    if not gpu_ids:
        print("No GPUs found.")
        return

    mode_desc = f"{args.mem:.1f} GB each" if args.mem else f"max (reserve {args.reserve_mb} MB)"
    compute_desc = "no compute" if args.no_compute else "with compute"
    print(f"Occupying GPUs: {gpu_ids} — {mode_desc}, {compute_desc}")

    threads = []
    for gid in gpu_ids:
        if gid >= torch.cuda.device_count():
            print(f"GPU {gid} not available (only {torch.cuda.device_count()} visible), skip.")
            continue
        # Pre-init CUDA context on each device from main thread.
        torch.cuda.set_device(gid)
        torch.zeros(1, device=f"cuda:{gid}")
    for gid in gpu_ids:
        if gid >= torch.cuda.device_count():
            continue
        t = threading.Thread(target=_occupy_one_gpu,
                             args=(gid, args.reserve_mb, args.mem, not args.no_compute),
                             daemon=True)
        t.start()
        threads.append(t)

    if not threads:
        print("No GPUs occupied.")
        return

    print(f"\nAll {len(threads)} GPUs locked. Press Ctrl+C to release.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nReleasing GPUs and exiting...")


if __name__ == "__main__":
    occupy_gpu()
