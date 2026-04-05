"""
多卡 NCCL 通信测试脚本
用法: torchrun --nproc_per_node=3 test_nccl.py
"""
import os
import time
import torch
import torch.distributed as dist


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "N/A").split(",")[local_rank] \
        if "CUDA_VISIBLE_DEVICES" in os.environ else str(local_rank)

    print(f"[Rank {rank}] 物理 GPU {gpu_id} | {torch.cuda.get_device_name(device)} | "
          f"显存: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
    dist.barrier()

    # ---------- 测试 1: 小张量 allreduce ----------
    print(f"[Rank {rank}] 测试 1: 小张量 allreduce (1024 元素)...", flush=True)
    t = torch.ones(1024, device=device) * (rank + 1)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    expected = sum(range(1, world_size + 1))
    assert torch.allclose(t, torch.full_like(t, expected)), f"Rank {rank} 小张量 allreduce 结果错误!"
    print(f"[Rank {rank}] 测试 1 通过 ✓", flush=True)
    dist.barrier()

    # ---------- 测试 2: 大张量 allreduce (模拟梯度同步) ----------
    sizes = [100_000_000, 462_551_040]  # 100M 和实际训练中的大小
    for numel in sizes:
        mem_gb = numel * 4 / 1024**3  # bf16=2, float32=4
        print(f"[Rank {rank}] 测试 2: 大张量 allreduce ({numel:,} 元素, ~{mem_gb:.1f} GB float32)...", flush=True)
        dist.barrier()

        t = torch.randn(numel, device=device)
        torch.cuda.synchronize()
        start = time.time()
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        elapsed = time.time() - start

        bw = numel * 4 / elapsed / 1e9  # GB/s
        print(f"[Rank {rank}] 测试 2 通过 ✓ ({numel:,} 元素) 耗时: {elapsed:.3f}s, 带宽: {bw:.1f} GB/s", flush=True)
        del t
        torch.cuda.empty_cache()
        dist.barrier()

    # ---------- 测试 3: broadcast (逐卡广播) ----------
    print(f"[Rank {rank}] 测试 3: broadcast 测试...", flush=True)
    dist.barrier()
    for src in range(world_size):
        t = torch.ones(1024 * 1024, device=device) * (src + 1) if rank == src \
            else torch.zeros(1024 * 1024, device=device)
        dist.broadcast(t, src=src)
        assert torch.allclose(t, torch.full_like(t, src + 1)), \
            f"Broadcast 从 rank {src} 失败!"
        print(f"[Rank {rank}] broadcast from rank {src} 通过 ✓", flush=True)
        dist.barrier()

    # ---------- 测试 3b: reduce_scatter ----------
    print(f"[Rank {rank}] 测试 3b: reduce_scatter...", flush=True)
    input_t = torch.ones(world_size * 1024 * 1024, device=device) * (rank + 1)
    output_t = torch.zeros(1024 * 1024, device=device)
    dist.reduce_scatter_tensor(output_t, input_t, op=dist.ReduceOp.SUM)
    assert torch.allclose(output_t, torch.full_like(output_t, expected)), \
        f"Rank {rank} reduce_scatter 结果错误!"
    print(f"[Rank {rank}] 测试 3b 通过 ✓", flush=True)
    dist.barrier()

    # ---------- 测试 4: 多轮连续 allreduce (模拟训练循环) ----------
    n_iters = 20
    print(f"[Rank {rank}] 测试 4: 连续 {n_iters} 轮 allreduce (462M 元素)...", flush=True)
    dist.barrier()
    t = torch.randn(462_551_040, device=device)
    times = []
    for i in range(n_iters):
        torch.cuda.synchronize()
        start = time.time()
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        times.append(elapsed)
        if rank == 0 and (i + 1) % 5 == 0:
            print(f"  第 {i+1}/{n_iters} 轮: {elapsed:.3f}s", flush=True)
    del t
    torch.cuda.empty_cache()

    avg = sum(times) / len(times)
    mx = max(times)
    print(f"[Rank {rank}] 测试 4 通过 ✓ 平均: {avg:.3f}s, 最大: {mx:.3f}s", flush=True)
    dist.barrier()

    # ---------- 汇总 ----------
    if rank == 0:
        print("\n========== 所有测试通过 ==========")
        if mx > 10:
            print(f"⚠ 警告: 最大单轮耗时 {mx:.1f}s，可能存在通信瓶颈")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
