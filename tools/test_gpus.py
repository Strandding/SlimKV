import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
import os

def setup(rank, world_size, gpu_ids):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Map the logical rank to a physical GPU ID.
    physical_gpu = gpu_ids[rank]
    torch.cuda.set_device(physical_gpu)
    
    # Initialize the process group with the NCCL GPU communication backend.
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """Destroy the distributed environment."""
    dist.destroy_process_group()

def test_multi_gpu(rank, world_size, gpu_ids):
    # 1. Initialize.
    setup(rank, world_size, gpu_ids)
    physical_gpu = gpu_ids[rank]
    device_name = torch.cuda.get_device_name(physical_gpu)
    print(f"[进程 {rank}] 成功绑定物理显卡 {physical_gpu}: {device_name}")

    # 2. Test GPU memory allocation.
    try:
        # Allocate a 1024x1024 matrix on the selected GPU.
        tensor = torch.ones(1024, 1024).cuda(physical_gpu)
        print(f"[进程 {rank}] 显存分配正常。")
    except Exception as e:
        print(f"[进程 {rank}] 显存分配失败: {e}")
        cleanup()
        return

    # 3. Test multi-GPU NCCL communication (All-Reduce).
    # Each process creates a scalar tensor with value 1.
    comm_tensor = torch.ones(1).cuda(physical_gpu)
    
    # Wait until all processes reach this barrier.
    dist.barrier()
    
    # Perform All-Reduce with a sum operation.
    # Successful communication produces world_size on every GPU.
    dist.all_reduce(comm_tensor, op=dist.ReduceOp.SUM)

    if comm_tensor.item() == world_size:
        print(f"[进程 {rank}] NCCL 通信正常！All-Reduce 结果: {comm_tensor.item()}")
    else:
        print(f"[进程 {rank}] NCCL 通信异常。预期值 {world_size}，实际值 {comm_tensor.item()}")

    # 4. Clean up.
    cleanup()

def main():
    parser = argparse.ArgumentParser(description="多 GPU 显存及 NCCL 通信测试工具")
    parser.add_argument('--gpus', type=str, required=True, 
                        help="需要测试的物理 GPU ID 列表，用逗号分隔，例如: '0,1,3'")
    args = parser.parse_args()

    # Parse the requested GPU IDs.
    try:
        gpu_ids = [int(id.strip()) for id in args.gpus.split(',')]
    except ValueError:
        print("错误: GPU ID 格式不正确，请确保使用逗号分隔的数字，例如 '0,1'。")
        return

    world_size = len(gpu_ids)
    
    if world_size < 2:
        print("提示: 至少需要指定 2 张显卡才能进行多卡通信测试。")
        return

    # Check CUDA availability and validate the GPU IDs.
    if not torch.cuda.is_available():
        print("错误: 当前环境无法检测到 CUDA，请检查显卡驱动或 PyTorch 安装。")
        return
        
    max_gpus = torch.cuda.device_count()
    for gid in gpu_ids:
        if gid >= max_gpus:
            print(f"错误: 指定的 GPU ID {gid} 越界。系统共检测到 {max_gpus} 张卡 (ID: 0 到 {max_gpus-1})。")
            return

    print(f"=========================================")
    print(f"开始测试，参与测试的显卡 ID: {gpu_ids} (共 {world_size} 张)")
    print(f"=========================================")
    
    # Launch the multiprocess test with multiprocessing.
    mp.spawn(test_multi_gpu,
             args=(world_size, gpu_ids),
             nprocs=world_size,
             join=True)
    
    print(f"=========================================")
    print("测试流程结束。如果上述进程均报告'NCCL 通信正常'，则多卡状态良好。")

if __name__ == "__main__":
    main()