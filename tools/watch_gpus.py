import torch
import time
import multiprocessing
from pynvml import *

# Configuration.
TARGET_GPUS = [0,1,2,3,4, 5, 6, 7]
MEM_THRESHOLD_GB = 5  # Treat GPU memory usage below 5 GB as idle.
POLLING_INTERVAL = 6 # Check every 60 seconds.

def occupy_gpu(gpu_id):
    """Hold memory and compute resources on one GPU."""
    try:
        torch.cuda.set_device(gpu_id)
        print(f"[{time.strftime('%H:%M:%S')}] 🚀 开始占用显卡 {gpu_id}...")
        
        # 1. Hold GPU memory by allocating tensors until allocation fails.
        tensors = []
        try:
            while True:
                # Allocate 1 GB per tensor.
                tensors.append(torch.randn(1024 * 1024 * 256, device=f'cuda:{gpu_id}'))
        except RuntimeError: 
            print(f"[{time.strftime('%H:%M:%S')}] ✅ 显卡 {gpu_id} 显存已填满。")

        # 2. Hold compute resources with repeated matrix multiplication.
        a = torch.randn(5000, 5000, device=f'cuda:{gpu_id}')
        b = torch.randn(5000, 5000, device=f'cuda:{gpu_id}')
        
        while True:
            torch.matmul(a, b)
            
    except Exception as e:
        print(f"显卡 {gpu_id} 占用过程中出错: {e}")

def monitor_and_run():
    nvmlInit()
    active_processes = {}

    print(f"监听启动：目标显卡 {TARGET_GPUS}，阈值 {MEM_THRESHOLD_GB}GB")
    
    try:
        while True:
            for gpu_id in TARGET_GPUS:
                handle = nvmlDeviceGetHandleByIndex(gpu_id)
                info = nvmlDeviceGetMemoryInfo(handle)
                
                # Convert to GB.
                used_gb = info.used / 1024**3
                
                # Start only below the threshold and without an existing holder process.
                if used_gb < MEM_THRESHOLD_GB:
                    if gpu_id not in active_processes or not active_processes[gpu_id].is_alive():
                        p = multiprocessing.Process(target=occupy_gpu, args=(gpu_id,))
                        p.daemon = True
                        p.start()
                        active_processes[gpu_id] = p
                        print(f"检测到显卡 {gpu_id} 空闲 (当前已用: {used_gb:.2f}GB)，已启动占用程序。")
                else:
                    # Keep the process-management logic simple; do not terminate processes automatically.
                    pass
            
            time.sleep(POLLING_INTERVAL)
    except KeyboardInterrupt:
        print("\n正在停止监听...")
    finally:
        nvmlShutdown()

if __name__ == "__main__":
    # Use multiprocessing only under the __main__ guard.
    monitor_and_run()
