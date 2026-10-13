import torch
import time
import multiprocessing
from pynvml import *

# 配置参数
TARGET_GPUS = [0,1,2,3,4, 5, 6, 7]
MEM_THRESHOLD_GB = 5  # 阈值：低于5GB认为显卡已空闲
POLLING_INTERVAL = 6 # 每60秒检测一次

def occupy_gpu(gpu_id):
    """占用特定显卡的显存和算力"""
    try:
        torch.cuda.set_device(gpu_id)
        print(f"[{time.strftime('%H:%M:%S')}] 🚀 开始占用显卡 {gpu_id}...")
        
        # 1. 占用显存：通过不断分配张量直到显存溢出边缘
        tensors = []
        try:
            while True:
                # 每次申请 1GB 的张量
                tensors.append(torch.randn(1024 * 1024 * 256, device=f'cuda:{gpu_id}'))
        except RuntimeError: 
            print(f"[{time.strftime('%H:%M:%S')}] ✅ 显卡 {gpu_id} 显存已填满。")

        # 2. 占用算力：进行无限循环的矩阵乘法
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
                
                # 转换为 GB
                used_gb = info.used / 1024**3
                
                # 如果占用低于阈值且该卡当前没有占用进程在运行
                if used_gb < MEM_THRESHOLD_GB:
                    if gpu_id not in active_processes or not active_processes[gpu_id].is_alive():
                        p = multiprocessing.Process(target=occupy_gpu, args=(gpu_id,))
                        p.daemon = True
                        p.start()
                        active_processes[gpu_id] = p
                        print(f"检测到显卡 {gpu_id} 空闲 (当前已用: {used_gb:.2f}GB)，已启动占用程序。")
                else:
                    # 如果手动停掉了占用，这里可以保持逻辑简单，不自动杀死进程
                    pass
            
            time.sleep(POLLING_INTERVAL)
    except KeyboardInterrupt:
        print("\n正在停止监听...")
    finally:
        nvmlShutdown()

if __name__ == "__main__":
    # 使用 multiprocessing 必须在 if __name__ == "__main__" 下
    monitor_and_run()
