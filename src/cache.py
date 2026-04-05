import torch
import sys
import time
import argparse

def occupy_gpu():
    parser = argparse.ArgumentParser(description="GPU 显存占用工具")
    parser.add_argument('--gpus', type=str, default='4,5,6,7', help='指定 GPU 编号，用逗号分隔 (例如: 0,1,2)')
    parser.add_argument('--mem', type=float, default=25, help='每张显卡占用的显存大小 (单位: GB)')
    
    args = parser.parse_args()
    
    # 解析 GPU 列表
    gpu_ids = [int(x.strip()) for x in args.gpus.split(',')]
    # 计算字节数 (1GB = 1024^3 bytes)
    # float32 占用 4 bytes，所以元素个数 = 字节数 / 4
    num_elements = int((args.mem * 1024**3) / 4)

    tensors = []

    try:
        for gpu_id in gpu_ids:
            if gpu_id >= torch.cuda.device_count():
                print(f"错误: 找不到 GPU {gpu_id}")
                continue
            
            device = torch.device(f"cuda:{gpu_id}")
            # 创建全零张量并移至目标显卡
            print(f"正在尝试在 GPU {gpu_id} 上占用 {args.mem} GB...")
            data = torch.zeros((num_elements,), device=device, dtype=torch.float32)
            tensors.append(data)
            print(f"GPU {gpu_id} 占用成功！")

        print("\n所有指定显存已锁定。按 Ctrl+C 停止占用并释放显存。")
        
        # 保持程序运行
        while True:
            time.sleep(1)
            
    except torch.cuda.OutOfMemoryError:
        print("\n错误: 显存不足，无法申请指定大小的空间。")
    except KeyboardInterrupt:
        print("\n正在释放显存并退出...")

if __name__ == "__main__":
    occupy_gpu()