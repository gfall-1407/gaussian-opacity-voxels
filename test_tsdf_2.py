import os
import sys

# 专门针对 Windows 用户的补丁
if os.name == 'nt':
    # 你的 CUDA bin 路径
    cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin"
    
    if os.path.exists(cuda_path):
        try:
            # 这是一个 Python 3.8+ 特有的函数，专门解决找不到 DLL 的问题
            os.add_dll_directory(cuda_path)
            print(f"[System] 已成功添加 CUDA 路径: {cuda_path}")
        except Exception as e:
            print(f"[Warning] 添加 CUDA 路径失败: {e}")
    else:
        print(f"[Error] 找不到路径: {cuda_path}，请检查是否正确安装 CUDA Toolkit")

# 必须在上面那段代码之后，再导入 open3d 和 torch
import open3d as o3d
import open3d.core as o3c
import torch

print(f"Open3D CUDA: {o3c.cuda.is_available()}")