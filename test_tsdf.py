import open3d as o3d
import open3d.core as o3c
import numpy as np

def create_synthetic_depth_map(width, height, fx, fy, cx, cy, sphere_radius=0.5, sphere_center_z=1.5):
    """
    生成一个合成深度图（背景墙 + 前景球体）
    """
    x = np.arange(width)
    y = np.arange(height)
    xx, yy = np.meshgrid(x, y)

    # 计算简单的距离场
    dist_from_center = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    
    # 初始化深度为背景 (2.0米)
    depth = np.full((height, width), 2.0, dtype=np.float32)
    
    # 在图像中心画一个球体 (半径约 150 像素)
    radius_px = 150
    mask = dist_from_center < radius_px
    
    # 简化的球面深度计算
    # z = z_center - sqrt(R^2 - r^2)
    # 注意：这里为了演示简单，直接在像素空间模拟球体隆起
    dome_height = np.sqrt(np.maximum(0, radius_px**2 - dist_from_center[mask]**2)) / 500.0
    depth[mask] = sphere_center_z - dome_height
    
    return depth

def main():
    # 1. 设置设备
    device = o3c.Device("CUDA:0")
    print(f"Using device: {device}")

    # 2. 相机参数
    width, height = 640, 480
    fx, fy = 525.0, 525.0
    cx, cy = 319.5, 239.5
    
    # 内参矩阵 Tensor
    intrinsic = o3c.Tensor([[fx, 0, cx],
                            [0, fy, cy],
                            [0, 0, 1]], dtype=o3c.float64, device=device)
    
    # 外参矩阵 (单位矩阵，代表相机不动)
    extrinsic = o3c.Tensor(np.eye(4), dtype=o3c.float64, device=device)

    # 3. 初始化 VoxelBlockGrid (稀疏 TSDF)
    # 这是实现 GPU TSDF 的核心类
    voxel_size = 0.01
    block_resolution = 16
    block_count = 10000
    
    vbg = o3d.t.geometry.VoxelBlockGrid(
        attr_names=('tsdf', 'weight', 'color'),
        attr_dtypes=(o3c.float32, o3c.uint16, o3c.uint16),
        attr_channels=((1), (1), (3)),
        voxel_size=voxel_size,
        block_resolution=block_resolution,
        block_count=block_count,
        device=device
    )

    print("开始融合...")

    # 4. 循环融合
    for i in range(10):
        # A. 生成数据 (CPU Numpy)
        # 让球体每帧稍微移动一点点，模拟动态扫描或者多帧效果
        move_z = 1.5 + (i * 0.01) 
        depth_np = create_synthetic_depth_map(width, height, fx, fy, cx, cy, sphere_center_z=move_z)
        
        color_np = np.zeros((height, width, 3), dtype=np.uint8)
        color_np[:, :, 0] = 100 + i * 10  # 颜色渐变
        color_np[:, :, 1] = 255 - i * 10

        # B. 传输到 GPU Tensor
        depth_tensor = o3c.Tensor(depth_np, device=device)
        color_tensor = o3c.Tensor(color_np, device=device)

        # C. 融合步骤 (重要)
        # VoxelBlockGrid 是稀疏的，融合前需要计算哪些 Block 在视锥体内
        
        # 1. 计算当前帧视锥体内的 Block 坐标
        depth_scale = 1.0
        depth_max = 3.0
        sdf_trunc = 0.04
        
        frustum_block_coords = vbg.compute_unique_block_coordinates(
            depth_tensor, intrinsic, extrinsic, depth_scale, depth_max, trunc_voxel_multiplier=8.0
        )

        # # 2. 这里的 integrate 函数通常会自动激活哈希表中的块，
        # #    但在某些底层实现中，也可以显式调用 vbg.hashmap().activate_block_coords(...)
        # #    Open3D 的 integrate 接口直接处理：
        # vbg.integrate(
        #     frustum_block_coords, 
        #     depth_tensor, 
        #     color_tensor, 
        #     intrinsic, 
        #     intrinsic, # 某些版本需要两个内参 (depth_intrinsic, color_intrinsic)
        #     extrinsic, 
        #     depth_scale, 
        #     depth_max,
        #     trunc_voxel_multiplier=8.0
        # )
        
        print(f"Frame {i+1} integrated. Active blocks: {vbg.hashmap().size()}")

    # 5. 提取 Mesh
    print("Extracting surface mesh...")
    mesh = vbg.extract_triangle_mesh(weight_threshold=1.0, estimated_vertex_count=100000)
    
    # 6. 转回 Legacy 格式以便可视化
    mesh_legacy = mesh.to_legacy()
    print(f"Result mesh: {len(mesh_legacy.vertices)} vertices.")

    if len(mesh_legacy.vertices) > 0:
        mesh_legacy.compute_vertex_normals()
        o3d.visualization.draw_geometries([mesh_legacy], window_name="VoxelBlockGrid Result")
    else:
        print("Warning: Generated mesh is empty.")

if __name__ == "__main__":
    main()