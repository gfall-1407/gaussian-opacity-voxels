import torch
import tinycudann as tcnn
import time
import torch.nn.functional as F
import os

# --- 检查依赖 ---
try:
    import skimage.measure
    import plotly.graph_objects as go
    import kaleido # 导入 kaleido 来激活 .write_image()
except ImportError:
    print("\n--- 警告 ---")
    print("你的依赖没装全。请运行:")
    print("pip install scikit-image plotly kaleido")
    print("--------------\n")
    exit()

# --- 0. 检查 GPU ---
if not torch.cuda.is_available():
    print("没找到 CUDA GPU。tiny-cuda-nn 睡大觉了。")
    exit()
device = "cuda:0"
print(f"在 {device} 上运行...")

# --- 1. TCNN 配置 ---
# 我们要学 SDF (1维输出)
config = {
    "encoding": {
        "otype": "HashGrid",
        "n_levels": 16,
        "n_features_per_level": 2, # F=2, 经典默认值
        "log2_hashmap_size": 19,
        "base_resolution": 16,
        "per_level_scale": 1.5 # 1.5-2.0 都行
    },
    "network": {
        "otype": "FullyFusedMLP",
        "activation": "ReLU",
        "output_activation": "None", # 直接输出SDF值
        "n_neurons": 64,
        "n_hidden_layers": 1,
    }
}

# 实例化SDF场 (3维输入, 1维输出)
sdf_field = tcnn.Network(
    n_input_dims=3,
    n_output_dims=1, # s 值
    network_config=config
).to(device)

print(f"SDF 场创建完毕。参数量: {sum(p.numel() for p in sdf_field.parameters())}")

# --- 2. 训练 ---
optimizer = torch.optim.Adam(sdf_field.parameters(), lr=1e-3)
N = 10000  # 每轮训练的点
sphere_center = torch.tensor([0.5, 0.5, 0.5], device=device)
sphere_radius = 0.3 # 球的半径

print("开始训练一个球体 SDF...")
start_time = time.time()
for i in range(10000): # 训练200轮
    optimizer.zero_grad()

    # 在 [0, 1] 空间中随机采样点
    query_points = torch.rand(N, 3, device=device)

    # 计算"真实"的SDF值: 距离 - 半径
    distances = torch.norm(query_points - sphere_center, dim=-1, keepdim=True)
    true_sdf = distances - sphere_radius

    # 查询TCNN
    predicted_sdf = sdf_field(query_points)
    
    # 用 .float() 解决 FP16/FP32 冲突, 然后算 Loss
    loss = F.mse_loss(predicted_sdf.float(), true_sdf.float())
    
    loss.backward()
    optimizer.step()
    
    if (i + 1) % 20 == 0:
        print(f"Step {i+1}/200, Loss: {loss.item():.7f}")

print(f"训练完成! 耗时: {time.time() - start_time:.2f} 秒")

# --- 3. 可视化 (提取网格并保存) ---
print("开始生成可视化网格...")

GRID_SIZE = 128 # 128^3 的体素网格
t = torch.linspace(0, 1, GRID_SIZE, device=device)
# 创建网格坐标
grid_x, grid_y, grid_z = torch.meshgrid(t, t, t, indexing="ij")
grid_xyz = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)

sdf_values = []
with torch.no_grad():
    # 批量查询，防止显存爆炸
    for batch in grid_xyz.split(8192):
        sdf_values.append(sdf_field(batch).float())

# 把结果拼回来，放到CPU上，转成Numpy
sdf_volume = torch.cat(sdf_values).reshape(GRID_SIZE, GRID_SIZE, GRID_SIZE).cpu().numpy()
print("SDF 体素场已生成。")

# 运行 Marching Cubes 算法
# 我们要提取 s=0 的等势面
print("正在运行 Marching Cubes...")
verts, faces, _, _ = skimage.measure.marching_cubes(sdf_volume, level=0.0)

# --- 4. 用 Plotly 绘图并保存 ---
print("正在用 Plotly 生成3D网格...")
fig = go.Figure(data=[
    go.Mesh3d(
        x=verts[:, 0],
        y=verts[:, 1],
        z=verts[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        opacity=0.9,
        color='lightblue', # 换个清爽的颜色
        name='SDF Isosurface'
    )
])

fig.update_layout(
    title='TCNN 学到的SDF (s=0)',
    scene=dict(
        xaxis_title='X (0 -> 128)',
        yaxis_title='Y (0 -> 128)',
        zaxis_title='Z (0 -> 128)',
        aspectratio=dict(x=1, y=1, z=1) # 保证比例正确
    )
)

output_filename = "tcnn_sdf_sphere.png"
print(f"正在保存图像到 '{output_filename}'...")

# 保存为PNG图片
try:
    fig.write_image(output_filename, width=1000, height=800, scale=2)
    print(f"成功保存! 文件在: {os.path.abspath(output_filename)}")
except Exception as e:
    print("\n--- 保存失败 ---")
    print(f"错误: {e}")
    print("请确保 'kaleido' 库已正确安装 (pip install kaleido)")