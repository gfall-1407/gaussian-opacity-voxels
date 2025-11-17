import torch
import tinycudann as tcnn
import time
import torch.nn.functional as F
import os

import skimage.measure
import plotly.graph_objects as go
import trimesh

device = "cuda:0"

config = {
    "encoding": {
        "otype": "HashGrid",
        "n_levels": 16,
        "n_features_per_level": 2,
        "log2_hashmap_size": 19,
        "base_resolution": 16,
        "per_level_scale": 1.5
    },
    "network": {
        "otype": "FullyFusedMLP",
        "activation": "ReLU",
        "output_activation": "None",
        "n_neurons": 64,
        "n_hidden_layers": 1,
    }
}
sdf_field = tcnn.Network(
    n_input_dims=3,
    n_output_dims=1,
    network_config=config
).to(device)

print(f"SDF 场创建完毕。参数量: {sum(p.numel() for p in sdf_field.parameters())}")

optimizer = torch.optim.Adam(sdf_field.parameters(), lr=1e-3)
N = 10000
sphere_center = torch.tensor([0.5, 0.5, 0.5], device=device)
sphere_radius = 0.3

start_time = time.time()
for i in range(N):
    optimizer.zero_grad()
    query_points = torch.rand(N, 3, device=device)
    distances = torch.norm(query_points - sphere_center, dim=-1, keepdim=True)
    true_sdf = distances - sphere_radius
    predicted_sdf = sdf_field(query_points)
    loss = F.mse_loss(predicted_sdf.float(), true_sdf.float())
    loss.backward()
    optimizer.step()
    if (i + 1) % 20 == 0:
        print(f"Step {i+1}/200, Loss: {loss.item():.7f}")

GRID_SIZE = 512
t = torch.linspace(0, 1, GRID_SIZE, device=device)
grid_x, grid_y, grid_z = torch.meshgrid(t, t, t, indexing="ij")
grid_xyz = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)
sdf_values = []
with torch.no_grad():
    for batch in grid_xyz.split(8192):
        sdf_values.append(sdf_field(batch).float())
sdf_volume = torch.cat(sdf_values).reshape(GRID_SIZE, GRID_SIZE, GRID_SIZE).cpu().numpy()
verts, faces, _, _ = skimage.measure.marching_cubes(sdf_volume, level=0.0)
mesh = trimesh.Trimesh(vertices=verts, faces=faces)
mesh.export("sphere_sdf_mesh.ply")