import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import open3d as o3d
import open3d.core as o3c

import numpy as np

import matplotlib.pyplot as plt

def to_cam_open3d(viewpoint_stack):
    camera_traj = []
    for i, viewpoint_cam in enumerate(viewpoint_stack):
        W = viewpoint_cam.image_width
        H = viewpoint_cam.image_height
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W-1) / 2],
            [0, H / 2, 0, (H-1) / 2],
            [0, 0, 0, 1]]).float().cuda().T
        intrins =  (viewpoint_cam.projection_matrix @ ndc2pix)[:3,:3].T
        intrinsic=o3d.camera.PinholeCameraIntrinsic(
            width=viewpoint_cam.image_width,
            height=viewpoint_cam.image_height,
            cx = intrins[0,2].item(),
            cy = intrins[1,2].item(), 
            fx = intrins[0,0].item(), 
            fy = intrins[1,1].item()
        )

        extrinsic=np.asarray((viewpoint_cam.world_view_transform.T).cpu().numpy())
        camera = o3d.camera.PinholeCameraParameters()
        camera.extrinsic = extrinsic
        camera.intrinsic = intrinsic
        camera_traj.append(camera)

    return camera_traj

def focus_point_fn(poses: np.ndarray) -> np.ndarray:
  """Calculate nearest point to all focal axes in poses."""
  directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
  m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
  mt_m = np.transpose(m, [0, 2, 1]) @ m
  focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
  return focus_pt

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, mesh_name, depth_trunc, voxel_size, sdf_trunc, if_mesh = False):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    rgbmaps = []
    median_depthmaps = []
    viewpoint_stack = []

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering_pkg = render(view, gaussians, pipeline, background)["render"]
        rendering = rendering_pkg[0:3, :, :]
        mean_depth = rendering_pkg[3:4, :, :]
        median_depth = rendering_pkg[4:5, :, :]
        rgbmaps.append(rendering.cpu())
        median_depthmaps.append(median_depth.cpu())
        viewpoint_stack.append(view)
    
    if if_mesh:
        torch.cuda.empty_cache()
        c2ws = np.array([np.linalg.inv(np.asarray((cam.world_view_transform.T).cpu().numpy())) for cam in viewpoint_stack])
        poses = c2ws[:,:3,:] @ np.diag([1, -1, -1, 1])
        center = (focus_point_fn(poses))
        radius = np.linalg.norm(c2ws[:,:3,3] - center, axis=-1).min()
        center = torch.from_numpy(center).float().cuda()

        depth_trunc = (radius * 2.0) if depth_trunc < 0  else depth_trunc
        voxel_size = (depth_trunc / 1024) if voxel_size < 0 else voxel_size
        sdf_trunc = 5.0 * voxel_size if sdf_trunc < 0 else sdf_trunc
        
        device = o3c.Device("CUDA:0" if o3c.cuda.is_available() else "CPU:0")
        vbg = o3d.t.geometry.VoxelBlockGrid(
            attr_names=('tsdf', 'weight', 'color'),
            attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
            attr_channels=((1), (1), (3)),
            voxel_size=voxel_size,
            block_resolution=16,
            block_count=500000,  # 显存允许的话，可以给大一点，避免溢出
            device=device
        )

        for i, cam_o3d in tqdm(enumerate(to_cam_open3d(viewpoint_stack)), desc="TSDF integration progress (Tensor API)"):
            print("i")
            rgb_t = rgbmaps[i].permute(1, 2, 0).clip(0.0, 1.0) * 255.0
            rgb_t = rgb_t.to(torch.uint8).contiguous()
            depth_t = median_depthmaps[i]
            if depth_t.ndim == 3:
                depth_t = depth_t.permute(1, 2, 0)
            if depth_t.ndim == 2:
                depth_t = depth_t.unsqueeze(-1)

            depth_t = depth_t.to(torch.float32).contiguous()

            # --- PyTorch -> Open3D 转换 ---
            # [修改] 显式 .to(device) 确保完全匹配
            o3d_color = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(rgb_t)).to(device)
            o3d_depth = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(depth_t)).to(device)

            # --- 矩阵转换 (核心修复点) ---
            # [修改] 显式指定 dtype=o3c.float32。
            # 这一步非常关键！Numpy 默认是 float64，Open3D CUDA 遇到 float64 会直接报错。
            intrinsic = o3c.Tensor(cam_o3d.intrinsic.intrinsic_matrix, dtype=o3c.float32, device=device)
            extrinsic = o3c.Tensor(cam_o3d.extrinsic, dtype=o3c.float32, device=device)
            frustum_block_coords = vbg.compute_unique_block_coordinates(
                o3d_depth, intrinsic, extrinsic, depth_scale=1.0, depth_max=depth_trunc
            )
            vbg.integrate(
                frustum_block_coords,
                o3d_depth,
                o3d_color,
                intrinsic,
                extrinsic,
                depth_scale=1.0,
                depth_max=depth_trunc
            )
        

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        name = 'fuse.ply'
        depth_trunc = args.depth_trunc
        voxel_size = args.voxel_size
        sdf_trunc = args.sdf_trunc

        render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, name, depth_trunc, voxel_size, sdf_trunc, True)



if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--voxel_size", default=-1.0, type=float, help='Mesh: voxel size for TSDF')
    parser.add_argument("--depth_trunc", default=-1.0, type=float, help='Mesh: Max depth range for TSDF')
    parser.add_argument("--sdf_trunc", default=-1.0, type=float, help='Mesh: truncation value for TSDF')
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # ender_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)
    # 检查 CUDA 是否真的可用
    print(f"Open3D CUDA Available: {o3c.cuda.is_available()}")
    device = o3c.Device("CUDA:0" if o3c.cuda.is_available() else "CPU:0")
    print(f"Current Device: {device}") 
    # 如果这里打印的是 CPU:0，说明你的 Open3D 没认出显卡，或者驱动有问题。