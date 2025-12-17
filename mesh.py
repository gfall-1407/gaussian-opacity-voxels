#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

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
import numpy as np

import matplotlib.pyplot as plt

def save_depth_2_point_cloud(depth, FoVx, FoVy, filename):
    """
    image: (3, H, W) numpy array, 范围 [0, 1] 或 [0, 255]
    depth: (H, W) numpy array
    viewpoint_cam: 包含相机参数的对象
    filename: 保存路径 (.ply)
    """
    # 1. 获取图像尺寸
    H, W = depth.shape
    
    # 2. 计算相机内参 (焦距)
    # Gaussian Splatting 代码中通常存储的是 FoVx 和 FoVy
    fx = W / (2 * np.tan(FoVx / 2))
    fy = H / (2 * np.tan(FoVy / 2))
    cx = W / 2.0
    cy = H / 2.0

    # 3. 创建像素坐标网格
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    u = u.flatten()
    v = v.flatten()
    z = depth.flatten()

    # 4. 过滤掉深度无效的点 (比如深度极小或极大)
    valid_mask = (z > 0.00) # 根据场景调整阈值
    u = u[valid_mask]
    v = v[valid_mask]
    z = z[valid_mask]

    # 5. 反投影：从 2D 像素 -> 3D 相机坐标系
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    # 注意：Gaussian Splatting 的相机坐标系通常是 Y向下，Z向前
    # 如果生成的点云上下颠倒，可以尝试 y = -y 
    
    # 堆叠 xyz
    xyz = np.stack([x, y, z], axis=1)

    # 6. 转世界坐标系 (可选)
    # 如果你想看它在世界中的位置，需要乘以相机外参的逆 (c2w)
    # viewpoint_cam.world_view_transform 通常是 w2c
    # c2w = torch.inverse(viewpoint_cam.world_view_transform).cpu().numpy()
    # R = c2w[:3, :3]
    # t = c2w[:3, 3]
    # xyz = xyz @ R.T + t

    # 8. 写入 PLY 文件头
    num_points = xyz.shape[0]
    header = f"""ply
        format ascii 1.0
        element vertex {num_points}
        property float x
        property float y
        property float z
        end_header
    """
    
    # 9. 保存数据
    with open(filename, 'w') as f:
        f.write(header)
        for i in range(num_points):
            f.write(f"{xyz[i,0]:.4f} {xyz[i,1]:.4f} {xyz[i,2]:.4f}\n")
            
    print(f"Point cloud saved to {filename}")


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

        # d_np = mean_depth.detach().cpu().numpy()
        # d_map = d_np.squeeze()
        # plt.imsave('test/mean_depth_' + str(idx) +'.png', d_map, cmap='plasma')
        # save_depth_2_point_cloud(d_map, view.FoVx, view.FoVy, 'test/d_' + str(idx) +'.ply')
        
        # md_np = median_depth.detach().cpu().numpy()
        # md_map = md_np.squeeze()
        # plt.imsave('test/median_depth_' + str(idx) +'.png', md_map, cmap='plasma')
        # save_depth_2_point_cloud(md_map, view.FoVx, view.FoVy, 'test/md_' + str(idx) +'.ply')
    
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
        
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length = voxel_size,
            sdf_trunc = sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

        for i, cam_o3d in tqdm(enumerate(to_cam_open3d(viewpoint_stack)), desc="TSDF integration progress"):
            rgb = rgbmaps[i]
            depth = median_depthmaps[i]
            
            # if we have mask provided, use it
            # if mask_backgrond and (viewpoint_stack[i].gt_alpha_mask is not None):
            #     depth[(viewpoint_stack[i].gt_alpha_mask < 0.5)] = 0

            # make open3d rgbd
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.asarray(np.clip(rgb.permute(1,2,0).cpu().numpy(), 0.0, 1.0) * 255, order="C", dtype=np.uint8)),
                o3d.geometry.Image(np.asarray(depth.permute(1,2,0).cpu().numpy(), order="C")),
                depth_trunc = depth_trunc, convert_rgb_to_intensity=False,
                depth_scale = 1.0
            )

            volume.integrate(rgbd, intrinsic=cam_o3d.intrinsic, extrinsic=cam_o3d.extrinsic)
        mesh = volume.extract_triangle_mesh()
        o3d.io.write_triangle_mesh(os.path.join('test/', 'fuse.ply'), mesh)

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

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)