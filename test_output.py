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

from flask import json
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

import json

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
    T_05_mean = 0
    T_05_min = 0
    T_05_max = 0
    T_05_alpha_mean = 0
    T_05_alpha_min = 0
    T_05_alpha_max = 0
    T_max_mean = 0
    T_max_min = 0
    T_max_max = 0
    alpha_max_mean = 0
    alpha_max_min = 0
    alpha_max_max = 0
    count = 0
    
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        image = rendering[0:3, :, :]
        depth = rendering[3:4, :, :]
        T_05_np = rendering[7, :, :].cpu().numpy()
        T_05_np = T_05_np[T_05_np!=0]
        T_05_alpha_np = rendering[8, :, :].cpu().numpy()
        T_05_alpha_np = T_05_alpha_np[T_05_alpha_np!=0]
        T_max_np = rendering[9, :, :].cpu().numpy()
        T_max_np = T_max_np[T_max_np!=0]
        alpha_max_np = rendering[10, :, :].cpu().numpy()
        alpha_max_np = alpha_max_np[alpha_max_np!=0]
        T_05_mean += np.mean(T_05_np)
        T_05_min += np.min(T_05_np)
        T_05_max += np.max(T_05_np)
        T_05_alpha_mean += np.mean(T_05_alpha_np)
        T_05_alpha_min += np.min(T_05_alpha_np)
        T_05_alpha_max += np.max(T_05_alpha_np)
        T_max_mean += np.mean(T_max_np)
        T_max_min += np.min(T_max_np)
        T_max_max += np.max(T_max_np)
        alpha_max_mean += np.mean(alpha_max_np)
        alpha_max_min += np.min(alpha_max_np)
        alpha_max_max += np.max(alpha_max_np)
        # print("\n[ITER {}] T_05 mean: {}, T_05 min: {}, T_05 max: {}, T_05_alpha mean: {}, T_05_alpha min: {}, T_05_alpha max: {}".format(iteration, np.mean(T_05_np), np.min(T_05_np), np.max(T_05_np), np.mean(T_05_alpha_np), np.min(T_05_alpha_np), np.max(T_05_alpha_np)))
        count += 1
    
    T_05_mean /= count
    T_05_min /= count
    T_05_max /= count
    T_05_alpha_mean /= count
    T_05_alpha_min /= count
    T_05_alpha_max /= count
    T_max_mean /= count
    T_max_min /= count
    T_max_max /= count
    alpha_max_mean /= count
    alpha_max_min /= count
    alpha_max_max /= count

    data_to_save = {
        "T_05_mean": T_05_mean,
        "T_05_min": T_05_min,
        "T_05_max": T_05_max,
        "T_05_alpha_mean": T_05_alpha_mean,
        "T_05_alpha_min": T_05_alpha_min,
        "T_05_alpha_max": T_05_alpha_max,
        "T_max_mean": T_max_mean,
        "T_max_min": T_max_min,
        "T_max_max": T_max_max,
        "alpha_max_mean": alpha_max_mean,
        "alpha_max_min": alpha_max_min,
        "alpha_max_max": alpha_max_max
    }

    file_name = "T_05.json"
    with open(os.path.join(model_path, file_name), 'w') as f:
        json.dump(data_to_save, f, indent=4)
    print("\n[ITER {}] AVG T_05 mean: {}, AVG T_05 min: {}, AVG T_05 max: {}, AVG T_05_alpha mean: {}, AVG T_05_alpha min: {}, AVG T_05_alpha max: {}".format(iteration, T_05_mean, T_05_min, T_05_max, T_05_alpha_mean, T_05_alpha_min, T_05_alpha_max))
    
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

        if not skip_train:
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