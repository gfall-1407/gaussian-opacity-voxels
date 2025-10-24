import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from govs_renderer import render
import sys
from scene import Scene, GovsModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from PIL import Image
import numpy as np

def count_voxels_in_ellipsoids(govs, scene_center, scene_extent):
    """
    计算每个球体包含多少个体素网格中心
    （所有3DGS都是球体，只需判断距离是否小于半径）
    
    Args:
        govs: GovsModel对象，包含球体参数
        scene_center: 场景中心点
        scene_extent: 场景半径
    
    Returns:
        voxel_counts: 每个球体包含的体素数量 (N,)
    """
    # 获取球体参数
    means = govs.get_xyz  # (N, 3) 球体中心
    scales = govs.get_scaling  # (N, 3) 球体尺度，scale[0]是半径
    
    # 生成体素网格中心点
    resolution = govs.opacity_field_resolution
    grid_size = resolution + 1  # 网格数量
    
    # 体素网格范围: 场景中心 ± 场景半径
    voxel_min = scene_center - scene_extent
    voxel_max = scene_center + scene_extent
    # 保证voxel_min和voxel_max为Tensor类型
    if not torch.is_tensor(voxel_min):
        voxel_min = torch.tensor(voxel_min, device=means.device)
    if not torch.is_tensor(voxel_max):
        voxel_max = torch.tensor(voxel_max, device=means.device)
    
    # 创建体素网格中心点坐标
    x = torch.linspace(voxel_min[0].item(), voxel_max[0].item(), grid_size, device="cuda")
    y = torch.linspace(voxel_min[1].item(), voxel_max[1].item(), grid_size, device="cuda")
    z = torch.linspace(voxel_min[2].item(), voxel_max[2].item(), grid_size, device="cuda")
    
    # 创建网格 (grid_size, grid_size, grid_size, 3)
    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
    voxel_centers = torch.stack([grid_x, grid_y, grid_z], dim=-1)  # (grid_size, grid_size, grid_size, 3)
    voxel_centers = voxel_centers.reshape(-1, 3)  # (grid_size^3, 3)
    
    num_ellipsoids = means.shape[0]
    num_voxels = voxel_centers.shape[0]
    voxel_counts = torch.zeros(num_ellipsoids, dtype=torch.long, device="cuda")
    
    print(f"检查 {num_ellipsoids} 个球体和 {num_voxels} 个体素网格中心...")
    print(f"体素网格范围: [{voxel_min[0].item():.3f}, {voxel_max[0].item():.3f}] x [{voxel_min[1].item():.3f}, {voxel_max[1].item():.3f}] x [{voxel_min[2].item():.3f}, {voxel_max[2].item():.3f}]")
    
    # 统计球心是否在体素网格包围盒内
    not_in_grid_count = 0
    for i in range(num_ellipsoids):
        center = means[i]  # (3,) 球心
        radius = scales[i, 0]  # 球体半径（scale的第0个分量）
        
        # 计算点到球心的距离
        diff = voxel_centers - center  # (num_voxels, 3)
        distances = torch.norm(diff, dim=1)  # (num_voxels,) 欧式距离
        
        # 判断是否在球内（距离 <= 半径）
        inside = distances <= radius
        voxel_counts[i] = inside.sum()

        # 判断球心是否在体素网格包围盒内
        if not torch.all(center >= voxel_min) or not torch.all(center <= voxel_max):
            not_in_grid_count += 1        
        if (i + 1) % 100 == 0:
            print(f"  已处理 {i + 1}/{num_ellipsoids} 个球体...")
    
    print(f"\n有 {not_in_grid_count} 个球心未在体素网格包围盒内！")
    return voxel_counts

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    govs = GovsModel(dataset.sh_degree)
    scene = Scene(dataset, govs)
    govs.training_setup(opt)
    
    if False:
        # 计算每个球体包含的体素数量
        print("\n" + "="*60)
        print("计算每个球体包含的体素网格中心数量...")
        print("="*60)
        voxel_counts = count_voxels_in_ellipsoids(govs, scene.cameras_center, scene.cameras_extent)
        print(f"\n统计结果:")
        print(f"  球体总数: {len(voxel_counts)}")
        print(f"  平均每个球体包含体素数: {voxel_counts.float().mean():.2f}")
        print(f"  最小值: {voxel_counts.min()}")
        print(f"  最大值: {voxel_counts.max()}")
        print(f"  中位数: {voxel_counts.float().median():.2f}")
        print("="*60 + "\n")
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render(viewpoint_cam, govs, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        if iteration % 1 == 0:
            image_np = image.detach().cpu().numpy()
            image_np = np.transpose(image_np, (1, 2, 0))
            array = np.array(image_np*255.0, dtype=np.byte)  
            image_save = Image.fromarray(array, "RGB")  
            image_save.save("test/" + str(iteration) + ".png" )
        
        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        # loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
