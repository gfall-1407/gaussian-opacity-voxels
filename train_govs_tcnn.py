import sys
import torch
from tqdm import tqdm
from random import randint
from scene.govs_tcnn_scene import Scene, GovsTCNNModel
from govs_render_tcnn import render
from utils.general_utils import safe_state
from utils.loss_utils import l1_loss, ssim, compute_tv_loss_3d
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

import skimage.measure
import trimesh
from PIL import Image
import numpy as np

from utils.general_utils import inverse_sigmoid_python

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    govs = GovsTCNNModel(dataset.sh_degree)
    scene = Scene(dataset, govs)
    govs.training_setup(opt)
    
    # 计算每个球体包含的体素数量
    # print("\n" + "="*60)
    # print("计算每个球体包含的体素网格中心数量...")
    # print("="*60)
    # voxel_counts = count_voxels_in_ellipsoids(govs, scene.cameras_center, scene.cameras_extent)
    # print(f"\n统计结果:")
    # print(f"  球体总数: {len(voxel_counts)}")
    # print(f"  平均每个球体包含体素数: {voxel_counts.float().mean():.2f}")
    # print(f"  最小值: {voxel_counts.min()}")
    # print(f"  最大值: {voxel_counts.max()}")
    # print(f"  中位数: {voxel_counts.float().median():.2f}")
    # print("="*60 + "\n")
    
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

        govs.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            govs.oneupSHdegree()
        
        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render(viewpoint_cam, govs, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        if iteration % 500 == 0:
            image_np = image.detach().cpu().numpy()
            image_np = np.transpose(image_np, (1, 2, 0))
            array = np.array(image_np*255.0, dtype=np.byte)  
            image_save = Image.fromarray(array, "RGB")  
            image_save.save("test/" + str(iteration) + ".png" )
        
        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            # training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            # if (iteration in saving_iterations):
            #     print("\n[ITER {}] Saving Gaussians".format(iteration))
            #     scene.save(iteration)

            # Densification
            # if iteration < opt.densify_until_iter:
            #     # Keep track of max radii in image-space for pruning
            #     govs.max_radii2D[visibility_filter] = torch.max(govs.max_radii2D[visibility_filter], radii[visibility_filter])
            #     govs.add_densification_stats(viewspace_point_tensor, visibility_filter)

            # if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
            #     size_threshold = 20 if iteration > opt.opacity_reset_interval else None
            #     govs.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
            #     if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
            #         govs.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
               govs.optimizer.step()
               govs.optimizer.zero_grad(set_to_none = True)
            
            # if (iteration in checkpoint_iterations):
            #     print("\n[ITER {}] Saving Checkpoint".format(iteration))
            #     torch.save((govs.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            if iteration % 500 == 0:
                GRID_SIZE = 128
                t = torch.linspace(0, 1, GRID_SIZE, device="cuda")
                grid_x, grid_y, grid_z = torch.meshgrid(t, t, t, indexing="ij")
                grid_xyz = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)
                sdf_values = []
                for batch in grid_xyz.split(8192):
                    sdf_values.append(govs._opacity_field(batch).float())
                # sdf_volume = torch.cat(sdf_values).reshape(GRID_SIZE, GRID_SIZE, GRID_SIZE).cpu().numpy()
                # print(sdf_volume.min(), sdf_volume.max())
                #verts, faces, _, _ = skimage.measure.marching_cubes(sdf_volume, level=0)
                #mesh = trimesh.Trimesh(vertices=verts, faces=faces)
                #mesh.export("output_mesh_"+ str(iteration) +".ply")

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
