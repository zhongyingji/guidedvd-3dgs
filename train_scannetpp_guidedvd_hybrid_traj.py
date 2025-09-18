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
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import numpy as np
import os
import copy
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision
import random
from random import randint
from utils.loss_utils import l1_loss, l1_loss_mask, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from lpipsPyTorch import lpips

from scene.cameras import PseudoCamera
from utils.viewcrafter_wrapper import LossGuidance, ViewCrafterWrapper
from utils.midas_depth_estimator import MiDasDepthEstimator
from utils.inpainted_depth_to_pointcloud import depth_to_point_cloud
from utils.vgg_loss import VggLoss
from utils.easy_renderer import EasyRenderer
from torch import Tensor
from third_party.ViewCrafter.utils_vc.pvd_utils import save_video


def training(dataset, opt, pipe, args):
    testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from = args.test_iterations, \
            args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(args)

    scene = Scene(args, gaussians, shuffle=False)
    assert not scene.shuffle
    
    gaussians.training_setup(opt)
    
    # renderer
    if args.dataset == "Replica": 
        assert args.dataset == "Replica"
        split = args.source_path.split("/")
        scene_name, seq = split[-2], split[-1]
        model_path="./output/replica_baseline/{}/{}/".format(scene_name, seq)
    
    elif args.dataset == "Scannetpp": 
        scene_name = args.source_path.split("/")[-1]
        model_path = "./output/scannetpp_baseline/{}".format(scene_name)
    
    easy_renderer = EasyRenderer(
        model_path=model_path, 
        iteration=opt.iterations)
    print("Easy renderer set up done. ")

    loss_guidance_fn = LossGuidance(
        ddim_steps=opt.guidance_ddim_steps, 
        recur_steps=opt.guidance_recur_steps, iter_steps=1, 
        recon_loss=opt.guidance_recon_loss, 
        save_dir=os.path.join(scene.model_path, "vd/", "pred_x0/"), 
        ssim_guidance=opt.guidance_with_ssim, 
        lpips_guidance=opt.guidance_with_lpips, 
        device="cuda:{}".format(opt.guidance_gpu_id), 
        verbose=opt.guidance_verbose, 
        mean_loss=opt.guidance_mean_loss, 
        scale_guidance_weight=opt.scale_guidance_weight, 
    )
    
    monodepth_est = None
    if opt.append_pcd_from_video_diffusion: 
        monodepth_est = MiDasDepthEstimator(device="cuda")

    vc_wrapper = ViewCrafterWrapper(
        train_cams=scene.scene_info_train_cams, 
        save_dir=os.path.join(scene.model_path, "vd/"), 
        viewcrafter_root_path="./third_party/ViewCrafter/", 
        H=320 if not opt.scannetpp_newres else 320, 
        W=448 if not opt.scannetpp_newres else 512, 
        loss_guidance_fn=loss_guidance_fn, 
        setup_diffusion=not opt.guidance_videos_from_file, 
        device="cuda:{}".format(opt.guidance_gpu_id)
    )
    vc_wrapper.vc_opts.ddim_steps = opt.guidance_ddim_steps
    loss_guidance_fn.set_hw(vc_wrapper.vc_opts.height, vc_wrapper.vc_opts.width)

    vc_wrapper.vc_opts.center_scale = opt.guidance_vc_center_scale

    if opt.scannetpp_newres: 
        if args.dataset == "Scannetpp": 
            vc_wrapper.vc_opts.height = 320
            vc_wrapper.vc_opts.width = 512
            print("=> scannetpp video diffusion resolution: {}x{}. ".format(vc_wrapper.vc_opts.height, vc_wrapper.vc_opts.width))

    tmp_cams = [scene.scene_info_all_cams[0]]
    _, intrinsics, H, W = vc_wrapper.parse_cameras(tmp_cams)
    intrinsic = intrinsics[0]
    fovx = tmp_cams[0].FovX
    fovy = tmp_cams[0].FovY


    # Trajectory Initialization Strategy Eq.(7) in the paper. 
    if opt.use_trajectory_pool: 
        # first render several poses to decide the trajectory
        print("=> Render several views for each training view to decide the trajectory.")
        traj_save_dir = os.path.join(scene.model_path, "define_traj/")
        scale_traj_save_dir = os.path.join(scene.model_path, "define_traj_scale/")
        scale3_traj_save_dir = os.path.join(scene.model_path, "define_traj_scale3/")
        trajectory_pool = {}
        mask_thsh = 0.1 * H * W
        top_k = 3
        top_k_center_scale = 2
        top_k_center_scale3 = 1
        d_theta = [-30, -15, 0, 15, 30] if opt.guidance_vc_center_scale != 1 else [-15,-7.5, 0, 7.5]
        
        original_center_scale = vc_wrapper.vc_opts.center_scale

        for train_idx in range(len(scene.scene_info_train_indices)): 
            traj_save_dir_i = os.path.join(traj_save_dir, str(scene.scene_info_train_indices[train_idx]))
            os.makedirs(traj_save_dir_i, exist_ok=True)

            scale_traj_save_dir_i = os.path.join(scale_traj_save_dir, str(scene.scene_info_train_indices[train_idx]))
            os.makedirs(scale_traj_save_dir_i, exist_ok=True)

            scale3_traj_save_dir_i = os.path.join(scale3_traj_save_dir, str(scene.scene_info_train_indices[train_idx]))
            os.makedirs(scale3_traj_save_dir_i, exist_ok=True)

            # ---------scale 1---------
            vc_wrapper.vc_opts.center_scale = original_center_scale

            c2w_candidates, others = vc_wrapper.get_candidate_poses(
                # d_phi=[-30,-15,15,30], d_theta=[-30,-15,15,30], 
                d_phi=[-30,-15,0,15,30], d_theta=d_theta, 
                fovx=fovx, fovy=fovy, 
                which_train_view=train_idx)
            c2w_candidates = c2w_candidates.cpu().numpy()
            print("=> Rendering {} poses for view {}... ".format(c2w_candidates.shape[0], train_idx))
            gs_render_results = []
            gs_render_alphas = []
            for i in range(c2w_candidates.shape[0]): 
                c2w_i = c2w_candidates[i]
                w2c_i = np.linalg.inv(c2w_i)
                gs_render_result_i, gs_render_alpha_i, gs_render_depth_i = easy_renderer.render(w2c_i, intrinsic, H, W)
                gs_render_results.append(gs_render_result_i.clamp(0, 1)) # [3, H, W]
                gs_render_alphas.append((gs_render_alpha_i.clamp(0, 1)<0.7).to(torch.float32)) # [1, H, W]
                torchvision.utils.save_image(gs_render_results[-1], os.path.join(traj_save_dir_i, "{}.png".format(i)))
                torchvision.utils.save_image(gs_render_alphas[-1], os.path.join(traj_save_dir_i, "{}_mask.png".format(i)))
            
            gs_render_results = torch.stack(gs_render_results, 0) # [n, 3, H, W]
            gs_render_alphas = torch.stack(gs_render_alphas, 0) # [n, 1, H, W]
            n_cand = gs_render_results.shape[0]

            # erosion
            gs_render_alphas = vc_wrapper.process_mask(gs_render_alphas)

            mask_regions = gs_render_alphas.view(n_cand, -1).sum(-1) # [n, ]
            filtered_indices = (mask_regions < mask_thsh).nonzero(as_tuple=True)[0]
            sorted_indices = torch.argsort(mask_regions[filtered_indices], descending=True)[:top_k]
            topk_indices = filtered_indices[sorted_indices] # [topk, ]
            print("selected indices: ", topk_indices.tolist())
            with open(os.path.join(traj_save_dir_i, "topk.txt"), "w") as output:
                output.write(str(topk_indices.tolist()))

            # selected_poses = c2w_candidates[topk_indices] # [topk, 4, 4]
            select_c2w_trajs = []
            for j in topk_indices: 
                interp_traj = vc_wrapper.interpolate_trajectory(
                    others["c2ws"], others["d_phis"][j], others["d_thetas"][j], others["d_rs"][j]) # [25, 4, 4]
                select_c2w_trajs.append(
                    [j, (torch.bmm(others["transform_back"].unsqueeze(0).expand_as(interp_traj), interp_traj)).cpu().numpy(), vc_wrapper.vc_opts.center_scale, 1]
                )
            
            # [topk, video_length, 4, 4]

            trajectory_pool[train_idx] = select_c2w_trajs

            # ---------scale 2---------
            # scale the center scale
            vc_wrapper.vc_opts.center_scale = original_center_scale / 3.

            c2w_candidates, others = vc_wrapper.get_candidate_poses(
                # d_phi=[-30,-15,15,30], d_theta=[-30,-15,15,30], 
                d_phi=[-30,-15,0,15,30], d_theta=d_theta, 
                fovx=fovx, fovy=fovy, 
                which_train_view=train_idx)
            c2w_candidates = c2w_candidates.cpu().numpy()
            print("=> Rendering {} poses for view {} (center scaled)... ".format(c2w_candidates.shape[0], train_idx))
            gs_render_results = []
            gs_render_alphas = []
            for i in range(c2w_candidates.shape[0]): 
                c2w_i = c2w_candidates[i]
                w2c_i = np.linalg.inv(c2w_i)
                gs_render_result_i, gs_render_alpha_i, gs_render_depth_i = easy_renderer.render(w2c_i, intrinsic, H, W)
                gs_render_results.append(gs_render_result_i.clamp(0, 1)) # [3, H, W]
                gs_render_alphas.append((gs_render_alpha_i.clamp(0, 1)<0.7).to(torch.float32)) # [1, H, W]
                torchvision.utils.save_image(gs_render_results[-1], os.path.join(scale_traj_save_dir_i, "{}.png".format(i)))
                torchvision.utils.save_image(gs_render_alphas[-1], os.path.join(scale_traj_save_dir_i, "{}_mask.png".format(i)))
            
            gs_render_results = torch.stack(gs_render_results, 0) # [n, 3, H, W]
            gs_render_alphas = torch.stack(gs_render_alphas, 0) # [n, 1, H, W]
            n_cand = gs_render_results.shape[0]

            # erosion
            gs_render_alphas = vc_wrapper.process_mask(gs_render_alphas)

            mask_regions = gs_render_alphas.view(n_cand, -1).sum(-1) # [n, ]
            filtered_indices = (mask_regions < mask_thsh).nonzero(as_tuple=True)[0]
            sorted_indices = torch.argsort(mask_regions[filtered_indices], descending=True)[:top_k_center_scale]
            topk_indices = filtered_indices[sorted_indices] # [topk, ]
            print("selected indices: ", topk_indices.tolist())
            with open(os.path.join(scale_traj_save_dir_i, "topk.txt"), "w") as output:
                output.write(str(topk_indices.tolist()))

            # selected_poses = c2w_candidates[topk_indices] # [topk, 4, 4]
            select_c2w_trajs = []
            for j in topk_indices: 
                interp_traj = vc_wrapper.interpolate_trajectory(
                    others["c2ws"], others["d_phis"][j], others["d_thetas"][j], others["d_rs"][j]) # [25, 4, 4]
                select_c2w_trajs.append(
                    [j, (torch.bmm(others["transform_back"].unsqueeze(0).expand_as(interp_traj), interp_traj)).cpu().numpy(), vc_wrapper.vc_opts.center_scale, 2]
                )

            trajectory_pool[train_idx].extend(select_c2w_trajs)
            
            # ---------scale 3---------
            # scale the center scale
            vc_wrapper.vc_opts.center_scale = original_center_scale / 10.

            c2w_candidates, others = vc_wrapper.get_candidate_poses(
                # d_phi=[-30,-15,15,30], d_theta=[-30,-15,15,30], 
                d_phi=[-30,-15,0,15,30], d_theta=d_theta, 
                fovx=fovx, fovy=fovy, 
                which_train_view=train_idx)
            c2w_candidates = c2w_candidates.cpu().numpy()
            print("=> Rendering {} poses for view {} (center scaled 3)... ".format(c2w_candidates.shape[0], train_idx))
            gs_render_results = []
            gs_render_alphas = []
            for i in range(c2w_candidates.shape[0]): 
                c2w_i = c2w_candidates[i]
                w2c_i = np.linalg.inv(c2w_i)
                gs_render_result_i, gs_render_alpha_i, gs_render_depth_i = easy_renderer.render(w2c_i, intrinsic, H, W)
                gs_render_results.append(gs_render_result_i.clamp(0, 1)) # [3, H, W]
                gs_render_alphas.append((gs_render_alpha_i.clamp(0, 1)<0.7).to(torch.float32)) # [1, H, W]
                torchvision.utils.save_image(gs_render_results[-1], os.path.join(scale3_traj_save_dir_i, "{}.png".format(i)))
                torchvision.utils.save_image(gs_render_alphas[-1], os.path.join(scale3_traj_save_dir_i, "{}_mask.png".format(i)))
            
            gs_render_results = torch.stack(gs_render_results, 0) # [n, 3, H, W]
            gs_render_alphas = torch.stack(gs_render_alphas, 0) # [n, 1, H, W]
            n_cand = gs_render_results.shape[0]

            # erosion
            gs_render_alphas = vc_wrapper.process_mask(gs_render_alphas)

            mask_regions = gs_render_alphas.view(n_cand, -1).sum(-1) # [n, ]
            filtered_indices = (mask_regions < mask_thsh).nonzero(as_tuple=True)[0]
            sorted_indices = torch.argsort(mask_regions[filtered_indices], descending=True)[:top_k_center_scale3]
            topk_indices = filtered_indices[sorted_indices] # [topk, ]
            print("selected indices: ", topk_indices.tolist())
            with open(os.path.join(scale3_traj_save_dir_i, "topk.txt"), "w") as output:
                output.write(str(topk_indices.tolist()))

            # selected_poses = c2w_candidates[topk_indices] # [topk, 4, 4]
            select_c2w_trajs = []
            for j in topk_indices: 
                interp_traj = vc_wrapper.interpolate_trajectory(
                    others["c2ws"], others["d_phis"][j], others["d_thetas"][j], others["d_rs"][j]) # [25, 4, 4]
                select_c2w_trajs.append(
                    [j, (torch.bmm(others["transform_back"].unsqueeze(0).expand_as(interp_traj), interp_traj)).cpu().numpy(), vc_wrapper.vc_opts.center_scale, 3]
                )

            trajectory_pool[train_idx].extend(select_c2w_trajs)
        
        # dummy code above 
        vc_wrapper.vc_opts.center_scale = original_center_scale
        print("=> Recover center scale: ", vc_wrapper.vc_opts.center_scale)
        
        trajectory_pool_shuffle = copy.deepcopy(trajectory_pool)
        for k, v in trajectory_pool_shuffle.items(): 
            random.shuffle(trajectory_pool_shuffle[k])


    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")

    viewpoint_stack, pseudo_stack = None, None
    pseudo_stack_alltime = []

    vd_generated_indices = np.arange(len(scene.scene_info_train_indices))
    np.random.shuffle(vd_generated_indices)
    vd_generated_indices = vd_generated_indices.tolist()

    if opt.pseudo_cam_lpips: 
        percep_loss_fn = VggLoss("cuda")

    txt_traj_warmup = True

    ema_loss_for_log = 0.0
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        # Every 1000 its we increase the levels of SH up to a maximum degree
        # if iteration % 500 == 0:
        # # if (iteration >= 5000) and (iteration % 1000 == 0): 
        #     gaussians.oneupSHdegree()

        # Pick a random Camera
        viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss_mask(image, gt_image)
        loss = ((1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)))
        
        
        pseudo_cam = None
        pseudo_loss = torch.Tensor([0.]).to(loss.device)
        if iteration % args.sample_pseudo_interval == 0 and \
            iteration > args.start_sample_pseudo and \
            iteration < args.end_sample_pseudo and \
            not (pseudo_stack is None and len(pseudo_stack_alltime) == 0): 

            # L14-16 in Algorithm2 of the paper. 
            if np.random.rand()>0.5 and len(pseudo_stack_alltime) > 0: 
                pseudo_stack_copy = pseudo_stack_alltime.copy()
            else: 
                pseudo_stack_copy = pseudo_stack.copy()
            
            pseudo_cam = pseudo_stack_copy.pop(randint(0, len(pseudo_stack_copy) - 1))
            render_pkg_pseudo = render(pseudo_cam, gaussians, pipe, background)
            render_image_pseudo = render_pkg_pseudo["render"] # [3, H, W]

            visibility_filter_pseudo_cam = render_pkg_pseudo["visibility_filter"]
            viewspace_point_tensor_pseudo_cam = render_pkg_pseudo["viewspace_points"]
            radii_pseudo_cam = render_pkg_pseudo["radii"]
            depth_pseudo_cam = render_pkg_pseudo["depth"] # [1, H, W]

            pseudo_gt = pseudo_cam.pseudo_gt.cuda()
            pseudo_loss = l1_loss(render_image_pseudo, pseudo_gt)
            if opt.pseudo_cam_ssim: 
                pseudo_loss_ssim = 1.0 - ssim(render_image_pseudo, pseudo_gt)
                pseudo_loss = (1.0 - opt.lambda_dssim) * pseudo_loss + opt.lambda_dssim * pseudo_loss_ssim
            if opt.pseudo_cam_lpips: 
                pseudo_loss_lpips = percep_loss_fn(render_image_pseudo[None].clamp(0,1), pseudo_gt[None].clamp(0,1))
                pseudo_loss = pseudo_loss + opt.pseudo_cam_lpips_weight * pseudo_loss_lpips
            
            pseudo_cam_weight = opt.pseudo_cam_weight
            if opt.pseudo_cam_weight_decay: 
                step_in_cur_interval = iteration % opt.guidance_vd_iter
                w = np.clip(step_in_cur_interval * 1.0 / (1 if (opt.guidance_vd_iter < 1) else opt.guidance_vd_iter), 0, 1)
                pseudo_cam_weight = opt.pseudo_cam_weight_start * (1.-w) + w * opt.pseudo_cam_weight_end
        
            loss = loss + pseudo_cam_weight * pseudo_loss
            # L18 in Algorithm 2 of the paper. 

        loss.backward()

        with torch.no_grad(): 
            if iteration % 100 == 0: 
                print_str = "[Iter {}] Loss: {}  L1: {}  pseudo_l1: {} ".format(iteration, loss.item(), Ll1.item(), pseudo_loss.item())
                print(print_str)
            
            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss,
                            testing_iterations, scene, render, (pipe, background))

            if iteration > first_iter and (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if iteration > first_iter and (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration),
                           scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            # Densification
            if  iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                if pseudo_cam is not None: 
                    # novel pose image
                    gaussians.max_radii2D[visibility_filter_pseudo_cam] = torch.max(gaussians.max_radii2D[visibility_filter_pseudo_cam], radii_pseudo_cam[visibility_filter_pseudo_cam])
                    gaussians.add_densification_stats_with_novel_pose(viewspace_point_tensor, visibility_filter, viewspace_point_tensor_pseudo_cam, visibility_filter_pseudo_cam)
                else: 
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                    
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.prune_threshold, scene.cameras_extent, size_threshold, iteration)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            gaussians.update_learning_rate(iteration)

            # if (iteration - args.start_sample_pseudo - 1) % opt.opacity_reset_interval == 0 and \
            #         iteration > args.start_sample_pseudo:
            if iteration % opt.opacity_reset_interval == 0: 
                gaussians.reset_opacity()

        
        if (iteration-1) % opt.guidance_vd_iter == 0 and iteration < args.end_sample_pseudo: 
            print("=> Running video diffusion at iteration {} ...".format(iteration))
            
            # if opt.guidance_random_traj: 
            #     r = np.random.rand()

            #     if opt.guidance_no_wave_traj: 
            #         if r < 0.5: 
            #             traj_txt = "./test/trajs/loop2.txt"
            #         else: 
            #             traj_txt = "./test/trajs/loop1.txt"
            #     else: 
            #         if r < 0.33: 
            #             traj_txt = "./test/trajs/loop2.txt"
            #         elif r < 0.66: 
            #             traj_txt = "./test/trajs/loop1.txt"
            #         else: 
            #             traj_txt = "./test/trajs/wave1.txt"
                    
            #     vc_wrapper.vc_opts.traj_txt = os.path.join(vc_wrapper.viewcrafter_root_path, traj_txt)

            #     print("=> Trajectory file: ", vc_wrapper.vc_opts.traj_txt)

            traj_txt = "./test/trajs/loop2.txt"
            vc_wrapper.vc_opts.traj_txt = os.path.join(vc_wrapper.viewcrafter_root_path, traj_txt)


            vc_wrapper.update_save_dir(iteration)
            loss_guidance_fn.update_save_dir(iteration)

            if len(vd_generated_indices) == 0: 
                vd_generated_indices = np.arange(len(scene.scene_info_train_indices))
                np.random.shuffle(vd_generated_indices)
                vd_generated_indices = vd_generated_indices.tolist()

                txt_traj_warmup = False

            which_train_view = vd_generated_indices.pop()
            
            if opt.use_trajectory_pool:  
                if not txt_traj_warmup: 
                    if len(trajectory_pool_shuffle[which_train_view]) == 0: 
                        trajectory_pool_shuffle[which_train_view] = copy.deepcopy(trajectory_pool[which_train_view])
                        random.shuffle(trajectory_pool_shuffle[which_train_view])
                    
                    # defined_camera_traj_c2ws = trajectory_pool[which_train_view] # [topk, 25, 4, 4]
                    # defined_camera_traj_c2ws = defined_camera_traj_c2ws[np.random.choice(len(defined_camera_traj_c2ws))]
                    defined_camera_traj_c2ws = trajectory_pool_shuffle[which_train_view].pop()
                    interp_idx, defined_camera_traj_c2ws, cur_traj_center_scale, cur_traj_center_scale_idx = defined_camera_traj_c2ws

                    vc_wrapper.vc_opts.center_scale = cur_traj_center_scale

                    pc_render_results, camera_traj_c2ws = vc_wrapper.preprocess_video_diffusion(
                        which_train_view, 
                        torch.from_numpy(defined_camera_traj_c2ws).to(vc_wrapper.device), 
                        pc_render_single_view=not opt.guidance_pc_render_all_views)
                else: 
                    interp_idx = 0
                    pc_render_results, camera_traj_c2ws = vc_wrapper.preprocess_video_diffusion(
                        which_train_view, 
                        pc_render_single_view=not opt.guidance_pc_render_all_views)
            
            camera_traj_c2ws = camera_traj_c2ws.cpu().numpy() # pc_render_results: [n_frames, H, W, 3]
            
            
            # use renderer to render outputs
            gs_render_results = []
            gs_render_alphas = []
            gs_render_depths = []
            if opt.guidance_with_training_gs and iteration >= opt.guidance_with_training_gs_startiter: 
                # use the current training gs to render guidance images. But we do not use this. 
                # we still use the baseline 3dgs to render guidance images. 

                print("=> Use the on-train gs to render guidance images at iteration {}. ".format(iteration))
                for i in range(camera_traj_c2ws.shape[0]): 
                    c2w_i = camera_traj_c2ws[i]
                    w2c_i = np.linalg.inv(c2w_i)
                    cam = PseudoCamera(
                        R=w2c_i[:3, :3].T, T=w2c_i[:3, 3], 
                        FoVx=viewpoint_cam.FoVx, FoVy=viewpoint_cam.FoVy,
                        width=viewpoint_cam.image_width, height=viewpoint_cam.image_height, 
                        pseudo_gt=None
                    )
                    gs_render_pkg = render(cam, gaussians, pipe, background)
                    gs_render_result_i = gs_render_pkg["render"] # [3, H, W]

                    if opt.guidance_with_training_gs_decide_mask: 
                        gs_render_alpha_i = gs_render_pkg["alpha"] # [1, H, W]
                    else: 
                        _, gs_render_alpha_i, _ = easy_renderer.render(w2c_i, intrinsic, H, W)
                    
                    gs_render_depth_i = gs_render_pkg["depth"] # [1, H, W]
                    gs_render_results.append(gs_render_result_i.clamp(0, 1)) # [3, H, W]
                    gs_render_alphas.append(gs_render_alpha_i.clamp(0, 1)) # [1, H, W]
                    gs_render_depths.append(gs_render_depth_i)
                
            else: 
                for i in range(camera_traj_c2ws.shape[0]): 
                    c2w_i = camera_traj_c2ws[i]
                    w2c_i = np.linalg.inv(c2w_i)
                    gs_render_result_i, gs_render_alpha_i, gs_render_depth_i = easy_renderer.render(w2c_i, intrinsic, H, W)
                    gs_render_results.append(gs_render_result_i.clamp(0, 1)) # [3, H, W]
                    gs_render_alphas.append(gs_render_alpha_i.clamp(0, 1)) # [1, H, W]
                    gs_render_depths.append(gs_render_depth_i)
                

            gs_render_results = torch.stack(gs_render_results, 0) # [n, 3, H, W]
            save_video(gs_render_results.permute(0, 2, 3, 1), os.path.join(vc_wrapper.save_dir, 'gs_render.mp4'))

            gs_render_alphas = torch.stack(gs_render_alphas, 0) # [n, 1, H, W]
            # gs_render_alphas = (gs_render_alphas < 0.7).to(torch.float32)
            gs_render_alphas = (gs_render_alphas < 0.9).to(torch.float32)
            # gs_render_alphas = vc_wrapper.process_mask2(gs_render_alphas)
            save_video(gs_render_alphas.expand_as(gs_render_results).permute(0, 2, 3, 1), os.path.join(vc_wrapper.save_dir, 'gs_render_alpha.mp4'))

            gs_render_depths = torch.stack(gs_render_depths, 0) # [n, 1, H, W]
            tmp_d = gs_render_depths * (1.-gs_render_alphas)
            tmp_d = (tmp_d - tmp_d.min()) / (tmp_d.max() - tmp_d.min()) # [n, 1, H, W]
            save_video(tmp_d.expand_as(gs_render_results).permute(0, 2, 3, 1), os.path.join(vc_wrapper.save_dir, 'gs_render_depth.mp4'))
            
            
            videos_dir = os.path.join(scene.model_path, "video_files/", str(which_train_view))
            os.makedirs(videos_dir, exist_ok=True)
            video_file = os.path.join(videos_dir, "{}.pth".format(interp_idx))

            diffusion_results = vc_wrapper.run_video_diffusion(
                pc_render_results, 
                guidance_images=gs_render_results.to(vc_wrapper.device), 
                guidance_masks=1.-gs_render_alphas.to(vc_wrapper.device), 
                guidance_depths=gs_render_depths.to(vc_wrapper.device), # NOTE: raw prediction
                no_guidance=opt.no_guidance)
                # [n_frames, 3, H, W], 0-1. torch.

            diffusion_results = diffusion_results.to(torch.float32)
            diffusion_results = F.interpolate(diffusion_results, 
                                                size=(gs_render_results.shape[2], gs_render_results.shape[3]), mode='bilinear', align_corners=False)
            
            if opt.guidance_save_videos: 
                if not txt_traj_warmup: 
                    videos_dir = os.path.join(scene.model_path, "video_files_scale{}/".format(cur_traj_center_scale_idx), str(which_train_view))
                else: 
                    videos_dir = os.path.join(scene.model_path, "video_files_warmup/", str(which_train_view))
                os.makedirs(videos_dir, exist_ok=True)
                video_file = os.path.join(videos_dir, "{}.pth".format(interp_idx))
                torch.save(diffusion_results, video_file)
                print("=> Save video file at {}. ".format(video_file))

            
            if opt.append_pcd_from_video_diffusion: 
                with torch.no_grad(): 
                    rel_depth = monodepth_est.get_rel_depth(2*diffusion_results.to(monodepth_est.device)-1.) # [N, H, W]
                    scale, shift = monodepth_est.get_scaleshift(rel_depth, gs_render_depths[:, 0], 1.-gs_render_alphas[:, 0])
                    metric_depth = monodepth_est.convert_rel_to_real(rel_depth, scale, shift) # [N, H, W]
                metric_depth_min = metric_depth.view(metric_depth.shape[0], -1).min(-1)[0]
                metric_depth_max = metric_depth.view(metric_depth.shape[0], -1).max(-1)[0]
                tmp_m_d = ((metric_depth - metric_depth_min[:, None, None]) / (metric_depth_max[:, None, None] - metric_depth_min[:, None, None]))[:, None] # [n, 1, H, W]
                save_video(tmp_m_d.expand_as(gs_render_results).permute(0, 2, 3, 1), os.path.join(vc_wrapper.save_dir, 'diffusion_monodepth.mp4'))
                

                # project the pointcloud based on metric_depth
                append_frame_gap = 5
                print("=> Inpaint the point cloud of with predicted depth. Select the points every {} frame. ".format(append_frame_gap))
                append_pts, append_rgbs = [], []
                for i in range(diffusion_results.shape[0]): 
                    # if i % append_frame_gap != 0:
                    #     continue
                    rgb_i = diffusion_results[i].permute(1, 2, 0).cpu().numpy()
                    depth_i = metric_depth[i].cpu().numpy()
                    mask_i = gs_render_alphas[i, 0].cpu().numpy()
                    append_pts_i, append_rgbs_i = depth_to_point_cloud(
                        depth_i, intrinsic, camera_traj_c2ws[i], mask_i, rgb_i)
                    append_pts.append(append_pts_i[::append_frame_gap]) # in range 0-1
                    append_rgbs.append(append_rgbs_i[::append_frame_gap])
                append_pts = np.concatenate(append_pts, 0).astype(np.float32)
                append_rgbs = np.concatenate(append_rgbs, 0).astype(np.float32)
                print("=> Appended points in total: {}. ".format(append_pts.shape[0]))

                # TODO: filter out nan points
                append_pts = torch.from_numpy(append_pts)
                append_rgbs = torch.from_numpy(append_rgbs)
                tmp1 = append_pts.sum(1)
                tmp2 = append_rgbs.sum(1)
                invalid1 = torch.isnan(tmp1) | torch.isinf(tmp1)
                invalid2 = torch.isnan(tmp2) | torch.isinf(tmp2)
                valid = ~(invalid1 | invalid2)

                append_pts = append_pts[valid].cpu().numpy()
                append_rgbs = append_rgbs[valid].cpu().numpy()
                print("=> Appended points in total (filter invalid): {}. ".format(append_pts.shape[0]))

                # append to gaussian
                gaussians.add_points(append_pts, append_rgbs)

            
            diffusion_results = diffusion_results.cpu()
            pseudo_stack = []
            for i in range(camera_traj_c2ws.shape[0]): 
                if i == 0: 
                    continue # skip the first frame

                c2w_i = camera_traj_c2ws[i]
                w2c_i = np.linalg.inv(c2w_i)
                
                cam = PseudoCamera(
                    R=w2c_i[:3, :3].T, T=w2c_i[:3, 3], 
                    FoVx=viewpoint_cam.FoVx, FoVy=viewpoint_cam.FoVy,
                    width=viewpoint_cam.image_width, height=viewpoint_cam.image_height, 
                    pseudo_gt=diffusion_results[i, :3], 
                    mask=gs_render_alphas[i].cpu(), # [1, h, w]
                )

                pseudo_stack.append(cam)
                if np.random.rand() > 0.8: # 20%
                    pseudo_stack_alltime.append(cam)
            
            print("=> Generated sequence around idx {} is pushed to the stack. Stack length: {}. ".format(which_train_view, len(pseudo_stack)))


    

def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer



def training_report(tb_writer, iteration, Ll1, loss, l1_loss, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        # tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                              {'name': 'train', 'cameras' : scene.getTrainCameras()})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test, psnr_test, ssim_test, lpips_test = 0.0, 0.0, 0.0, 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 8):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()

                    _mask = None
                    _psnr = psnr(image, gt_image, _mask).mean().double()
                    _ssim = ssim(image, gt_image, _mask).mean().double()
                    _lpips = lpips(image, gt_image, _mask, net_type='vgg')
                    psnr_test += _psnr
                    ssim_test += _ssim
                    lpips_test += _lpips
                psnr_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {} ".format(
                    iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)

    parser.add_argument("--test_iterations", nargs="+", type=int, default=[10_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[10_00, 10_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[10_00, 10_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--train_bg", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print(args.test_iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args)

    # All done
    print("\nTraining complete.")