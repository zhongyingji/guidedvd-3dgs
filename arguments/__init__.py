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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group


class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images_8"
        self.dataset = "LLFF"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.n_views = 6
        self.dust3r_min_conf_thr = 1
        
        self.demo_setting = False

        self.replica_use_project_cam = False # use project cam for replica during training baseline
        
        
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.use_confidence = False
        self.use_color = True

        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 10_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 10_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000 # 2000
        self.densify_from_iter = 500
        self.prune_from_iter = 500
        self.densify_until_iter = 10_000 # 8000
        self.densify_grad_threshold = 0.0005
        self.prune_threshold = 0.005
        self.start_sample_pseudo = 2000
        self.end_sample_pseudo = 9500
        self.sample_pseudo_interval = 1
        self.dist_thres = 10.

        self.project_cam_prob = 0.8
        self.project_cam_weight = 0.05 # use project cam for replica during training baseline

        self.pseudo_cam_weight = 0.05
        self.pseudo_cam_ssim = False
        self.pseudo_cam_lpips = True
        self.pseudo_cam_lpips_weight = 0.1

        self.pseudo_cam_weight_decay = False
        self.pseudo_cam_weight_start = 10.0
        self.pseudo_cam_weight_end = 0.05

        
        self.use_trajectory_pool = True

        # guidance param
        self.guidance_recon_loss = "l2"
        self.w_guidance_recon_loss = 0.5
        self.guidance_gpu_id = 1
        self.guidance_vd_iter = 260 # run video diffusion every iters
        self.guidance_ddim_steps = 50
        self.guidance_pc_render_all_views = False
        self.guidance_recur_steps = 1
        self.guidance_vc_center_scale = 1.

        self.no_guidance = False # for ablation on directly using vd to generate samples
        self.guidance_random_traj = False
        self.guidance_no_wave_traj = False
        
        self.guidance_with_training_gs = False
        self.guidance_with_training_gs_startiter = 5999
        self.guidance_with_training_gs_decide_mask = False # use the training gs to decide mask

        self.guidance_with_ssim = False
        self.guidance_mean_loss = False
        self.guidance_with_lpips = False
        self.guidance_verbose = False

        self.guidance_videos_from_file = False
        self.guidance_save_videos = True # set it to False if there is not enough disk for saving videos
        
        self.append_pcd_from_video_diffusion = False # project the point cloud of holes

        self.scale_guidance_weight = False

        self.scannetpp_newres = False

        # replace projection with gs render
        self.replace_diffusion_input_with_gsrender = False

        super().__init__(parser, "Optimization Parameters")


def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        model_path = args_cmdline.model_path
        # cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        cfgfilepath = os.path.join(model_path, "cfg_args")

        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)


def get_combined_args_without_cmdlne(parser : ArgumentParser, trained_model_path):
    cmdlne_string = ""
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)
    try:
        model_path = trained_model_path
        cfgfilepath = os.path.join(model_path, "cfg_args")

        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)