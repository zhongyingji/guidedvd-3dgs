import os, sys
sys.path.append("../")
import numpy as np
import torch
import torchvision
from scene import Scene
from scene.cameras import PseudoCamera
from gaussian_renderer import render, GaussianModel
from utils.general_utils import safe_state
from utils.graphics_utils import focal2fov
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args_without_cmdlne


class EasyRenderer(): 
    def __init__(self, model_path, iteration=10000): 
        
        self.model_path = model_path
        self.iteration = iteration

        parser = ArgumentParser(description="Easy renderer")
        model = ModelParams(parser, sentinel=True) # init a a series of add_argument...
        pipeline = PipelineParams(parser) # init a a series of add_argument...
        
        # print("()()()", vars(model))
        # print("()()()", vars(pipeline))
        # args = get_combined_args(parser, trained_model_path=model_path) # combine 
        args = get_combined_args_without_cmdlne(parser, trained_model_path=model_path) # combine 
        # print("before set: ", args)


        model_param = model.extract(args)
        pipeline_param = pipeline.extract(args)

        args.iteration = iteration # additional parameter

        # print("after set args: ", args)
        # print("model_param: ", model_param)
        # print(vars(model_param))

        self.model_param = model_param
        self.pipeline_param = pipeline_param
        # print("***", vars(self.model_param), vars(self.pipeline_param))

        # safe_state(args.quiet)

        with torch.no_grad(): 
            self.gaussians = GaussianModel(args)
            # self.scene = Scene(args, self.gaussians, load_iteration=args.iteration, shuffle=False)
            
            ply_file = os.path.join(model_path, "point_cloud", "iteration_10000", "point_cloud.ply") # trained gs
            self.gaussians.load_ply(ply_file)

            bg_color = [1,1,1] if model_param.white_background else [0, 0, 0]
            self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        print("EasyRenderer from {} set done. ".format(model_path))

    def render(self, w2c, intrinsic, h, w): 
        # w2c: [4, 4]. np.array
        # intrinsic: [3, 3]. np.array
        with torch.no_grad(): 
            view = self.make_gs_view_format(w2c, intrinsic, h, w)
            rendering = render(view, self.gaussians, self.pipeline_param, self.background)
            # [3, H, W]
        return rendering["render"], rendering["alpha"], rendering["depth"]
    
    def make_gs_view_format(self, w2c, intrinsic, h, w): 
        focal_length_x = intrinsic[0, 0]
        focal_length_y = intrinsic[1, 1]
        FovX = focal2fov(focal_length_x, w)
        FovY = focal2fov(focal_length_y, h)

        R = np.transpose(w2c[:3, :3])
        T = w2c[:3, 3]
        
        pseudo_cam = PseudoCamera(R=R, T=T, FoVx=FovX, FoVy=FovY, width=w, height=h)
        return pseudo_cam


if __name__ == "__main__": 
    model_path = "../output/replica_dust3r_minconf1_nopseudo_nodepth_0poslr_nodensify_withpointersect_highpointersectweight/office_2/Sequence_2/"
    iteration = 10000

    easy_renderer = EasyRenderer(model_path, iteration)
    
    cam_file = "../utils/cam_poses.pt"
    cam = torch.load(cam_file)
    
    idx = 0
    for idx in range(400, 900)[::6]: 
        intrinsic = cam['intrinsic'][0][idx]
        c2w = cam['H_c2w'][0][idx]
        w2c = np.linalg.inv(c2w)
        h, w = cam['height_px'], cam['width_px']
        
        rendering = easy_renderer.render(w2c, intrinsic, h, w, idx)
        rgb = rendering["render"]
        print(rgb.shape) # [3, h, w]

        torchvision.utils.save_image(rgb, "./{}.png".format(idx))

    