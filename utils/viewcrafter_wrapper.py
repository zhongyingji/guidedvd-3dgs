import os, sys
import math
from datetime import datetime
import torch
import torch.nn.functional as F
import numpy as np
from scipy import ndimage

sys.path.append("../")
from utils.graphics_utils import fov2focal
from utils.image_utils import psnr
from utils.loss_utils import ssim_noavg, ssim
from utils.vgg_loss import VggLoss
from lpipsPyTorch import lpips
from torchmetrics.functional.regression import pearson_corrcoef
from utils.midas_depth_estimator import MiDasDepthEstimator

try: 
    sys.path.append("./third_party/ViewCrafter/extern/dust3r/")
    from dust3r.inference import inference, load_model
    from dust3r.utils.image import load_images
    from dust3r.image_pairs import make_pairs
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
    from dust3r.utils.device import to_numpy

    sys.path.append("./third_party/ViewCrafter") # some places in the package import lvdm
    from third_party.ViewCrafter.viewcrafter import ViewCrafter
    from third_party.ViewCrafter.configs.infer_config import get_parser
    from third_party.ViewCrafter.utils_vc.pvd_utils import save_video, generate_traj_txt_my, world_point_to_obj_my, sphere2pose, txt_interpolation

except ImportError:
    sys.path.append("../third_party/ViewCrafter/extern/dust3r/")
    from dust3r.inference import inference, load_model
    from dust3r.utils.image import load_images
    from dust3r.image_pairs import make_pairs
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
    from dust3r.utils.device import to_numpy

    sys.path.append("../third_party/ViewCrafter") # some places in the package import lvdm
    from third_party.ViewCrafter.viewcrafter import ViewCrafter
    from third_party.ViewCrafter.configs.infer_config import get_parser
    from third_party.ViewCrafter.utils_vc.pvd_utils import save_video, generate_traj_txt_my, world_point_to_obj_my, sphere2pose, txt_interpolation

from pytorch3d.renderer import PerspectiveCameras


class LossGuidance: 
    def __init__(
            self, 
            ddim_steps, 
            recur_steps=2, 
            iter_steps=0, 
            recon_loss="l2", 
            w_recon_loss=0.5, 
            save_dir=None, 
            ssim_guidance=False, 
            lpips_guidance=False, 
            device="cuda:0", 
            verbose=False, 
            mean_loss=False, 
            scale_guidance_weight=False, 
        ): 

        self.ddim_steps = ddim_steps

        self.recur_steps = recur_steps
        self.iter_steps = iter_steps

        self.root_save_dir = save_dir
        if self.root_save_dir is not None: 
            os.makedirs(self.root_save_dir, exist_ok=True)
        self.save_dir = save_dir

        self.ssim_guidance = ssim_guidance
        self.lpips_guidance = lpips_guidance
        if self.lpips_guidance: 
            self.lpips_fn = VggLoss(device)
        
        self.verbose = verbose

        self.guidance_images = None
        self.guidance_latents = None
        self.guidance_masks = None

        self.mean_loss = mean_loss
        assert mean_loss is False, "Important to set it to False. "
        # NOTE: should be set to False, due to strange bug in bp

        self.recon_fn = torch.square
        self.w_recon = w_recon_loss
        
        self.scale_guidance_weight = scale_guidance_weight
        if scale_guidance_weight: 
            self.guidance_weight_fn = lambda step: learning_rate_decay(
                                                step,
                                                lr_init=0.01,
                                                lr_final=1.0,
                                                max_steps=2500)
            print("=> Guidance weight is scaled like the learning rate decay. ")
        
    
    def set_hw(self, H, W): 
        self.H = H
        self.W = W

    def set_guidance_images(self, guidance_images): 
        # guidance_images: [n_frames, 3, H, W]. rendered from 3dgs
        guidance_images = F.interpolate(guidance_images, size=(self.H, self.W), mode='bilinear', align_corners=False)
        self.guidance_images = guidance_images.clamp(0, 1) # [0, 1]
    
        self.guidance_latents = None # TODO: get the vae latents
    
    def set_guidance_masks(self, guidance_masks): 
        # guidance_masks: [n_frames, 1, H, W]. 
        guidance_masks = F.interpolate(guidance_masks, size=(self.H, self.W), mode='nearest')
        self.guidance_masks = guidance_masks
    
    def set_guidance_depths(self, guidance_depths): 
        # guidance_depths: [n_frames, 1, H, W]. 
        guidance_depths = F.interpolate(guidance_depths, size=(self.H, self.W), mode='nearest')
        self.guidance_depths = guidance_depths

    def __call__(
        self, 
        diffused_images, 
        ddim_index, 
        batch_idx_start, 
        batch_idx_end, 
    ): 
        # diffused_images: [3, 1, H, W], [-1, 1]. diffused_images is sampled by batch_idx_start:batch_idx_end
        
        diffused_images = diffused_images.permute(1, 0, 2, 3) # [1, 3, H, W]
        diffused_images = (diffused_images + 1.)/2.
        diffused_images = diffused_images.clamp(0, 1)

        loss_dict = {}

        guidance_masks = None
        if self.guidance_masks is None: 
            guidance_masks = torch.ones_like(diffused_images)
        else: 
            guidance_masks = self.guidance_masks[batch_idx_start:batch_idx_end] # [1, 1, H, W]
            guidance_masks = guidance_masks.expand_as(diffused_images) # [1, 3, H, W]
        
        loss_recon = self.w_recon * self.recon_fn((diffused_images-self.guidance_images[batch_idx_start:batch_idx_end])) * guidance_masks
        numel = guidance_masks.sum()
        loss_dict["recon"] = loss_recon.sum() if not self.mean_loss else loss_recon.mean()
        # guidance loss Eq.(6) in the paper. 
        
        if self.ssim_guidance: 
            loss_ssim = 1.0-ssim_noavg(
                diffused_images.to(torch.float32), self.guidance_images[batch_idx_start:batch_idx_end].to(torch.float32), 
                mask=guidance_masks)
            loss_ssim = loss_ssim.sum() if not self.mean_loss else loss_ssim.mean()
            loss_dict["recon"] = 0.8*loss_dict["recon"] + 0.2*loss_ssim

        if self.lpips_guidance: 
            loss_lpips = self.lpips_fn(diffused_images.to(torch.float32), self.guidance_images[batch_idx_start:batch_idx_end].to(torch.float32), mask=guidance_masks)
            loss_dict["recon"] = loss_dict["recon"] + numel * loss_lpips * 0.001

        if self.verbose: 
            if batch_idx_start == 0: 
                print("=> in loss guidance. ", loss_recon.shape, loss_recon.abs().sum())
        
        return loss_dict, numel

    def update_save_dir(self, train_iter): 
        self.save_dir = os.path.join(self.root_save_dir, "train_iter"+str(train_iter))
        if self.save_dir is not None: 
            os.makedirs(self.save_dir, exist_ok=True)
        
        self.current_train_iter = train_iter

    def save_pred_x0(self, pred_x0, ddim_index): 
        # [1, 3, n_frames, H, W]
        if self.save_dir is None: 
            return
        
        is_depth_map = (pred_x0.shape[1] == 1)
        
        if is_depth_map: 
            pred_x0 = (pred_x0 - pred_x0.min()) / (pred_x0.max() - pred_x0.min()) # [1, 1, n_frames, H, W], 0-1
            pred_x0 = pred_x0[0, 0] # [n_frames, H, W]
            d_min = pred_x0.view(pred_x0.shape[0], -1).min(-1)[0]
            d_max = pred_x0.view(pred_x0.shape[0], -1).max(-1)[0]
            pred_x0 = (pred_x0-d_min[:, None, None])/(d_max[:, None, None]-d_min[:, None, None])
            pred_x0 = pred_x0[..., None].expand(-1, -1, -1, 3)
            save_video(pred_x0, os.path.join(self.save_dir, 'pred_depth_step{}.mp4'.format(ddim_index)))

        else: 
            pred_x0 = torch.clamp(pred_x0[0].permute(1, 2, 3, 0), -1., 1.) # [-1, 1]
            save_video((pred_x0+1.)/2., os.path.join(self.save_dir, 'pred_x0_step{}.mp4'.format(ddim_index)))
        
        
class ViewCrafterWrapper: 
    def __init__(
            self, 
            train_cams, 
            save_dir, 
            viewcrafter_root_path, 
            H=320, W=448, 
            loss_guidance_fn=None, 
            setup_diffusion=True, 
            setup_dust3r=True, 
            device="cuda:0"): 
        
        parser = get_parser() # infer config.py
        opts, _ = parser.parse_known_args()
        # if opts.exp_name == None:
        #     prefix = datetime.now().strftime("%Y%m%d_%H%M")
        #     opts.exp_name = f'{prefix}_{os.path.splitext(os.path.basename(opts.image_dir))[0]}'
        
        self.setup_train_views(train_cams)
        self.train_cams = train_cams
        
        self.vc_opts = opts
        self.init_H, self.init_W = H, W
        self.hard_code_vc_opts(viewcrafter_root_path)
        self.device = device

        self.root_save_dir = save_dir
        os.makedirs(self.root_save_dir, exist_ok=True)
        self.save_dir = save_dir

        self.view_crafter = ViewCrafter(
            self.vc_opts, 
            setup_diffusion=setup_diffusion, 
            device=self.device)

        self.dust3r = load_model(self.vc_opts.dust3r_path, self.device)
        
        self.setup_dust3r = setup_dust3r
        if setup_dust3r: 
            self.run_dust3r_train_views()
        else: 
            image_names = [cam.image_path for cam in self.train_cams]
            images = load_images(image_names, size=512, force_1024=True, wh=(self.vc_opts.width, self.vc_opts.height))
            self.shape = images[0]['true_shape']
            self.d_H, self.d_W = int(self.shape[0][0]), int(self.shape[0][1]) # H=368, W=512
            print("=======> No dust3r running. Replace pc render results with gs rendering. <=======")
    
        self.viewcrafter_root_path = viewcrafter_root_path

        self.loss_guidance_fn = loss_guidance_fn
    
    def update_save_dir(self, train_iter): 
        self.save_dir = os.path.join(self.root_save_dir, "train_iter"+str(train_iter))
        if self.save_dir is not None: 
            os.makedirs(self.save_dir, exist_ok=True)
        
    def hard_code_vc_opts(self, viewcrafter_root_path): 

        self.vc_opts.video_length = 25
        # self.vc_opts.ckpt_path = "./checkpoints/model_frame{}_512.ckpt".format(self.vc_opts.video_length)
        self.vc_opts.ckpt_path = "./checkpoints/model.ckpt".format(self.vc_opts.video_length)

        self.vc_opts.traj_txt = "./test/trajs/loop2.txt"
        self.vc_opts.mode = "single_view_txt"
        self.vc_opts.center_scale = 1.
        self.vc_opts.elevation = 5
        self.vc_opts.seed = 123
        self.vc_opts.ddim_steps = 50

        # actually the parameter of camera can be set in the program
        self.vc_opts.d_theta = -30
        self.vc_opts.d_phi = 45
        self.vc_opts.d_r = -.5 
        
        self.vc_opts.height = self.init_H
        self.vc_opts.width = self.init_W

        self.vc_opts.config = "configs/inference_pvd_1024.yaml"

        self.vc_opts.known_c2w = True
        self.vc_opts.known_focal = True

        # temporal solution for reading files under the root of ViewCrafter
        self.vc_opts.traj_txt = os.path.join(viewcrafter_root_path, self.vc_opts.traj_txt)
        self.vc_opts.ckpt_path = os.path.join(viewcrafter_root_path, self.vc_opts.ckpt_path)
        self.vc_opts.dust3r_path = os.path.join(viewcrafter_root_path, self.vc_opts.dust3r_path)
        self.vc_opts.config = os.path.join(viewcrafter_root_path, self.vc_opts.config)

    def delete_all_tensors(self): 
        del self.dust3r, self.scene, self.c2ws, self.principal_points, self.focals, self.pcd, self.depth
        torch.cuda.empty_cache()

    def setup_train_views(self, train_cams): 
        self.train_poses, self.intrinsic, self.H, self.W = self.parse_cameras(train_cams) # [n, 4, 4], [n, 3, 3]
        self.train_poses = self.train_poses.astype(np.float32)
    
    def run_dust3r_train_views(self): 
        known_c2w, known_focal = [], []
        n_poses = self.train_poses.shape[0]
        for idx in range(n_poses): 
            known_c2w.append(self.train_poses[idx])
            known_focal.append(self.intrinsic[idx][0, 0]/(max(self.H, self.W)/512.))
        
        image_names = [cam.image_path for cam in self.train_cams]
        images = load_images(image_names, size=512, force_1024=True, wh=(self.vc_opts.width, self.vc_opts.height))
        pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
        output = inference(pairs, self.dust3r, self.device, batch_size=self.vc_opts.batch_size)

        mode = GlobalAlignerMode.PointCloudOptimizer
        scene = global_aligner(output, device=self.device, mode=mode)

        if self.vc_opts.known_c2w: 
            print("=> Preset poses. ")
            scene.preset_pose(known_c2w)
            scene.preset_focal(known_focal)
        
        print("=> Running dust3r on training views...")
        loss = scene.compute_global_alignment(init='mst', niter=self.vc_opts.niter, schedule=self.vc_opts.schedule, lr=self.vc_opts.lr)
        
        self.d_images = images
        
        self.scene = scene
        self.c2ws = scene.get_im_poses().detach() # [n, 4, 4]
        self.principal_points = scene.get_principal_points().detach() #cx cy
        self.focals = scene.get_focals().detach() # [n_views, 1]
        self.pcd = [i.detach() for i in scene.get_pts3d(clip_thred=self.vc_opts.dpt_trd)]
        self.depth = [i.detach() for i in scene.get_depthmaps()]
        self.shape = images[0]['true_shape']
        self.d_H, self.d_W = int(self.shape[0][0]), int(self.shape[0][1]) # H=368, W=512
        
        
        print("=>> pp: ", self.principal_points)

        save_dict = {
            "d_images": self.d_images, 
            "scene": self.scene, 
            "c2ws": self.c2ws, 
            "principal_points": self.principal_points, 
            "focals": self.focals, 
            "pcd": self.pcd, 
            "depth": self.depth, 
            "shape": self.shape, 
            "d_H": self.d_H, 
            "d_W": self.d_W, 
            "shape": self.shape
        }
        # torch.save(save_dict, save_dict_file)
        
        print("=> Done. ")
    
    def get_camera_traj(self, c2w, H, W, focal, pp): 
        # based on one camera pose
        # NOTE: c2w have to be transformed
        
        traj_back = False
        if self.vc_opts.mode == "single_view_txt": 
            with open(self.vc_opts.traj_txt, 'r') as file:
                lines = file.readlines()
                phi = [float(i) for i in lines[0].split()]
                theta = [float(i) for i in lines[1].split()]
                r = [float(i) for i in lines[2].split()]
            camera_traj, num_views, camera_traj_c2ws = generate_traj_txt_my(
                c2w, H, W, focal, pp, phi, theta, r, 
                self.vc_opts.video_length, self.device, 
                viz_traj=False, save_dir=None)
            
            if phi[-1]==0. and theta[-1]==0. and r[-1]==0.: 
                traj_back = True
        else: 
            raise NotImplementedError
        
        return camera_traj, num_views, camera_traj_c2ws, traj_back
    
    def get_candidate_poses(self, d_phi, d_theta, fovx, fovy,  
                            which_train_view=5, pc_render_single_view=True, ignore_0_0=False):
        idx = which_train_view

        print("Get the candidate poses around idx: {}. ".format(idx))

        if pc_render_single_view: 
            imgs = np.array(self.scene.imgs)[[idx]]
            pcd = [self.pcd[idx]]
        else: 
            imgs = np.array(self.scene.imgs)
            pcd = torch.stack(self.pcd, 0)
        
        img_ori = (self.d_images[idx]['img_ori'].squeeze(0).permute(1,2,0)+1.)/2. # [512,704,3] [0,1]
        c2ws = self.c2ws[[idx]]
        principal_points = self.principal_points[[idx]]
        focals = self.focals[[idx]]
        depth = [self.depth[idx]]
        H, W = self.d_H, self.d_W

        depth_avg = depth[-1][H//2,W//2]
        radius = depth_avg*self.vc_opts.center_scale

        c2ws, pcd, transform_back = world_point_to_obj_my(
            poses=c2ws, points=torch.stack(pcd) if pc_render_single_view else pcd, 
            k=-1, r=radius, elevation=self.vc_opts.elevation, device=self.device)
        masks = None

        # fovx_degree = min((fovx / (math.pi/2.))*90, 30*2)
        # fovy_degree = min((fovy / (math.pi/2.))*90, 30*2)


        # d_phi = [-fovx_degree/2., -fovx_degree/4., -fovx_degree/8, 0, fovx_degree/8, fovx_degree/4., fovx_degree/2.]
        # d_theta = [-fovy_degree/2., -fovy_degree/4., -fovy_degree/8., 0, fovy_degree/8., fovy_degree/4., fovy_degree/2.]
        # print("=> fov: ", d_phi)

        d_phis = []
        d_thetas = []
        d_rs = []
        for d_phi_j in d_phi: 
            for d_theta_j in d_theta: 
                if ignore_0_0 and d_phi_j == 0 and d_theta_j == 0: 
                    print("=> ignore 0, 0 at get_candidate_poses!")
                d_phis.append(d_phi_j)
                d_thetas.append(d_theta_j)
                d_rs.append(0)
    
        c2w_candidates = []
        for th, ph, r in zip(d_thetas, d_phis, d_rs): 
            c2w_new = sphere2pose(c2ws, np.float32(th), np.float32(ph), np.float32(r), self.device)
            c2w_candidates.append(c2w_new)
        c2w_candidates = torch.cat(c2w_candidates, dim=0)

        c2w_candidates = torch.bmm(transform_back.unsqueeze(0).expand_as(c2w_candidates), c2w_candidates)
        
        return c2w_candidates, \
            {"c2ws": c2ws, "d_phis": d_phis, "d_thetas": d_thetas, "d_rs": d_rs, "transform_back": transform_back}

    def interpolate_trajectory(self, c2w, d_phi, d_theta, d_r): 
        frame = self.vc_opts.video_length
        
        thetas = np.linspace(0, d_theta, frame)
        phis = np.linspace(0, d_phi, frame)
        rs = np.linspace(0, d_r*c2w[0,2,3].cpu(), frame)
        
        c2w_traj = []
        for th, ph, r in zip(thetas, phis, rs):
            c2w_new = sphere2pose(c2w, np.float32(th), np.float32(ph), np.float32(r), self.device)
            c2w_traj.append(c2w_new)
        c2w_traj = torch.cat(c2w_traj, dim=0)

        return c2w_traj # [25, 4, 4]

    def interpolate_trajectory_loopclosure(self, c2w, d_phi, d_theta, d_r): 
        frame = self.vc_opts.video_length
        
        d_r = d_r*c2w[0,2,3].cpu()
        theta_list = [0, d_theta/2., d_theta, 0]
        phi_list = [0, d_phi/2., d_phi, 0]
        r_list = [0, d_r/2., d_r, 0]
        
        thetas = txt_interpolation(theta_list, frame, mode="smooth")
        thetas[0] = theta_list[0]
        thetas[-1] = theta_list[-1]

        phis = txt_interpolation(phi_list, frame, mode="smooth")
        phis[0] = phi_list[0]
        phis[-1] = phi_list[-1]

        rs = txt_interpolation(r_list, frame, mode="smooth")
        rs[0] = r_list[0]
        rs[-1] = r_list[-1]

        c2w_traj = []
        for th, ph, r in zip(thetas, phis, rs):
            c2w_new = sphere2pose(c2w, np.float32(th), np.float32(ph), np.float32(r), self.device)
            c2w_traj.append(c2w_new)
        c2w_traj = torch.cat(c2w_traj, dim=0)

        return c2w_traj # [25, 4, 4]

    def preprocess_video_diffusion(self, which_train_view=5, defined_camera_traj_c2ws=None, pc_render_single_view=True):
 
        # idx = np.random.choice(self.train_poses.shape[0]) # sample

        idx = which_train_view
        
        print("=> Camera trajectory is only supported around training views. Current idx: {}. ".format(idx))
        
        if pc_render_single_view: 
            imgs = np.array(self.scene.imgs)[[idx]]
            pcd = [self.pcd[idx]]
        else: 
            imgs = np.array(self.scene.imgs)
            pcd = torch.stack(self.pcd, 0)
        
        img_ori = (self.d_images[idx]['img_ori'].squeeze(0).permute(1,2,0)+1.)/2. # [512,704,3] [0,1]
        c2ws = self.c2ws[[idx]]
        principal_points = self.principal_points[[idx]]
        focals = self.focals[[idx]]
        depth = [self.depth[idx]]
        H, W = self.d_H, self.d_W
        # print("===> check type: ", pcd[0].shape, type(c2ws), c2ws.shape, type(principal_points), principal_points.shape, type(focals), focals.shape)

        depth_avg = depth[-1][H//2,W//2]
        radius = depth_avg*self.vc_opts.center_scale

        c2ws, pcd, transform_back = world_point_to_obj_my(
            poses=c2ws, points=torch.stack(pcd) if pc_render_single_view else pcd, 
            k=-1, r=radius, elevation=self.vc_opts.elevation, device=self.device)
        masks = None
        
        if defined_camera_traj_c2ws is None: 
            camera_traj, num_views, camera_traj_c2ws, traj_back = self.get_camera_traj(c2ws, H, W, focals, principal_points)
        else: 
            ori_traj_c2ws = defined_camera_traj_c2ws.clone()
            # be very careful here: 
            defined_camera_traj_c2ws = torch.bmm(torch.linalg.inv(transform_back).unsqueeze(0).expand_as(defined_camera_traj_c2ws), 
                                                 defined_camera_traj_c2ws)


            R, T = defined_camera_traj_c2ws[:, :3, :3], defined_camera_traj_c2ws[:, :3, 3:]
            R = torch.stack([-R[:,:, 0], -R[:,:, 1], R[:,:, 2]], 2)
            new_c2w = torch.cat([R, T], 2)
            w2c = torch.linalg.inv(torch.cat((new_c2w, torch.Tensor([[[0,0,0,1]]]).to(self.device).repeat(new_c2w.shape[0],1,1)),1))
            R_new, T_new = w2c[:,:3, :3].permute(0,2,1), w2c[:,:3, 3]
            image_size = ((H, W),)  # (h, w)
            camera_traj = PerspectiveCameras(focal_length=focals, principal_point=principal_points, 
                                             in_ndc=False, image_size=image_size, R=R_new, T=T_new, 
                                             device=self.device)

            camera_traj_c2ws = defined_camera_traj_c2ws
            num_views = defined_camera_traj_c2ws.shape[0]
            traj_back = False
        
        if pc_render_single_view: 
            render_results, viewmask = self.view_crafter.run_render(
                [pcd[-1]], [imgs[-1]], masks, H, W, camera_traj, num_views)
        else: 
            render_results, viewmask = self.view_crafter.run_render(
                [pcd[j] for j in range(pcd.shape[0])], [imgs[j] for j in range(imgs.shape[0])], 
                masks, H, W, camera_traj, num_views)
        
        render_results = F.interpolate(render_results.permute(0,3,1,2), 
                                       size=(self.vc_opts.height, self.vc_opts.width), 
                                       mode='bilinear', align_corners=False).permute(0,2,3,1)
        # [n_frames, H, W, 3]
        print("=> Point cloud render done. ")


        render_results[0] = img_ori
        if self.vc_opts.mode == 'single_view_txt':
            if traj_back:
                render_results[-1] = img_ori
        save_video(render_results, os.path.join(self.save_dir, 'render0.mp4'))

        camera_traj_c2ws = torch.bmm(transform_back.unsqueeze(0).expand_as(camera_traj_c2ws), camera_traj_c2ws)
        if defined_camera_traj_c2ws is not None: 
            camera_traj_c2ws = ori_traj_c2ws # avoid some numerical issue
        
        return render_results, camera_traj_c2ws

    def run_video_diffusion(
            self, 
            point_cloud_render_results, 
            guidance_images, guidance_masks=None, 
            guidance_depths=None, no_guidance=False): 
        # guidance_images: [n_frames, 3, H, W]. rendered from 3dgs
        # remember to alight two resolutions
        
        print("=> Running video diffusion...")
        if self.loss_guidance_fn is not None: 
            self.loss_guidance_fn.set_guidance_images(guidance_images)
            if guidance_masks is not None: 
                self.loss_guidance_fn.set_guidance_masks(guidance_masks)
            if guidance_depths is not None: 
                self.loss_guidance_fn.set_guidance_depths(guidance_depths)

        diffusion_results = self.view_crafter.run_diffusion(point_cloud_render_results, self.loss_guidance_fn, no_guidance)
        result_images = (diffusion_results + 1.0) / 2.0
        print("=> Video diffusion done. ")

        save_video(result_images, os.path.join(self.save_dir, 'diffusion0.mp4'))
        
        result_images = result_images.permute(0, 3, 1, 2) # [n_frames, 3, H, W]
        return result_images
        
    def parse_cameras(self, cams): 
        n_cams = len(cams)
        poses_w2c = []
        poses_c2w = []
        intrinsics = []
        for i in range(n_cams): 
            cam = cams[i]
            # fovx, fovy = cam.FoVx, cam.FoVy
            # w, h = cam.image_width, cam.image_height
            fovx, fovy = cam.FovX, cam.FovY
            w, h = cam.width, cam.height

            fx = fov2focal(fovx, w)
            fy = fov2focal(fovy, h)
            intrinsic = np.array([
                [fx, 0, w//2], 
                [0, fy, h//2], 
                [0, 0, 1]], dtype=np.float32)
            intrinsics.append(intrinsic)
            Rt = np.zeros((4, 4))
            Rt[:3, :3] = cam.R.transpose()
            Rt[:3, 3] = cam.T
            Rt[3, 3] = 1.0
            poses_w2c.append(Rt)
            poses_c2w.append(np.linalg.inv(Rt))
        return np.stack(poses_c2w, 0), np.stack(intrinsics, 0), h, w
    
    def decide_unobserved_regions(self, gs_render_results): 
        # gs_render_results: [N, 3, H, W]. tensor, 0-1

        device = gs_render_results.device
        gs_render_results = gs_render_results.sum(1) # [N, H, W]
        gs_render_results = gs_render_results.cpu().numpy()
        masks = []
        for i in range(gs_render_results.shape[0]): 
            mask_i = ((gs_render_results[i] == 0.)).astype(np.float32)
            mask_i = self.mask_erosion(mask_i, size=3) # [H, W]
            mask_i = self.mask_dilation(mask_i)
            masks.append(mask_i[None]) # [1, H, W]
        masks = np.stack(masks, 0)
        masks = torch.from_numpy(masks).to(device)
        return masks
    
    def mask_erosion(self, input_mask, size=3): 
        input_mask = ndimage.binary_erosion(input_mask, structure=np.ones((size, size))).astype(input_mask.dtype)
        return input_mask
    
    def mask_dilation(self, input_mask, size=5): 
        # 0-1
        input_mask = ndimage.binary_dilation(input_mask, structure=np.ones((size, size))).astype(input_mask.dtype)
        return input_mask
    
    def process_mask(self, input_mask): 
        # [N, 1, H, W]. 0-1, float32
        device = input_mask.device
        input_mask = input_mask[:, 0].cpu().numpy() # [N, H, W]
        masks = []
        for i in range(input_mask.shape[0]): 
            mask_i = self.mask_erosion(input_mask[i], size=5)
            # mask_i = self.mask_dilation(mask_i, size=5)
            masks.append(mask_i[None])
        masks = np.stack(masks, 0) # [N, 1, H, W]
        masks = torch.from_numpy(masks).to(device)
        return masks

    def process_mask2(self, input_mask): 
        # [N, 1, H, W]. 0-1, float32
        device = input_mask.device
        input_mask = input_mask[:, 0].cpu().numpy() # [N, H, W]
        masks = []
        for i in range(input_mask.shape[0]): 
            mask_i = self.mask_erosion(input_mask[i], size=5)
            mask_i = self.mask_dilation(mask_i, size=10)
            masks.append(mask_i[None])
        masks = np.stack(masks, 0) # [N, 1, H, W]
        masks = torch.from_numpy(masks).to(device)
        return masks


def log_lerp(t, v0, v1):
    """Interpolate log-linearly from `v0` (t=0) to `v1` (t=1)."""
    if v0 <= 0 or v1 <= 0:
        raise ValueError(f'Interpolants {v0} and {v1} must be positive.')
    lv0 = np.log(v0)
    lv1 = np.log(v1)
    return np.exp(np.clip(t, 0, 1) * (lv1 - lv0) + lv0)


def learning_rate_decay(step,
                        lr_init,
                        lr_final,
                        max_steps,
                        lr_delay_steps=0,
                        lr_delay_mult=1):
    """Continuous learning rate decay function.
  The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
  is log-linearly interpolated elsewhere (equivalent to exponential decay).
  If lr_delay_steps>0 then the learning rate will be scaled by some smooth
  function of lr_delay_mult, such that the initial learning rate is
  lr_init*lr_delay_mult at the beginning of optimization but will be eased back
  to the normal learning rate when steps>lr_delay_steps.
  Args:
    step: int, the current optimization step.
    lr_init: float, the initial learning rate.
    lr_final: float, the final learning rate.
    max_steps: int, the number of steps during optimization.
    lr_delay_steps: int, the number of steps to delay the full learning rate.
    lr_delay_mult: float, the multiplier on the rate when delaying it.
  Returns:
    lr: the learning for current step 'step'.
  """
    if lr_delay_steps > 0:
        delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
            0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1))
    else:
        delay_rate = 1.
    return delay_rate * log_lerp(step / max_steps, lr_init, lr_final)
