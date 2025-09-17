import numpy as np
import torch
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class MiDasDepthEstimator(): 
    def __init__(self, device="cuda:0"): 
        # self.midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
        self.midas = torch.hub.load("/home/ma-user/.cache/torch/hub/intel-isl_MiDaS_master", "DPT_Hybrid", trust_repo=True, source="local")
        self.midas.to(device)
        self.midas.eval()
        # self.midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.midas_transforms = torch.hub.load("/home/ma-user/.cache/torch/hub/intel-isl_MiDaS_master", "transforms", trust_repo=True, source="local")
        self.device = device
    
    def get_rel_depth(self, img): 
        # img: [n, 3, H, W]. [-1, 1]
        if img.dim() == 4: 
            h, w = img.shape[2:]
        else: 
            h, w = img.shape[1:]
            img = img[None]

        # norm_img = (img - 0.5) / 0.5
        norm_img = img.clamp(-1., 1.)
        norm_img = torch.nn.functional.interpolate(
            norm_img,
            size=(384, 512),
            mode="bicubic",
            align_corners=False)
        
        prediction = self.midas(norm_img) # [n, 384, 512]
        prediction = torch.nn.functional.interpolate(
                        prediction.unsqueeze(1),
                        size=(h, w),
                        mode="bicubic",
                        align_corners=False).squeeze() # [n, H, W]
        
        return prediction
    
    def requires_grad(self, requires_grad): 
        for param in self.midas.parameters(): 
            param.requires_grad = requires_grad

    """https://github.com/isl-org/MiDaS/issues/26"""
    def get_scaleshift(
            self, 
            midas_depth, 
            real_depth, 
            mask, 
            lstsq_strategy="each"): 
        # midas_depth: [N, H, W]
        # real_depth: [N, H, W]
        # mask: [N, H, W]
        print("=> in scaleshift: ", midas_depth.shape, real_depth.shape, mask.shape)
        n_img = midas_depth.shape[0]

        # normalize midas depth to 0...1
        
        # midas_depth = midas_depth * mask

        max_midas_depth = torch.max(midas_depth.view(n_img, -1), 1)[0] # [N, ]
        mdias_depth = midas_depth / max_midas_depth[:, None, None] # [N, H, W]

        midas_depth = midas_depth.view(n_img, -1)
        real_depth = real_depth.view(n_img, -1)
        mask = mask.view(n_img, -1)

        device = midas_depth.device
        midas_depth = midas_depth.cpu().numpy()
        real_depth = real_depth.cpu().numpy()
        mask = mask.cpu().numpy()
        mask = mask.astype(np.bool)
        
        scale, shift = None, None

        if lstsq_strategy == "first": 
            # only use the first frame to get the scale and shift
            raise NotImplementedError
        
        elif lstsq_strategy == "each": 
            # each frame is assigned with an independent scale and shift
            scale, shift = [], []
            
            for i in range(n_img): 
                x = midas_depth[i][mask[i]]
                y = 1./real_depth[i][mask[i]]
                A = np.vstack([x, 1-x]).T

                s, t = np.linalg.lstsq(A, y, rcond=None)[0]

                min_depth = 1/s
                max_depth = 1/t

                A = (1 / min_depth) - (1/ max_depth)
                B = 1 / max_depth

                scale.append(A)
                shift.append(B)

            scale = torch.from_numpy(np.array(scale)).to(device)
            shift = torch.from_numpy(np.array(shift)).to(device)

        elif lstsq_strategy == "all": 
            # all frames share a single scale/shift
            raise NotImplementedError
        
        else: 
            raise NotImplementedError
        
        
        return scale, shift # [N, ], [N, ]
    

    def convert_rel_to_real(self, midas_depth, scale, shift):
        # midas_depth: [N, H, W]. midas tenosr. 
        # scale/shift: [N, ]. tensor. 
        # each image should have a different scale/trans
        real_depth = 1. / (scale[:, None, None] * midas_depth + shift[:, None, None])
        return real_depth

        
        
        