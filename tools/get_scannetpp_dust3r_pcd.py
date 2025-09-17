import os, sys

sys.path.append("./")
import numpy as np
from scene.dataset_readers import readColmapSceneInfo
from utils.graphics_utils import fov2focal
from tools.dust3r_to_colmap import convert_dust3r_to_colmap

dust3r_model_path = "third_party/ViewCrafter/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
scenes = ['8a20d62ac0', '94ee15e8ba', '7831862f02', 'a29cccc784']

for scene in scenes: 
    path = f"dataset/scannetpp/{scene}"
    images = "dslr/undistorted_images"
    dataset = "Scannetpp"
    eval = True
    n_views = 6
    dust3r_min_conf_thr = 1
    demo_setting = False

    scene_id = path.split("/")[-1]
    scene = scene_id

    if demo_setting: 
        img_dir_for_dust3r = os.path.join("./dust3r_results/Scannetpp_{}v_thr{}_trimeshsave_minconf1_demosetting/{}".format(n_views, int(dust3r_min_conf_thr), scene))
    else: 
        img_dir_for_dust3r = os.path.join("./dust3r_results/Scannetpp_{}v_thr{}_trimeshsave_minconf1/{}".format(n_views, int(dust3r_min_conf_thr), scene))
    
    train_cam_infos = readColmapSceneInfo(path, images, dataset, eval, n_views, 
                                            dust3r_min_conf_thr, demo_setting, get_dust3r_pcd=True)
    
    train_img_paths = [train_cam_info.image_path for train_cam_info in train_cam_infos]

    known_c2w, known_focal = [], []
    for train_cam_info in train_cam_infos: 
        Rt = np.zeros((4, 4))
        Rt[:3, :3] = train_cam_info.R.transpose()
        Rt[:3, 3] = train_cam_info.T
        Rt[3, 3] = 1.0
        
        C2W = np.linalg.inv(Rt)
        known_c2w.append(C2W)
        known_focal.append(fov2focal(train_cam_info.FovX, train_cam_info.width) / (max(train_cam_info.width, train_cam_info.height)/512))
    
    scale_factor = 1
    os.makedirs(img_dir_for_dust3r, exist_ok=True)
    convert_dust3r_to_colmap(image_files=train_img_paths, save_dir=img_dir_for_dust3r, 
                                        min_conf_thr=dust3r_min_conf_thr, model_path=dust3r_model_path, 
                                        known_c2w=known_c2w, known_focal=known_focal, no_mask_pc=False, 
                                        scale_factor=scale_factor)