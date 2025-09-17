import os, sys

sys.path.append("./")
import numpy as np
from PIL import Image
from scene.dataset_readers import readColmapSceneInfo
from scene.pcd2img import project_point_cloud_to_image
from utils.graphics_utils import fov2focal

scenes = ['room_0/Sequence_2', 'room_1/Sequence_1', 'room_2/Sequence_1', 'office_2/Sequence_2', 'office_3/Sequence_1', 'office_4/Sequence_2']

for scene in scenes: 
    path = f"dataset/Replica/{scene}"
    images = "rgb"
    dataset = "Replica"
    eval = True
    n_views = 6
    dust3r_min_conf_thr = 1
    demo_setting = False
    gap = 6

    split = path.split("/")
    scene, seq = split[-2], split[-1]
    scene, scene_id = scene.split("_")[0], scene.split("_")[1]
    seq_id = seq.split("_")[1]
    scene = scene+scene_id+"_seq"+seq_id

    projected_dir = os.path.join("./projected_dir/", scene)
    os.makedirs(projected_dir, exist_ok=True)
    
    scene_info = readColmapSceneInfo(path, images, dataset, eval, n_views, 
                                            dust3r_min_conf_thr, demo_setting, get_dust3r_pcd=False)
    cam_infos = scene_info.all_cameras
    pcd = scene_info.point_cloud
    pcd_pos = pcd.points
    pcd_colors = (pcd.colors * 255).astype(np.uint8)
    
    
    for cam_info in cam_infos[::gap]: 
        w2c = np.zeros((4, 4))
        w2c[:3, :3] = cam_info.R.transpose()
        w2c[:3, 3] = cam_info.T
        w2c[3, 3] = 1.0 # [4, 4]
        
        fx = fov2focal(cam_info.FovX, cam_info.width)
        fy = fov2focal(cam_info.FovY, cam_info.height)

        K = np.array([
            [fx, 0, 0.5*cam_info.width],
            [0, fy, 0.5*cam_info.height],
            [0, 0, 1]
        ]) # [3, 3]

        projected_image, projected_mask = project_point_cloud_to_image(pcd_pos, pcd_colors, K, w2c, cam_info.width, cam_info.height)
        projected_image = Image.fromarray(projected_image)
        projected_mask = Image.fromarray((projected_mask*255).astype(np.uint8))
        image_name = cam_info.image_name
        save_path = os.path.join(projected_dir, image_name+".png")
        save_path_mask = os.path.join(projected_dir, image_name+"_mask.png")
        print("saving at: ", save_path, save_path_mask)
        projected_image.save(save_path)
        projected_mask.save(save_path_mask)
