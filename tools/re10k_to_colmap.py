import os
import cv2
import numpy as np
from PIL import Image
import imageio
import math
from pathlib import Path

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

def save_cameras(intrinsics, sparse_path, H, W):
    # Save cameras.txt
    # cameras_file = sparse_path / 'cameras.txt'
    cameras_file = os.path.join(sparse_path, 'cameras.txt')
    with open(cameras_file, 'w') as cameras_file:
        cameras_file.write("# Camera list with one line of data per camera:\n")
        cameras_file.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for i, intrinsic in enumerate(intrinsics):
            cameras_file.write(f"{i} PINHOLE {W} {H} {intrinsic[0, 0]} {intrinsic[1, 1]} {intrinsic[0, 2]} {intrinsic[1, 2]}\n")
            
def save_imagestxt(world2cam, sparse_path):
     # Save images.txt
    # images_file = sparse_path / 'images.txt'
    images_file = os.path.join(sparse_path, 'images.txt')
    # Generate images.txt content
    with open(images_file, 'w') as images_file:
        images_file.write("# Image list with two lines of data per image:\n")
        images_file.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        images_file.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for i in range(world2cam.shape[0]):
            # Convert rotation matrix to quaternion
            rotation_matrix = world2cam[i, :3, :3]
            qw, qx, qy, qz = rotmat2qvec(rotation_matrix)
            tx, ty, tz = world2cam[i, :3, 3]
            images_file.write(f"{i} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {i} {i}.png\n")
            images_file.write("\n") # Placeholder for points, assuming no points are associated with images here



def load_re10k_data(scene_dir): 
    import json
    blender2opencv = np.array(
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    )
    example_path = scene_dir / "transforms.json"
    with open(example_path, "r") as f:
        meta_data = json.load(f)
    
    store_h, store_w = meta_data["h"], meta_data["w"]
    fx, fy, cx, cy = (
        meta_data["fl_x"],
        meta_data["fl_y"],
        meta_data["cx"],
        meta_data["cy"],
    )

    w2cs = []
    intrinsics = []
    for frame in meta_data["frames"]: 
        intrinsics.append(
            np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        )

        w2c = np.linalg.inv(np.array(frame["transform_matrix"]) @ blender2opencv)
        w2cs.append(w2c)
    
    intrinsics = np.stack(intrinsics, 0)
    w2cs = np.stack(w2cs, 0)
    return w2cs, intrinsics, store_h, store_w


def pipeline(scene_dir):
    poses_w2c, intrinsics, H, W = load_re10k_data(scene_dir)
    sparse_path = os.path.join(scene_dir, "sparse/0")
    os.makedirs(sparse_path, exist_ok=True)
    
    save_cameras(intrinsics, sparse_path, H, W)
    save_imagestxt(poses_w2c, sparse_path)
    
    print(scene_dir)


data_dir = '/home/ma-user/work/zhongyj/code/guidedvd_release/dataset/re10k'
read_all_scenes = os.listdir(data_dir)
all_scenes = []
for per_scene in read_all_scenes: 
    if ".DS_Store" in per_scene: 
        continue
    all_scenes.append(per_scene)
print(all_scenes)
for per_scene in all_scenes: 
    scene_dir = Path(os.path.join(data_dir, per_scene))
    print(scene_dir)
    pipeline(scene_dir)