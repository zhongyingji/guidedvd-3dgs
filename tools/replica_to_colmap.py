import os
import cv2
import numpy as np
from PIL import Image
import imageio
import math

convert_mat = np.array([
    [1, 0, 0, 0], 
    [0, 1, 0, 0], 
    [0, 0, 1, 0], 
    [0, 0, 0, 1]
])

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



def load_replica_data(basedir): 
    poses_c2w = []
    poses_w2c = []
    with open(os.path.join(basedir, 'traj_w_c.txt'), 'r') as fp:
        for line in fp:
            tokens = line.split(' ')
            tokens = [float(token) for token in tokens]
            tokens = np.array(tokens).reshape(4, 4)
            poses_c2w.append(tokens)
            poses_w2c.append(np.linalg.inv(tokens))
    poses_c2w =  np.stack(poses_c2w, 0)
    poses_w2c =  np.stack(poses_w2c, 0)
    nframes = poses_c2w.shape[0]
    
    
    fname = os.path.join(basedir, 'rgb', "rgb_0.png")
    img = imageio.imread(fname)
    H, W = img.shape[:2]
    hfov = 90
    focal = W / 2.0 / math.tan(math.radians(hfov / 2.0))

    K = np.array([
        [focal, 0, 0.5*W],
        [0, focal, 0.5*H],
        [0, 0, 1]
    ])
    intrinsics = np.repeat(K[None], nframes, 0)
    intrinsics = intrinsics[:, :3, :3].astype(np.float32) # [n, 3, 3]
    poses_w2c = poses_w2c[:, :3].astype(np.float32) # [n, 3, 4]

    return poses_w2c, intrinsics, H, W


def pipeline(scene, base_path):
    path = os.path.join(base_path, scene)

    poses_w2c, intrinsics, H, W = load_replica_data(path)
    sparse_path = os.path.join(path, "sparse/0")
    os.makedirs(sparse_path, exist_ok=True)
    
    save_cameras(intrinsics, sparse_path, H, W)
    save_imagestxt(poses_w2c, sparse_path)
    
    print(path)


for scene in ['office_2/Sequence_2', 'office_3/Sequence_1', 'office_4/Sequence_2', 'room_0/Sequence_2', 'room_1/Sequence_1', 'room_2/Sequence_1']: 
    pipeline(scene, base_path = '/home/ma-user/work/zhongyj/code/guidedvd_release/dataset/Replica')  # please use absolute path!
