import os, json
import cv2
import torch
import numpy as np
from PIL import Image
import imageio
import math

def auto_orient_and_center_poses(
    poses,
    method,
    center_method):  

    """Orients and centers the poses.

    We provide three methods for orientation:

    - pca: Orient the poses so that the principal directions of the camera centers are aligned
        with the axes, Z corresponding to the smallest principal component.
        This method works well when all of the cameras are in the same plane, for example when
        images are taken using a mobile robot.
    - up: Orient the poses so that the average up vector is aligned with the z axis.
        This method works well when images are not at arbitrary angles.
    - vertical: Orient the poses so that the Z 3D direction projects close to the
        y axis in images. This method works better if cameras are not all
        looking in the same 3D direction, which may happen in camera arrays or in LLFF.

    There are two centering methods:

    - poses: The poses are centered around the origin.
    - focus: The origin is set to the focus of attention of all cameras (the
        closest point to cameras optical axes). Recommended for inward-looking
        camera configurations.

    Args:
        poses: The poses to orient.
        method: The method to use for orientation.
        center_method: The method to use to center the poses.

    Returns:
        Tuple of the oriented poses and the transform matrix.
    """

    origins = poses[..., :3, 3]

    mean_origin = torch.mean(origins, dim=0)
    translation_diff = origins - mean_origin

    if center_method == "poses":
        translation = mean_origin
    elif center_method == "focus":
        translation = focus_of_attention(poses, mean_origin)
    elif center_method == "none":
        translation = torch.zeros_like(mean_origin)
    else:
        raise ValueError(f"Unknown value for center_method: {center_method}")

    if method == "pca":
        _, eigvec = torch.linalg.eigh(translation_diff.T @ translation_diff)
        eigvec = torch.flip(eigvec, dims=(-1,))

        if torch.linalg.det(eigvec) < 0:
            eigvec[:, 2] = -eigvec[:, 2]

        transform = torch.cat([eigvec, eigvec @ -translation[..., None]], dim=-1)
        oriented_poses = transform @ poses

        if oriented_poses.mean(dim=0)[2, 1] < 0:
            oriented_poses[:, 1:3] = -1 * oriented_poses[:, 1:3]
    elif method in ("up", "vertical"):
        up = torch.mean(poses[:, :3, 1], dim=0)
        up = up / torch.linalg.norm(up)
        if method == "vertical":
            # If cameras are not all parallel (e.g. not in an LLFF configuration),
            # we can find the 3D direction that most projects vertically in all
            # cameras by minimizing ||Xu|| s.t. ||u||=1. This total least squares
            # problem is solved by SVD.
            x_axis_matrix = poses[:, :3, 0]
            _, S, Vh = torch.linalg.svd(x_axis_matrix, full_matrices=False)
            # Singular values are S_i=||Xv_i|| for each right singular vector v_i.
            # ||S|| = sqrt(n) because lines of X are all unit vectors and the v_i
            # are an orthonormal basis.
            # ||Xv_i|| = sqrt(sum(dot(x_axis_j,v_i)^2)), thus S_i/sqrt(n) is the
            # RMS of cosines between x axes and v_i. If the second smallest singular
            # value corresponds to an angle error less than 10° (cos(80°)=0.17),
            # this is probably a degenerate camera configuration (typical values
            # are around 5° average error for the true vertical). In this case,
            # rather than taking the vector corresponding to the smallest singular
            # value, we project the "up" vector on the plane spanned by the two
            # best singular vectors. We could also just fallback to the "up"
            # solution.
            if S[1] > 0.17 * math.sqrt(poses.shape[0]):
                # regular non-degenerate configuration
                up_vertical = Vh[2, :]
                # It may be pointing up or down. Use "up" to disambiguate the sign.
                up = up_vertical if torch.dot(up_vertical, up) > 0 else -up_vertical
            else:
                # Degenerate configuration: project "up" on the plane spanned by
                # the last two right singular vectors (which are orthogonal to the
                # first). v_0 is a unit vector, no need to divide by its norm when
                # projecting.
                up = up - Vh[0, :] * torch.dot(up, Vh[0, :])
                # re-normalize
                up = up / torch.linalg.norm(up)

        rotation = rotation_matrix(up, torch.Tensor([0, 0, 1]))
        transform = torch.cat([rotation, rotation @ -translation[..., None]], dim=-1)
        oriented_poses = transform @ poses
    elif method == "none":
        transform = torch.eye(4)
        transform[:3, 3] = -translation
        transform = transform[:3, :]
        oriented_poses = transform @ poses
    else:
        raise ValueError(f"Unknown value for method: {method}")

    return oriented_poses, transform


def rotation_matrix(a, b):
    """Compute the rotation matrix that rotates vector a to vector b.

    Args:
        a: The vector to rotate.
        b: The vector to rotate to.
    Returns:
        The rotation matrix.
    """
    a = a / torch.linalg.norm(a)
    b = b / torch.linalg.norm(b)
    v = torch.cross(a, b)
    c = torch.dot(a, b)
    # If vectors are exactly opposite, we add a little noise to one of them
    if c < -1 + 1e-8:
        eps = (torch.rand(3) - 0.5) * 0.01
        return rotation_matrix(a + eps, b)
    s = torch.linalg.norm(v)
    skew_sym_mat = torch.Tensor(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ]
    )
    return torch.eye(3) + skew_sym_mat + skew_sym_mat @ skew_sym_mat * ((1 - c) / (s**2 + 1e-8))


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



def load_scannetpp_data(basedir): 
    meta_path = os.path.join(basedir, "dslr/nerfstudio/transforms_undistorted.json")
    with open(meta_path, encoding="UTF-8") as file: 
        meta = json.load(file)
    
    # get the intrinsics
    intrinsics = np.eye(3)
    intrinsics[0, 0] = meta["fl_x"]
    intrinsics[1, 1] = meta["fl_y"]
    intrinsics[0, 2] = meta["cx"]
    intrinsics[1, 2] = meta["cy"]
    H, W = meta["h"], meta["w"]
    focal = intrinsics[0, 0] # useless
    
    data_dir = os.path.join(basedir, "dslr/undistorted_images")
    image_filenames = []
    poses = []
    frames_suffix = [] # get the number
    frames = meta["frames"] + meta["test_frames"]
    frames.sort(key=lambda x: x["file_path"])

    for idx, frame in enumerate(frames): 
        filepath = frame["file_path"]
        frames_suffix.append(int(filepath[3:8])) # '03870'
        fname = os.path.join(data_dir, filepath)
        image_filenames.append(fname)
        poses.append(np.array(frame["transform_matrix"]))
    
    # process the pose
    poses = torch.from_numpy(np.array(poses).astype(np.float32)).cuda()
    poses, transform_matrix = auto_orient_and_center_poses(poses, method="up", center_method="poses")

    # scale poses
    scale_factor = 1.0
    config_scale_factor = 1.0
    auto_scale_poses = True
    if auto_scale_poses: 
        scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))
    scale_factor *= config_scale_factor
    poses[:, :3, 3] *= scale_factor
    poses = poses.cpu().numpy().astype(np.float32) # [n, 3, 4]

    # NOTE: very important
    # convert from opengl (ruback) to opencv (rdforward)
    R = poses[:, :3, :3] # [n, 3, 3]
    T = poses[:, :3, 3:] # [n, 3, 1]
    R = np.stack([R[:, :, 0], -R[:, :, 1], -R[:, :, 2]], -1) # [n, 3, 3]
    poses = np.concatenate([R, T], -1) # [n, 3, 4]
 

    pad_row = np.array([[0, 0, 0, 1]], dtype=np.float32) # [1, 4]
    pad_row = np.repeat(pad_row[None], poses.shape[0], 0) # [n, 1, 4]
    poses = np.concatenate([poses, pad_row], 1)


    poses_w2c = []
    for idx in range(poses.shape[0]): 
        poses_w2c_i = np.linalg.inv(poses[idx])
        poses_w2c.append(poses_w2c_i)

    poses_w2c = np.stack(poses_w2c, 0)
    poses_w2c = poses_w2c.astype(np.float32) # [n, 4, 4]
    
    intrinsics = np.repeat(intrinsics[None], poses.shape[0], 0)
    intrinsics = intrinsics[:, :3, :3].astype(np.float32) # [n, 3, 3]

    return poses_w2c, intrinsics, H, W
    

def pipeline(scene, base_path):
    path = os.path.join(base_path, scene)

    poses_w2c, intrinsics, H, W = load_scannetpp_data(path)
    sparse_path = os.path.join(path, "sparse/0")
    os.makedirs(sparse_path, exist_ok=True)
    
    save_cameras(intrinsics, sparse_path, H, W)
    save_imagestxt(poses_w2c, sparse_path)
    
    print(path)


if __name__ == "__main__": 
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    for scene in ['8a20d62ac0', '94ee15e8ba', '7831862f02', 'a29cccc784']: 
        pipeline(scene, base_path = '/home/ma-user/work/zhongyj/code/guidedvd_release/dataset/scannetpp')  # please use absolute path!


