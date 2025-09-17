import os, sys
from typing import NamedTuple, Optional
import cv2  # Assuming OpenCV is used for image saving
import numpy as np
import torch
import trimesh
from scene.gaussian_model import BasicPointCloud
from PIL import Image
from scene.colmap_loader import rotmat2qvec
from utils.graphics_utils import focal2fov, fov2focal
# from scene.dataset_readers import storePly
from plyfile import PlyData, PlyElement
from pathlib import Path

sys.path.append("./third_party/ViewCrafter/extern/dust3r/")
from dust3r.inference import inference, load_model
from dust3r.utils.image import load_images
from dust3r.utils.device import to_numpy
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode


class CameraInfo(NamedTuple):
    uid: int
    R: np.ndarray
    T: np.ndarray
    FovY: np.ndarray
    FovX: np.ndarray
    image: np.ndarray
    image_path: str
    image_name: str
    width: int
    height: int
    mask: Optional[np.ndarray] = None
    mono_depth: Optional[np.ndarray] = None


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    render_cameras: Optional[list[CameraInfo]] = None
    

def init_filestructure(save_path):
    save_path = Path(save_path)
    save_path.mkdir(exist_ok=True, parents=True)
    
    images_path = save_path / 'images'
    masks_path = save_path / 'masks'
    sparse_path = save_path / 'sparse/0'
    depth_maps_path = save_path / 'depth_maps'
    
    images_path.mkdir(exist_ok=True, parents=True)
    masks_path.mkdir(exist_ok=True, parents=True)    
    sparse_path.mkdir(exist_ok=True, parents=True)
    depth_maps_path.mkdir(exist_ok=True, parents=True)
    
    return save_path, images_path, masks_path, sparse_path, depth_maps_path


def save_images_masks(imgs, masks, depth_maps, images_path, masks_path, depth_maps_path):
    # Saving images and optionally masks/depth maps
    for i, (image, mask, depth_map) in enumerate(zip(imgs, masks, depth_maps)):
        image_save_path = images_path / f"{i}.png"
        mask_save_path = masks_path / f"{i}.png"
        depth_map_save_path = depth_maps_path / f"{i}.npy"

        image[~mask] = 1.
        rgb_image = cv2.cvtColor(image*255, cv2.COLOR_BGR2RGB)
        cv2.imwrite(str(image_save_path), rgb_image)
        
        mask = np.repeat(np.expand_dims(mask, -1), 3, axis=2)*255
        Image.fromarray(mask.astype(np.uint8)).save(mask_save_path)

        with open(depth_map_save_path, "wb") as f:
            np.save(f, depth_map)


def read_masks(masks_path): 
    mask_files = os.listdir(masks_path)
    mask_files = sorted(mask_files, key=lambda x: int(x.split(".")[0]))
    masks = []
    for mask_file in mask_files: 
        mask = cv2.imread(os.path.join(masks_path, mask_file))[..., 0] # [h, w]
        masks.append(mask)
    return np.stack(masks, 0) # [n, h, w]


def read_depth_maps(depth_maps_path): 
    depth_files = os.listdir(depth_maps_path)
    depth_maps = []
    depth_maps = sorted(depth_maps, key=lambda x: int(x.split(".")[0]))
    for depth_file in depth_files: 
        with open(os.path.join(depth_maps_path, depth_file), "rb") as f: 
            depth_map = np.load(f)
        depth_maps.append(depth_map)
    return np.stack(depth_maps, 0)
    

def save_cameras(focals, principal_points, sparse_path, imgs_shape):
    # Save cameras.txt
    cameras_file = sparse_path / 'cameras.txt'
    with open(cameras_file, 'w') as cameras_file:
        cameras_file.write("# Camera list with one line of data per camera:\n")
        cameras_file.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for i, (focal, pp) in enumerate(zip(focals, principal_points)):
            cameras_file.write(f"{i} PINHOLE {imgs_shape[2]} {imgs_shape[1]} {focal[0]} {focal[0]} {pp[0]} {pp[1]}\n")


def save_imagestxt(world2cam, sparse_path):
     # Save images.txt
    images_file = sparse_path / 'images.txt'
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


def save_pointcloud_with_normals(imgs, pts3d, msk, sparse_path):
    pc = get_pc(imgs, pts3d, msk)  # Assuming get_pc is defined elsewhere and returns a trimesh point cloud

    # Define a default normal, e.g., [0, 1, 0]
    default_normal = [0, 1, 0]

    # Prepare vertices, colors, and normals for saving
    vertices = pc.vertices
    colors = pc.colors
    normals = np.tile(default_normal, (vertices.shape[0], 1))

    save_path = sparse_path / 'points3D.ply'

    # Construct the header of the PLY file
    header = """ply
format ascii 1.0
element vertex {}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property float nx
property float ny
property float nz
end_header
""".format(len(vertices))

    # Write the PLY file
    with open(save_path, 'w') as ply_file:
        ply_file.write(header)
        for vertex, color, normal in zip(vertices, colors, normals):
            ply_file.write('{} {} {} {} {} {} {} {} {}\n'.format(
                vertex[0], vertex[1], vertex[2],
                int(color[0]), int(color[1]), int(color[2]),
                normal[0], normal[1], normal[2]
            ))
            

def get_pc(imgs, pts3d, mask):
    imgs = to_numpy(imgs)
    pts3d = to_numpy(pts3d)
    mask = to_numpy(mask)
    
    pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
    col = np.concatenate([p[m] for p, m in zip(imgs, mask)])
    
    pts = pts.reshape(-1, 3)# [::3]
    col = col.reshape(-1, 3)# [::3]
    
    #mock normals:
    normals = np.tile([0, 1, 0], (pts.shape[0], 1))
    
    pct = trimesh.PointCloud(pts, colors=col)
    pct.vertices_normal = normals  # Manually add normals to the point cloud
    
    return pct#, pts


def save_pointcloud(imgs, pts3d, msk, sparse_path):
    save_path = sparse_path / 'points3D.ply'
    pc = get_pc(imgs, pts3d, msk)
    
    pc.export(save_path)


def process_dust3r(image_files, 
    device="cuda:0", 
    batch_size=1, 
    schedule='cosine', 
    lr=0.01, # NOTE: 
    niter=300, 
    min_conf_thr=3, 
    model_path="./dust3r_ckpt/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth", 
    known_c2w=None, known_focal=None): 

    def inv(mat):
        """ Invert a torch or numpy matrix
        """
        if isinstance(mat, torch.Tensor):
            return torch.linalg.inv(mat)
        if isinstance(mat, np.ndarray):
            return np.linalg.inv(mat)
        raise ValueError(f'bad matrix type = {type(mat)}')

    model = load_model(model_path, device)
    images = load_images(image_files, size=512)
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=batch_size)
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
    
    if known_c2w is not None: 
        scene.preset_pose(known_c2w)
    if known_focal is not None: 
        scene.preset_focal(known_focal)

    loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)

    scene = scene.clean_pointcloud()

    intrinsics = scene.get_intrinsics().detach().cpu().numpy()
    cam2world = scene.get_im_poses().detach().cpu().numpy() # [n_views, 4, 4]
    principal_points = scene.get_principal_points().detach().cpu().numpy()
    focals = scene.get_focals().detach().cpu().numpy() # [n_views, 1]
    imgs = np.array(scene.imgs)
    pts3d = [i.detach() for i in scene.get_pts3d()]
    depth_maps = [i.detach().cpu().numpy() for i in scene.get_depthmaps()]

    scene.min_conf_thr = float(scene.conf_trf(torch.tensor(min_conf_thr)))
    masks = to_numpy(scene.get_masks())
    
    # scale the pose
    # pose_centers = cam2world[:, :3, 3] # [n, 3]
    # avg_pose_centers = np.mean(pose_centers, axis=0, keepdims=True) # [1, 3]
    # dist = np.linalg.norm(pose_centers - avg_pose_centers, axis=0, keepdims=True) # [3, ]
    # diagonal = np.min(dist)
    # scale_factor = diagonal
    # cam2world[:, :3, 3] /= scale_factor
    # pts3d = [t_pts3d/scale_factor for t_pts3d in pts3d]
    # depth_maps = [dmap/scale_factor for dmap in depth_maps]
    
    # average the focal length
    avg_focal = np.mean(focals)
    focals = np.ones_like(focals) * avg_focal
    
    world2cam = inv(cam2world)

    return imgs, masks, focals, principal_points, world2cam, pts3d, depth_maps


def save_pointcloud_with_normals(imgs, pts3d, msk, sparse_path):
    pc = get_pc(imgs, pts3d, msk)  # Assuming get_pc is defined elsewhere and returns a trimesh point cloud

    # Define a default normal, e.g., [0, 1, 0]
    default_normal = [0, 1, 0]

    # Prepare vertices, colors, and normals for saving
    vertices = pc.vertices
    colors = pc.colors
    normals = np.tile(default_normal, (vertices.shape[0], 1))

    save_path = sparse_path / 'points3D.ply'

    # Construct the header of the PLY file
    header = """ply
format ascii 1.0
element vertex {}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property float nx
property float ny
property float nz
end_header
""".format(len(vertices))

    # Write the PLY file
    with open(save_path, 'w') as ply_file:
        ply_file.write(header)
        for vertex, color, normal in zip(vertices, colors, normals):
            ply_file.write('{} {} {} {} {} {} {} {} {}\n'.format(
                vertex[0], vertex[1], vertex[2],
                int(color[0]), int(color[1]), int(color[2]),
                normal[0], normal[1], normal[2]
            ))


def get_pc(imgs, pts3d, mask):
    imgs = to_numpy(imgs)
    pts3d = to_numpy(pts3d)
    mask = to_numpy(mask)
    
    pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
    col = np.concatenate([p[m] for p, m in zip(imgs, mask)])
    
    pts = pts.reshape(-1, 3)[::3]
    col = col.reshape(-1, 3)[::3]
    
    #mock normals:
    normals = np.tile([0, 1, 0], (pts.shape[0], 1))
    
    pct = trimesh.PointCloud(pts, colors=col)
    pct.vertices_normal = normals  # Manually add normals to the point cloud
    
    return pct#, pts

def save_pointcloud(imgs, pts3d, msk, sparse_path):
    save_path = sparse_path / 'points3D.ply'
    pc = get_pc(imgs, pts3d, msk)
    
    pc.export(save_path)


def convert_dust3r_to_colmap(image_files, save_dir, 
                            min_conf_thr=3, 
                            model_path="./dust3r_ckpt/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth", 
                            known_c2w=None, known_focal=None, no_mask_pc=False, scale_factor=1): 
    
    print("Writing dust3r predictions to {}...".format(save_dir))
    imgs, masks, focals, principal_points, world2cam, pts3d, depth_maps = process_dust3r(
                            image_files, min_conf_thr=min_conf_thr, model_path=model_path, 
                            known_c2w=known_c2w, known_focal=known_focal)

    save_path, images_path, masks_path, sparse_path, depth_maps_path = init_filestructure(save_dir)
    save_images_masks(imgs, masks, depth_maps, images_path, masks_path, depth_maps_path)

    print(focals, [principal_points, imgs.shape]) # [n, 1], [n, 2], [n, h, w, 3]
    save_cameras(focals, principal_points, sparse_path, imgs_shape=imgs.shape) # TODO: check the shape
    save_imagestxt(world2cam, sparse_path)
    # save_pointcloud(imgs, pts3d, masks, sparse_path)
    # save_pointcloud_with_normals((imgs*255).astype(np.uint8), pts3d, masks, sparse_path)
    # save_pointcloud_with_normals(imgs, pts3d, masks, sparse_path)
    


    # imgs = to_numpy(imgs)
    # imgs = (imgs*255).astype(np.uint8)
    # pts3d = to_numpy(pts3d)
    # masks = to_numpy(masks)
    # depth_maps = to_numpy(depth_maps)

    # if not no_mask_pc: 
    #     pts = np.concatenate([p[m] for p, m in zip(pts3d, masks)])
    #     col = np.concatenate([p[m] for p, m in zip(imgs, masks)])
    # else: 
    #     pts = np.concatenate(pts3d)
    #     col = np.concatenate(imgs)
    
    # pts = pts.reshape(-1, 3)[::10]
    # col = col.reshape(-1, 3)[::10]

    # storePly(os.path.join(sparse_path, "points3D.ply"), pts, col)
    



    save_pointcloud_with_normals(imgs, pts3d, masks, sparse_path)
    
    print("Write done. ")

    return


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)