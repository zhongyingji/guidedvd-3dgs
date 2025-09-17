import os
import numpy as np
import torch
from PIL import Image
import sys
try: 
    from scene.dataset_readers import fetchPly
except: 
    sys.path.append('../')
    from scene.dataset_readers import fetchPly
import trimesh

def depth_to_point_cloud(depth_map, intrinsic_matrix, c2w, mask, rgb_map):
    # rgb_map: [h, w, 3]
    # depth_map: [h, w]

    # Get the image dimensions
    H, W = depth_map.shape

    # Create a grid of (u, v) coordinates
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    u = u.flatten()
    v = v.flatten()

    # Flatten the depth map, mask, and rgb map
    depth = depth_map.flatten()
    mask = mask.flatten()
    rgb_map = rgb_map.reshape(-1, 3)

    # Apply the mask
    u = u[mask == 1]
    v = v[mask == 1]
    depth = depth[mask == 1]
    rgb = rgb_map[mask == 1]

    # Intrinsic matrix components
    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]

    # Convert pixel coordinates and depth to camera coordinates
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth

    # Stack into a 3xN array of 3D points in camera coordinates
    points_camera = np.vstack((x, y, z, np.ones_like(z)))

    # Transform to world coordinates using the c2w matrix
    points_world = c2w @ points_camera

    # Drop the homogeneous coordinate
    points_world = points_world[:3, :].T

    # Return points with their corresponding RGB colors
    return points_world, rgb


def save_pointcloud_with_normals(imgs, pts3d, save_ply_file):
    pc = get_pc(imgs, pts3d)  # Assuming get_pc is defined elsewhere and returns a trimesh point cloud

    # Define a default normal, e.g., [0, 1, 0]
    default_normal = [0, 1, 0]

    # Prepare vertices, colors, and normals for saving
    vertices = pc.vertices
    colors = pc.colors
    normals = np.tile(default_normal, (vertices.shape[0], 1))

    # save_path = sparse_path / 'points3D.ply'

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
    with open(save_ply_file, 'w') as ply_file:
        ply_file.write(header)
        for vertex, color, normal in zip(vertices, colors, normals):
            ply_file.write('{} {} {} {} {} {} {} {} {}\n'.format(
                vertex[0], vertex[1], vertex[2],
                int(color[0]), int(color[1]), int(color[2]),
                normal[0], normal[1], normal[2]
            ))

def get_pc(col, pts):
# def get_pc(imgs, pts3d, mask):
    # imgs = to_numpy(imgs)
    # pts3d = to_numpy(pts3d)
    # mask = to_numpy(mask)
    
    # pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
    # col = np.concatenate([p[m] for p, m in zip(imgs, mask)])
    
    # pts = pts.reshape(-1, 3)[::3]
    # col = col.reshape(-1, 3)[::3]

    
    #mock normals:
    normals = np.tile([0, 1, 0], (pts.shape[0], 1))
    
    pct = trimesh.PointCloud(pts, colors=col)
    pct.vertices_normal = normals  # Manually add normals to the point cloud
    
    return pct#, pts


if __name__ == "__main__": 
    SCENE = ["office2_seq2", "office3_seq1", "office4_seq2",  "room0_seq2", "room1_seq1", "room2_seq1"]
    CAM_FILE = ["office_2/Sequence_2", "office_3/Sequence_1", "office_4/Sequence_2", "room_0/Sequence_2", "room_1/Sequence_1", "room_2/Sequence_1"]
    # SCENE = ["room0_seq2"]
    # CAM_FILE = ["room_0/Sequence_2"]
    
    root_inpaint_dir = "./inpaint_dir"
    root_mask_dir = "../pointcloudrender_dir_minconf1/"

    for s, f in zip(SCENE, CAM_FILE): 
        ply_path = "../dust3r_results/Replica_6v_thr1_trimeshsave_minconf1/{}/sparse/0/points3D.ply".format(s)
        ply = fetchPly(ply_path)
        ply_pts = ply.points
        ply_cols = ply.colors

        new_ply_dir = "../dust3r_results/Replica_6v_thr1_trimeshsave_minconf1_addinpaint/{}/sparse/0".format(s)
        os.makedirs(new_ply_dir, exist_ok=True)
        new_ply_path = os.path.join(new_ply_dir, "points3D.ply")

        rgb_dir = os.path.join(root_inpaint_dir, s, "rgb/")
        depth_dir = os.path.join(root_inpaint_dir, s, "depth/")
        mask_dir = os.path.join(root_mask_dir, s)

        cam_file = "../replica_cam_poses/{}/cam_poses.pt".format(f)
        cam_data = torch.load(cam_file)

        poses = cam_data["H_c2w"][0, ::6] # [150, 4, 4]
        poses = np.array(poses).astype(np.float32)
        H, W = cam_data['height_px'], cam_data['width_px']
        intrinsic = cam_data['intrinsic'][0, 0]
        intrinsic = np.array(intrinsic).astype(np.float32)

        pts = []
        cols = []
        for idx in range(poses.shape[0]): 
            i = 6*idx
            c2w_i = poses[idx]
            
            mask_file = os.path.join(mask_dir, "rgb_{}_mask_erosion.png".format(i))
            inpaint_depth_file = os.path.join(depth_dir, "depth_inpaint_{}.npy".format(i))
            inpaint_rgb_file = os.path.join(rgb_dir, "rgb_inpaint_{}.png".format(i))

            mask = Image.open(mask_file).convert('L')
            mask = np.array(mask) / 255.

            inpaint_depth = np.load(inpaint_depth_file)

            inpaint_rgb = Image.open(inpaint_rgb_file).convert('RGB')
            inpaint_rgb = np.array(inpaint_rgb) / 255.

            add_pts, add_rgbs = depth_to_point_cloud(inpaint_depth, intrinsic, c2w_i, mask, inpaint_rgb)
            # in range 0-1
            print(":> Check shape of pose {}: {}, {}".format(i, add_pts.shape, add_rgbs.shape))

            pts.append(add_pts[::10])
            cols.append(add_rgbs[::10])
        
        pts = np.concatenate(pts, 0)
        cols = np.concatenate(cols, 0)

        # pts = pts[::3]
        # cols = cols[::3]
        print(":> Check shape of concat: ", pts.shape, cols.shape)

        all_pts = np.concatenate([ply_pts, pts], 0)
        all_cols = np.concatenate([ply_cols, cols], 0)

        save_pointcloud_with_normals(all_cols, all_pts, new_ply_path)