import numpy as np
import cv2

def project_point_cloud_to_image(point_cloud, colors, intrinsics, extrinsics, width, height, near=0.1, far=1000.0):
    """
    Projects a 3D point cloud into a 2D image plane using camera intrinsics and extrinsics.
    Considers near and far clipping planes and handles depth buffering to ensure the nearest point's color is used.

    :param point_cloud: (N, 3) numpy array containing 3D points
    :param colors: (N, 3) numpy array containing RGB colors for each point
    :param intrinsics: (3, 3) numpy array containing camera intrinsics matrix
    :param extrinsics: (4, 4) numpy array containing camera extrinsics matrix
    :param width: width of the output image
    :param height: height of the output image
    :param near: near clipping plane
    :param far: far clipping plane
    :return: (H, W, 3) numpy array representing the image with projected points, (H, W) numpy array indicating the mask
    """
    # Create an empty image, mask, and depth buffer
    image = np.zeros((height, width, 3), dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)
    depth_buffer = np.full((height, width), np.inf)

    # Convert point cloud to homogeneous coordinates
    num_points = point_cloud.shape[0]
    homogeneous_points = np.hstack((point_cloud, np.ones((num_points, 1))))

    # Apply extrinsics (world to camera transformation)
    camera_points = (extrinsics @ homogeneous_points.T).T
    valid_depth_mask = (camera_points[:, 2] > near) & (camera_points[:, 2] < far)

    # Filter points based on depth
    camera_points = camera_points[valid_depth_mask]
    colors = colors[valid_depth_mask]

    # Apply intrinsics (camera to image transformation)
    image_points = (intrinsics @ camera_points[:, :3].T).T
    image_points[:, 0] /= image_points[:, 2]
    image_points[:, 1] /= image_points[:, 2]

    # Round and cast to integers
    u = np.round(image_points[:, 0]).astype(int)
    v = np.round(image_points[:, 1]).astype(int)
    z = camera_points[:, 2]

    # Filter points that project within the image bounds
    valid_indices = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u = u[valid_indices]
    v = v[valid_indices]
    z = z[valid_indices]
    colors = colors[valid_indices]

    # Create a linear index array for depth comparison and color assignment
    linear_indices = v * width + u

    # Use argsort to sort by depth
    sorted_indices = np.argsort(z)
    u = u[sorted_indices]
    v = v[sorted_indices]
    z = z[sorted_indices]
    colors = colors[sorted_indices]
    linear_indices = linear_indices[sorted_indices]

    # Update image, mask, and depth buffer
    unique_indices, unique_positions = np.unique(linear_indices, return_index=True)
    image[v[unique_positions], u[unique_positions]] = colors[unique_positions]
    mask[v[unique_positions], u[unique_positions]] = 1
    depth_buffer[v[unique_positions], u[unique_positions]] = z[unique_positions]

    return image, mask

# def visualize_image(image):
#     """
#     Visualizes the image with projected points.

#     :param image: The image onto which the points are projected
#     """
#     cv2.imshow('Projected Points', image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


# if __name__ == "__main__": 

#     # Example data
#     point_cloud = np.array([
#         [1.0, 0.5, 2.0],
#         [1.5, -0.5, 2.5],
#         [2.0, 1.0, 3.0],
#         [1.0, 0.5, 2.1], # Added a point slightly behind the first point
#     ])

#     # Corresponding RGB colors (example values)
#     colors = np.array([
#         [255, 0, 0],  # Red
#         [0, 255, 0],  # Green
#         [0, 0, 255],  # Blue
#         [255, 255, 0], # Yellow, should be ignored as it's behind the red point
#     ])

#     # Camera intrinsics (example values)
#     intrinsics = np.array([
#         [800, 0, 320],
#         [0, 800, 240],
#         [0, 0, 1]
#     ])

#     # Example camera extrinsics (example values)
#     extrinsics = np.array([
#         [1, 0, 0, 0],
#         [0, 1, 0, 0],
#         [0, 0, 1, 0],
#         [0, 0, 0, 1]
#     ])

#     # Image dimensions and clipping planes
#     width = 640
#     height = 480
#     near = 0.1
#     far = 100.0

#     # Project point cloud onto image
#     image = project_point_cloud_to_image(point_cloud, colors, intrinsics, extrinsics, width, height, near, far)

#     # Visualize the result
#     visualize_image(image)