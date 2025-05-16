import matplotlib.pyplot as plt
import cv2
import numpy as np
import sys
import os
import json
from PIL import Image

from src.segment import YOLOSegmentation
from src.geometry import Model2D, Model3D


device = "cpu"
image_id = 7


image_path = f'{image_id}.jpg'
image_pil = Image.open(image_path).convert("RGB")

# Load pretrained YOLO-segment
segmentator = YOLOSegmentation(yolo_ckpt='yolo_seg.pt')
segmentator.device = device

# Generate masks
masks = segmentator.create_masks(image_path)


# [Optional] Load saved masks
mask_path = f'{image_id}_mask_0.png'
mask = Image.open(mask_path)
mask = np.array(mask)
if len(mask.shape) == 3:
    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY) 
_, bin_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

# Initialize 2D model to extract poly
plane_model = Model2D()
best_cnt, best_poly = plane_model.create_poly(bin_mask, image_pil, visualize=True)
print(f"Polygon shape: {best_poly.shape}")

# Simplify poly to 4 vertices
image_vertices = plane_model.create_hexagon(best_poly)
print(f"Simplified polygon shape: {image_vertices.shape}")

# [Optional] Visualize poly
plane_model.vis_poly(bin_mask, np.int32(image_vertices.reshape(image_vertices.shape[0],1,2)))


# Load 3D model
world_model = Model3D()

# Load camera intrinsics
K = world_model.load_k(f'{image_id}.json')

# Load depth map
depth_path = f'{image_id}.png'
depth_map = Image.open(depth_path)
depth_map = np.array(depth_map)
# depth_map = depth_map/(1e-5+np.max(depth_map))

depth_scale = 1e-4
z_raw = depth_map[image_pil.size[0]//2, image_pil.size[1]//2]
# print("Raw depth value:", z_raw)
# print("Converted depth (m):", z_raw * depth_scale)

# # PnPSolve 3D pose
# world_model = Model3D()
# image_pts, object_pts, ray, depth_c = world_model.get_brick_center(image_vertices, depth_map, intrinsics)
# world_model.draw_vertex_labels(image_pil, image_pts, object_pts)
# R, tvec, normal = world_model.solve_brick_pose(image_pts, object_pts, intrinsics, depth_c, ray)
# world_model.project_brick(R, tvec, normal, image_pts, object_pts, intrinsics, image_pil)
# world_model.draw_normal_vector(image_pil, tvec, normal, intrinsics)


inner_poly = plane_model.shrink_polygon(image_vertices, shrink_factor=0.8)

# Get inner mask points to avoid boundary issues
inner_mask = plane_model.polygon_to_mask(mask.shape, inner_poly)


# Project face points to 3D space via depth map
points_3d = world_model.project_face(depth_map, inner_mask, K, scale=depth_scale)

# PCA to fit 2D plane
centroid, normal, best_inliers = world_model.fit_plane(points_3d, ransac_thresh=0.004, min_inliers=300)


# Get the in-mask 2d direction of the longest edge
axis_2d = world_model.fit_axis(inner_mask)


x, y, z    = centroid
vx, vy, vz = normal
pose = np.array([x, y, z, vx, vy, vz])


world_model.vis_plane(points_3d, centroid, normal, mask=None, K=None, patch_size=0.1)
world_model.vis_pose(image_pil, inner_mask, centroid, normal, axis_2d, K,
                                scale=100.0, color_centroid=(255, 0, 0),
                                color_normal=(0, 0, 255), color_axis=(255, 255, 0))


image_overlay = world_model.overlay_points(image_pil, points_3d, centroid, normal, K, patch_size=0.2)
image_overlay.show()