import matplotlib.pyplot as plt
import cv2
import numpy as np
import sys
import os
import json
from PIL import Image

from src.detect import Detection
from src.segment import Segmentation
from src.geometry import Model2D, Model3D


device = "cpu"

# idx = 0

# sample = {
#     "id": idx,
#     "image_path": f"rgb/{idx}.jpg",
#     "intrinsics": f"jsons{idx}.png",
#     "depth": f"depth/{idx}.png"
# }

# image_path = os.path.join('input', 'data', sample['image_path'])
# depth_path = os.path.join('input', 'data', sample['depth'])


# detector = Detection(device=device,
#                      mode='resnet', 
#                      box_threshold=0.3)



# bboxes, confidences, labels, orig_img = detector.detect(image_path)
# print(f"{len(bboxes)} objects detected")

# bboxes = detector.sort_boxes(bboxes)

# image_pil, cropped_depth = detector.crop(orig_img, bboxes[0], depth_path)

# detector.vis_box(bboxes[0], orig_img)


# segmentator = Segmentation(sam2_checkpoint_name='sam2.1_hiera_large.pt', 
#                            sam2_config_name='sam2.1_hiera_l.yaml', 
#                            threshold=1e-4, 
#                            area_threshold=0.5)


# image_path = 'brick_1.jpg'
image_path = '7.jpg'
image_pil = Image.open(image_path).convert("RGB")

mask_path = '7_mask_0.png'
mask = Image.open(mask_path)
mask = np.array(mask)
if len(mask.shape) == 3:
    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY) 
_, bin_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

# bin_mask, mask, image_pil = segmentator.create_mask(image_path, visualize=True)



plane_model = Model2D()
best_cnt, best_poly = plane_model.create_poly(bin_mask, image_pil, visualize=True)

print(f"Polygon shape: {best_poly.shape}")


image_vertices = plane_model.create_hexagon(best_poly)
print(f"Simplified polygon shape: {image_vertices.shape}")
# print(f"Simplified polygon points: {hexagon}")

plane_model.vis_poly(bin_mask, np.int32(image_vertices.reshape(image_vertices.shape[0],1,2)))

world_model = Model3D()

with open('7.json', 'r') as f:
    intrinsics = json.load(f)

depth_path = '7.png'
depth_map = Image.open(depth_path)
depth_map = np.array(depth_map)
depth_map = depth_map/(1e-5+np.max(depth_map))

image_pts, object_pts, ray, depth_c = world_model.get_brick_center(image_vertices, depth_map, intrinsics)

world_model.draw_vertex_labels(image_pil, image_pts, object_pts)

R, tvec, normal = world_model.solve_brick_pose(image_pts, object_pts, intrinsics, depth_c, ray)

world_model.project_brick(R, tvec, normal, image_pts, object_pts, intrinsics, image_pil)
world_model.draw_normal_vector(image_pil, tvec, normal, intrinsics)