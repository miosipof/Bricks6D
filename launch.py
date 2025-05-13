import matplotlib.pyplot as plt
import cv2
import numpy as np
import sys
import os
import json
from PIL import Image

from src.detect import Detection
from src.segment import Segmentation
from src.geometry import Model2D, VertexSolver, Model3D


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


segmentator = Segmentation(sam2_checkpoint_name='sam2.1_hiera_large.pt', 
                           sam2_config_name='sam2.1_hiera_l.yaml', 
                           threshold=1e-4, 
                           area_threshold=0.5)


image_path = 'brick_1.jpg'
# image_path = '7_crop.jpg'
bin_mask, mask, image_pil = segmentator.create_mask(image_path, visualize=True)

plane_model = Model2D()
best_cnt, best_poly = plane_model.create_poly(bin_mask, image_pil, visualize=True)

print(f"Polygon shape: {best_poly.shape}")


hexagon = plane_model.create_hexagon(best_poly)
print(f"Simplified polygon shape: {hexagon.shape}")
# print(f"Simplified polygon points: {hexagon}")

plane_model.vis_poly(bin_mask, np.int32(hexagon.reshape(hexagon.shape[0],1,2)))

vertex_solver = VertexSolver()
vertex, debug = vertex_solver.find_missing_vertex(hexagon)
print(f"Solved for missing vertex: {vertex}")
vertex_solver.draw_vertex(mask, hexagon, debug)

world_model = Model3D()

with open('7.json', 'r') as f:
    intrinsics = json.load(f)

# depth_path = '7_crop.png'
# depth_map = Image.open(depth_path)
# depth_map = np.array(depth_map)
# depth_map = depth_map/(1e-5+np.max(depth_map))

h,w = image_pil.size
depth_map = np.full((h,w), 1.0, dtype=np.float32)

image_vertices = np.vstack([hexagon, np.round(vertex)])
# image_vertices = hexagon

img_pts, obj_pts, orient_ok, ray, depth_c, face_type = world_model.get_brick_center(image_vertices, depth_map, intrinsics)

world_model.draw_vertex_labels(image_pil, img_pts, obj_pts)

img_pts = img_pts[:-1]
R, tvec, normal = world_model.solve_brick_pose(img_pts, obj_pts, intrinsics, depth_c, ray)

world_model.project_brick(R, tvec, normal, img_pts, obj_pts, intrinsics, image_pil)
world_model.draw_normal_vector(image_pil, tvec, normal, intrinsics)