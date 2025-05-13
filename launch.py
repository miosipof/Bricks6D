import matplotlib.pyplot as plt
import cv2
import numpy as np
import sys

from src.segment import Segmentation
from src.geometry import Model2D, VertexSolver, Model3D

segmentator = Segmentation(sam2_checkpoint_name='sam2.1_hiera_large.pt', 
                           sam2_config_name='sam2.1_hiera_l.yaml', 
                           threshold=1e-4, 
                           area_threshold=0.5)



bin_mask, mask, image_pil = segmentator.create_mask('brick_1.jpg', visualize=True)

plane_model = Model2D()
best_cnt, best_poly = plane_model.create_poly(bin_mask, image_pil, visualize=True)

print(f"Polygon shape: {best_poly.shape}")


hexagon = plane_model.create_hexagon(best_poly)
print(f"Simplified polygon shape: {hexagon.shape}")
# print(f"Simplified polygon points: {hexagon}")

plane_model.vis_poly(bin_mask, np.int32(hexagon.reshape(hexagon.shape[0],1,2)))



## Naive isometric logic for missing vertex reconstruction
# quadA, quadB = plane_model.split_brick_face(hexagon)

# print("Quad A vertices:\n", quadA)
# print("Quad B vertices:\n", quadB)


# canvas = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
# plane_model.draw_quads(canvas, quadA, quadB)

# plt.figure(figsize=(6, 6))
# plt.title("")
# plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
# plt.axis("off")
# plt.show()

vertex_solver = VertexSolver()
vertex, debug = vertex_solver.find_missing_vertex(hexagon)
print(f"Solved for missing vertex: {vertex}")

# print(debug)
# print(hexagon,vertex)

"""
missing vertex should be insert between anchor1_idx and anchor2_idx ???
"""

# vertex_solver.draw_vertex(mask, hexagon, debug)




world_model = Model3D()

intrinsics = {"fx": 646.8458251953125, "fy": 646.8458251953125, "cx": 660.7882690429688, "cy": 364.16741943359375}

h,w = image_pil.size
depth_map = np.full((h,w), 2.0, dtype=np.float32)


image_vertices = np.vstack([hexagon, np.round(vertex)])
# image_vertices = hexagon


img_pts, obj_pts, orient_ok, ray, depth_c, face_type = world_model.get_brick_center(image_vertices, depth_map, intrinsics)

# pose = {
#     "image_points": img_pts,
#     "object_points": obj_pts,
#     "orientation_ok": orient_ok,
#     "ray_to_centroid": ray,
#     "depth_centroid": depth_c
# }
# print(pose)

# world_model.draw_vertex_labels(image_pil, img_pts, obj_pts)


# sys.exit()
img_pts = img_pts[:-1]

R, tvec, normal = world_model.solve_brick_pose(img_pts, obj_pts, intrinsics, depth_c, ray)

print("Translation (m):", tvec.flatten())
print("Normal vector   :", normal)


world_model.project_brick(R, tvec, normal, img_pts, obj_pts, intrinsics, image_pil)

world_model.draw_normal_vector(image_pil, tvec, normal, intrinsics)