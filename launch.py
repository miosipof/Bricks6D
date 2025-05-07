import matplotlib.pyplot as plt
import cv2
import numpy as np

from src.segment import Segmentation
from src.geometry import Model2D

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
plane_model.vis_poly(bin_mask, np.int32(hexagon.reshape(hexagon.shape[0],1,2)))


quadA, quadB = plane_model.split_brick_face(hexagon)

# print("Quad A vertices:\n", quadA)
# print("Quad B vertices:\n", quadB)


canvas = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
plane_model.draw_quads(canvas, quadA, quadB)

plt.figure(figsize=(6, 6))
plt.title("")
plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
