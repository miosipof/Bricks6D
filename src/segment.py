import os
import torch
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import logging

# from .. import sam2
from sam2.utils.track_utils import sample_points_from_masks
from sam2.utils.video_utils import create_video_from_images
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from resources.constants import ROOT_DIR, BASE_DIR, OUTPUT_DIR


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)




class Segmentation():
    def __init__(self, sam2_checkpoint_name='sam2.1_hiera_large.pt', sam2_config_name='sam2.1_hiera_l.yaml', threshold=1e-4, area_threshold=0.5):
        self.device = ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold # Threshold for binary mask
        self.area_threshold = area_threshold # Threshold to reject small masks

        """
        Initialize and set up segmentation model
        """
        sam2_config_dir = os.path.join(ROOT_DIR, 'configs')
        checkpoint_dir = os.path.join(BASE_DIR, 'checkpoints')

        sam2_checkpoint = os.path.join(checkpoint_dir,sam2_checkpoint_name)
        sam2_model_cfg = os.path.join(sam2_config_dir,sam2_config_name)
      
        self.sam2_image_model = build_sam2(sam2_config_name, sam2_checkpoint, device=self.device)
        log.info(f"SAM2 image model loaded from {sam2_checkpoint}")

        self.image_predictor = SAM2ImagePredictor(self.sam2_image_model, device=self.device)
        log.info(f"image_predictor loaded from SAM2 image model")


    """
    Clean mask to filter out noise
    """
    def clean_mask(self, mask):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        clean_mask  = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        clean_mask  = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel, iterations=2)    

        return clean_mask    
    

    """
    Check object area and convexity
    """
    def check_mask(self, bin_mask, box_area):

        positive_area = bin_mask.sum()
        print(positive_area,box_area)
        if positive_area / box_area < self.area_threshold: 
            log.info(f"Inverting mask: area {positive_area} is below {box_area*self.area_threshold:.2f} = {box_area}*{self.area_threshold}")
            return 1-bin_mask


        # positive_area = bin_mask.sum()
        # if positive_area / box_area < self.area_threshold: 
        #     log.info(f"Mask rejected: area {positive_area} is below {box_area}*{self.area_threshold}")
        #     return None    

        # # Create envelope contour
        # contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # largest_contour = max(contours, key=cv2.contourArea)

        # # Check convexity
        # is_convex = cv2.isContourConvex(largest_contour)
        

        # # If not convex, probbaly the mask is inverted:
        # if not is_convex:    
        #     log.info(f"Contour is not convex, trying to invert mask")
        #     bin_mask_inverted = 1 - bin_mask
        #     contours_inv, _ = cv2.findContours(bin_mask_inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #     largest_inv = max(contours_inv, key=cv2.contourArea)
        #     is_convex_inv = cv2.isContourConvex(largest_inv)

        #     if is_convex_inv:
        #         log.info(f"Inverted mask is convex, proceeding with it")
        #         bin_mask = bin_mask_inverted
        #     else:
        #         log.info(f"Neither inverted mask is convex, mask rejected")
        #         return None        
            

    """
    Generate mask from bbox image crop
    """
    def create_mask(self, image_path, visualize=False):

        image_pil = Image.open(image_path).convert("RGB")
        image_np = np.array(image_pil)
        self.image_predictor.set_image(image_pil)
        image_tensor = torch.tensor(image_np).permute(2, 0, 1).float() / 255.0

        h,w = image_pil.size

        # Assume the image has been already cropped to bbox
        x1 = 0
        y1 = 0
        x2 = w
        y2 = h        

        # Predict masks
        sam_masks, scores, _ = self.image_predictor.predict(
            box=np.array([x1, y1, x2, y2]),
            multimask_output=False,
            return_logits=False,
        )

        mask = self.clean_mask(sam_masks[0])

        bin_mask = (mask > self.threshold).astype(np.uint8)
        bin_mask = self.check_mask(bin_mask, abs((x2-x1)*(y2-y1)))
        
        if visualize:
            self.vis_mask(image_path,image_np,bin_mask)

        return bin_mask, mask, image_pil



    """
    Save mask .png and overlay on the original image
    """
    def vis_mask(self,image_path,image_np,bin_mask):
        
        Image.fromarray((bin_mask * 255).astype(np.uint8)).save(os.path.join('', OUTPUT_DIR, image_path.replace(".jpg", "_mask.png")))
        vis_image = image_np.copy()

        # Overlay mask on vis_image
        mask_color = np.zeros_like(vis_image)
        mask_color[bin_mask > 0] = (0, 255, 0)  # Green
        mask_color[bin_mask == 0] = (0, 0, 0)  # Black

        vis_image = cv2.addWeighted(vis_image, 1.0, mask_color, 0.3, 0)

        vis_path = os.path.join('', OUTPUT_DIR, image_path.replace(".jpg", "_overlay.jpg"))
        cv2.imwrite(vis_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))



