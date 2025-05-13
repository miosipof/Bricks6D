import torch
import torchvision
import gc
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np

from resources.constants import RESNET_WEIGHTS
from ultralytics import YOLO


class Detection:
    def __init__(self, device, mode, box_threshold=0.3):

        if mode == "resnet":
            self.model = torchvision.models.detection.retinanet_resnet50_fpn(weights=RESNET_WEIGHTS)
            self.mode = 'resnet'
        elif mode == "yolo":
            self.mode = 'yolo'
            self.model = YOLO("yolov8n.pt")
        
        self.device = device
        self.model.to(self.device).eval()
        self.transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        self.box_threshold = box_threshold

    def detect(self, image_path):
        image_pil = Image.open(image_path).convert("RGB")
        image_np = np.array(image_pil)
        image_tensor = torch.tensor(image_np).permute(2, 0, 1).float() / 255.0
        input_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)

        if self.mode == 'resnet':
            with torch.no_grad():
                preds = self.model(input_tensor)[0]

            threshold = self.box_threshold
            keep = preds['scores'] >= threshold
            bboxes = preds['boxes'][keep]
            confidences = preds['scores'][keep]
            labels = preds['labels'][keep]

            del image_tensor, image_np
            gc.collect()
            torch.cuda.empty_cache()
            
            print(f"[RetinaNet] {len(bboxes)} boxes detected")

        elif self.mode == 'yolo':
            results = self.model(image_np)
            bboxes = []
            confidences = []
            labels = []
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf.item()
                cls = int(box.cls.item())
                if conf >= self.box_threshold:
                    bboxes.append([x1, y1, x2, y2])
                    confidences.append(conf)
                    labels.append(cls)
                    
        return bboxes, confidences, labels, image_pil
    
    def sort_boxes(self, bboxes):
        def area(box):
            x1, y1, x2, y2 = box
            return max(0, x2 - x1) * max(0, y2 - y1)

        sorted_bboxes = sorted(bboxes, key=area, reverse=True)
        return sorted_bboxes
    
    def crop(self, pil_image, bbox, depth_path):
        """
        Crop both a PIL image and its corresponding depth map using a bounding box.

        Parameters
        ----------
        pil_image : PIL.Image         – RGB or grayscale image
        bbox      : tuple[int, int, int, int] – (x1, y1, x2, y2)
        depth_map : PIL.Image         – (H, W) depth map in meters

        Returns
        -------
        cropped_img   : PIL.Image     – cropped image
        cropped_depth : np.ndarray    – cropped depth map (same shape as image)
        """
        
        depth_pil = Image.open(depth_path)
        depth_np = np.array(depth_pil)
        
        x1, y1, x2, y2 = map(int, bbox)

        # Crop image
        cropped_img = pil_image.crop((x1, y1, x2, y2))

        # Crop depth map
        cropped_depth = depth_np[y1:y2, x1:x2].copy()

        del depth_pil, depth_np
        gc.collect()
        torch.cuda.empty_cache()

        return cropped_img, cropped_depth

    def vis_box(self, bbox, pil_image, color="red", width=3):
        img_with_box = pil_image.copy()
        draw = ImageDraw.Draw(img_with_box)
        bbox = tuple(map(int, bbox))
        draw.rectangle(bbox, outline=color, width=width)

        # Show using matplotlib
        plt.figure(figsize=(6, 6))
        plt.imshow(img_with_box)
        plt.title("Image with Bounding Box")
        plt.axis("off")
        plt.show()
