import os
import cv2
import supervision as sv
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import pandas as pd
import numpy as np

class SignatureCropper:
    def __init__(self, hf_token=None):
        # Use token from argument or environment
        hf_token = hf_token or os.getenv("HUGGINGFACE_TOKEN")
        if not hf_token:
            raise ValueError("Hugging Face token is required. Pass it as an argument or set HUGGINGFACE_TOKEN env variable.")

        model_path = hf_hub_download(
            repo_id="tech4humans/yolov8s-signature-detector", 
            filename="yolov8s.pt",
            token=hf_token
        )
        self.model = YOLO(model_path)

    def sign_detect(self, images_dir, target_dir):
        images = os.listdir(images_dir)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        with open("signature_logs.txt", "a") as f:
            for img in images:
                image_path = os.path.join(images_dir, img)
                image = cv2.imread(image_path)

                results = self.model(image_path)
                detections = sv.Detections.from_ultralytics(results[0])
                boxes = np.array(detections.xyxy)

                num_boxes = len(results[0].boxes)
                
                f.write(f"{img},{num_boxes}\n")
                f.flush()

                if num_boxes == 0:
                    print(f"No detections found for {img}")
                    continue

                if num_boxes > 1:
                    x_min = int(np.min(boxes[:, 0]))
                    y_min = int(np.min(boxes[:, 1]))
                    x_max = int(np.max(boxes[:, 2]))
                    y_max = int(np.max(boxes[:, 3]))
                else:
                    x_min, y_min, x_max, y_max = map(int, boxes[0])

                cropped = image[y_min:y_max, x_min:x_max]
                
                if cropped.size > 0:
                    cv2.imwrite(os.path.join(target_dir, f"crop_{os.path.splitext(img)[0]}.jpg"), cropped)
