import cv2
import supervision as sv

from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import os
import pandas as pd
import numpy as np

model_path = hf_hub_download(
  repo_id="tech4humans/yolov8s-signature-detector", 
  filename="yolov8s.pt"
)

model = YOLO(model_path)

images = os.listdir("./Signatures/testing_sign")

# data = pd.read_csv("signature_logs.txt", names=["img","#boxes"])
# error = data[data["#boxes"] > 1]
# images = error["img"].tolist()

# with open("signature_logs.txt", "a") as f:
#     for img in images[0:1]:
#         image_path = f"./Signatures/testing_sign/{img}"
#         image = cv2.imread(image_path)

#         results = model(image_path)
#         detections = sv.Detections.from_ultralytics(results[0])
#         boxes = np.array(detections.xyxy)

#         box_annotator = sv.BoxAnnotator()
#         # annotated_image = box_annotator.annotate(scene=image, detections=detections)

#         # cv2.imshow("Detections", annotated_image)
        
#         f.write(f"{img},{len(results[0].boxes)}\n")

#             # cv2.imwrite(f"./annotated_img/annotated_{img}", annotated_image)
#         # for i, box in enumerate(detections.xyxy):
#         # print(box)
#         print(boxes)
#         if len(results[0].boxes) > 1:
#             x_min = int(np.min(boxes[:,0]))
#             y_min = int(np.min(boxes[:,1]))
#             x_max = int(np.max(boxes[:,2]))
#             y_max = int(np.max(boxes[:,3]))
#         else:
#             x_min, y_min, x_max, y_max = map(int, boxes)
#         cropped = image[y_min:y_max, x_min:x_max]
#         # cv2.imshow(f"Cropped {i}", cropped)
#         cv2.imwrite(f"./cropped_output_1/crop_{img.split('.')[0]}.jpg", cropped)



with open("signature_logs.txt", "a") as f:
    for img in images[100:]:
        image_path = f"./Signatures/testing_sign/{img}"
        image = cv2.imread(image_path)

        results = model(image_path)
        detections = sv.Detections.from_ultralytics(results[0])
        boxes = np.array(detections.xyxy)

        num_boxes = len(results[0].boxes)
        
        f.write(f"{img},{num_boxes}\n")
        f.flush()

        # print(boxes)

        # Skip if no boxes detected
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
        
        # Only save if crop is valid (non-zero size)
        if cropped.size > 0:
            cv2.imwrite(f"./cropped_output_1/crop_{img.split('.')[0]}.jpg", cropped)