# Automatic Signature Detection and Cropping Algorithm
## ✒️ SignatureCropper

A Python module to detect and crop signatures from images using a YOLOv8 model fine-tuned on a custom signature dataset.

Created by **Saransh Saini** and **Aayush Kumawat**, undergraduate students at **IIT Madras** in the **BS Data Science and Applications** program.

---

## 📸 Overview

SignatureCropper leverages a YOLOv8 model fine-tuned on 1,000 annotated signature images to accurately detect and crop signature regions from scanned documents or photos. It can be easily integrated into document processing pipelines, archival tools, or signature verification systems.

---

## 🔧 Features

- Detects signatures in various document layouts
- Automatically crops and saves detected signature regions
- Handles multiple detections in an image
- Logs detection counts for auditing

---

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/signature-cropper.git
   cd signature-cropper
    ```
2. **(Optional) Create and activate a virtual environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    ```
3. **Install required dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## 🔑 Hugging Face Token
This project requires a Hugging Face token to download the custom YOLOv8 model.
- Create an account at https://huggingface.co
- Generate an access token from your account settings
- Set it as an environment variable:
```bash
export HUGGINGFACE_TOKEN=your_token_here
```
Or pass it directly when initializing the class:
```bash
cropper = SignatureCropper(hf_token="your_token_here")
```

## 🚀 Usage
#### Step 1: Prepare your input and output folders
- Place all input images in a directory, e.g., ```input_images/```
- Create an empty directory (or let the script create one) for cropped signatures, e.g., ```output_crops/```
#### Step 2: Run the cropper
```python
from signature_cropper import SignatureCropper

cropper = SignatureCropper()
cropper.sign_detect("input_images", "output_crops")
```
- Logs will be saved in ```signature_logs.txt```
- Cropped images will be saved as ```crop_<original_filename>.jpg``` in the output folder

### 🤝 Authors
Saransh Saini – [https://www.linkedin.com/in/saranshsaini48/](https://www.linkedin.com/in/saranshsaini48/)
Aayush Kumawat – [https://www.linkedin.com/in/aayush-kumawat-1a1641277/](https://www.linkedin.com/in/aayush-kumawat-1a1641277/)

### 📜 License
This project is licensed under the MIT License. Feel free to use and modify it as needed.

### 💡 Acknowledgements
 - [YOLOv8s - Handwritten Signature Detection by Samuel Lima](https://huggingface.co/tech4humans/yolov8s-signature-detector)
 - [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
 - [Supervision by Roboflow](https://github.com/roboflow/supervision)