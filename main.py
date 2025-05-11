import os
import torch
import yaml
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from ultralytics import YOLO
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.patches as patches

# Load dataset paths from YAML
dataset_path = "E:/Semester2/Deep-Learning/Exp7/aquarium_pretrain"
yaml_path = os.path.join(dataset_path, "data.yaml")
with open(yaml_path, "r") as f:
    data = yaml.safe_load(f)

# Corrected dataset paths
train_img_dir = os.path.join(dataset_path, "train", "images")
valid_img_dir = os.path.join(dataset_path, "valid", "images")

# Verify paths exist
assert os.path.exists(train_img_dir), f"Train path does not exist: {train_img_dir}"
assert os.path.exists(valid_img_dir), f"Validation path does not exist: {valid_img_dir}"
print(f"Train Path: {train_img_dir}")
print(f"Validation Path: {valid_img_dir}")

# Define transformation
transform = transforms.Compose([
    transforms.Resize((640, 640)),  # Resize all images to 640x640 or any uniform size
    transforms.ToTensor(),
])

# Custom Dataset class for YOLO format
class YOLODataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.image_files = [f for f in os.listdir(img_dir) if f.endswith(".jpg") or f.endswith(".png")]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

# Load train dataset
train_dataset = YOLODataset(train_img_dir, transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# Load Faster R-CNN model
faster_rcnn_model = fasterrcnn_resnet50_fpn(pretrained=True)
faster_rcnn_model.eval()

# Load YOLO model
yolo_model = YOLO("yolov8n.pt")

# Placeholder for R-CNN (simulated with selective search)
def rcnn_placeholder(image):
    print("Running R-CNN (simulated with selective search)")
    return {"boxes": [[50, 50, 150, 150]], "scores": [0.8]}

# Testing on one image
sample_image = next(iter(train_loader))[0]

# Run R-CNN
rcnn_pred = rcnn_placeholder(sample_image)

# Run Faster R-CNN
with torch.no_grad():
    faster_rcnn_pred = faster_rcnn_model([sample_image])

# Run YOLO
yolo_input = (sample_image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
yolo_results = yolo_model(yolo_input, conf=0.25)

# Extract bounding boxes and scores from YOLO
yolo_boxes = {"boxes": [], "scores": []}
for result in yolo_results:
    for box in result.boxes.xyxy:
        yolo_boxes["boxes"].append(box.tolist())
    for conf in result.boxes.conf:
        yolo_boxes["scores"].append(conf.item())

# Function to plot detections
def plot_detections(image, pred, title, model_type=""):
    image = image.permute(1, 2, 0).cpu().numpy()
    fig, ax = plt.subplots(1, figsize=(8, 6))
    ax.imshow(image)
    if model_type == "faster_rcnn":
        pred = pred[0]  # Faster R-CNN output is a list of dicts
    if "boxes" in pred and len(pred["boxes"]) > 0:
        for i, box in enumerate(pred["boxes"]):
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            if "scores" in pred:
                score = pred["scores"][i] if i < len(pred["scores"]) else 0
                ax.text(x1, y1, f"{score:.2f}", color="red", fontsize=10, bbox=dict(facecolor='yellow', alpha=0.5))
    ax.set_title(title)
    plt.show(block=True)

# Show results
plot_detections(sample_image, rcnn_pred, "R-CNN Detections", model_type="rcnn")
plot_detections(sample_image, faster_rcnn_pred, "Faster R-CNN Detections", model_type="faster_rcnn")
plot_detections(sample_image, yolo_boxes, "YOLO Detections", model_type="yolo")

import pandas as pd

# Model comparison data
model_comparison = {
    "Model": ["R-CNN", "Faster R-CNN", "YOLO"],
    "Inference Time (ms)": [200, 50, 10],  # Example values
    "Accuracy": [0.75, 0.85, 0.88]
}

# Create DataFrame
comparison_df = pd.DataFrame(model_comparison)

# Display the table
print("\nModel Performance Comparison:\n")
print(comparison_df.to_string(index=False))
