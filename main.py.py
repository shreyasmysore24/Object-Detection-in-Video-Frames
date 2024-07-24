# please install YOLOv5 from github to run this code!

import cv2
import torch
import os
import sys
from pathlib import Path
from torchvision import transforms
import numpy as np

# Add the YOLOv5 directory to the Python path

# Add the YOLOv5 directory to the Python path
yolo_path = 'C:/Users/shrey/Desktop/yolov5-master/yolov5-master'
sys.path.append(yolo_path)

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes, colorstr

# Load the YOLOv5 model
weights = Path(yolo_path) / 'yolov5x.pt'  # Path to the weights file
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DetectMultiBackend(weights, device=device)

# Ensure CUDA is being used
print(f"Using device: {device}")

# Load the YOLOv5 model
# weights = Path(yolo_path) / 'yolov5x.pt'  # Path to the weights file
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = DetectMultiBackend(weights, device=device)
model.eval()  # Set the model to evaluation mode

# Define the transformation to convert frames to tensors
def transform(frame):
    # Resize to 640x640 and convert to tensor
    transform_pipeline = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((640, 640)),  # Resize frame to 640x640
        transforms.ToTensor(),
    ])
    return transform_pipeline(frame)

# Generate colors for each class
def generate_colors(num_colors):
    colors = []
    for i in range(num_colors):
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        colors.append(color)
    return colors

# Function to detect objects in a frame and draw bounding boxes and labels
def detect_objects(frame):
    img = transform(frame).unsqueeze(0).to(device)  # Add batch dimension and move to device
    pred = model(img)
    pred = non_max_suppression(pred)
    detections = pred[0]
    if detections is not None:
        detections[:, :4] = scale_boxes(img.shape[2:], detections[:, :4], frame.shape).round()
        num_classes = len(model.names)
        colors = generate_colors(num_classes)  # Generate colors for each class
        for *xyxy, conf, cls in detections:
            x1, y1, x2, y2 = map(int, xyxy)
            label = f'{model.names[int(cls)]} {conf:.2f}'
            color = colors[int(cls)]
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # Add label
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return frame

# Create output directory
output_dir = 'detected_objects2'
os.makedirs(output_dir, exist_ok=True)

# Capture video
video_path = r'C:\Users\shrey\Videos\AI4SEE sample.mp4'  # your video path
cap = cv2.VideoCapture(video_path)

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects and draw bounding boxes and labels
    output_frame = detect_objects(frame)

    # Save the processed frame
    output_path = os.path.join(output_dir, f'frame{frame_count}.jpg')
    cv2.imwrite(output_path, output_frame)

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
