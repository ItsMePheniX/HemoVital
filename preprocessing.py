import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from tqdm import tqdm

# Define paths
DATASET_PATH = "path/to/your/dataset"  # Change this to your dataset path
OUTPUT_PATH = "processed_data"

# Create output directory
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Transformation for TimeSformer (resize, normalize)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def extract_frames(video_path, output_folder, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in np.linspace(0, total_frames - 1, num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        frame = transform(frame)
        frames.append(frame)

    cap.release()
    return torch.stack(frames) if len(frames) == num_frames else None

# Process all videos
for class_name in os.listdir(DATASET_PATH):
    class_path = os.path.join(DATASET_PATH, class_name)
    output_class_path = os.path.join(OUTPUT_PATH, class_name)
    os.makedirs(output_class_path, exist_ok=True)

    for video_name in tqdm(os.listdir(class_path), desc=f"Processing {class_name}"):
        video_path = os.path.join(class_path, video_name)
        frames = extract_frames(video_path, output_class_path)
        if frames is not None:
            torch.save(frames, os.path.join(output_class_path, video_name.replace(".mp4", ".pt")))
