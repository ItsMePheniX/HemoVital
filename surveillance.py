import cv2
import torch
import numpy as np
from transformers import AutoModelForVideoClassification, AutoFeatureExtractor
from collections import deque
from datetime import datetime

# Load pre-trained TimeSformer model
MODEL_NAME = r"E:\Codes\timesformer"
model = AutoModelForVideoClassification.from_pretrained(MODEL_NAME)
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)

# Set model to evaluation mode
model.eval()

# Open video stream
vid1 = r"E:\Codes\Projects\human-activity-recognition\test\example3.mp4"
cap = cv2.VideoCapture(vid1)

# Parameters for TimeSformer
num_frames = 8  # Adjust based on model's expected input
frame_queue = deque(maxlen=num_frames)

def preprocess_frame(frame):
    """Preprocess frame for TimeSformer input"""
    frame = cv2.resize(frame, (224, 224))  # Resize to match model input
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
    frame = np.transpose(frame, (2, 0, 1))  # Convert to (C, H, W) format
    frame = frame / 255.0  # Normalize pixel values
    return frame

def log_activity(activity):
    """Log detected activity with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("activity_log.txt", "a") as f:
        f.write(f"{timestamp}: Activity {activity}\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame and add to queue
    processed_frame = preprocess_frame(frame)
    frame_queue.append(processed_frame)

    # Only predict when enough frames are collected
    if len(frame_queue) == num_frames:
        input_tensor = torch.tensor([frame_queue]).float()  # Shape: (1, num_frames, C, H, W)

        with torch.no_grad():
            output = model(input_tensor)

        predicted_index = torch.argmax(output.logits).item()

        # Display predicted class index on screen
        cv2.putText(frame, f"Predicted Activity: {predicted_index}", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Log detected activity
        log_activity(predicted_index)

    # Show video feed
    cv2.imshow("Elderly Activity Monitoring", frame)

    # Press 'f' to exit
    if cv2.waitKey(10) & 0xFF == ord('f'):
        break

cap.release()
cv2.destroyAllWindows()