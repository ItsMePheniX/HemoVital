import os
import cv2
import torch
import numpy as np
from transformers import TimesformerForVideoClassification, AutoProcessor
import time

# Specify the model path
MODEL_PATH = r"C:\Users\vadit\.cache\huggingface\hub\models--facebook--timesformer-base-finetuned-k400\snapshots\8aaf40ea7d3d282dcb0a5dea01a198320d15d6c0"

def load_model():
    """Load the TimeSformer model and processor"""
    print(f"Loading model from {MODEL_PATH}...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = TimesformerForVideoClassification.from_pretrained(MODEL_PATH)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)
    
    return model, processor, device

def extract_frames(video_path, num_frames=8):
    """Extract frames from a video file"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        raise ValueError(f"Video has no frames: {video_path}")
    
    indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)
    
    frames = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    
    cap.release()
    
    if not frames:
        raise ValueError(f"Could not extract frames from: {video_path}")
    
    return frames

def capture_webcam_frames(num_frames=8, delay=0.5):
    """Capture frames from webcam"""
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise ValueError("Could not open webcam")
    
    frames = []
    print(f"Capturing {num_frames} frames from webcam...")
    
    for i in range(num_frames):
        print(f"Frame {i+1}/{num_frames}")
        ret, frame = cap.read()
        
        if not ret:
            break
        
        cv2.imshow("Webcam", frame)
        cv2.waitKey(1)
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        
        if i < num_frames - 1:
            time.sleep(delay)
    
    cap.release()
    cv2.destroyAllWindows()
    
    return frames

def predict(model, processor, device, frames, top_k=5):
    inputs = processor(frames, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    top_probs, top_indices = torch.topk(probs, top_k)
    
    results = []
    for i in range(top_k):
        idx = top_indices[0, i].item()
        prob = top_probs[0, i].item()
        class_name = model.config.id2label[idx]
        results.append((class_name, prob))
    
    return results

def main():
    model, processor, device = load_model()
    
    choice = input("Use [1] webcam or [2] video file? (1/2): ").strip()
    
    try:
        if choice == "1":
            frames = capture_webcam_frames()
        elif choice == "2":
            video_path = input("Enter path to video file: ").strip()
            if not os.path.isfile(video_path):
                print(f"Error: File not found: {video_path}")
                return
            frames = extract_frames(video_path)
        else:
            print("Invalid choice. Please enter 1 or 2.")
            return
        
        print("Running inference...")
        predictions = predict(model, processor, device, frames)
        
        print("\nTop predictions:")
        for i, (class_name, prob) in enumerate(predictions, 1):
            print(f"{i}. {class_name}: {prob:.4f} ({prob*100:.2f}%)")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()