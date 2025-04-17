import os
import cv2
import time
import numpy as np
import torch
# from transformers import TimesformerFeatureExtractor, TimesformerForVideoClassification
from transformers import AutoImageProcessor, TimesformerForVideoClassification
from collections import deque
from datetime import datetime
import threading

ACTIVITY_CATEGORIES = {
    "emergency": [
        "falling", "stumbling", "fainting", "collapsing"
    ],
    "concerning": [
        "lying down", "not moving", "writhing", "shaking"
    ],
    "monitor_duration": [
        "sitting", "standing still", "sleeping"
    ],
    "movement_patterns": [
        "irregular gait", "shuffling", "limping", "struggling"
    ]
}

ACTIVITY_THRESHOLDS = {
    "sitting_max": 3600,
    "no_movement_max": 1800,
    "sleeping_daytime_max": 7200,
}

ALERT_PRIORITY = {
    "emergency": "CRITICAL",
    "concerning": "HIGH",
    "duration": "MEDIUM",
    "pattern": "LOW"
}


class ElderlyMonitor:
    def __init__(self):
        """Initialize the TimesFormer model for elderly surveillance"""
        print("Loading TimesFormer model for elderly monitoring...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.processor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")
        self.model = TimesformerForVideoClassification.from_pretrained("facebook/timesformer-base-finetuned-k400")
        self.model.to(self.device)

        self.frame_buffer = deque(maxlen=8)
        self.alert_cooldown = 30
        self.last_alert_time = 0

        self.concern_activities = ["falling", "lying down", "stumbling", "fainting"]

        if not os.path.exists("incidents"):
            os.makedirs("incidents")

    def add_frame(self, frame):
        """Add a frame to the processing buffer"""
        frame = cv2.resize(frame, (224, 224))

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        self.frame_buffer.append(frame)

    def predict_activity(self):
        """Predict activity from collected frames"""
        if len(self.frame_buffer) < 8:
            return "collecting frames..."

        try:
            inputs = self.processor(list(self.frame_buffer), return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

            predicted_class_idx = logits.argmax(-1).item()
            activity = self.model.config.id2label[predicted_class_idx]

            if any(concern in activity.lower() for concern in self.concern_activities):
                self._handle_concerning_activity(activity)

            return activity

        except Exception as e:
            print(f"Error in activity prediction: {e}")
            return "error in prediction"

    def _handle_concerning_activity(self, activity):
        """Handle detection of concerning activity"""
        current_time = time.time()

        if current_time - self.last_alert_time < self.alert_cooldown:
            return

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        print(f"ALERT: Detected {activity} at {timestamp}")

        incident_path = f"incidents/incident_{timestamp}.mp4"
        self._save_incident(incident_path)

        self.last_alert_time = current_time

    def _save_incident(self, path):
        """Save current frames as video clip"""
        if len(self.frame_buffer) < 8:
            return

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        height, width = self.frame_buffer[0].shape[:2]
        writer = cv2.VideoWriter(path, fourcc, 15, (width, height))

        for frame in self.frame_buffer:
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(bgr_frame)

        writer.release()
        print(f"Incident saved to {path}")

if __name__ == "__main__":
    print("=" * 50)
    print("HemoVital Elderly Monitoring System")
    print("=" * 50)
    print("Controls:")
    print(" - Press 'q' to exit")
    print(" - Press 's' to save a screenshot")
    print("=" * 50)
    
    camera_index = 0
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print("Error: Cannot open camera. Check connection.")
        exit()
    
    monitor = ElderlyMonitor()
    
    cv2.namedWindow("Elderly Monitoring", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Elderly Monitoring", 800, 600)
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame. Retrying...")
                time.sleep(0.5)
                continue
            
            display_frame = frame.copy()
            
            if frame_count % 3 == 0:
                monitor.add_frame(frame.copy())
                activity = monitor.predict_activity()
            
            frame_count += 1
            
            cv2.putText(display_frame, f"Activity: {activity}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Press 'q' to quit", (10, display_frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow("Elderly Monitoring", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Exiting...")
                break
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                screenshot_path = f"incidents/screenshot_{timestamp}.jpg"
                cv2.imwrite(screenshot_path, frame)
                print(f"Screenshot saved: {screenshot_path}")
                
    except KeyboardInterrupt:
        print("Interrupted by user")
        
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Monitoring system shutdown")