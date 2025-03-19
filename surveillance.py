import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from datetime import datetime

# Load MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Load Your Trained Model
model = tf.keras.models.load_model("your_model.h5")  # Update with correct path

# Define activity labels
ACTIVITY_LABELS = ["Sitting", "Standing", "Walking", "Falling"]

# Initialize video capture
cap = cv2.VideoCapture(0)  # Change to "video.mp4" for a file

def extract_landmarks(results):
    """Extracts 33 pose landmarks from MediaPipe results."""
    if results.pose_landmarks:
        return np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten()
    return np.zeros(33 * 3)  # Return zero array if no landmarks detected

# Logging function
def log_activity(activity):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("activity_log.txt", "a") as f:
        f.write(f"{timestamp}: {activity}\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    
    # Extract pose landmarks
    landmarks = extract_landmarks(results)
    landmarks = landmarks.reshape(1, -1)  # Reshape for model input
    
    # Predict activity
    prediction = model.predict(landmarks)
    activity = ACTIVITY_LABELS[np.argmax(prediction)]
    
    # Display result on frame
    cv2.putText(frame, f"Activity: {activity}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Elderly Activity Monitoring", frame)
    
    # Log detected activity
    log_activity(activity)

    # Exit with 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
