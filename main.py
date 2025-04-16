import cv2
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from activity_model import ActivityRecognizer
from utils import log_event
import threading
import serial  # For serial communication with ECG device

# Display startup banner
print("="*50)
print("HemoVital ECG Analysis System")
print("Using AnoBeat model for anomaly detection")
print("="*50)
print("Controls:")
print(" - Press 'q' to exit")
print(" - Press 's' to save a screenshot")
print("="*50)

# Create output directory for screenshots
if not os.path.exists("screenshots"):
    os.makedirs("screenshots")

# Initialize system components
recognizer = ActivityRecognizer()
last_action = None
last_anomaly_time = 0
anomaly_cooldown = 10
fps_values = deque(maxlen=30)  # Track FPS
last_time = time.time()

# Create visualization window
plt.figure(figsize=(12, 6))
plt.ion()  # Interactive mode

# Function to read ECG data from actual hardware
def read_ecg_data():
    """Read ECG data from actual hardware"""
    # Example using Arduino with serial connection
    if hasattr(read_ecg_data, 'serial_port'):
        try:
            if read_ecg_data.serial_port.in_waiting > 0:
                line = read_ecg_data.serial_port.readline().decode('utf-8').strip()
                return float(line)
        except Exception as e:
            print(f"Serial read error: {e}")
    else:
        try:
            # Initialize serial port - adjust COM port and baud rate as needed
            read_ecg_data.serial_port = serial.Serial('COM3', 9600, timeout=1)
            time.sleep(2)  # Allow connection to establish
        except Exception as e:
            print(f"Serial connection error: {e}")
    
    return 0.0  # Default value if read fails

print("System initialized. Starting ECG monitoring...")
log_event("ECG monitoring started")

try:
    while True:
        start_time = time.time()
        
        # Get ECG sample
        ecg_sample = read_ecg_data()
        recognizer.add_ecg_sample(ecg_sample)
        
        # Run analysis
        action = recognizer.predict()
        
        if action != last_action:
            log_event(f"ECG status: {action}")
            last_action = action
        
        # Handle anomaly detection
        if "arrhythmia" in action.lower():
            if time.time() - last_anomaly_time > anomaly_cooldown:
                log_event("⚠️ CARDIAC ANOMALY DETECTED ⚠️")
                last_anomaly_time = time.time()
                
                # Save screenshot of ECG
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                plt.savefig(f"screenshots/anomaly_{timestamp}.png")
        
        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - start_time)
        fps_values.append(fps)
        avg_fps = sum(fps_values) / len(fps_values)
        
        # Update visualization every 10 iterations to avoid slowing down processing
        if int(time.time() * 10) % 5 == 0:
            current_ecg, reconstruction, error = recognizer.get_current_data()
            
            if current_ecg is not None:
                plt.clf()
                plt.plot(current_ecg, 'b-', label="ECG Signal")
                plt.plot(reconstruction, 'r--', label="Reconstruction")
                plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                plt.legend()
                plt.title(f"ECG Analysis - Status: {action} | Error: {error:.6f} | FPS: {avg_fps:.1f}")
                plt.ylim(-1.5, 1.5)
                plt.pause(0.01)
        
        # Check for key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Exiting...")
            break
        elif key == ord('s'):
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            plt.savefig(f"screenshots/capture_{timestamp}.png")
            print(f"Screenshot saved: capture_{timestamp}.png")

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    # Cleanup
    log_event("ECG monitoring stopped")
    print("HemoVital ECG system shutdown complete.")
    plt.close('all')