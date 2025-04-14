import cv2
import time
from collections import deque
from activity_model import ActivityRecognizer
from utils import log_event

#! In the below section enter your correct url
URL = ''
cap = cv2.VideoCapture(URL)

frame_buffer = deque(maxlen=8)
recognizer = ActivityRecognizer()
last_action = None
last_fall_time = 0
fall_cooldown = 10

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    resized = cv2.resize(frame, (224, 224))
    frame_buffer.append(resized)

    if len(frame_buffer) == 8:
        action = recognizer.predict(list(frame_buffer))

        if action != last_action:
            log_event(action)
            last_action = action

        if "fall" in action.lower():
            if time.time() - last_fall_time > fall_cooldown:
                log_event("⚠️ FALL DETECTED ⚠️")
                last_fall_time = time.time()

        cv2.putText(frame, f"Activity: {action}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255) if "fall" in action.lower() else (0, 255, 0), 2)

    cv2.imshow("Monitoring Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
