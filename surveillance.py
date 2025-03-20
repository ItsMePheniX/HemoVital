import cv2
import torch
import numpy as np
from torchvision import transforms
from transformers import TimeSformerForVideoClassification


MODEL_PATH = "har_timesformer.pth"
model = TimeSformerForVideoClassification.from_pretrained(
    "facebook/timesformer-base-finetuned-k400", num_labels=7
)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval().cuda()


CLASS_LABELS = [
    "Clapping",
    "Meeting and Splitting",
    "Sitting",
    "Standing Still",
    "Walking",
    "Walking while Reading",
    "Walking while Using Phone",
]


cap = cv2.VideoCapture(0)  
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

frame_buffer = []
FPS = 16  

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    
    frame = transform(frame)
    frame_buffer.append(frame)

    
    if len(frame_buffer) == FPS:
        
        video_tensor = torch.stack(frame_buffer).unsqueeze(0).cuda()

        
        with torch.no_grad():
            outputs = model(video_tensor).logits
            predicted_class = outputs.argmax(1).item()

       
        cv2.putText(
            frame,
            f"Activity: {CLASS_LABELS[predicted_class]}",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        frame_buffer = []  

   
    cv2.imshow("HAR - Real-Time", frame)

    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()