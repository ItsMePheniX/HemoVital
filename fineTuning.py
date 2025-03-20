import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForVideoClassification
import cv2
import os
import numpy as np
from tqdm import tqdm

# -------------------------
# 1️⃣ Define the Dataset Class
# -------------------------
class HARVideoDataset(Dataset):
    def __init__(self, video_paths, labels, transform=None):
        self.video_paths = video_paths  
        self.labels = labels  
        self.transform = transform  

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

       
        frames = self.load_video(video_path)

        if self.transform:
            frames = self.transform(frames)

        return frames, torch.tensor(label, dtype=torch.long)

    def load_video(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (224, 224))  
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
            frame = frame / 255.0  
            frames.append(frame)
        
        cap.release()
        frames = np.array(frames).astype(np.float32)  
        frames = torch.tensor(frames).permute(3, 0, 1, 2) 

        return frames

# -------------------------
# 2️⃣ Load Dataset Paths
# -------------------------
video_dir = r"E:\DataSets\HAR"  
classes = [
    "Clapping",
    "Meet and Split",
    "Sitting",
    "Standing Still",
    "Walking",
    "Walking While Reading Book",
    "Walking While Using Phone"
]

video_paths = []
labels = []

for idx, class_name in enumerate(classes):
    class_dir = os.path.join(video_dir, class_name)
    for video in os.listdir(class_dir):
        video_paths.append(os.path.join(class_dir, video))
        labels.append(idx)  


dataset = HARVideoDataset(video_paths, labels)

# -------------------------
# 3️⃣ Create DataLoader
# -------------------------
train_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

# -------------------------
# 4️⃣ Load Pretrained Model (TimeSformer)
# -------------------------
model = AutoModelForVideoClassification.from_pretrained(
    r"E:\Codes\timesformer", num_labels=len(classes)
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# -------------------------
# 5️⃣ Define Loss and Optimizer
# -------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

# -------------------------
# 6️⃣ Define Training Loop
# -------------------------
def train_model(model, dataloader, optimizer, criterion, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss, correct, total = 0, 0, 0
        for videos, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            videos, labels = videos.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(videos).logits  
            loss = criterion(outputs, labels)  
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        print(f"Epoch {epoch+1} | Loss: {total_loss/len(dataloader):.4f} | Accuracy: {correct/total:.4f}")

# -------------------------
# 7️⃣ Train Model
# -------------------------
if __name__ == "__main__":
    if len(dataset) == 0:
        raise ValueError("Dataset is empty! Please check dataset path.")
    
    train_model(model, train_loader, optimizer, criterion, epochs=5)
    model.save_pretrained("E:/Codes/HAR_model")  
