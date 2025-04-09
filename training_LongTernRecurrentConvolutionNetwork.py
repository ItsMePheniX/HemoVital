import os
import cv2
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import LabelEncoder

# Reproducibility
SEED = 27
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configs
IMAGE_HEIGHT, IMAGE_WIDTH = 256, 256
SEQUENCE_LENGTH = 10
BATCH_SIZE = 4
EPOCHS = 30
MAX_VIDEOS_PER_CLASS = 50
DATASET_DIR = r"E:\DataSets\HAR"
CLASSES_LIST = sorted([d.name for d in os.scandir(DATASET_DIR) if d.is_dir()])

# Image preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
    transforms.ToTensor(),
])

# Dataset class
class VideoDataset(Dataset):
    def __init__(self, dataset_dir, classes, sequence_length, transform=None):
        self.dataset_dir = dataset_dir
        self.classes = classes
        self.sequence_length = sequence_length
        self.transform = transform
        self.samples = []
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.classes)
        self._load_dataset()

    def _load_dataset(self):
        for class_name in self.classes:
            class_dir = os.path.join(self.dataset_dir, class_name)
            files = os.listdir(class_dir)[:MAX_VIDEOS_PER_CLASS]
            for file_name in files:
                video_path = os.path.join(class_dir, file_name)
                self.samples.append((video_path, class_name))

    def __len__(self):
        return len(self.samples)

    def _extract_frames(self, video_path):
        frames = []
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if frame_count < self.sequence_length:
            return None

        skip = max(int(frame_count / self.sequence_length), 1)

        for i in range(self.sequence_length):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * skip)
            success, frame = cap.read()
            if not success:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)

        cap.release()

        if len(frames) == self.sequence_length:
            return torch.stack(frames)
        return None

    def __getitem__(self, idx):
        video_path, class_name = self.samples[idx]
        label = self.label_encoder.transform([class_name])[0]
        frames_tensor = self._extract_frames(video_path)
        if frames_tensor is None:
            return self.__getitem__((idx + 1) % len(self.samples))  # Try next sample
        return frames_tensor, torch.tensor(label, dtype=torch.long)

# LRCN Model
class LRCN(nn.Module):
    def __init__(self, num_classes):
        super(LRCN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(4), nn.Dropout(0.25),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(4), nn.Dropout(0.25),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout(0.25),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.flatten = nn.Flatten()
        self.lstm = nn.LSTM(64 * 4 * 4, 32, batch_first=True)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        b, t, c, h, w = x.size()
        x = x.view(b * t, c, h, w)
        x = self.conv(x)
        x = self.flatten(x)
        x = x.view(b, t, -1)
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out

# Train loop
def train(model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for videos, labels in train_loader:
        videos, labels = videos.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(videos)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# Eval loop
def evaluate(model, val_loader, criterion):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for videos, labels in val_loader:
            videos, labels = videos.to(device), labels.to(device)
            outputs = model(videos)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

# Dataset and DataLoader
dataset = VideoDataset(DATASET_DIR, CLASSES_LIST, SEQUENCE_LENGTH, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Model, Loss, Optimizer
model = LRCN(len(CLASSES_LIST)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
for epoch in range(EPOCHS):
    train_loss = train(model, train_loader, criterion, optimizer)
    val_acc = evaluate(model, val_loader, criterion)
    print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")

# Save model
torch.save(model.state_dict(), "lrcn_model.pth")
print("Model saved as lrcn_model.pth")