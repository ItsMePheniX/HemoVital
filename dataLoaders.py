from torch.utils.data import Dataset, DataLoader
import os
import torch


OUTPUT_PATH = r"E:\Codes\HAR Model"
class HARVideoDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.classes = sorted(os.listdir(data_path))
        self.samples = []
        for class_id, class_name in enumerate(self.classes):
            class_dir = os.path.join(data_path, class_name)
            for file_name in os.listdir(class_dir):
                self.samples.append((os.path.join(class_dir, file_name), class_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        video_tensor = torch.load(video_path)
        return video_tensor, label


dataset = HARVideoDataset(OUTPUT_PATH)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)
