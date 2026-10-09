import os
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt


# ✅ Device setup (CPU only)
device = torch.device("cpu")

# 📦 Dataset Loader
class FetalCRLDataset(Dataset):
    def __init__(self, json_file, transform=None):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        image = cv2.imread(entry['file_name'], cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (512, 512))
        image = np.expand_dims(image, axis=0)  # Add channel dimension

        keypoints = np.array(entry['crl'], dtype=np.float32).flatten()  # 2 points → 4 values

        image = torch.tensor(image, dtype=torch.float32)
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(keypoints, dtype=torch.float32)

# 🧠 Model Architecture
class CRLKeypointModel(nn.Module):
    def __init__(self):
        super(CRLKeypointModel, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 4)  # 2 keypoints × 2 coords

    def forward(self, x):
        return self.backbone(x)

# 🏋️ Training Loop
def train_model(model, dataloader, epochs=20):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

# 📊 Evaluation & Visualization
def visualize_predictions(model, dataloader, scale_mm_per_pixel=0.1):
    model.eval()
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            outputs = model(images).cpu().numpy()

            for i in range(len(images)):
                img = images[i].cpu().numpy().squeeze()
                preds = outputs[i].reshape(-1, 2)

                x1, y1 = preds[0]
                x2, y2 = preds[1]

                crl_length = np.linalg.norm(preds[0] - preds[1]) * scale_mm_per_pixel
                gest_age = 8.052 * np.sqrt(crl_length) + 23.73
                edd_days = 280 - gest_age

                plt.imshow(img, cmap='gray')
                plt.plot([x1, x2], [y1, y2], 'r-', linewidth=2)
                plt.title(f"CRL: {crl_length:.2f} mm | GA: {gest_age:.1f} days | EDD in {edd_days:.1f} days")
                plt.axis('off')
                plt.show()

# 🚀 Run Everything
if __name__ == "__main__":
    transform = transforms.Normalize(mean=[0.5], std=[0.5])
    dataset = FetalCRLDataset(r"C:\Users\vasan\Videos\crl.v1i.coco\crl_keypoints_cleaned.json", transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = CRLKeypointModel().to(device)
    train_model(model, dataloader, epochs=50)
    visualize_predictions(model, dataloader)