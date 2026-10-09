from ultralytics import YOLO
import torch
import time
import os

# Load trained model
model_path = r"runs\detect\train2\weights\best.pt"
model = YOLO(model_path)

# Run validation (this calculates mAP, precision, recall)
metrics = model.val(data="data.yaml", split="val")

# Extract metrics
map50 = metrics.box.map50
map5095 = metrics.box.map
precision = metrics.box.mp
recall = metrics.box.mr

# Measure FPS
img = torch.zeros((1,3,640,640))

start = time.time()
for _ in range(100):
    model(img)
end = time.time()

fps = 100/(end-start)

# Model size
model_size = os.path.getsize(model_path)/(1024*1024)

print("\n----- MODEL METRICS -----")
print(f"mAP@50: {map50:.3f}")
print(f"mAP@50-95: {map5095:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"FPS: {fps:.2f}")
print(f"Model Size: {model_size:.2f} MB")