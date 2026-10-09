## ---(Mon Oct 13 20:21:29 2025)---
from ultralytics import YOLO
import os
import torch

def train_model():
    # Initialize model
    model = YOLO('yolov8n.pt')  # You can use yolov8s.pt, yolov8m.pt for better accuracy
    
    # Training configuration
    training_config = {
        'data': r"C:\Users\vasan\Videos\My First Project.v1i.yolov8\data.yaml",
        'epochs': 100,
        'imgsz': 640,
        'batch': 16,
        'device': '0' if torch.cuda.is_available() else 'cpu',
        'workers': 4,
        'patience': 10,
        'save': True,
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'auto',
        'lr0': 0.01,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'plots': True
    }
    
    # Start training
    results = model.train(**training_config)
    
    # Validate the model
    metrics = model.val()
    print(f"mAP50-95: {metrics.box.map}")
    print(f"mAP50: {metrics.box.map50}")
    
    return model

if __name__ == "__main__":
    trained_model = train_model()
    # Save the trained model
    trained_model.export(format='onnx')  # Optional: export to ONNX
from ultralytics import YOLO
import os
import torch

def train_model():
    # Initialize model
    model = YOLO('yolov8n.pt')  # You can use yolov8s.pt, yolov8m.pt for better accuracy
    
    # Training configuration
    training_config = {
        'data': r"C:\Users\vasan\Videos\My First Project.v1i.yolov8\data.yaml",
        'epochs': 2,
        'imgsz': 640,
        'batch': 16,
        'device': '0' if torch.cuda.is_available() else 'cpu',
        'workers': 4,
        'patience': 10,
        'save': True,
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'auto',
        'lr0': 0.01,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'plots': True
    }
    
    # Start training
    results = model.train(**training_config)
    
    # Validate the model
    metrics = model.val()
    print(f"mAP50-95: {metrics.box.map}")
    print(f"mAP50: {metrics.box.map50}")
    
    return model

if __name__ == "__main__":
    trained_model = train_model()
    # Save the trained model
    trained_model.export(format='onnx')