from ultralytics import YOLO

# Load model (YOLOv8n is lightweight; use YOLOv8m or YOLOv8l for better accuracy)
model = YOLO('yolov8n.pt')

# Train
model.train(data=r'C:\Users\vasan\Videos\My First Project.v1i.yolov8\data.yaml', epochs=20, imgsz=640, batch=16)
