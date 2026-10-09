import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load trained model
model = YOLO(r'C:\Users\vasan\Videos\My First Project.v1i.yolov8\runs\detect\train2\weights\best.pt')  # Adjust path if needed

# Load video
video_path = r'C:\Users\vasan\Videos\My First Project.v1i.yolov8\ultrasound_video_slow1.mp4'
cap = cv2.VideoCapture(video_path)

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference
    results = model(frame)

    # Annotate frame
    annotated_frame = results[0].plot()

    # Save frame to disk (optional)
    cv2.imwrite(f'output/frame_{frame_count:04d}.jpg', annotated_frame)

    # Display using€ matplotlib
    plt.imshow(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
    plt.title(f"Fetal Plane Detection - Frame {frame_count}")
    plt.axis('off')
    plt.pause(0.001)  # Small pause to simulate video playback
    plt.clf()

    frame_count += 1

cap.release()
plt.close()
