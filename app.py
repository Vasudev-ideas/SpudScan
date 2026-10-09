from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
from ultralytics import YOLO
import os
import tempfile
import logging

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load your YOLO model (replace with your trained model path)
model = YOLO(r'C:\Users\vasan\Videos\My First Project.v1i.yolov8\ultrasound_video_slow1.mp4')  # e.g., 'best.pt'

# Allowed file extensions
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov'}

def allowed_file(filename, extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in extensions

@app.route('/detect_image', methods=['POST'])
def detect_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
        return jsonify({'error': 'Invalid file type'}), 400
    
    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
        file.save(temp_file.name)
        temp_path = temp_file.name
    
    try:
        # Run YOLO inference
        results = model(temp_path)
        
        # Extract detections (e.g., bounding boxes, labels)
        detections = []
        for result in results:
            for box in result.boxes:
                detections.append({
                    'label': result.names[int(box.cls)],
                    'confidence': float(box.conf),
                    'bbox': box.xyxy.tolist()[0]  # [x1, y1, x2, y2]
                })
        
        os.unlink(temp_path)  # Clean up
        return jsonify({'detections': detections})
    
    except Exception as e:
        logging.error(f"Error processing image: {str(e)}")
        os.unlink(temp_path)
        return jsonify({'error': 'Processing failed'}), 500

@app.route('/detect_video', methods=['POST'])
@app.route('/')
def index():
    return send_file('index.html')
def detect_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename, ALLOWED_VIDEO_EXTENSIONS):
        return jsonify({'error': 'Invalid file type'}), 400
    
    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        file.save(temp_file.name)
        temp_path = temp_file.name
    
    try:
        cap = cv2.VideoCapture(temp_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        detections_summary = {}
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every 10th frame for efficiency (adjust as needed)
            if frame_idx % 10 == 0:
                results = model(frame)
                for result in results:
                    for box in result.boxes:
                        label = result.names[int(box.cls)]
                        if label not in detections_summary:
                            detections_summary[label] = []
                        detections_summary[label].append({
                            'frame': frame_idx,
                            'confidence': float(box.conf),
                            'bbox': box.xyxy.tolist()[0]
                        })
            
            frame_idx += 1
        
        cap.release()
        os.unlink(temp_path)
        return jsonify({'summary': detections_summary, 'total_frames': frame_count, 'fps': fps})
    
    except Exception as e:
        logging.error(f"Error processing video: {str(e)}")
        os.unlink(temp_path)
        return jsonify({'error': 'Processing failed'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)