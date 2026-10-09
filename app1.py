from flask import Flask, request, send_file, jsonify
from ultralytics import YOLO
import cv2
import os
import uuid

app = Flask(__name__)

# Load YOLO model
model = YOLO(r'C:\Users\vasan\Videos\My First Project.v1i.yolov8\runs\detect\train2\weights\best.pt')
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def process_image(input_path, output_path):
    results = model(input_path)
    img = cv2.imread(input_path)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = f"{model.names[cls]} {conf:.2f}"

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imwrite(output_path, img)


def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = f"{model.names[cls]} {conf:.2f}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    ext = file.filename.split(".")[-1].lower()

    file_id = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_DIR, f"{file_id}.{ext}")
    output_path = os.path.join(OUTPUT_DIR, f"pred_{file_id}.{ext}")

    file.save(input_path)

    # Image formats
    if ext in ["jpg", "jpeg", "png"]:
        process_image(input_path, output_path)

    # Video formats
    elif ext in ["mp4", "avi", "mov"]:
        process_video(input_path, output_path)

    else:
        return jsonify({"error": "Unsupported file format"}), 400

    return send_file(output_path, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
