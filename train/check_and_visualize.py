# check_and_visualize.py
import os, glob, cv2
from pathlib import Path

# Update these
IMAGES_DIR = r"C:\Users\vasan\Videos\My First Project.v1i.yolov8\train\images"   # path to train images
LABELS_DIR = r"C:\Users\vasan\Videos\My First Project.v1i.yolov8\train\labels"   # path to train labels
OUT_DIR = "debug_overlays"
CLASS_NAMES = ['Abdominal Plane','Biparietal Plane','Femur Plane','Heart Plane','No Plane','Spine Plane']

os.makedirs(OUT_DIR, exist_ok=True)

def draw_bbox(img, cls, xc, yc, w, h):
    H, W = img.shape[:2]
    x_center = xc * W
    y_center = yc * H
    bw = w * W
    bh = h * H
    x1 = int(x_center - bw/2)
    y1 = int(y_center - bh/2)
    x2 = int(x_center + bw/2)
    y2 = int(y_center + bh/2)
    cv2.rectangle(img, (x1,y1),(x2,y2), (0,255,0), 2)
    cv2.putText(img, CLASS_NAMES[int(cls)], (x1, max(15,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

counts = [0]*len(CLASS_NAMES)
bad_files = []
sampleed = 0
for img_path in glob.glob(os.path.join(IMAGES_DIR, "*")):
    base = Path(img_path).stem
    label_path = os.path.join(LABELS_DIR, base + ".txt")
    if not os.path.exists(label_path):
        continue
    img = cv2.imread(img_path)
    H, W = img.shape[:2]
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                bad_files.append(label_path)
                continue
            cls, xc, yc, w, h = parts
            cls_i = int(cls)
            xc, yc, w, h = map(float, [xc,yc,w,h])
            # basic checks
            if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < w <= 1 and 0 < h <= 1):
                bad_files.append(label_path)
            else:
                counts[cls_i] += 1
                if sampleed < 50:
                    draw_bbox(img, cls_i, xc, yc, w, h)
    # save a debug overlay for the first few
    if sampleed < 50:
        cv2.imwrite(os.path.join(OUT_DIR, base + "_debug.jpg"), img)
        sampleed += 1

print("Per-class counts:")
for i,n in enumerate(counts):
    print(f"{i} {CLASS_NAMES[i]} : {n}")
if bad_files:
    print("Files with possible label issues:", bad_files[:10])
else:
    print("No obvious label format issues found.")
