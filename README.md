# SpudScan


**Automated detection and localization of six predefined fetal ultrasound planes during the second and third trimesters of pregnancy using YOLOv8.**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Model](https://img.shields.io/badge/Model-YOLOv8-orange)
![Task](https://img.shields.io/badge/Task-Ultrasound%20Object%20Detection-purple)
![Status](https://img.shields.io/badge/Status-Research%20Project-lightgrey)

## Overview

Fetal ultrasound examination is an important component of prenatal assessment during the second and third trimesters. Identifying standard anatomical planes in ultrasound images is a relevant computer vision task because image quality, anatomical variation, acquisition angle, and operator experience can affect plane recognition.

This project explores the use of the **YOLOv8 object detection architecture** to detect and localize six predefined fetal ultrasound planes in images acquired during the second and third trimesters.

The model processes an ultrasound image and generates predictions for the trained classes, including bounding-box coordinates and confidence scores. These predictions can support research into automated ultrasound image analysis and quality-assessment workflows.

The system is intended as a research and development project. Its actual capabilities depend on the training dataset, class definitions, annotation quality, model configuration, and independent evaluation results.

## Objectives

The primary objectives of this project are to:

* Develop a YOLOv8-based object detection pipeline for fetal ultrasound images.
* Detect six predefined standard ultrasound planes.
* Localize the detected planes using bounding boxes.
* Support images from the second and third trimesters.
* Evaluate detection performance using standard object detection metrics.
* Visualize model predictions for qualitative inspection.
* Establish a reproducible workflow for training, validation, testing, and inference.
* Investigate the potential use of automated plane detection in ultrasound image analysis.

## Target Ultrasound Planes

The model is designed for six target classes. Their exact definitions should match the dataset annotations and the project's clinical or research protocol.

| Class ID | Plane name | Description                                        |
| -------- | ---------- | -------------------------------------------------- |
| 0        | `PLANE_1`  | Replace with the first annotated ultrasound plane  |
| 1        | `PLANE_2`  | Replace with the second annotated ultrasound plane |
| 2        | `PLANE_3`  | Replace with the third annotated ultrasound plane  |
| 3        | `PLANE_4`  | Replace with the fourth annotated ultrasound plane |
| 4        | `PLANE_5`  | Replace with the fifth annotated ultrasound plane  |
| 5        | `PLANE_6`  | Replace with the sixth annotated ultrasound plane  |

**Important:** These are placeholders, not confirmed anatomical class names. Replace them with the six actual target planes, using consistent terminology throughout the dataset configuration, code, results, and README.

## Key Features

* **Six-class detection:** Supports six configured ultrasound-plane categories.
* **YOLOv8 architecture:** Uses a real-time object detection framework.
* **Bounding-box localization:** Predicts the location of target regions within ultrasound images.
* **Multi-trimester scope:** Intended for evaluation on second- and third-trimester images, subject to dataset coverage.
* **Confidence-based predictions:** Produces confidence scores for detections.
* **Prediction visualization:** Supports inspection of detected regions and class labels.
* **Model evaluation:** Enables class-wise and overall performance analysis.
* **Reproducible experimentation:** Supports documented training configurations and evaluation protocols.
* **Potential deployment:** Can be adapted for inference applications after appropriate technical validation.

## Technology Stack

| Component               | Technology                                   |
| ----------------------- | -------------------------------------------- |
| Programming language    | Python                                       |
| Object detection        | Ultralytics YOLOv8                           |
| Deep learning framework | PyTorch                                      |
| Image processing        | OpenCV / Pillow, as needed                   |
| Annotation format       | YOLO bounding-box format                     |
| Evaluation              | Precision, recall, mAP and inference latency |
| Visualization           | Matplotlib / Ultralytics plotting utilities  |
| Optional deployment     | ONNX or another supported export format      |

## System Architecture

The proposed workflow consists of the following stages:

```text
Fetal Ultrasound Images
          |
          v
Dataset Validation
          |
          v
Image and Annotation Preprocessing
          |
          v
YOLOv8 Model Training
          |
          v
Trained Model Checkpoint
          |
          v
Ultrasound Image Inference
          |
          v
Six-Class Plane Predictions
          |
          v
Bounding Boxes, Class Labels
and Confidence Scores
          |
          v
Evaluation and Visualization
```

The pipeline describes the intended workflow. The final implementation should reflect the scripts and components actually present in the repository.

## Repository Structure

A recommended organization for the project is:

```text
yolov8-ultrasound-plane-detection/
├── dataset/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── labels/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── data.yaml
├── configs/
│   └── train.yaml
├── src/
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── utils.py
├── tests/
├── examples/
├── docs/
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

This is a suggested structure, not a claim about the current repository contents. Adapt it to the actual implementation.

Datasets, patient-related information, and large model checkpoints should not be committed to a public repository unless sharing is authorized and appropriate. Use suitable artifact storage or release assets when needed.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/yolov8-ultrasound-plane-detection.git
cd yolov8-ultrasound-plane-detection
```

Replace `YOUR_USERNAME` with the relevant GitHub account or organization.

### 2. Create a virtual environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

If a tested dependency file is available:

```bash
pip install -r requirements.txt
```

For an initial Ultralytics installation:

```bash
pip install ultralytics
```

For GPU training, install a compatible PyTorch build according to the official [PyTorch installation guide](https://pytorch.org/get-started/locally/).

Verify the installation:

```bash
python -c "import ultralytics; print(ultralytics.__version__)"
```

Record the tested Python, PyTorch, and Ultralytics versions to make the experiment reproducible.

## Dataset Preparation

The dataset should contain ultrasound images with corresponding annotations for the target planes.

### Recommended directory structure

```text
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
```

### Dataset splits

* **Training set:** Used to optimize model parameters.
* **Validation set:** Used to monitor performance and select model configurations.
* **Test set:** Reserved for final evaluation after model development.

The split strategy should prevent leakage between sets. When multiple frames come from the same examination or patient, consider splitting by patient or examination rather than randomly splitting individual images. Otherwise, the model may appear to perform better than it would on genuinely unseen cases.

### Annotation format

YOLO object detection labels use one line per bounding box:

```text
class_id x_center y_center width height
```

Coordinates are normalized to the image width and height.

Example:

```text
2 0.512 0.463 0.245 0.318
```

This is an illustrative annotation only and does not represent an actual ultrasound image.

Ensure that:

* Every class ID is between `0` and `5`.
* The class mapping is consistent across all annotations.
* Bounding boxes accurately represent the intended target region.
* Image and label filenames match.
* Missing labels and malformed annotations are handled correctly.
* The dataset contains sufficient examples of each target class and acquisition condition.

### Second- and third-trimester representation

Document the gestational-age range represented by the dataset, the number of examinations in each trimester, and the distribution of the six target classes.

If trimester-specific performance is an intended objective, evaluate the model separately on second-trimester and third-trimester images, provided each evaluation group contains sufficient independent cases.

Do not assume that a model trained on both trimesters will perform equally well across them without measuring that performance.

## Dataset Configuration

Create `dataset/data.yaml` with the dataset paths and class names.

```yaml
path: ./dataset

train: images/train
val: images/val
test: images/test

names:
  0: PLANE_1
  1: PLANE_2
  2: PLANE_3
  3: PLANE_4
  4: PLANE_5
  5: PLANE_6
```

Replace the placeholder class names with the exact six class labels used in the annotations.

## Model Training

The following example demonstrates fine-tuning a YOLOv8 nano checkpoint for six-class object detection.

### Command-line training

```bash
yolo detect train model=yolov8n.pt data=dataset/data.yaml epochs=100 imgsz=640 batch=16 project=runs/detect name=ultrasound-plane-detection
```

### Python training

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="dataset/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    project="runs/detect",
    name="ultrasound-plane-detection",
)
```

The values above are example hyperparameters, not a claim that these settings are optimal.

Adjust the image size, batch size, epochs, and other parameters based on hardware availability, dataset size, image characteristics, and validation results.

The training run generally saves model checkpoints and diagnostic results under the specified output directory.

## Model Validation

Evaluate the trained checkpoint on the validation set:

```bash
yolo detect val model=runs/detect/ultrasound-plane-detection/weights/best.pt data=dataset/data.yaml imgsz=640
```

Validation should assess both overall detection quality and performance for individual classes.

Recommended checks include:

* Precision and recall for each target plane.
* mAP at IoU 0.50 and across IoU thresholds from 0.50 to 0.95.
* Confusion patterns between visually similar classes.
* Missed detections and false-positive detections.
* Sensitivity to image quality, acquisition conditions, and trimester.
* Performance on cases that differ from the training data.

## Inference

### Run inference on an image

```bash
yolo detect predict model=runs/detect/ultrasound-plane-detection/weights/best.pt source=path/to/ultrasound.jpg conf=0.25 save=True
```

Replace the image path with the actual input file.

### Run inference using Python

```python
from ultralytics import YOLO

model = YOLO(
    "runs/detect/ultrasound-plane-detection/weights/best.pt"
)

results = model.predict(
    source="path/to/ultrasound.jpg",
    conf=0.25,
    save=True,
)

for result in results:
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        coordinates = box.xyxy[0].tolist()

        class_name = result.names[class_id]

        print({
            "class_name": class_name,
            "confidence": confidence,
            "bounding_box_xyxy": coordinates,
        })
```

The output includes the predicted class, confidence score, and bounding-box coordinates in pixel units.

The confidence threshold is configurable and should be selected using validation data appropriate to the intended use case.

### Expected output

For a supported input image, the detector may return:

* The predicted ultrasound-plane class.
* The confidence score for each retained detection.
* Bounding-box coordinates.
* An annotated image for visual inspection.

A successful detection does not establish that an image is clinically adequate, that all relevant anatomy is visible, or that any diagnosis is correct.

## Evaluation Metrics

Evaluate the model using a defined protocol and an independent test set.

| Metric            | Purpose                                                               |
| ----------------- | --------------------------------------------------------------------- |
| Precision         | Measures the proportion of predicted detections that are correct      |
| Recall            | Measures the proportion of ground-truth objects successfully detected |
| mAP@0.50          | Measures average precision at an IoU threshold of 0.50                |
| mAP@0.50:0.95     | Measures average precision across multiple IoU thresholds             |
| Per-class metrics | Identifies differences in performance between the six classes         |
| Confusion matrix  | Helps identify errors between target categories                       |
| Inference latency | Measures prediction time under specified conditions                   |
| Model size        | Measures the storage footprint of the checkpoint                      |

### Performance reporting

Populate the following table only after running and verifying the experiments.

| Metric                        | Result         |
| ----------------------------- | -------------- |
| Dataset size                  | To be measured |
| Independent test examinations | To be reported |
| Precision                     | To be measured |
| Recall                        | To be measured |
| mAP@0.50                      | To be measured |
| mAP@0.50:0.95                 | To be measured |
| Second-trimester performance  | To be measured |
| Third-trimester performance   | To be measured |
| Inference latency             | To be measured |
| Evaluation hardware           | To be reported |

Report per-class results and the evaluation conditions wherever possible. State whether the measurements are image-level or patient/examination-level and explain the test split methodology.

## Model Artifacts

The training workflow may produce a structure similar to:

```text
runs/
└── detect/
    └── ultrasound-plane-detection/
        ├── weights/
        │   ├── best.pt
        │   └── last.pt
        ├── results.csv
        ├── results.png
        └── confusion_matrix.png
```

The exact files depend on the installed Ultralytics version and configuration.

* `best.pt`: Checkpoint selected as best according to the training workflow.
* `last.pt`: Checkpoint saved at the final training epoch.
* `results.csv`: Available training and validation statistics.
* Diagnostic plots: Useful for examining convergence and class-specific errors.

Do not commit large checkpoints directly to Git unless appropriate. Consider GitHub Releases, Git LFS, or approved model-artifact hosting. Respect any restrictions on distributing pretrained weights.

## Reproducibility

For every reported experiment, document:

* Dataset version and provenance.
* Six class definitions and annotation protocol.
* Number of independent examinations and images.
* Train, validation, and test split methodology.
* Initial model checkpoint.
* Python, PyTorch, and Ultralytics versions.
* Training configuration and random seeds.
* Input image resolution and preprocessing.
* Confidence and IoU thresholds.
* Hardware and inference benchmarking conditions.

Where feasible, include the training configuration and evaluation scripts so that other developers can reproduce the results.

## Limitations

* Detection performance depends on dataset quality, annotation consistency, and coverage of the intended population and acquisition conditions.
* The model may fail on poor-quality images, unusual anatomical appearances, incomplete views, or out-of-distribution inputs.
* Performance may differ between the second and third trimesters.
* Similar-looking planes may be confused.
* Confidence scores are not calibrated probabilities of clinical correctness.
* Detecting a region does not establish diagnostic accuracy or clinical acceptability.
* Results from a single dataset may not generalize to different devices, hospitals, or acquisition protocols.

The model should be considered a research prototype unless and until it has undergone appropriate independent validation and the relevant clinical, regulatory, privacy, and deployment requirements have been addressed.

## Future Improvements

Potential extensions include:

* Improving class balance and annotation consistency.
* Increasing dataset diversity across acquisition conditions.
* Evaluating trimester-specific generalization.
* Performing patient-level and external-dataset evaluation.
* Analyzing false positives, false negatives, and confusion between classes.
* Comparing YOLOv8 model sizes and alternative detection approaches.
* Benchmarking accuracy, inference latency, and memory requirements.
* Adding automated data validation and continuous integration.
* Developing a demonstration interface for authorized research images.
* Evaluating whether image-quality assessment or additional localization methods are appropriate for the intended research objective.

## Responsible Use and Data Privacy

Ultrasound images may contain sensitive medical information. Ensure that dataset access, storage, processing, and publication comply with applicable consent, institutional, and privacy requirements.

Do not upload identifiable patient images, patient records, or confidential datasets to a public repository. Share only authorized and appropriately de-identified examples.

This repository describes a technical research project and is not intended to provide medical diagnoses or replace qualified healthcare professionals.

## Contributing

Contributions that improve code quality, reproducibility, evaluation, and documentation are welcome.

Before submitting changes:

1. Explain the purpose of the change.
2. Run the relevant tests.
3. Document new dependencies and configuration options.
4. Avoid committing secrets, private data, or unnecessary artifacts.
5. Include reproducible evidence for any claimed performance improvement.

## License

Add a `LICENSE` file that reflects the intended distribution terms for the project. Verify the applicable licenses and restrictions for the source code, pretrained weights, dependencies, and dataset before publishing or redistributing them.

## Acknowledgments

This project uses the Ultralytics YOLO framework and PyTorch for deep learning and object detection.

* Ultralytics: https://github.com/ultralytics/ultralytics
* PyTorch: https://pytorch.org/

Review the relevant licenses and terms before distributing project artifacts.

---

**Project:** YOLOv8-Based Six-Plane Fetal Ultrasound Detection
**Model:** YOLOv8 (nano checkpoint shown in the example commands)
**Target scope:** Second- and third-trimester fetal ultrasound images
**Task:** Six-class object detection and localization
**Project status:** Research and development; update according to actual validation status.
