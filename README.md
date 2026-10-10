# Autonomous Real-Time Drone (UAV) Detection System

An AI-powered computer vision system designed to detect and track small unmanned aerial vehicles (drones) in real-time using optical camera feeds and state-of-the-art YOLO object detection.

![UAV Real-Time Detection Demo](assets/output_test.gif)

## 🔬 Core Research Objective
Designing an optimized, single-stage spatial detection pipeline tailored to isolate small-scale aerial intruders under complex atmospheric and background clutter, bypassing the limitations of active RF or high-cost RADAR arrays.

---

## Problem Overview

Commercial drones pose security and privacy risks to critical locations. Traditional detection methods have major limitations:
- **RF (Radio Frequency):** Fails if the drone is flying autonomously without a signal.
- **RADAR:** Expensive and easily confused by ground objects or birds.
- **Acoustics:** Fails in loud or windy environments.

**Our Approach:** Uses standard cameras and deep learning to visually identify drones instantly regardless of RF signals or ambient noise.

## Key Features & Advantages

- **High Accuracy:** Specifically trained to detect small aerial objects against complex sky and urban backgrounds.
- **Real-Time Speed:** Optimized to process video frames in under 10 milliseconds.
- **Cost-Effective:** Works with standard camera hardware instead of expensive RADAR equipment.


## ⚡ Performance Profiling & Edge Constraints

| Optimization Tier | Precision Mode | Target Hardware | Inference Latency | mAP@0.5 |
| :--- | :--- | :--- | :---: | :---: |
| **Baseline PyTorch** | FP32 | CPU (Intel/Apple Silicon) | ~45.2 ms | 92.1% |
| **Optimized Half-Precision**| FP16 | GPU (CUDA / TensorRT) | ~8.1 ms | 92.0% |

---

## 🚀 Model Training & Fine-Tuning

To support full research reproducibility, this repository includes the interactive training notebook (`train.ipynb`) and dataset configurations. 

You can execute the training pipeline directly via Jupyter or run it programmatically using the Ultralytics framework:

```python
from ultralytics import YOLO

# Load pre-trained medium model weights
model = YOLO('yolov8m.pt')

# Execute custom training pipeline for small-object aerial detection
results = model.train(
    data='custom_data.yaml',
    imgsz=640,
    epochs=250,
    batch=8,
    name='DroneDetection-YOLOV8'
)

```

## 📦 Model Weights & Checkpoints

Download the fine-tuned model weights directly from the [v1.0.0 Release Page](https://github.com/TheRabiyaQuadri/uav-detection-yolo/releases/tag/v1.0.0):

| Model Variant | Input Size | mAP@0.5 | Download Link |
| :--- | :---: | :---: | :---: |
| **YOLOv8m-UAV** | $640 \times 640$ | 92.0% | [Download `.pt` Weights](https://github.com/TheRabiyaQuadri/uav-detection-yolo/releases/download/v1.0.0/uav_model.pt) |

Place the downloaded `.pt` file into a `weights/` directory before executing `detect.py`.

## 🛠️ Engineering Execution
To run inference with custom weight loading and tensor validation:
```bash
# 1. Clone this repository
git clone https://github.com/TheRabiyaQuadri/uav-detection-yolo.git
cd uav-detection-yolo

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run detection on a video file
python detect.py --source sample_video.mp4 --weights weights/uav_model.pt
