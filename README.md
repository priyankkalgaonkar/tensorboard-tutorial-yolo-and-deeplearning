# TensorBoard Tutorial — YOLO + PyTorch + TensorFlow

This repository contains a complete Google Colab tutorial demonstrating how to use **TensorBoard** with:

- Ultralytics YOLO (PyTorch)
- Custom PyTorch training loops
- TensorFlow / Keras models

The notebook walks through enabling TensorBoard, writing logs, visualizing metrics, and integrating logging directly into model training workflows.

## File Included
- TensorBoard_Tutorial.ipynb: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1lB82opvIUYZ3HJgP5_RoZNRwnx2DU_Lf?usp=sharing)

## How to Use This Notebook

1. Open it using Google Colab.
2. Run: **Runtime → Run all**

TensorBoard will launch inside Colab and display:
- Loss curves
- mAP / Precision / Recall (YOLO)
- Learning rate schedules
- Confusion matrices
- Image samples
- Weight histograms
- Custom scalar plots
- TensorFlow summaries

---

# Tutorial Overview

## SECTION 1 — TensorBoard with YOLO (PyTorch)
- Shows how to train Ultralytics YOLO models with TensorBoard logging enabled.
- Automatically logs:
  - Training/validation loss  
  - Precision/Recall  
  - mAP scores  
  - Learning rate  
  - Predictions / images  
- Launch TensorBoard directly from Colab.

Example from the notebook:

```python
!yolo train model=yolov8n.pt data=coco128.yaml epochs=20 imgsz=640 project=tb_yolo name=run1 tensorboard=True
```

Start TensorBoard:

```python
%load_ext tensorboard
%tensorboard --logdir runs
```

## SECTION 2 — Custom TensorBoard Logging (PyTorch)
Shows how to manually log:
- Scalars
- Images
- Histograms
- Model graph
- Custom training-loop metrics

Uses:

```python
from torch.utils.tensorboard import SummaryWriter
```

Example:

```python
writer.add_scalar("Loss/train", loss, epoch)
```

## SECTION 3 — TensorBoard with TensorFlow/Keras
Demonstrates:
- Enabling TensorBoard callback
- Logging training metrics
- Viewing graphs and distributions

Example:

```python
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs_tf", histogram_freq=1)
model.fit(x_train, y_train, epochs=5, callbacks=[tensorboard_callback])
```

## Running TensorBoard in Colab
The notebook uses:
```perl
%load_ext tensorboard
%tensorboard --logdir <log_directory>
```

TensorBoard will open in an interactive window inside Colab.

## Requirements (if running locally)
Install:
```nginx
pip install torch torchvision tensorboard tensorflow ultralytics
```

Start TensorBoard locally:
```css
tensorboard --logdir runs
```

## Contributing
Pull requests are welcome—especially TensorBoard examples for:
- YOLO training variations
- Custom PyTorch models
- Additional TensorFlow workflows
