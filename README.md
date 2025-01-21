# Multiple Person and Mobile Detection with Warning Display

This repository contains a Python script for detecting multiple persons and mobile phones in an image or video feed. The detection system utilizes YOLOv11, a state-of-the-art object detection model, and highlights objects with a confidence score threshold of **60%**. If a mobile phone is detected, a warning is displayed on the screen.

## Features
- Detect multiple persons and mobile phones in real-time or from static images.
- Highlights objects with a bounding box and their respective confidence scores.
- Displays a warning on the screen if a mobile phone is detected.

## Requirements
Before running the script, ensure you have the following dependencies installed:

```bash
pip install ultralytics opencv-python

```
make sure you download yolo11n.pt from yolo ultralytics website.
