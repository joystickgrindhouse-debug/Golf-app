# AI Golf Swing Coach

## Overview
A web-based AI Golf Swing Coach application that analyzes golf swing posture using MediaPipe pose detection and OpenCV. Users can upload photos or use their camera to get real-time feedback on their golf swing mechanics.

## Tech Stack
- **Backend**: Python 3.12, Flask
- **AI/ML**: MediaPipe Tasks API (v0.10+), OpenCV (headless)
- **Frontend**: Vanilla HTML/CSS/JavaScript
- **Production Server**: Gunicorn

## Project Structure
```
├── app.py                    # Flask application with pose analysis logic
├── templates/
│   └── index.html            # Main web interface
├── static/
│   ├── css/style.css         # Styling
│   └── js/app.js             # Frontend JavaScript
├── models/
│   └── pose_landmarker_lite.task  # MediaPipe pose detection model
├── Code/                     # Original Python scripts (reference)
│   ├── basic.py              # MediaPipe pose basic example
│   ├── head_only.py          # Head tracking for golf swing
│   ├── single_rsp.py         # Rock-paper-scissors gesture recognition
│   └── Development Log/      # Development notes
└── replit.md                 # This file
```

## Features
- Upload golf swing images for analysis
- Live camera capture and analysis
- Pose landmark detection and visualization
- Feedback on:
  - Shoulder alignment
  - Hip alignment
  - Knee flex angle
  - Arm position
  - Spine tilt

## Running the App
- **Development**: `python app.py` (starts on port 5000)
- **Production**: `gunicorn --bind=0.0.0.0:5000 --reuse-port app:app`

## Important Notes
- Uses MediaPipe Tasks API (not the legacy `solutions` API) - the model file must be present at `models/pose_landmarker_lite.task`
- The model is downloaded from Google's storage and is ~5.6MB
- OpenCV is used in headless mode (no GUI display)
- The original scripts in `Code/` use webcam/GUI display - they won't run directly in Replit

## Dependencies
- flask
- opencv-python-headless
- mediapipe (>=0.10)
- numpy
- gunicorn

## System Dependencies (Nix)
- xorg.libxcb
- libGL
- xorg.libX11
- xorg.libXext
- xorg.libXrender
