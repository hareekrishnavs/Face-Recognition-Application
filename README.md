# FaceVault - AI Face Recognition Application

A real-time face recognition web application built with Flask, SocketIO, and dual AI model support. Switch between **ArcFace** (CNN classifier) and **InsightFace** (embedding similarity) at runtime without restarting the app.

## Features

- **Dual Model Support** - Toggle between ArcFace and InsightFace from the UI
- **Real-Time Recognition** - Live camera feed with bounding boxes, confidence scores, and identity labels
- **Face Enrollment** - Add new people directly from the camera with a few clear captures
- **Activity Logging** - SQLite-backed detection log with CSV export
- **Stability Tracking** - Alignment guidance, blur detection, and multi-frame consensus before prediction
- **Gallery View** - See all enrolled users at a glance

## How the Models Work

| | ArcFace | InsightFace |
|---|---|---|
| **Detection** | OpenCV Haar Cascade | InsightFace `buffalo_l` (RetinaFace) |
| **Recognition** | Custom CNN (`FaceClassifierCNN`) with softmax + margin-based confidence | 512-d embeddings with cosine similarity against prototype vectors |
| **Model File** | `models/bestModel.pth` | `models/face_index.npz` + `buffalo_l` (auto-downloaded) |
| **Decision** | Known if confidence >= threshold AND margin >= recognition_margin | Known if cosine similarity >= threshold |

## Project Structure

```
Face-Recognition-Application/
├── app/
│   ├── main.py                  # Flask + SocketIO server, camera loop, API routes
│   ├── face_engine.py           # Dual-model recognition engine
│   ├── enrollment.py            # Unknown face enrollment workflow
│   ├── activity_log.py          # SQLite activity logging
│   ├── config.py                # App configuration management
│   ├── config.json              # Runtime config (auto-generated)
│   ├── requirements_web.txt     # Python dependencies
│   └── frontend/
│       ├── index.html           # Main UI
│       └── static/
│           ├── css/app.css      # Dark-themed dashboard styles
│           └── js/
│               ├── app.js       # Camera, detection overlay, model toggle
│               ├── activity.js  # Activity log rendering
│               ├── enrollment.js# Enrollment panel logic
│               ├── gallery.js   # Enrolled users gallery
│               └── neural.js    # Animated background
├── models/
│   ├── bestModel.pth            # Trained ArcFace CNN weights
│   ├── face_index.npz           # InsightFace embedding index
│   ├── labelMap.json            # Class index to name mapping
│   └── model_metadata.json      # Training metadata & thresholds
├── training/
│   ├── model.py                 # FaceClassifierCNN architecture
│   ├── train.py                 # Training script
│   └── dataset.py               # Dataset loader
├── utils/
│   ├── config.py                # Global paths and hyperparameters
│   └── insightface_backend.py   # InsightFace wrapper (detection + embeddings)
└── dataset/                     # Training images (not in repo)
    ├── raw/
    ├── processed/
    └── captured/                # Photos added via web UI enrollment
```

## Setup

### Prerequisites

- Python 3.10+
- pip

### Install Dependencies

```bash
pip install -r app/requirements_web.txt
```

This installs both ArcFace (PyTorch) and InsightFace dependencies. If you only want ArcFace, you can skip `insightface` and `onnxruntime` -- the app will run with the InsightFace toggle disabled.

### Run

```bash
cd app
python3 main.py
```

Open **http://localhost:5001** in your browser.

> **Note:** Port 5000 is used by AirPlay Receiver on macOS. The app defaults to port 5001.

## Usage

1. **Start Camera** - Click "ACTIVATE CAMERA" to begin the live feed
2. **Recognition** - Face your camera. The app detects, aligns, and identifies faces
3. **Switch Models** - Use the ArcFace/InsightFace toggle in the top nav bar to switch engines at runtime
4. **Enroll New Person** - When an unknown face is detected, an enrollment panel appears. Enter a name to add them
5. **Adjust Threshold** - Use the threshold slider (0.40 - 0.95) to tune sensitivity
6. **Export Logs** - Click "Export" in the activity panel to download detection history as CSV
7. **Clear Result** - Click "CLEAR RESULT" to reset the view and scan again

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Main UI |
| GET | `/video_feed` | MJPEG camera stream |
| POST | `/api/camera/start` | Start camera |
| POST | `/api/camera/stop` | Stop camera |
| GET | `/api/camera/status` | Camera state |
| POST | `/api/model/switch` | Switch model (`{"model_type": "arcface"}` or `"insightface"`) |
| GET | `/api/model/info` | Active model and availability info |
| POST | `/api/threshold` | Update recognition threshold |
| GET | `/api/config` | Get full config |
| POST | `/api/config` | Update config |
| GET | `/api/known_users` | List enrolled users |
| POST | `/api/remove_user` | Remove a user (`{"name": "..."}`) |
| POST | `/api/enroll/confirm` | Confirm enrollment (`{"name": "..."}`) |
| POST | `/api/enroll/cancel` | Cancel enrollment |
| GET | `/api/activity_log` | Recent detections |
| GET | `/api/activity_log/export` | Export as CSV |
| POST | `/api/activity_log/clear` | Clear log |

## Database

The app uses SQLite (`app/facevault.db`). It auto-creates on first run and clears on every restart -- no setup required.

## Branches

| Branch | Description |
|---|---|
| `main` | Base project |
| `facevault` | ArcFace CNN model with dark-themed UI |
| `insightface` | InsightFace embedding model |
| `combined-models` | Both models merged with runtime toggle |
