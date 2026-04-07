# FaceVault - AI Face Recognition Application

A real-time face recognition web application built with Flask, SocketIO, and dual AI model support. Switch between **CNN** (ArcFace classifier) and **InsightFace** (embedding similarity) at runtime without restarting the app.

## Features

- **Dual Model Support** - Toggle between CNN and InsightFace from the UI at runtime
- **Real-Time Recognition** - Live camera feed with bounding boxes, confidence scores, and identity labels
- **Face Enrollment** - Add new people directly from the camera; InsightFace index rebuilds automatically in the background
- **Background Retraining** - When a new person is enrolled, the InsightFace model retrains itself without freezing the UI
- **Training Status Banner** - Shows "Training new face, please wait..." while retraining and "Training complete" when done
- **Activity Logging** - SQLite-backed detection log with CSV export
- **Stability Tracking** - Alignment guidance, blur detection, and multi-frame consensus before prediction
- **Gallery View** - See all enrolled users at a glance
- **No Demo Mode** - Adding new people never breaks the live recognition pipeline

## How the Models Work

| | CNN (ArcFace) | InsightFace |
|---|---|---|
| **Detection** | OpenCV Haar Cascade | `buffalo_l` RetinaFace |
| **Recognition** | Custom CNN (`FaceClassifierCNN`) with softmax + margin-based confidence | 512-d embeddings with cosine similarity against prototype vectors |
| **Model File** | `models/bestModel.pth` + `models/labelMap.json` | `models/face_index.npz` (auto-updated on enrollment) |
| **Decision** | Known if confidence >= threshold AND margin >= recognition_margin | Known if cosine similarity >= threshold |
| **New person support** | Requires retraining CNN (offline) | Automatic — index rebuilds in background after enrollment |

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
│   ├── bestModel.pth            # Trained CNN weights (fixed, 5 classes)
│   ├── face_index.npz           # InsightFace embedding index (updated on enrollment)
│   ├── labelMap.json            # CNN class index → name mapping (CNN ground truth, not modified by InsightFace)
│   └── model_metadata.json      # Training metadata & thresholds
├── training/
│   ├── model.py                 # FaceClassifierCNN architecture
│   ├── train.py                 # Training script
│   └── dataset.py               # Dataset loader
├── preprocessing/
│   └── dataset.py               # Image preprocessing pipeline
├── utils/
│   ├── config.py                # Global paths and hyperparameters
│   └── insightface_backend.py   # InsightFace wrapper (detection + embeddings)
└── dataset/                     # Training images (not in repo — gitignored)
    ├── raw/
    ├── processed/
    └── captured/                # Face crops added via web UI enrollment
```

## Setup

### Prerequisites

- Python 3.10+
- pip

### Install Dependencies

```bash
pip install -r app/requirements_web.txt
```

This installs both CNN (PyTorch) and InsightFace dependencies. If you only want the CNN model, you can skip `insightface` and `onnxruntime` — the app will run with the InsightFace toggle disabled.

### Run

```bash
python3 app/main.py
```

Open **http://localhost:5001** in your browser.

> **Note:** Port 5000 is used by AirPlay Receiver on macOS. The app defaults to port 5001.

## Usage

1. **Start Camera** - Click "ACTIVATE CAMERA" to begin the live feed
2. **Recognition** - Face your camera. The app detects, aligns, and identifies faces in real time
3. **Switch Models** - Use the CNN / InsightFace toggle in the top nav bar to switch engines at runtime
4. **Enroll New Person** - When an unknown face is detected, an enrollment panel appears. Enter a name to add them. InsightFace retrains automatically in the background — a status banner appears while training is in progress
5. **Adjust Threshold** - Use the threshold slider (0.40 – 0.95) to tune recognition sensitivity
6. **Export Logs** - Click "Export" in the activity panel to download detection history as CSV
7. **Clear Result** - Click "CLEAR RESULT" to reset the view and scan again

## Enrollment Flow (InsightFace)

```
Unknown face detected
  → Enrollment panel appears
  → User enters name → POST /api/enroll/confirm
  → Face crops saved + embeddings extracted immediately (person recognizable right away)
  → Background thread starts
      → Banner: "Training new face, please wait..."
      → All captured images re-processed, prototypes rebuilt
      → Banner: "Training complete" (auto-dismisses after 4s)
```

The CNN model is **not affected** by enrollment. Its `labelMap.json` and `bestModel.pth` remain unchanged.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Main UI |
| GET | `/video_feed` | MJPEG camera stream |
| POST | `/api/camera/start` | Start camera |
| POST | `/api/camera/stop` | Stop camera |
| GET | `/api/camera/status` | Camera state |
| POST | `/api/model/switch` | Switch model (`{"model_type": "cnn"}` or `"insightface"`) |
| GET | `/api/model/info` | Active model and availability info |
| POST | `/api/threshold` | Update recognition threshold |
| GET | `/api/config` | Get full config |
| POST | `/api/config` | Update config |
| GET | `/api/known_users` | List enrolled users |
| POST | `/api/remove_user` | Remove a user (`{"name": "..."}`) |
| POST | `/api/enroll/confirm` | Confirm enrollment (`{"name": "..."}`) — triggers background InsightFace retrain |
| POST | `/api/enroll/cancel` | Cancel enrollment |
| GET | `/api/activity_log` | Recent detections |
| GET | `/api/activity_log/export` | Export as CSV |
| POST | `/api/activity_log/clear` | Clear log |

## Socket.IO Events

| Event | Direction | Payload |
|---|---|---|
| `detection_result` | Server → Client | `{faces, status_message, status_level, stability_progress}` |
| `known_detected` | Server → Client | `{name, confidence}` |
| `unknown_detected` | Server → Client | `{face_crop_b64, timestamp}` |
| `enrollment_complete` | Server → Client | `{name, sample_image_b64, detection_count}` |
| `training_status` | Server → Client | `{status: "in_progress"\|"done"\|"error", message}` |
| `camera_status` | Server → Client | `{active, reason}` |
| `model_switched` | Server → Client | `{model_type}` |

## Database

The app uses SQLite (`app/facevault.db`). It auto-creates on first run — no setup required.

## Branches

| Branch | Description |
|---|---|
| `main` | Base project |
| `facevault` | CNN model with dark-themed UI |
| `insightface` | InsightFace embedding model |
| `combined-models` | Both models with runtime toggle, auto-enrollment retraining |
