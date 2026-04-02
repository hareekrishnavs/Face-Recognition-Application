# Face-Recognition-Application

## Web App

**FaceVault** — real-time AI face recognition with a dark, terminal-inspired UI.

### Quick start

```bash
cd app
pip install -r requirements_web.txt
python main.py
# Open http://localhost:5000
```

### Requirements

- Python 3.10+
- A webcam (index 0, 1, or 2)
- Trained model at `models/arcface_model.pth` (optional — app runs in Demo Mode without it)
- Pre-processed embeddings in `dataset/processed/train/` (one subfolder per person)

### Features

- Live MJPEG camera stream with canvas bounding-box overlay
- Real-time face detection and recognition via ArcFace embeddings
- Unknown-face enrollment flow with known-user verification
- Activity log with confidence bars, auto-scrolling, CSV export
- Known-users gallery with remove support
- Threshold slider (0.40–0.95) persisted to `app/config.json`
- Demo Mode when no model is present (full UI, simulated events)
- SQLite activity log at `app/facevault.db`