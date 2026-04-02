"""
FaceVault — Flask + SocketIO face recognition web application.

Run from the repo root:
    cd app
    pip install -r requirements_web.txt
    python main.py
    # Open http://localhost:5000

Paths (all resolved from repo root, one level above this file):
    dataset/processed/train/   — pre-trained embeddings
    dataset/raw/               — new enrollment images saved here
    models/arcface_model.pth   — trained weights

Demo mode: if the model file is absent, a placeholder video banner is shown
and all UI controls remain functional for development.
"""
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit

from activity_log import ActivityLog
from config import config
from enrollment import EnrollmentManager
from face_engine import engine as face_engine

# ── Paths ─────────────────────────────────────────────────────────────────────
APP_DIR   = Path(__file__).parent
REPO_ROOT = APP_DIR.parent
RAW_DIR   = REPO_ROOT / "dataset" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ── Flask / SocketIO setup ────────────────────────────────────────────────────
app = Flask(
    __name__,
    static_folder="frontend/static",
    template_folder="frontend",
)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")
CORS(app)

activity_log   = ActivityLog()
enrollment_mgr = EnrollmentManager()

# Sync initial threshold from persisted config
face_engine.set_threshold(config.threshold)

# ── Shared state ──────────────────────────────────────────────────────────────
_state_lock  = threading.Lock()
_latest_jpeg = None  # type: Optional[bytes]
_demo_mode   = not face_engine.model_loaded


# ── Demo frame generator ──────────────────────────────────────────────────────

def _make_demo_frame() -> bytes:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (14, 18, 24)
    # grid lines for visual interest
    for i in range(0, 640, 40):
        cv2.line(frame, (i, 0), (i, 480), (25, 32, 42), 1)
    for i in range(0, 480, 40):
        cv2.line(frame, (0, i), (640, i), (25, 32, 42), 1)
    cv2.putText(frame, "FACEVAULT", (195, 215),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 217, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "DEMO MODE  /  no model loaded", (170, 255),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (136, 153, 170), 1, cv2.LINE_AA)
    cv2.putText(frame, datetime.now().strftime("%H:%M:%S"), (285, 290),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (74, 85, 104), 1, cv2.LINE_AA)
    _, buf = cv2.imencode(".jpg", frame)
    return buf.tobytes()


# ── Camera / detection worker ─────────────────────────────────────────────────

def _camera_worker():
    global _latest_jpeg, _demo_mode

    cap = None
    if not _demo_mode:
        for idx in range(3):
            c = cv2.VideoCapture(idx)
            if c.isOpened():
                cap = c
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                print(f"[Camera] Opened index {idx}")
                break
        if cap is None:
            print("[Camera] No webcam found — switching to demo mode.")
            _demo_mode = True

    while True:
        # ── Demo mode ─────────────────────────────────────────────────────────
        if _demo_mode:
            with _state_lock:
                _latest_jpeg = _make_demo_frame()
            # Emit simulated events so the UI can be previewed
            socketio.emit("detection", {"faces": [], "frame_w": 640, "frame_h": 480})
            time.sleep(0.5)
            continue

        # ── Live camera ───────────────────────────────────────────────────────
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        frame_h, frame_w = frame.shape[:2]
        detections = face_engine.process_frame(frame)

        # Store MJPEG frame (no server-side annotation — canvas handles it)
        _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        with _state_lock:
            _latest_jpeg = jpg.tobytes()

        # Emit detection overlay data
        faces_payload = [
            {
                "name":       d["name"],
                "confidence": d["confidence"],
                "bbox":       d["bbox"],
                "is_known":   d["is_known"],
            }
            for d in detections
        ]
        socketio.emit("detection", {
            "faces":   faces_payload,
            "frame_w": frame_w,
            "frame_h": frame_h,
        })

        # Log detections with debounce
        for det in detections:
            if activity_log.should_log(det["name"]):
                activity_log.log(det["name"], det["confidence"], det["is_known"])
                socketio.emit("log_entry", {
                    "timestamp":  datetime.now(timezone.utc).isoformat(),
                    "name":       det["name"],
                    "confidence": det["confidence"],
                    "is_known":   det["is_known"],
                })
                socketio.emit("stats_update", activity_log.get_stats())

        # ── Enrollment trigger ────────────────────────────────────────────────
        for det in detections:
            if not det["is_known"]:
                triggered = enrollment_mgr.on_unknown_frame(det["face_crop_b64"])
                if triggered:
                    socketio.emit("unknown_detected", {
                        "face_crop_b64": det["face_crop_b64"],
                        "timestamp":    datetime.now(timezone.utc).isoformat(),
                    })
                break  # one unknown trigger per frame is enough

        # ── Verification window ───────────────────────────────────────────────
        if enrollment_mgr.is_verifying():
            if enrollment_mgr.check_verify_timeout():
                socketio.emit("verification_timeout", {})
            else:
                for det in detections:
                    if det["is_known"]:
                        enrollment_mgr.set_verified(det["name"])
                        socketio.emit("verification_approved", {
                            "verifier":      det["name"],
                            "face_crop_b64": enrollment_mgr.get_crop_b64(),
                        })
                        break

        # ── Collect enrollment frames ─────────────────────────────────────────
        if enrollment_mgr.is_verifying() or enrollment_mgr.is_verified():
            for det in detections:
                if not det["is_known"]:
                    x, y, w, h = det["bbox"]
                    crop = frame[y : y + h, x : x + w]
                    enrollment_mgr.collect_frame(crop)
                    break

        time.sleep(0.033)   # ~30 fps cap

    if cap:
        cap.release()


threading.Thread(target=_camera_worker, daemon=True).start()


# ── HTTP Routes ───────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    """MJPEG stream — works directly in an <img> tag."""
    def _generate():
        while True:
            with _state_lock:
                frame = _latest_jpeg
            if frame:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                )
            time.sleep(0.033)

    return Response(
        _generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/threshold", methods=["POST"])
def api_threshold():
    data  = request.get_json(silent=True) or {}
    value = float(data.get("value", data.get("threshold", config.threshold)))
    config.threshold = value
    face_engine.set_threshold(config.threshold)
    return jsonify({"threshold": config.threshold})


@app.route("/api/activity_log")
def api_activity_log():
    return jsonify(activity_log.get_recent(100))


@app.route("/api/known_users")
def api_known_users():
    return jsonify(face_engine.get_known_users())


@app.route("/api/stats")
def api_stats():
    stats = activity_log.get_stats()
    stats.update({
        "known_count":  len(face_engine.get_known_names()),
        "model_loaded": face_engine.model_loaded,
        "threshold":    config.threshold,
        "demo_mode":    _demo_mode,
    })
    return jsonify(stats)


@app.route("/api/enroll/approve", methods=["POST"])
def api_enroll_approve():
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"error": "Name is required"}), 400

    frames = enrollment_mgr.get_frames()

    # Save crops to disk (so the team can retrain later)
    saved = enrollment_mgr.finalize(name, RAW_DIR)

    # Hot-enroll: use saved paths if available, otherwise raw numpy crops
    face_engine.add_person(name, saved if saved else frames)

    socketio.emit("enrollment_complete", {
        "name":        name,
        "image_count": len(saved or frames),
    })
    return jsonify({"enrolled": name, "images": len(saved or frames)})


@app.route("/api/enroll/reject", methods=["POST"])
def api_enroll_reject():
    enrollment_mgr.reset()
    return jsonify({"rejected": True})


@app.route("/api/remove_user", methods=["POST"])
def api_remove_user():
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"error": "Name is required"}), 400
    face_engine.remove_person(name)
    return jsonify({"removed": name})


@app.route("/api/export_csv")
def api_export_csv():
    return Response(
        activity_log.export_csv(),
        mimetype="text/csv",
        headers={
            "Content-Disposition": 'attachment; filename="facevault_log.csv"'
        },
    )


# ── SocketIO Events ───────────────────────────────────────────────────────────

@socketio.on("connect")
def on_connect():
    stats = activity_log.get_stats()
    stats.update({
        "known_count":  len(face_engine.get_known_names()),
        "model_loaded": face_engine.model_loaded,
        "threshold":    config.threshold,
        "demo_mode":    _demo_mode,
    })
    emit("init", stats)


@socketio.on("set_threshold")
def on_set_threshold(value):
    v = float(value)
    config.threshold = v
    face_engine.set_threshold(config.threshold)
    emit("threshold_updated", {"threshold": config.threshold})


@socketio.on("start_verification")
def on_start_verification():
    enrollment_mgr.start_verification()
    emit("verification_started", {})


@socketio.on("enrollment_response")
def on_enrollment_response(data):
    if not data.get("approved", False):
        enrollment_mgr.reset()
        emit("enrollment_cancelled", {})


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("[FaceVault] Starting on http://localhost:5000")
    if _demo_mode:
        print("[FaceVault] WARNING: model not found — running in Demo Mode.")
    socketio.run(app, host="0.0.0.0", port=5001, debug=False, allow_unsafe_werkzeug=True)
