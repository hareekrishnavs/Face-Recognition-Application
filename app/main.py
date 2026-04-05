import time
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

import cv2
from flask import Flask, Response, jsonify, render_template, request
from flask_cors import CORS
from flask_socketio import SocketIO

from activity_log import ActivityLog
from config import load_config, save_config
from enrollment import EnrollmentController
from face_engine import FaceEngine

APP_ROOT = Path(__file__).resolve().parent

app = Flask(
    __name__,
    static_folder="frontend/static",
    template_folder="frontend",
)
app.config["SECRET_KEY"] = "facevault-secret"
CORS(app)

socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# Global singletons
config: Dict[str, Any] = load_config()
face_engine = FaceEngine(threshold=float(config.get("threshold", 0.65)))
activity_log = ActivityLog()
enrollment = EnrollmentController(face_engine, socketio)

# Camera state
_camera_lock = Lock()
_camera_active: bool = False
_camera_thread = None
_camera_capture: Optional[Any] = None
_latest_frame: Optional[Any] = None
_idle_deadline: Optional[float] = None
_idle_warning_sent: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_camera_active(active: bool, reason: Optional[str] = None) -> None:
    global _camera_active, _idle_deadline, _idle_warning_sent
    _camera_active = active
    if active:
        timeout = float(config.get("idle_timeout_seconds", 30))
        _idle_deadline = time.time() + timeout
        _idle_warning_sent = False
    else:
        _idle_deadline = None
        _idle_warning_sent = False

    socketio.emit("camera_status", {"active": bool(active), "reason": reason or ""})
    if not active and reason and reason not in ("detected", "manual"):
        socketio.emit("camera_stopped", {"reason": reason})


def _open_camera() -> bool:
    global _camera_capture
    if _camera_capture is not None:
        return True

    index = int(config.get("camera_index", 0))
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return False

    _camera_capture = cap
    return True


def _release_camera() -> None:
    global _camera_capture
    if _camera_capture is not None:
        _camera_capture.release()
        _camera_capture = None


def _camera_loop() -> None:
    """Continuous detection loop — runs until the user clicks STOP CAMERA
    or the frontend 5-second countdown fires stop_camera.

    - Emits detection_result every frame (canvas overlay stays live).
    - Logs each unique name ONCE per camera session (not repeatedly).
    - Emits known_detected toast ONCE per person per session.
    - Triggers enrollment panel once per unknown face.
    """
    global _camera_active, _latest_frame

    session_logged: set = set()    # names already logged this session
    session_toasted: set = set()   # names already toasted this session

    while _camera_active:
        with _camera_lock:
            cap = _camera_capture
        if cap is None:
            break

        grabbed, frame = cap.read()
        if not grabbed or frame is None:
            time.sleep(0.05)
            continue

        _latest_frame = frame
        now = time.time()

        # ── Run face engine ──────────────────────────────────────────
        detections = face_engine.process_frame(frame)

        # ── Build payload (strip internal blobs) ────────────────────
        sendable_faces = []
        for det in detections:
            name       = det.get("name") or "Unknown"
            is_known   = bool(det.get("is_known"))
            confidence = float(det.get("confidence", 0.0))
            bbox       = det.get("bbox", {})
            sendable_faces.append(
                {"name": name, "confidence": confidence,
                 "bbox": bbox, "is_known": is_known}
            )

            # Log — ONCE per session per unique name
            log_key = name if is_known else "Unknown"
            if log_key not in session_logged:
                session_logged.add(log_key)
                record = activity_log.log(name, confidence, is_known)
                socketio.emit("log_entry", record)

        # ── Canvas overlay ───────────────────────────────────────────
        # Only send known (above-threshold) faces to the canvas.
        # Unknown faces are still logged and trigger enrollment but are
        # not drawn — satisfying: "recognise only if confidence >= threshold".
        canvas_faces = [f for f in sendable_faces if f["is_known"]]
        socketio.emit(
            "detection_result",
            {"faces": canvas_faces, "frame_ts": datetime.utcnow().isoformat()},
        )

        if detections:
            # Known-face toast — ONCE per session per person
            known_faces = [d for d in detections if d.get("is_known")]
            if known_faces:
                kname = known_faces[0].get("name")
                if kname not in session_toasted:
                    session_toasted.add(kname)
                    socketio.emit(
                        "known_detected",
                        {
                            "name": kname,
                            "confidence": float(known_faces[0].get("confidence", 0.0)),
                        },
                    )

            # Enrollment panel — only fires when enrollment state is idle
            enrollment.on_frame_processed(frame, detections)

        time.sleep(0.03)

    # ── Cleanup ──────────────────────────────────────────────────────
    _release_camera()
    if _camera_active:
        _set_camera_active(False, reason="manual")


def _ensure_camera_thread() -> None:
    global _camera_thread
    if _camera_thread is None or not _camera_active:
        if not _open_camera():
            _set_camera_active(False, reason="unavailable")
            return
        _set_camera_active(True)
        _camera_thread = socketio.start_background_task(_camera_loop)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def index() -> Any:
    return render_template("index.html")


@app.route("/video_feed")
def video_feed() -> Response:
    """MJPEG stream endpoint.  Serves the last captured frame even after
    the camera loop stops so the UI can show a frozen detection frame."""

    def generate():
        boundary = "--frame"
        while True:
            frame = _latest_frame
            if frame is None or not _camera_active:
                time.sleep(0.05)
                if _latest_frame is not None and not _camera_active:
                    # Serve frozen last frame once more then wait
                    ok, buffer = cv2.imencode(".jpg", _latest_frame)
                    if ok:
                        jpg_bytes = buffer.tobytes()
                        yield (
                            f"{boundary}\r\n"
                            f"Content-Type: image/jpeg\r\n"
                            f"Content-Length: {len(jpg_bytes)}\r\n\r\n".encode()
                            + jpg_bytes
                            + b"\r\n"
                        )
                continue
            ok, buffer = cv2.imencode(".jpg", frame)
            if not ok:
                continue
            jpg_bytes = buffer.tobytes()
            yield (
                f"{boundary}\r\n"
                f"Content-Type: image/jpeg\r\n"
                f"Content-Length: {len(jpg_bytes)}\r\n\r\n".encode()
                + jpg_bytes
                + b"\r\n"
            )
            time.sleep(0.03)

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/camera/start", methods=["POST"])
def api_camera_start() -> Any:
    _ensure_camera_thread()
    return jsonify({"active": bool(_camera_active)})


@app.route("/api/camera/stop", methods=["POST"])
def api_camera_stop() -> Any:
    global _camera_active
    _camera_active = False
    _set_camera_active(False, reason="manual")
    return jsonify({"active": False})


@app.route("/api/camera/status", methods=["GET"])
def api_camera_status() -> Any:
    return jsonify({"active": bool(_camera_active)})


@app.route("/api/threshold", methods=["POST"])
def api_threshold() -> Any:
    data = request.get_json(silent=True) or {}
    value = float(data.get("value", config.get("threshold", 0.65)))
    face_engine.set_threshold(value)
    config["threshold"] = float(value)
    save_config(config)
    return jsonify({"threshold": face_engine.threshold})


@app.route("/api/activity_log/clear", methods=["POST"])
def api_activity_clear() -> Any:
    activity_log._clear_on_start()
    return jsonify({"status": "cleared"})


@app.route("/api/activity_log", methods=["GET"])
def api_activity_log() -> Any:
    entries = activity_log.get_recent(limit=100)
    return jsonify(entries)


@app.route("/api/activity_log/export", methods=["GET"])
def api_activity_export() -> Any:
    csv_data = activity_log.export_csv()
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=facevault_activity.csv",
        },
    )


@app.route("/api/known_users", methods=["GET"])
def api_known_users() -> Any:
    users = face_engine.get_known_users()
    return jsonify(users)


@app.route("/api/remove_user", methods=["POST"])
def api_remove_user() -> Any:
    data = request.get_json(silent=True) or {}
    name = str(data.get("name", "")).strip()
    if name:
        face_engine.remove_person(name)
        socketio.emit("user_removed", {"name": name})
    return jsonify({"status": "ok"})


@app.route("/api/enroll/start", methods=["POST"])
def api_enroll_start() -> Any:
    return jsonify({"status": "ok"})


@app.route("/api/enroll/confirm", methods=["POST"])
def api_enroll_confirm() -> Any:
    data = request.get_json(silent=True) or {}
    name = str(data.get("name", "")).strip()
    if not name:
        return jsonify({"error": "name_required"}), 400

    frames = enrollment.pending_frames
    new_user = enrollment.capture_and_enroll(name, frames)

    if new_user is not None:
        socketio.emit("enrollment_complete", new_user)
    return jsonify({"status": "ok", "user": new_user})


@app.route("/api/enroll/cancel", methods=["POST"])
def api_enroll_cancel() -> Any:
    enrollment.cancel()
    return jsonify({"status": "cancelled"})


@app.route("/api/config", methods=["GET"])
def api_config_get() -> Any:
    cfg = load_config()
    cfg["demo_mode"] = bool(face_engine.demo_mode)
    return jsonify(cfg)


@app.route("/api/config", methods=["POST"])
def api_config_set() -> Any:
    data = request.get_json(silent=True) or {}
    cfg = load_config()
    cfg.update({k: v for k, v in data.items() if k in cfg})
    save_config(cfg)

    if "threshold" in data:
        face_engine.set_threshold(float(data["threshold"]))

    return jsonify(load_config())


# ---------------------------------------------------------------------------
# Socket.IO events
# ---------------------------------------------------------------------------


@socketio.on("start_camera")
def sio_start_camera() -> None:
    _ensure_camera_thread()


@socketio.on("stop_camera")
def sio_stop_camera() -> None:
    global _camera_active
    _camera_active = False
    _set_camera_active(False, reason="manual")


@socketio.on("set_threshold")
def sio_set_threshold(value: Any) -> None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return
    face_engine.set_threshold(v)
    config["threshold"] = float(v)
    save_config(config)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5001, debug=True)
