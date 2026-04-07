import time
import platform
from datetime import datetime, timezone
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
face_engine = FaceEngine(
    threshold=config.get("threshold"),
    recognitionMargin=config.get("recognition_margin"),
    supportShots=config.get("prototype_support_shots"),
)
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
_tracking_bbox: Optional[Dict[str, int]] = None
_stable_frames: int = 0
_stable_face_buffer: list = []
_active_camera_index: Optional[int] = None


def _reset_tracking() -> None:
    global _tracking_bbox, _stable_frames, _stable_face_buffer
    _tracking_bbox = None
    _stable_frames = 0
    _stable_face_buffer = []


def _bbox_iou(first: Dict[str, int], second: Dict[str, int]) -> float:
    x1 = max(first["x"], second["x"])
    y1 = max(first["y"], second["y"])
    x2 = min(first["x"] + first["w"], second["x"] + second["w"])
    y2 = min(first["y"] + first["h"], second["y"] + second["h"])
    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = float((x2 - x1) * (y2 - y1))
    first_area = float(first["w"] * first["h"])
    second_area = float(second["w"] * second["h"])
    union = max(1.0, first_area + second_area - intersection)
    return intersection / union


def _emit_capture_status(message: str, level: str = "info") -> None:
    socketio.emit("capture_status", {"message": message, "level": level})


def _get_alignment_feedback(
    frame_shape: tuple[int, int, int],
    bbox: Dict[str, int],
) -> tuple[bool, str]:
    frame_height, frame_width = frame_shape[:2]
    guide_scale = float(config.get("guide_box_scale", 0.42))
    tolerance = float(config.get("guide_box_tolerance", 0.18))

    guide_width = frame_width * guide_scale
    guide_height = frame_height * guide_scale
    guide_center_x = frame_width / 2.0
    guide_center_y = frame_height / 2.0

    face_center_x = bbox["x"] + (bbox["w"] / 2.0)
    face_center_y = bbox["y"] + (bbox["h"] / 2.0)

    max_offset_x = guide_width * tolerance
    max_offset_y = guide_height * tolerance

    offset_x = face_center_x - guide_center_x
    offset_y = face_center_y - guide_center_y

    if offset_x < -max_offset_x:
        return False, "Move slightly right."
    if offset_x > max_offset_x:
        return False, "Move slightly left."
    if offset_y < -max_offset_y:
        return False, "Move slightly down."
    if offset_y > max_offset_y:
        return False, "Move slightly up."

    return True, "Face aligned. Hold still."


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
    global _camera_capture, _active_camera_index
    if _camera_capture is not None:
        return True

    configured_index = int(config.get("camera_index", 1))
    prefer_builtin = bool(config.get("prefer_builtin_camera", True))

    candidate_indices = [configured_index]
    if platform.system() == "Darwin":
        preferred_order = [0, 1, 2, 3, 4] if prefer_builtin else [1, 0, 2, 3, 4]
        candidate_indices = []
        for index in preferred_order:
            if index not in candidate_indices:
                candidate_indices.append(index)
        if configured_index not in candidate_indices:
            candidate_indices.insert(0, configured_index)

    cap = None
    chosen_index = None
    for index in candidate_indices:
        backends = [cv2.CAP_AVFOUNDATION] if platform.system() == "Darwin" else [cv2.CAP_ANY]
        backends.append(cv2.CAP_ANY)

        for backend in backends:
            candidate = cv2.VideoCapture(index, backend)
            if candidate.isOpened():
                cap = candidate
                chosen_index = index
                break
            candidate.release()
        if cap is not None:
            break

    if cap is None:
        return False

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        return False

    _camera_capture = cap
    _active_camera_index = chosen_index
    return True


def _release_camera() -> None:
    global _camera_capture, _active_camera_index
    if _camera_capture is not None:
        _camera_capture.release()
        _camera_capture = None
    _active_camera_index = None


def _camera_loop() -> None:
    """Continuous detection loop — runs until the user clicks STOP CAMERA
    or the frontend 5-second countdown fires stop_camera.

    - Emits detection_result every frame (canvas overlay stays live).
    - Logs each unique name ONCE per camera session (not repeatedly).
    - Emits known_detected toast ONCE per person per session.
    - Triggers enrollment panel once per unknown face.
    """
    global _camera_active, _latest_frame, _tracking_bbox, _stable_frames, _stable_face_buffer

    session_logged: set = set()    # names already logged this session
    session_toasted: set = set()   # names already toasted this session
    _reset_tracking()
    camera_label = (
        f"Mobile camera selected (index {_active_camera_index}). Center your face."
        if _active_camera_index is not None
        else "Mobile camera ready. Center your face."
    )
    _emit_capture_status(camera_label, "info")

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

        raw_detections = face_engine.detect_faces(frame)
        selected_detection = None
        status_message = "Center your face in the frame."
        status_level = "info"
        stability_progress = 0.0

        if raw_detections:
            if len(raw_detections) > 1:
                _reset_tracking()
                selected_detection = max(
                    raw_detections,
                    key=lambda det: int(det["bbox"]["w"]) * int(det["bbox"]["h"]),
                )
                status_message = "Only one face should be in the frame."
                status_level = "warn"
            else:
                selected_detection = raw_detections[0]
                bbox = selected_detection["bbox"]
                face_bgr = selected_detection.get("face_bgr")
                blur_score = float(selected_detection.get("blur_score", 0.0))
                min_face_size = int(config.get("min_face_size", 110))
                stable_target = int(config.get("stability_frames_required", 4))
                blur_threshold = float(config.get("blur_threshold", 75.0))

                is_large_enough = min(bbox["w"], bbox["h"]) >= min_face_size
                is_aligned, alignment_message = _get_alignment_feedback(frame.shape, bbox)
                is_stable = _tracking_bbox is not None and _bbox_iou(_tracking_bbox, bbox) >= 0.88
                _tracking_bbox = bbox

                if not is_large_enough:
                    _stable_frames = 0
                    _stable_face_buffer = []
                    status_message = "Move a little closer to the camera."
                    status_level = "warn"
                elif not is_aligned:
                    _stable_frames = 0
                    _stable_face_buffer = []
                    status_message = alignment_message
                    status_level = "warn"
                elif blur_score < blur_threshold:
                    _stable_frames = 0
                    _stable_face_buffer = []
                    status_message = "Hold still. The face looks shaky."
                    status_level = "warn"
                else:
                    _stable_frames = _stable_frames + 1 if is_stable else 1
                    if is_stable and face_bgr is not None:
                        _stable_face_buffer.append(face_bgr.copy())
                    elif face_bgr is not None:
                        _stable_face_buffer = [face_bgr.copy()]

                    consensus_frames = int(config.get("prediction_consensus_frames", 5))
                    if len(_stable_face_buffer) > consensus_frames:
                        _stable_face_buffer = _stable_face_buffer[-consensus_frames:]

                    stability_progress = min(1.0, _stable_frames / max(1, stable_target))
                    if _stable_frames < stable_target or len(_stable_face_buffer) < consensus_frames:
                        status_message = f"{alignment_message} {_stable_frames}/{stable_target}"
                        status_level = "info"
                    else:
                        status_message = "Face captured. Predicting..."
                        status_level = "success"
                        stability_progress = 1.0
                        prediction = face_engine.predict_faces(_stable_face_buffer)
                        selected_detection = {**selected_detection, **prediction}
        else:
            _reset_tracking()
            status_message = "Center your face in the frame."
            status_level = "info"

        _emit_capture_status(status_message, status_level)

        sendable_faces = []
        for det in ([selected_detection] if selected_detection else []):
            name       = det.get("name") or "Unknown"
            is_known   = bool(det.get("is_known"))
            confidence = float(det.get("confidence", 0.0))
            bbox       = det.get("bbox", {})
            sendable_faces.append(
                {"name": name, "confidence": confidence,
                 "bbox": bbox, "is_known": is_known}
            )

            if "is_known" in det:
                log_key = name if is_known else "Unknown"
                if log_key not in session_logged:
                    session_logged.add(log_key)
                    record = activity_log.log(name, confidence, is_known)
                    socketio.emit("log_entry", record)

        socketio.emit(
            "detection_result",
            {
                "faces": sendable_faces,
                "frame_ts": datetime.now(timezone.utc).isoformat(),
                "status_message": status_message,
                "status_level": status_level,
                "stability_progress": stability_progress,
            },
        )

        if selected_detection and "is_known" in selected_detection:
            if selected_detection.get("is_known"):
                kname = selected_detection.get("name")
                if kname not in session_toasted:
                    session_toasted.add(kname)
                    socketio.emit(
                        "known_detected",
                        {
                            "name": kname,
                            "confidence": float(selected_detection.get("confidence", 0.0)),
                        },
                    )
                _set_camera_active(False, reason="detected")
                break

            enrollment.on_frame_processed(frame, [selected_detection])

        time.sleep(0.03)

    # ── Cleanup ──────────────────────────────────────────────────────
    _release_camera()
    _reset_tracking()
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


@app.route("/api/model/switch", methods=["POST"])
def api_model_switch() -> Any:
    data = request.get_json(silent=True) or {}
    model_type = str(data.get("model_type", "")).strip().lower()
    result = face_engine.switch_model(model_type)
    if "error" not in result:
        socketio.emit("model_switched", result)
    return jsonify(result)


@app.route("/api/model/info", methods=["GET"])
def api_model_info() -> Any:
    return jsonify(face_engine.get_model_info())


@app.route("/api/view/clear", methods=["POST"])
def api_view_clear() -> Any:
    global _latest_frame
    _latest_frame = None
    enrollment.cancel()
    _reset_tracking()
    return jsonify({"status": "cleared"})


@app.route("/api/config", methods=["GET"])
def api_config_get() -> Any:
    cfg = load_config()
    cfg["threshold"] = float(face_engine.threshold)
    cfg["demo_mode"] = bool(face_engine._check_demo_mode())
    cfg["model_type"] = face_engine.model_type
    model_info = face_engine.get_model_info()
    cfg["arcface_loaded"] = model_info["arcface_loaded"]
    cfg["insightface_loaded"] = model_info["insightface_loaded"]
    cfg["insightface_available"] = model_info["insightface_available"]
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


@socketio.on("switch_model")
def sio_switch_model(data: Any) -> None:
    model_type = str(data.get("model_type", "")).strip().lower() if isinstance(data, dict) else str(data)
    result = face_engine.switch_model(model_type)
    if "error" not in result:
        socketio.emit("model_switched", result)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5001, debug=True, allow_unsafe_werkzeug=True)
