from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional


class EnrollmentController:
    """Controls the unknown-face → enrollment flow.

    Simplified flow (no verification step):
      idle → pending (unknown detected) → naming → idle

    The verification step has been removed; any detected unknown face can be
    enrolled directly by entering a name.
    """

    def __init__(self, face_engine: Any, socketio: Any) -> None:
        self.engine = face_engine
        self.socketio = socketio
        self.pending_frames: List[Any] = []  # buffered face crops (BGR)
        self.state: str = "idle"  # idle | pending | naming

    def on_frame_processed(self, frame, detections: List[Dict[str, Any]]) -> None:
        """Called once per processed frame that contains detections.

        Triggers ``unknown_detected`` immediately on the first unknown face
        while the controller is idle.
        """

        unknowns = [d for d in detections if not d.get("is_known")]

        if unknowns and self.state == "idle":
            # Buffer the unknown face crop for later enrollment
            face_bgr = unknowns[0].get("face_bgr")
            if face_bgr is not None:
                self.pending_frames.append(face_bgr)
                if len(self.pending_frames) > 10:
                    self.pending_frames = self.pending_frames[-10:]

            self.state = "pending"
            self.socketio.emit(
                "unknown_detected",
                {
                    "face_crop_b64": unknowns[0].get("face_crop_b64"),
                    "timestamp": datetime.now().isoformat(),
                },
            )

    def capture_and_enroll(
        self, name: str, frames: Optional[List[Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Persist face images for a new person and update the embedding DB."""

        frames_to_use = frames if frames is not None else self.pending_frames
        self.engine.add_person(name, frames_to_use or [])

        users = self.engine.get_known_users()
        new_user = next((u for u in users if u.get("name") == name), None)

        self.state = "idle"
        self.pending_frames = []

        return new_user

    def cancel(self) -> None:
        """Abort any in-progress enrollment sequence."""
        self.state = "idle"
        self.pending_frames = []
