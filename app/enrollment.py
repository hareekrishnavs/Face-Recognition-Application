"""
enrollment.py — Unknown-face enrollment state machine for FaceVault.

States:
  IDLE       — no enrollment in progress
  DETECTING  — counting consecutive unknown-face frames
  PENDING    — unknown_detected emitted; waiting for user to click "Verify & Add"
  VERIFYING  — watching the camera for a known-face approver (15-s window)
  VERIFIED   — a known face confirmed the enrollment; collecting frames / awaiting name
"""
import time
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

TRIGGER_FRAMES    = 3      # consecutive unknown frames before emitting unknown_detected
COLLECT_MAX       = 10     # face-crop frames to collect for the new person
VERIFY_TIMEOUT    = 15.0   # seconds for a known face to appear
UNKNOWN_GAP_RESET = 2.0    # reset counter if no unknown seen within this many seconds


class State(Enum):
    IDLE      = auto()
    DETECTING = auto()
    PENDING   = auto()
    VERIFYING = auto()
    VERIFIED  = auto()


class EnrollmentManager:
    def __init__(self):
        self._state:          State          = State.IDLE
        self._count:          int            = 0
        self._last_seen:      float          = 0.0
        self._face_crop_b64:  Optional[str]  = None
        self._verifier:       Optional[str]  = None
        self._frames:         list[np.ndarray] = []
        self._verify_start:   float          = 0.0

    # ── Queries ───────────────────────────────────────────────────────────────

    @property
    def state(self) -> State:
        return self._state

    def is_pending(self) -> bool:
        return self._state == State.PENDING

    def is_verifying(self) -> bool:
        return self._state == State.VERIFYING

    def is_verified(self) -> bool:
        return self._state == State.VERIFIED

    def is_active(self) -> bool:
        return self._state != State.IDLE

    def get_crop_b64(self) -> Optional[str]:
        return self._face_crop_b64

    def get_verifier(self) -> Optional[str]:
        return self._verifier

    def get_frames(self) -> list[np.ndarray]:
        return list(self._frames)

    # ── State transitions ─────────────────────────────────────────────────────

    def on_unknown_frame(self, face_crop_b64: str) -> bool:
        """
        Call for every frame that contains an unknown face.
        Returns True exactly once when the trigger threshold is reached
        (i.e. the caller should emit unknown_detected).
        Ignored once an enrollment is already PENDING or further along.
        """
        if self._state not in (State.IDLE, State.DETECTING):
            return False

        now = time.monotonic()
        if now - self._last_seen > UNKNOWN_GAP_RESET:
            self._count = 0
            self._state = State.DETECTING

        self._last_seen = now
        self._count += 1

        if self._count >= TRIGGER_FRAMES:
            self._face_crop_b64 = face_crop_b64
            self._state = State.PENDING
            return True

        return False

    def start_verification(self):
        """Called when the user clicks 'Verify & Add'."""
        if self._state == State.PENDING:
            self._state        = State.VERIFYING
            self._verify_start = time.monotonic()
            self._verifier     = None

    def check_verify_timeout(self) -> bool:
        """Returns True if the 15-second verification window has expired."""
        if self._state == State.VERIFYING:
            if time.monotonic() - self._verify_start > VERIFY_TIMEOUT:
                self._state = State.PENDING   # fall back, let user retry
                return True
        return False

    def set_verified(self, verifier_name: str):
        """Called when a known face appears during the verification window."""
        if self._state == State.VERIFYING:
            self._verifier = verifier_name
            self._state    = State.VERIFIED

    def collect_frame(self, crop: np.ndarray):
        """
        Store a face-crop frame.  Only accepted while VERIFYING or VERIFIED,
        so we never accumulate stale frames during an idle/pending wait.
        """
        if self._state in (State.VERIFYING, State.VERIFIED):
            if len(self._frames) < COLLECT_MAX:
                self._frames.append(crop.copy())

    # ── Finalise / reset ──────────────────────────────────────────────────────

    def finalize(self, name: str, raw_dir: Path) -> list[Path]:
        """
        Save collected face crops to dataset/raw/<name>/.
        Returns the list of saved paths.
        Resets the manager to IDLE.
        """
        person_dir = raw_dir / name
        person_dir.mkdir(parents=True, exist_ok=True)

        saved: list[Path] = []
        for i, crop in enumerate(self._frames):
            ts   = int(time.time() * 1000)
            path = person_dir / f"web_{ts}_{i}.jpg"
            cv2.imwrite(str(path), crop)
            saved.append(path)

        self.reset()
        return saved

    def reset(self):
        self._state         = State.IDLE
        self._count         = 0
        self._last_seen     = 0.0
        self._face_crop_b64 = None
        self._verifier      = None
        self._frames        = []
        self._verify_start  = 0.0
