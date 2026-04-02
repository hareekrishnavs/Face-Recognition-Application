"""
activity_log.py — SQLite persistence for FaceVault.

DB file: app/facevault.db

Schema:
  detections(id, timestamp, name, confidence, is_known, session_id)
"""
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

DB_PATH    = Path(__file__).parent / "facevault.db"
SESSION_ID = str(uuid.uuid4())[:8]
_DEBOUNCE  = 3.0   # seconds — suppress repeated log entries for same person


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


def _init_db():
    with _conn() as c:
        c.executescript("""
            CREATE TABLE IF NOT EXISTS detections (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp  TEXT    NOT NULL,
                name       TEXT    NOT NULL,
                confidence REAL    NOT NULL,
                is_known   INTEGER NOT NULL DEFAULT 0,
                session_id TEXT    NOT NULL DEFAULT ''
            );
        """)


_init_db()


class ActivityLog:
    def __init__(self):
        self._debounce: dict[str, float] = {}

    # ── Debounce ──────────────────────────────────────────────────────────────

    def should_log(self, name: str) -> bool:
        now = time.monotonic()
        if now - self._debounce.get(name, 0.0) >= _DEBOUNCE:
            self._debounce[name] = now
            return True
        return False

    # ── Write ─────────────────────────────────────────────────────────────────

    def log(self, name: str, confidence: float, is_known: bool):
        ts = datetime.now(timezone.utc).isoformat()
        with _conn() as c:
            c.execute(
                "INSERT INTO detections (timestamp, name, confidence, is_known, session_id) "
                "VALUES (?, ?, ?, ?, ?)",
                (ts, name, round(confidence, 4), int(is_known), SESSION_ID),
            )

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_recent(self, limit: int = 100) -> list[dict]:
        with _conn() as c:
            rows = c.execute(
                "SELECT id, timestamp, name, confidence, is_known "
                "FROM detections ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_stats(self) -> dict:
        today = datetime.now(timezone.utc).date().isoformat()
        with _conn() as c:
            detected_today = c.execute(
                "SELECT COUNT(*) FROM detections WHERE timestamp LIKE ?",
                (f"{today}%",),
            ).fetchone()[0]
            sessions_today = c.execute(
                "SELECT COUNT(DISTINCT session_id) FROM detections WHERE timestamp LIKE ?",
                (f"{today}%",),
            ).fetchone()[0]
        return {
            "detected_today": detected_today,
            "sessions_today": sessions_today,
        }

    def export_csv(self) -> str:
        with _conn() as c:
            rows = c.execute(
                "SELECT id, timestamp, name, confidence, is_known FROM detections ORDER BY id DESC"
            ).fetchall()
        lines = ["id,timestamp,name,confidence,is_known"]
        for r in rows:
            lines.append(
                f"{r['id']},{r['timestamp']},{r['name']},"
                f"{r['confidence']:.4f},{r['is_known']}"
            )
        return "\n".join(lines)
