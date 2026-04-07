import csv
import io
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / "facevault.db"


class ActivityLog:
    def __init__(self) -> None:
        self._init_db()
        self._clear_on_start()
        # Short session identifier for grouping detections
        self.session_id = str(uuid.uuid4())[:8]

    def _clear_on_start(self) -> None:
        """Wipe all previous detections so every run starts with a fresh log."""
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("DELETE FROM detections")

    def _init_db(self) -> None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    name TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    is_known INTEGER NOT NULL,
                    session_id TEXT NOT NULL
                )
                """
            )

    def log(self, name: str, confidence: float, is_known: bool) -> dict:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "INSERT INTO detections (timestamp, name, confidence, is_known, session_id) VALUES (?,?,?,?,?)",
                (ts, name, round(float(confidence), 4), int(bool(is_known)), self.session_id),
            )
        return {
            "timestamp": ts,
            "name": name,
            "confidence": float(confidence),
            "is_known": bool(is_known),
        }

    def get_recent(self, limit: int = 100) -> list[dict]:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM detections ORDER BY id DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        # Reverse so oldest is first
        return [dict(r) for r in reversed(rows)]

    def get_stats(self) -> dict:
        today = datetime.now().strftime("%Y-%m-%d")
        with sqlite3.connect(DB_PATH) as conn:
            detected_today = conn.execute(
                "SELECT COUNT(*) FROM detections WHERE timestamp LIKE ?",
                (f"{today}%",),
            ).fetchone()[0]
            sessions_today = conn.execute(
                "SELECT COUNT(DISTINCT session_id) FROM detections WHERE timestamp LIKE ?",
                (f"{today}%",),
            ).fetchone()[0]
        return {"detected_today": detected_today, "sessions_today": sessions_today}

    def export_csv(self) -> str:
        rows = self.get_recent(limit=10000)
        out = io.StringIO()
        writer = csv.DictWriter(
            out,
            fieldnames=["id", "timestamp", "name", "confidence", "is_known", "session_id"],
        )
        writer.writeheader()
        writer.writerows(rows)
        return out.getvalue()
