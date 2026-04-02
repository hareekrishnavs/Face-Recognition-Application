"""
config.py — Application configuration with JSON persistence.
Threshold persists to app/config.json across restarts.
"""
import json
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "config.json"

_DEFAULTS: dict = {
    "threshold": 0.65,
}


class _Config:
    def __init__(self):
        self._data = dict(_DEFAULTS)
        self._load()

    def _load(self):
        if CONFIG_PATH.exists():
            try:
                self._data.update(json.loads(CONFIG_PATH.read_text()))
            except Exception:
                pass

    def _save(self):
        try:
            CONFIG_PATH.write_text(json.dumps(self._data, indent=2))
        except Exception:
            pass

    @property
    def threshold(self) -> float:
        return float(self._data.get("threshold", _DEFAULTS["threshold"]))

    @threshold.setter
    def threshold(self, value: float):
        self._data["threshold"] = round(max(0.40, min(0.95, float(value))), 2)
        self._save()

    def as_dict(self) -> dict:
        return dict(self._data)


config = _Config()
