import json
from pathlib import Path
from typing import Any, Dict

APP_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = APP_ROOT / "config.json"
MODEL_METADATA_PATH = APP_ROOT.parent / "models" / "model_metadata.json"

DEFAULT_CONFIG: Dict[str, Any] = {
    "idle_timeout_seconds": 30,
    "idle_warning_seconds": 10,
    "camera_index": 1,
    "prefer_builtin_camera": True,
    "stability_frames_required": 2,
    "prediction_consensus_frames": 3,
    "blur_threshold": 35.0,
    "min_face_size": 90,
    "guide_box_scale": 0.42,
    "guide_box_tolerance": 0.26,
}


def _load_model_runtime_defaults() -> Dict[str, Any]:
    defaults: Dict[str, Any] = {
        "threshold": 0.45,
    }
    if not MODEL_METADATA_PATH.exists():
        return defaults
    try:
        with MODEL_METADATA_PATH.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        if "unknown_threshold" in metadata:
            defaults["threshold"] = float(metadata["unknown_threshold"])
    except Exception:
        return defaults
    return defaults


def _merge_with_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    merged = DEFAULT_CONFIG.copy()
    merged.update(_load_model_runtime_defaults())
    for key, value in config.items():
        if key in DEFAULT_CONFIG or key == "threshold":
            merged[key] = value
    return merged


def load_config() -> Dict[str, Any]:
    """Load config.json, creating it with defaults if missing or invalid."""
    if CONFIG_PATH.exists():
        try:
            with CONFIG_PATH.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return _merge_with_defaults(data)
        except Exception:
            # Fall back to defaults on any parse error
            pass

    save_config(DEFAULT_CONFIG)
    return DEFAULT_CONFIG.copy()


def save_config(config: Dict[str, Any]) -> None:
    """Persist configuration, merging with defaults and writing pretty JSON."""
    merged = _merge_with_defaults(config)
    with CONFIG_PATH.open("w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, sort_keys=True)


def get_threshold() -> float:
    return float(load_config().get("threshold", 0.62))
