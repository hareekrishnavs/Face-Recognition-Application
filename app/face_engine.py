"""
face_engine.py — Inference wrapper for FaceVault.

FaceModel architecture replicated verbatim from training/liveDemo.py so
that torch.load_state_dict() can resolve the saved weights.

Model path:     <repo_root>/models/arcface_model.pth
Train embeddings from: <repo_root>/dataset/processed/train/
Sample thumbnails from: <repo_root>/dataset/raw/<name>/
"""
import base64
import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets, models, transforms

REPO_ROOT  = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO_ROOT / "models" / "arcface_model.pth"
TRAIN_DIR  = REPO_ROOT / "dataset" / "processed" / "train"
RAW_DIR    = REPO_ROOT / "dataset" / "raw"

EMBEDDING_SIZE = 512
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# ── Model (matches training/liveDemo.py exactly) ──────────────────────────────

class FaceModel(nn.Module):
    def __init__(self, embedding_size: int = EMBEDDING_SIZE):
        super().__init__()
        base           = models.resnet50(weights=None)
        self.backbone  = nn.Sequential(*list(base.children())[:-1])
        self.embedding = nn.Linear(base.fc.in_features, embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.embedding(x)
        return F.normalize(x)


# ── Engine ────────────────────────────────────────────────────────────────────

class FaceEngine:
    """
    Thread-safe face recognition engine.

    Loads FaceModel from models/arcface_model.pth and builds an in-memory
    embedding DB from dataset/processed/train/.  New persons are hot-enrolled
    via add_person() without retraining — their embeddings are added directly
    to the DB.

    If the model file is absent the engine enters stub/demo mode:
    process_frame() returns no detections, but the rest of the API still works.
    """

    def __init__(self):
        self._lock        = threading.Lock()
        self._model: Optional[FaceModel] = None
        self._db: dict[str, list[torch.Tensor]] = {}
        self.threshold    = 0.65
        self.model_loaded = False
        self._load()

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _load(self):
        if not MODEL_PATH.exists():
            print(f"[FaceEngine] {MODEL_PATH} not found — demo mode.")
            return
        try:
            m = FaceModel().to(DEVICE)
            m.load_state_dict(torch.load(str(MODEL_PATH), map_location=DEVICE))
            m.eval()
            with self._lock:
                self._model       = m
                self.model_loaded = True
            print(f"[FaceEngine] Model loaded from {MODEL_PATH}")
            self._build_db()
        except Exception as exc:
            print(f"[FaceEngine] Load error: {exc}")

    def _build_db(self):
        if not TRAIN_DIR.exists():
            print("[FaceEngine] Train dir missing — embedding DB empty.")
            return
        ds     = datasets.ImageFolder(str(TRAIN_DIR), transform=TRANSFORM)
        loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False)
        db: dict[str, list[torch.Tensor]] = {n: [] for n in ds.classes}
        with torch.no_grad():
            for imgs, labels in loader:
                embs = self._model(imgs.to(DEVICE))
                for emb, lbl in zip(embs, labels):
                    db[ds.classes[lbl.item()]].append(emb.cpu())
        with self._lock:
            self._db = db
        print(f"[FaceEngine] Embedding DB: {list(db.keys())}")

    # ── Public API ────────────────────────────────────────────────────────────

    def set_threshold(self, value: float):
        self.threshold = max(0.0, min(1.0, float(value)))

    def get_known_names(self) -> list[str]:
        with self._lock:
            return list(self._db.keys())

    def get_known_users(self) -> list[dict]:
        """Return [{name, sample_b64}] for every enrolled person."""
        with self._lock:
            names = list(self._db.keys())
        result = []
        for name in names:
            result.append({"name": name, "sample_b64": self._sample_b64(name)})
        return result

    def add_person(self, name: str, images: list):
        """
        Hot-enroll a new person.  `images` may be a mix of:
          - pathlib.Path / str   → loaded from disk
          - numpy.ndarray        → face crop (BGR)
        """
        if not self.model_loaded:
            print(f"[FaceEngine] add_person skipped — no model.")
            return
        embeddings: list[torch.Tensor] = []
        with torch.no_grad():
            for img in images:
                try:
                    if isinstance(img, (Path, str)):
                        pil = Image.open(img).convert("RGB")
                    elif isinstance(img, np.ndarray):
                        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    else:
                        continue
                    t = TRANSFORM(pil).unsqueeze(0).to(DEVICE)
                    embeddings.append(self._model(t)[0].cpu())
                except Exception as exc:
                    print(f"[FaceEngine] Skip image: {exc}")
        if embeddings:
            with self._lock:
                self._db[name] = embeddings
            print(f"[FaceEngine] Enrolled '{name}' ({len(embeddings)} embeddings)")

    def remove_person(self, name: str):
        with self._lock:
            self._db.pop(name, None)

    def process_frame(self, frame: np.ndarray) -> list[dict]:
        """
        Detect faces with Haar cascade and run ArcFace recognition.

        Returns a list of detection dicts:
          name, confidence, bbox [x,y,w,h], is_known, face_crop_b64
        """
        detections: list[dict] = []
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30)
        )

        for (x, y, w, h) in faces:
            crop       = frame[y : y + h, x : x + w]
            name, conf = "Unknown", 0.0

            if self.model_loaded:
                try:
                    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    inp = TRANSFORM(Image.fromarray(rgb)).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        emb = self._model(inp)[0].cpu()
                    name, conf = self._predict(emb)
                except Exception as exc:
                    print(f"[FaceEngine] Inference error: {exc}")

            # Encode thumbnail for socket/enrollment use
            thumb = cv2.resize(crop, (80, 80))
            _, buf = cv2.imencode(".jpg", thumb)
            face_crop_b64 = base64.b64encode(buf).decode()

            detections.append({
                "name":          name,
                "confidence":    round(conf, 4),
                "bbox":          [int(x), int(y), int(w), int(h)],
                "is_known":      name != "Unknown",
                "face_crop_b64": face_crop_b64,
            })

        return detections

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _predict(self, embedding: torch.Tensor) -> tuple[str, float]:
        best_name, best_score = "Unknown", -1.0
        with self._lock:
            db  = dict(self._db)
            thr = self.threshold
        for name, refs in db.items():
            for ref in refs:
                score = torch.dot(embedding, ref).item()
                if score > best_score:
                    best_score, best_name = score, name
        if best_score < thr:
            return "Unknown", max(0.0, best_score)
        return best_name, best_score

    def _sample_b64(self, name: str) -> Optional[str]:
        for base_dir in (RAW_DIR, TRAIN_DIR):
            person_dir = base_dir / name
            if not person_dir.exists():
                continue
            imgs = sorted(
                list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png"))
            )
            if imgs:
                try:
                    img = cv2.imread(str(imgs[0]))
                    if img is not None:
                        img = cv2.resize(img, (80, 80))
                        _, buf = cv2.imencode(".jpg", img)
                        return base64.b64encode(buf).decode()
                except Exception:
                    pass
        return None


# Module-level singleton — imported by main.py
engine = FaceEngine()
