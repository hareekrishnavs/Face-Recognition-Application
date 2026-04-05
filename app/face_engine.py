import base64
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets, transforms

REPO_ROOT = Path(__file__).resolve().parent.parent
APP_ROOT = Path(__file__).resolve().parent

# How many training images to use per class when building the embedding DB.
# Keeping this low speeds up startup significantly.
_MAX_SAMPLES_PER_CLASS = 20


def _build_smallcnn(num_classes: int) -> nn.Sequential:
    """Recreate the SmallCNNClassifier architecture from training/model.py."""
    return nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(128, 256, kernel_size=3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Dropout(p=0.5),
        nn.Linear(256, num_classes),
    )


class FaceEngine:
    """Wraps the SmallCNN face recognition model and Haar-cascade detector.

    - Uses bestModel.pth (SmallCNNClassifier trained on data/processed/train/).
    - The final Linear layer is stripped; the remaining 256-D activations are
      L2-normalised to form face embeddings, enabling cosine-similarity
      matching and zero-shot enrollment of new people.
    - Falls back to demo_mode if the model or dataset is missing.
    """

    def __init__(self, threshold: float = 0.65) -> None:
        self.threshold: float = max(0.40, min(0.95, float(threshold)))
        self.model: Optional[nn.Module] = None
        self.embedding_db: Dict[str, torch.Tensor] = {}
        self.sample_image_paths: Dict[str, Path] = {}
        self.detection_counts: Dict[str, int] = {}
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.demo_mode: bool = False
        self._demo_last_emit: float = 0.0
        self._demo_toggle: bool = False

        self._load_model()

    # ------------------------------------------------------------------
    # Model & embedding DB
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        model_path = REPO_ROOT / "models" / "bestModel.pth"
        label_map_path = REPO_ROOT / "models" / "labelMap.json"
        train_dir = REPO_ROOT / "data" / "processed" / "train"

        if not model_path.exists() or not label_map_path.exists():
            self.demo_mode = True
            return

        # Load label map: {"0": "Haree", "1": "HariHaran", ...}
        with open(label_map_path) as f:
            raw_map = json.load(f)
        label_map: Dict[int, str] = {int(k): v for k, v in raw_map.items()}
        num_classes = len(label_map)

        # Load the full SmallCNN with trained weights
        full_model = _build_smallcnn(num_classes).to(self.device)
        state = torch.load(model_path, map_location=self.device)
        full_model.load_state_dict(state)
        full_model.eval()

        # Strip the final nn.Linear classification head.
        # The remaining layers produce 256-D feature vectors.
        # Because these are references to the same nn.Module objects,
        # the loaded weights are preserved without re-loading.
        backbone_layers = list(full_model.children())[:-1]
        self.model = nn.Sequential(*backbone_layers).to(self.device)
        self.model.eval()

        # Build embedding DB from training images
        self._build_embedding_db(train_dir, label_map)

    def _build_embedding_db(
        self, train_dir: Path, label_map: Dict[int, str]
    ) -> None:
        if not train_dir.exists():
            self.demo_mode = not bool(self.embedding_db)
            return

        dataset = datasets.ImageFolder(train_dir)
        if len(dataset) == 0:
            self.demo_mode = True
            return

        self.embedding_db.clear()
        self.sample_image_paths.clear()
        self.detection_counts.clear()

        # Group sample paths by class name, capped per class
        class_samples: Dict[str, List[str]] = {}
        for sample_path, label_idx in dataset.samples:
            class_name = dataset.classes[label_idx]
            bucket = class_samples.setdefault(class_name, [])
            if len(bucket) < _MAX_SAMPLES_PER_CLASS:
                bucket.append(sample_path)
            if class_name not in self.sample_image_paths:
                self.sample_image_paths[class_name] = Path(sample_path)

        for class_name, paths in class_samples.items():
            embs: List[torch.Tensor] = []
            for sample_path in paths:
                try:
                    image = Image.open(sample_path).convert("RGB")
                except Exception:
                    continue

                tensor = self.transform(image).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    assert self.model is not None
                    feat = self.model(tensor)[0].cpu()
                    emb = F.normalize(feat.unsqueeze(0), p=2, dim=1)[0]
                embs.append(emb)

            if embs:
                self.embedding_db[class_name] = torch.stack(embs, dim=0)
                self.detection_counts[class_name] = 0

        if not self.embedding_db:
            self.demo_mode = True

    # ------------------------------------------------------------------
    # Frame processing
    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Process a single BGR frame.

        Returns a list of dicts:
        { name, confidence, bbox: {x,y,w,h}, is_known, face_crop_b64, face_bgr }
        """

        if self.demo_mode or self.model is None or not self.embedding_db:
            return self._process_frame_demo(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80),
        )

        results: List[Dict[str, Any]] = []

        for (x, y, w, h) in faces:
            x, y, w, h = int(x), int(y), int(w), int(h)
            face_bgr = frame[y : y + h, x : x + w]
            if face_bgr.size == 0:
                continue

            image = Image.fromarray(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB))
            tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                assert self.model is not None
                feat = self.model(tensor)[0].cpu()
                emb = F.normalize(feat.unsqueeze(0), p=2, dim=1)[0]

            name, confidence, is_known = self._match_embedding(emb)

            if is_known:
                self.detection_counts[name] = self.detection_counts.get(name, 0) + 1

            results.append(
                {
                    "name": name,
                    "confidence": float(confidence),
                    "bbox": {"x": x, "y": y, "w": w, "h": h},
                    "is_known": bool(is_known),
                    "face_crop_b64": self._face_to_b64(face_bgr),
                    "face_bgr": face_bgr,
                }
            )

        return results

    def _match_embedding(self, emb: torch.Tensor) -> Tuple[str, float, bool]:
        if not self.embedding_db:
            return "Unknown", 0.0, False

        best_name = "Unknown"
        best_score = 0.0

        for name, db_embs in self.embedding_db.items():
            if db_embs.numel() == 0:
                continue
            sims = F.cosine_similarity(emb.unsqueeze(0), db_embs, dim=1)
            score = float(sims.max().item())
            if score > best_score:
                best_score = score
                best_name = name

        is_known = best_score >= float(self.threshold)
        if not is_known:
            best_name = "Unknown"
        return best_name, best_score, is_known

    # ------------------------------------------------------------------
    # Demo mode
    # ------------------------------------------------------------------

    def _process_frame_demo(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Simulate detections when no trained model is available."""

        h, w = frame.shape[:2]
        now = time.time()

        # Emit at most once every 3 seconds
        if now - self._demo_last_emit < 3.0:
            return []

        self._demo_last_emit = now
        self._demo_toggle = not self._demo_toggle

        box_w, box_h = int(w * 0.28), int(h * 0.36)
        x = (w - box_w) // 2
        y = (h - box_h) // 3
        face_bgr = frame[y : y + box_h, x : x + box_w]

        if self._demo_toggle:
            name = "Alice"
            confidence = 0.89
            is_known = True
        else:
            name = "Unknown"
            confidence = 0.42
            is_known = False

        return [
            {
                "name": name,
                "confidence": confidence,
                "bbox": {"x": x, "y": y, "w": box_w, "h": box_h},
                "is_known": is_known,
                "face_crop_b64": self._face_to_b64(face_bgr),
                "face_bgr": face_bgr,
            }
        ]

    # ------------------------------------------------------------------
    # Enrollment helpers
    # ------------------------------------------------------------------

    def add_person(self, name: str, face_images_bgr: List[np.ndarray]) -> None:
        """Persist new person images and extend the in-memory embedding DB."""

        if not face_images_bgr:
            return

        safe_name = name.strip() or "unknown"
        person_dir = APP_ROOT / "captured" / safe_name
        person_dir.mkdir(parents=True, exist_ok=True)

        new_embs: List[torch.Tensor] = []
        ts = int(time.time())

        for idx, face_bgr in enumerate(face_images_bgr, start=1):
            if face_bgr is None or face_bgr.size == 0:
                continue

            filename = person_dir / f"web_{ts}_{idx}.jpg"
            cv2.imwrite(str(filename), face_bgr)

            image = Image.fromarray(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB))
            tensor = self.transform(image).unsqueeze(0).to(self.device)
            if self.model is not None:
                with torch.no_grad():
                    feat = self.model(tensor)[0].cpu()
                    emb = F.normalize(feat.unsqueeze(0), p=2, dim=1)[0]
                new_embs.append(emb)

            if safe_name not in self.sample_image_paths:
                self.sample_image_paths[safe_name] = filename

        if new_embs:
            stacked = torch.stack(new_embs, dim=0)
            if safe_name in self.embedding_db:
                self.embedding_db[safe_name] = torch.cat(
                    [self.embedding_db[safe_name], stacked], dim=0
                )
            else:
                self.embedding_db[safe_name] = stacked

        self.detection_counts.setdefault(safe_name, 0)

    def remove_person(self, name: str) -> None:
        """Remove a person from the in-memory embedding DB only."""
        self.embedding_db.pop(name, None)
        self.detection_counts.pop(name, None)

    def set_threshold(self, value: float) -> None:
        self.threshold = max(0.40, min(0.95, float(value)))

    def get_known_users(self) -> List[Dict[str, Any]]:
        users: List[Dict[str, Any]] = []

        if not self.embedding_db and self.demo_mode:
            users.append(
                {"name": "DemoUser", "sample_image_b64": None, "detection_count": 0}
            )
            return users

        for name in sorted(self.embedding_db.keys()):
            path = self.sample_image_paths.get(name)
            sample_b64: Optional[str] = None
            if path is not None and path.exists():
                img_bgr = cv2.imread(str(path))
                if img_bgr is not None:
                    sample_b64 = self._face_to_b64(img_bgr)

            users.append(
                {
                    "name": name,
                    "sample_image_b64": sample_b64,
                    "detection_count": int(self.detection_counts.get(name, 0)),
                }
            )

        return users

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _face_to_b64(self, face_bgr: np.ndarray) -> str:
        """Encode a BGR face crop as a base64 JPEG string."""
        if face_bgr is None or face_bgr.size == 0:
            return ""
        ok, buffer = cv2.imencode(".jpg", face_bgr)
        if not ok:
            return ""
        return base64.b64encode(buffer.tobytes()).decode("ascii")
