import base64
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.model import FaceClassifierCNN

DATASET_ROOT = REPO_ROOT / "dataset"
CAPTURED_DIR = DATASET_ROOT / "captured"
TRAIN_DIR = DATASET_ROOT / "processed" / "train"
MODEL_PATH = REPO_ROOT / "models" / "bestModel.pth"
LABEL_MAP_PATH = REPO_ROOT / "models" / "labelMap.json"
MODEL_METADATA_PATH = REPO_ROOT / "models" / "model_metadata.json"

_THRESHOLD_FLOOR = 0.45
_THRESHOLD_CEIL = 0.95
_DEFAULT_MARGIN = 0.12
_VALID_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class FaceEngine:
    def __init__(
        self,
        threshold: Optional[float] = None,
        recognitionMargin: Optional[float] = None,
        supportShots: Optional[int] = None,
    ) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model: Optional[FaceClassifierCNN] = None
        self.labelMap: Dict[int, str] = {}
        self.sample_image_paths: Dict[str, Path] = {}
        self.detection_counts: Dict[str, int] = {}
        self.metadata: Dict[str, Any] = self._load_metadata()
        self.imageSize = tuple(self.metadata.get("image_size", [160, 160]))
        self.threshold = self._resolve_threshold(threshold)
        self.recognitionMargin = self._resolve_margin(
            recognitionMargin
            if recognitionMargin is not None
            else self.metadata.get("recognition_margin", _DEFAULT_MARGIN)
        )
        self.transform = transforms.Compose(
            [
                transforms.Resize(self.imageSize),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.demo_mode = False
        self._demo_last_emit = 0.0
        self._demo_toggle = False
        self._load_model()

    def _load_metadata(self) -> Dict[str, Any]:
        if not MODEL_METADATA_PATH.exists():
            return {}
        try:
            with MODEL_METADATA_PATH.open("r", encoding="utf-8") as file:
                return json.load(file)
        except Exception:
            return {}

    def _resolve_threshold(self, threshold: Optional[float]) -> float:
        if threshold is None:
            threshold = self.metadata.get("unknown_threshold", 0.65)
        return max(_THRESHOLD_FLOOR, min(_THRESHOLD_CEIL, float(threshold)))

    def _resolve_margin(self, margin: Any) -> float:
        try:
            value = float(margin)
        except (TypeError, ValueError):
            value = _DEFAULT_MARGIN
        return max(0.02, min(0.20, value))

    def _load_model(self) -> None:
        if not MODEL_PATH.exists() or not LABEL_MAP_PATH.exists():
            self.demo_mode = True
            return

        with LABEL_MAP_PATH.open("r", encoding="utf-8") as file:
            raw_map = json.load(file)
        self.labelMap = {int(key): value for key, value in raw_map.items()}

        self.model = FaceClassifierCNN(numClasses=len(self.labelMap)).to(self.device)
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        self.model.eval()

        self.sample_image_paths = {
            name: self._find_sample_image(name) for name in self.labelMap.values()
        }
        self.detection_counts = {name: 0 for name in self.labelMap.values()}

    def _find_sample_image(self, name: str) -> Path:
        for root in (TRAIN_DIR / name, CAPTURED_DIR / name):
            if root.exists():
                for path in sorted(root.iterdir()):
                    if (
                        path.is_file()
                        and path.suffix.lower() in _VALID_IMAGE_SUFFIXES
                        and "__" not in path.stem
                    ):
                        return path
        return Path()

    def detect_faces(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        if self.demo_mode or self.model is None or not self.labelMap:
            return self._process_frame_demo(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80),
        )

        detections: List[Dict[str, Any]] = []
        for (x, y, w, h) in faces:
            face_bgr = frame[y : y + h, x : x + w]
            if face_bgr.size == 0:
                continue
            detections.append(
                {
                    "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
                    "face_crop_b64": self._face_to_b64(face_bgr),
                    "face_bgr": face_bgr,
                    "blur_score": self.measure_blur(face_bgr),
                }
            )
        return detections

    def process_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        detections = self.detect_faces(frame)
        results: List[Dict[str, Any]] = []
        for detection in detections:
            prediction = self.predict_face(detection.get("face_bgr"))
            merged = {**detection, **prediction}
            if merged.get("is_known"):
                name = str(merged.get("name"))
                self.detection_counts[name] = self.detection_counts.get(name, 0) + 1
            results.append(merged)
        return results

    def predict_face(self, face_bgr: Optional[np.ndarray]) -> Dict[str, Any]:
        if face_bgr is None or face_bgr.size == 0:
            return {"name": "Unknown", "confidence": 0.0, "is_known": False}
        return self.predict_faces([face_bgr])

    def predict_faces(self, face_images_bgr: List[np.ndarray]) -> Dict[str, Any]:
        if not face_images_bgr:
            return {"name": "Unknown", "confidence": 0.0, "is_known": False}
        if self.demo_mode or self.model is None or not self.labelMap:
            return {"name": "DemoUser", "confidence": 0.88, "is_known": True}

        logitsList: List[torch.Tensor] = []
        for face_bgr in face_images_bgr:
            logits = self._predict_logits(face_bgr)
            if logits is not None:
                logitsList.append(logits)

        if not logitsList:
            return {"name": "Unknown", "confidence": 0.0, "is_known": False}

        meanLogits = torch.stack(logitsList, dim=0).mean(dim=0)
        probabilities = F.softmax(meanLogits.unsqueeze(0), dim=1)[0]
        top2Values, top2Indices = probabilities.topk(k=min(2, probabilities.shape[0]))

        bestProb = float(top2Values[0].item())
        bestIndex = int(top2Indices[0].item())
        secondProb = float(top2Values[1].item()) if top2Values.shape[0] > 1 else 0.0
        margin = bestProb - secondProb

        isKnown = bestProb >= self.threshold and margin >= self.recognitionMargin
        name = self.labelMap.get(bestIndex, "Unknown") if isKnown else "Unknown"
        return {"name": name, "confidence": bestProb, "is_known": isKnown}

    def _predict_logits(self, face_bgr: np.ndarray) -> Optional[torch.Tensor]:
        if self.model is None:
            return None
        image = Image.fromarray(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB))
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)[0].cpu()
        return logits

    def measure_blur(self, face_bgr: np.ndarray) -> float:
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _process_frame_demo(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        height, width = frame.shape[:2]
        now = time.time()
        if now - self._demo_last_emit < 3.0:
            return []

        self._demo_last_emit = now
        self._demo_toggle = not self._demo_toggle

        boxWidth, boxHeight = int(width * 0.28), int(height * 0.36)
        x = (width - boxWidth) // 2
        y = (height - boxHeight) // 3
        face_bgr = frame[y : y + boxHeight, x : x + boxWidth]

        if self._demo_toggle:
            name = "DemoUser"
            confidence = 0.88
            isKnown = True
        else:
            name = "Unknown"
            confidence = 0.43
            isKnown = False

        return [
            {
                "name": name,
                "confidence": confidence,
                "bbox": {"x": x, "y": y, "w": boxWidth, "h": boxHeight},
                "is_known": isKnown,
                "face_crop_b64": self._face_to_b64(face_bgr),
                "face_bgr": face_bgr,
            }
        ]

    def add_person(self, name: str, face_images_bgr: List[np.ndarray]) -> None:
        if not face_images_bgr:
            return
        safeName = name.strip() or "unknown"
        personDir = CAPTURED_DIR / safeName
        personDir.mkdir(parents=True, exist_ok=True)
        timestamp = int(time.time())
        for index, face_bgr in enumerate(face_images_bgr, start=1):
            if face_bgr is None or face_bgr.size == 0:
                continue
            imagePath = personDir / f"web_{timestamp}_{index}.jpg"
            cv2.imwrite(str(imagePath), face_bgr)
            self.sample_image_paths.setdefault(safeName, imagePath)
        self.detection_counts.setdefault(safeName, 0)

    def remove_person(self, name: str) -> None:
        self.detection_counts.pop(name, None)
        self.sample_image_paths.pop(name, None)

    def set_threshold(self, value: float) -> None:
        self.threshold = self._resolve_threshold(value)

    def get_known_users(self) -> List[Dict[str, Any]]:
        if not self.labelMap and self.demo_mode:
            return [{"name": "DemoUser", "sample_image_b64": None, "detection_count": 0}]

        users: List[Dict[str, Any]] = []
        names = sorted(set(self.labelMap.values()) | set(self.sample_image_paths.keys()))
        for name in names:
            sampleB64: Optional[str] = None
            imagePath = self.sample_image_paths.get(name)
            if imagePath is not None and imagePath.exists():
                image = cv2.imread(str(imagePath))
                if image is not None:
                    sampleB64 = self._face_to_b64(image)
            users.append(
                {
                    "name": name,
                    "sample_image_b64": sampleB64,
                    "detection_count": int(self.detection_counts.get(name, 0)),
                }
            )
        return users

    def _face_to_b64(self, face_bgr: np.ndarray) -> str:
        if face_bgr is None or face_bgr.size == 0:
            return ""
        ok, buffer = cv2.imencode(".jpg", face_bgr)
        if not ok:
            return ""
        return base64.b64encode(buffer.tobytes()).decode("ascii")
