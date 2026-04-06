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
FACE_INDEX_PATH = REPO_ROOT / "models" / "face_index.npz"

_THRESHOLD_FLOOR = 0.20
_THRESHOLD_CEIL = 0.95
_DEFAULT_MARGIN = 0.12
_VALID_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Try to import insightface; it may not be installed
try:
    from utils.insightface_backend import InsightFaceBackend
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False

try:
    from utils.config import augmentationPlan
    _AUGMENTATION_SUFFIXES = set(augmentationPlan)
except Exception:
    _AUGMENTATION_SUFFIXES = set()


class FaceEngine:
    def __init__(
        self,
        threshold: Optional[float] = None,
        recognitionMargin: Optional[float] = None,
        supportShots: Optional[int] = None,
        model_type: str = "arcface",
    ) -> None:
        self.model_type = model_type  # "arcface" or "insightface"

        # ── ArcFace state ──
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.arcface_model: Optional[FaceClassifierCNN] = None
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # ── InsightFace state ──
        self.insightface_backend: Optional[Any] = None
        self.classNames: List[str] = []
        self.prototypes: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self.sampleEmbeddings: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self.sampleLabels: np.ndarray = np.empty((0,), dtype=np.int64)
        self.prototypeBlend = 0.65

        # ── Shared state ──
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
        self.demo_mode = False
        self.load_error: Optional[str] = None
        self._demo_last_emit = 0.0
        self._demo_toggle = False

        self._load_arcface_model()
        self._load_insightface_backend()
        self._load_insightface_index()

    # ── Metadata / config helpers ─────────────────────────────

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
            threshold = self.metadata.get("unknown_threshold", 0.45)
        return max(_THRESHOLD_FLOOR, min(_THRESHOLD_CEIL, float(threshold)))

    def _resolve_margin(self, margin: Any) -> float:
        try:
            value = float(margin)
        except (TypeError, ValueError):
            value = _DEFAULT_MARGIN
        return max(0.02, min(0.20, value))

    # ── Model loading ─────────────────────────────────────────

    def _load_arcface_model(self) -> None:
        if not MODEL_PATH.exists() or not LABEL_MAP_PATH.exists():
            return
        try:
            with LABEL_MAP_PATH.open("r", encoding="utf-8") as file:
                raw_map = json.load(file)
            label_map = {int(key): value for key, value in raw_map.items()}
            model = FaceClassifierCNN(numClasses=len(label_map)).to(self.device)
            model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
            model.eval()
            self.arcface_model = model
            if self.model_type == "arcface":
                self.labelMap = label_map
                self._refresh_sample_images()
        except Exception:
            pass

    def _load_insightface_backend(self) -> None:
        if not INSIGHTFACE_AVAILABLE:
            return
        try:
            self.insightface_backend = InsightFaceBackend()
        except Exception:
            self.insightface_backend = None

    def _load_insightface_index(self) -> None:
        if not FACE_INDEX_PATH.exists() or not LABEL_MAP_PATH.exists():
            return
        try:
            with LABEL_MAP_PATH.open("r", encoding="utf-8") as file:
                rawMap = json.load(file)
            label_map = {int(key): value for key, value in rawMap.items()}

            index = np.load(FACE_INDEX_PATH, allow_pickle=False)
            self.classNames = [str(name) for name in index["class_names"].tolist()]
            self.prototypes = index["prototypes"].astype(np.float32)
            self.sampleEmbeddings = index["sample_embeddings"].astype(np.float32)
            self.sampleLabels = index["sample_labels"].astype(np.int64)

            if self.model_type == "insightface":
                self.labelMap = label_map
                self._refresh_sample_images()
        except Exception as exc:
            self.load_error = str(exc)

    def _refresh_sample_images(self) -> None:
        self.sample_image_paths = {
            name: self._find_sample_image(name) for name in self.labelMap.values()
        }
        self.detection_counts = {name: 0 for name in self.labelMap.values()}

    def _check_demo_mode(self) -> bool:
        if self.model_type == "arcface":
            return self.arcface_model is None or not self.labelMap
        else:
            return (
                self.insightface_backend is None
                or not self.labelMap
                or self.prototypes.size == 0
            )

    # ── Model switching ───────────────────────────────────────

    def switch_model(self, model_type: str) -> Dict[str, Any]:
        if model_type not in ("arcface", "insightface"):
            return {"error": "Invalid model type. Use 'arcface' or 'insightface'."}

        if model_type == "insightface" and not INSIGHTFACE_AVAILABLE:
            return {"error": "InsightFace library is not installed."}
        if model_type == "insightface" and self.insightface_backend is None:
            return {"error": "InsightFace backend failed to load."}

        self.model_type = model_type

        # Reload label map for the active model
        if LABEL_MAP_PATH.exists():
            try:
                with LABEL_MAP_PATH.open("r", encoding="utf-8") as file:
                    rawMap = json.load(file)
                self.labelMap = {int(key): value for key, value in rawMap.items()}
            except Exception:
                self.labelMap = {}
        else:
            self.labelMap = {}

        self._refresh_sample_images()
        self.demo_mode = self._check_demo_mode()

        return {
            "model_type": self.model_type,
            "demo_mode": self.demo_mode,
            "label_count": len(self.labelMap),
        }

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "active_model": self.model_type,
            "arcface_loaded": self.arcface_model is not None,
            "insightface_loaded": self.insightface_backend is not None,
            "insightface_available": INSIGHTFACE_AVAILABLE,
            "demo_mode": self._check_demo_mode(),
        }

    # ── File helpers ──────────────────────────────────────────

    def _find_sample_image(self, name: str) -> Path:
        for root in (TRAIN_DIR / name, CAPTURED_DIR / name):
            if root.exists():
                for path in sorted(root.iterdir()):
                    if (
                        path.is_file()
                        and path.suffix.lower() in _VALID_IMAGE_SUFFIXES
                        and not self._is_augmented_file(path)
                    ):
                        return path
        return Path()

    def _is_augmented_file(self, path: Path) -> bool:
        stem = path.stem
        if "__" not in stem:
            return False
        suffix = stem.rsplit("__", 1)[-1]
        baseSuffix = suffix.rsplit("_", 1)[0] if "_" in suffix else suffix
        return suffix in _AUGMENTATION_SUFFIXES or baseSuffix in _AUGMENTATION_SUFFIXES

    # ── Detection ─────────────────────────────────────────────

    def detect_faces(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        if self._check_demo_mode():
            return self._process_frame_demo(frame)

        if self.model_type == "insightface":
            return self._detect_faces_insightface(frame)
        else:
            return self._detect_faces_arcface(frame)

    def _detect_faces_arcface(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80),
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

    def _detect_faces_insightface(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        faces = self.insightface_backend.detect(frame)
        detections: List[Dict[str, Any]] = []
        for face in faces:
            x1, y1, x2, y2 = [int(round(v)) for v in face.bbox]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            crop = frame[y1:y2, x1:x2]
            detections.append(
                {
                    "bbox": {"x": x1, "y": y1, "w": max(0, x2 - x1), "h": max(0, y2 - y1)},
                    "face_crop_b64": self._face_to_b64(crop),
                    "face_bgr": crop,
                    "blur_score": self.measure_blur(crop),
                    "embedding": self.insightface_backend._normalize(face.normed_embedding),
                }
            )
        return detections

    # ── Processing ────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        detections = self.detect_faces(frame)
        results: List[Dict[str, Any]] = []
        for detection in detections:
            if self.model_type == "insightface":
                prediction = self.predict_face(
                    detection.get("face_bgr"),
                    embedding=detection.get("embedding"),
                )
            else:
                prediction = self.predict_face(detection.get("face_bgr"))
            merged = {**detection, **prediction}
            if merged.get("is_known"):
                name = str(merged.get("name"))
                self.detection_counts[name] = self.detection_counts.get(name, 0) + 1
            results.append(merged)
        return results

    # ── Prediction ────────────────────────────────────────────

    def predict_face(
        self,
        face_bgr: Optional[np.ndarray],
        embedding: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        if face_bgr is None or face_bgr.size == 0:
            return {"name": "Unknown", "confidence": 0.0, "is_known": False}
        if self.model_type == "insightface":
            return self.predict_faces(
                [face_bgr],
                embeddings=[embedding] if embedding is not None else None,
            )
        else:
            return self.predict_faces([face_bgr])

    def predict_faces(
        self,
        face_images_bgr: List[np.ndarray],
        embeddings: Optional[List[Optional[np.ndarray]]] = None,
    ) -> Dict[str, Any]:
        if not face_images_bgr:
            return {"name": "Unknown", "confidence": 0.0, "is_known": False}
        if self._check_demo_mode():
            return {"name": "DemoUser", "confidence": 0.88, "is_known": True}

        if self.model_type == "insightface":
            return self._predict_insightface(face_images_bgr, embeddings)
        else:
            return self._predict_arcface(face_images_bgr)

    def _predict_arcface(self, face_images_bgr: List[np.ndarray]) -> Dict[str, Any]:
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

    def _predict_insightface(
        self,
        face_images_bgr: List[np.ndarray],
        embeddings: Optional[List[Optional[np.ndarray]]] = None,
    ) -> Dict[str, Any]:
        rows: List[np.ndarray] = []
        for index, face_bgr in enumerate(face_images_bgr):
            precomputed = None if embeddings is None or index >= len(embeddings) else embeddings[index]
            if precomputed is not None:
                rows.append(self.insightface_backend._normalize(precomputed))
                continue
            result = self.insightface_backend.extract_from_image(face_bgr, assume_aligned=True)
            if result is not None:
                rows.append(result.embedding)

        if not rows:
            return {"name": "Unknown", "confidence": 0.0, "is_known": False}

        query = self.insightface_backend.build_mean_embedding(rows)
        if query is None:
            return {"name": "Unknown", "confidence": 0.0, "is_known": False}

        prototypeScores = self.prototypes @ query
        sampleScoresByClass = np.full((len(self.classNames),), -1.0, dtype=np.float32)
        if self.sampleEmbeddings.size:
            sampleScores = self.sampleEmbeddings @ query
            for classIndex in range(len(self.classNames)):
                classRows = sampleScores[self.sampleLabels == classIndex]
                if classRows.size:
                    sampleScoresByClass[classIndex] = float(np.max(classRows))
                else:
                    sampleScoresByClass[classIndex] = float(prototypeScores[classIndex])
        else:
            sampleScoresByClass = prototypeScores.copy()

        combinedScores = (
            self.prototypeBlend * prototypeScores
            + (1.0 - self.prototypeBlend) * sampleScoresByClass
        )
        bestIndex = int(np.argmax(combinedScores))
        bestScore = float(combinedScores[bestIndex])
        isKnown = bestScore >= self.threshold
        name = self.classNames[bestIndex] if isKnown else "Unknown"
        return {"name": name, "confidence": bestScore, "is_known": isKnown}

    def _predict_logits(self, face_bgr: np.ndarray) -> Optional[torch.Tensor]:
        if self.arcface_model is None:
            return None
        image = Image.fromarray(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB))
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.arcface_model(tensor)[0].cpu()
        return logits

    # ── Utility ───────────────────────────────────────────────

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

    # ── Enrollment ────────────────────────────────────────────

    def add_person(self, name: str, face_images_bgr: List[np.ndarray]) -> None:
        if not face_images_bgr:
            return
        safeName = name.strip() or "unknown"
        personDir = CAPTURED_DIR / safeName
        personDir.mkdir(parents=True, exist_ok=True)
        timestamp = int(time.time())

        if self.model_type == "insightface" and self.insightface_backend is not None:
            self._add_person_insightface(safeName, face_images_bgr, personDir, timestamp)
        else:
            self._add_person_arcface(safeName, face_images_bgr, personDir, timestamp)

    def _add_person_arcface(
        self, name: str, face_images_bgr: List[np.ndarray], personDir: Path, timestamp: int
    ) -> None:
        for index, face_bgr in enumerate(face_images_bgr, start=1):
            if face_bgr is None or face_bgr.size == 0:
                continue
            imagePath = personDir / f"web_{timestamp}_{index}.jpg"
            cv2.imwrite(str(imagePath), face_bgr)
            self.sample_image_paths.setdefault(name, imagePath)
        self.detection_counts.setdefault(name, 0)

    def _add_person_insightface(
        self, name: str, face_images_bgr: List[np.ndarray], personDir: Path, timestamp: int
    ) -> None:
        newEmbeddings: List[np.ndarray] = []
        for index, face_bgr in enumerate(face_images_bgr, start=1):
            if face_bgr is None or face_bgr.size == 0:
                continue
            imagePath = personDir / f"web_{timestamp}_{index}.jpg"
            cv2.imwrite(str(imagePath), face_bgr)
            result = self.insightface_backend.extract_from_image(face_bgr, assume_aligned=True)
            if result is not None:
                newEmbeddings.append(result.embedding)
            self.sample_image_paths.setdefault(name, imagePath)

        if not newEmbeddings:
            return

        meanEmbedding = self.insightface_backend.build_mean_embedding(newEmbeddings)
        if meanEmbedding is None:
            return

        if name in self.classNames:
            classIndex = self.classNames.index(name)
            existing = self.sampleEmbeddings[self.sampleLabels == classIndex]
            merged = [*existing, *newEmbeddings] if existing.size else newEmbeddings
            proto = self.insightface_backend.build_mean_embedding(merged)
            if proto is not None:
                self.prototypes[classIndex] = proto
        else:
            self.classNames.append(name)
            self.prototypes = (
                np.vstack([self.prototypes, meanEmbedding])
                if self.prototypes.size
                else meanEmbedding.reshape(1, -1)
            )
            classIndex = len(self.classNames) - 1
            self.labelMap[classIndex] = name

        appendEmbeddings = np.stack(newEmbeddings, axis=0).astype(np.float32)
        appendLabels = np.full((appendEmbeddings.shape[0],), classIndex, dtype=np.int64)
        self.sampleEmbeddings = (
            np.vstack([self.sampleEmbeddings, appendEmbeddings])
            if self.sampleEmbeddings.size
            else appendEmbeddings
        )
        self.sampleLabels = (
            np.concatenate([self.sampleLabels, appendLabels])
            if self.sampleLabels.size
            else appendLabels
        )
        self.detection_counts.setdefault(name, 0)
        self._persist_index()

    def remove_person(self, name: str) -> None:
        # InsightFace index cleanup
        if name in self.classNames:
            index = self.classNames.index(name)
            keepClassMask = self.sampleLabels != index
            self.sampleEmbeddings = self.sampleEmbeddings[keepClassMask]
            self.sampleLabels = self.sampleLabels[keepClassMask]
            self.prototypes = np.delete(self.prototypes, index, axis=0)
            self.classNames.pop(index)
            self.labelMap = {idx: className for idx, className in enumerate(self.classNames)}
            remapped = self.sampleLabels.copy()
            remapped[remapped > index] -= 1
            self.sampleLabels = remapped
            self._persist_index()

        self.detection_counts.pop(name, None)
        self.sample_image_paths.pop(name, None)

    def set_threshold(self, value: float) -> None:
        self.threshold = self._resolve_threshold(value)

    def get_known_users(self) -> List[Dict[str, Any]]:
        if not self.labelMap and self._check_demo_mode():
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

    def _persist_index(self) -> None:
        if not self.classNames or self.prototypes.size == 0:
            return
        FACE_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            FACE_INDEX_PATH,
            class_names=np.asarray(self.classNames),
            prototypes=self.prototypes.astype(np.float32),
            sample_embeddings=self.sampleEmbeddings.astype(np.float32),
            sample_labels=self.sampleLabels.astype(np.int64),
        )
        with LABEL_MAP_PATH.open("w", encoding="utf-8") as file:
            json.dump({index: name for index, name in enumerate(self.classNames)}, file, indent=2)

    def _face_to_b64(self, face_bgr: np.ndarray) -> str:
        if face_bgr is None or face_bgr.size == 0:
            return ""
        ok, buffer = cv2.imencode(".jpg", face_bgr)
        if not ok:
            return ""
        return base64.b64encode(buffer.tobytes()).decode("ascii")
