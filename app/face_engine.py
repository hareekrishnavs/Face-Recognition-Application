import base64
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.config import augmentationPlan, faceIndexPath
from utils.insightface_backend import InsightFaceBackend

DATASET_ROOT = REPO_ROOT / "dataset"
CAPTURED_DIR = DATASET_ROOT / "captured"
TRAIN_DIR = DATASET_ROOT / "processed" / "train"
LABEL_MAP_PATH = REPO_ROOT / "models" / "labelMap.json"
MODEL_METADATA_PATH = REPO_ROOT / "models" / "model_metadata.json"

_THRESHOLD_FLOOR = 0.20
_THRESHOLD_CEIL = 0.90
_VALID_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_AUGMENTATION_SUFFIXES = set(augmentationPlan)


class FaceEngine:
    def __init__(
        self,
        threshold: Optional[float] = None,
        recognitionMargin: Optional[float] = None,
        supportShots: Optional[int] = None,
    ) -> None:
        self.backend = InsightFaceBackend()
        self.labelMap: Dict[int, str] = {}
        self.sample_image_paths: Dict[str, Path] = {}
        self.detection_counts: Dict[str, int] = {}
        self.metadata: Dict[str, Any] = self._load_metadata()
        self.threshold = self._resolve_threshold(threshold)
        self.prototypeBlend = 0.65
        self.classNames: List[str] = []
        self.prototypes: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self.sampleEmbeddings: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self.sampleLabels: np.ndarray = np.empty((0,), dtype=np.int64)
        self.demo_mode = False
        self.load_error: Optional[str] = None
        self._demo_last_emit = 0.0
        self._demo_toggle = False
        self._load_index()

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
            threshold = self.metadata.get("unknown_threshold", 0.40)
        return max(_THRESHOLD_FLOOR, min(_THRESHOLD_CEIL, float(threshold)))

    def _load_index(self) -> None:
        if not faceIndexPath.exists() or not LABEL_MAP_PATH.exists():
            self.demo_mode = True
            return
        try:
            with LABEL_MAP_PATH.open("r", encoding="utf-8") as file:
                rawMap = json.load(file)
            self.labelMap = {int(key): value for key, value in rawMap.items()}

            index = np.load(faceIndexPath, allow_pickle=False)
            self.classNames = [str(name) for name in index["class_names"].tolist()]
            self.prototypes = index["prototypes"].astype(np.float32)
            self.sampleEmbeddings = index["sample_embeddings"].astype(np.float32)
            self.sampleLabels = index["sample_labels"].astype(np.int64)
        except Exception as exc:
            self.demo_mode = True
            self.labelMap = {}
            self.load_error = str(exc)
            return

        self.sample_image_paths = {
            name: self._find_sample_image(name) for name in self.labelMap.values()
        }
        self.detection_counts = {name: 0 for name in self.labelMap.values()}
        self._persist_index()

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

    def detect_faces(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        if self.demo_mode or not self.labelMap:
            return self._process_frame_demo(frame)

        faces = self.backend.detect(frame)
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
                    "embedding": self.backend._normalize(face.normed_embedding),
                }
            )
        return detections

    def process_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        detections = self.detect_faces(frame)
        results: List[Dict[str, Any]] = []
        for detection in detections:
            prediction = self.predict_face(
                detection.get("face_bgr"),
                embedding=detection.get("embedding"),
            )
            merged = {**detection, **prediction}
            if merged.get("is_known"):
                name = str(merged.get("name"))
                self.detection_counts[name] = self.detection_counts.get(name, 0) + 1
            results.append(merged)
        return results

    def predict_face(
        self,
        face_bgr: Optional[np.ndarray],
        embedding: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        if face_bgr is None or face_bgr.size == 0:
            return {"name": "Unknown", "confidence": 0.0, "is_known": False}
        return self.predict_faces([face_bgr], embeddings=[embedding] if embedding is not None else None)

    def predict_faces(
        self,
        face_images_bgr: List[np.ndarray],
        embeddings: Optional[List[Optional[np.ndarray]]] = None,
    ) -> Dict[str, Any]:
        if not face_images_bgr:
            return {"name": "Unknown", "confidence": 0.0, "is_known": False}
        if self.demo_mode or not self.labelMap or self.prototypes.size == 0:
            return {"name": "DemoUser", "confidence": 0.88, "is_known": True}

        rows: List[np.ndarray] = []
        for index, face_bgr in enumerate(face_images_bgr):
            precomputed = None if embeddings is None or index >= len(embeddings) else embeddings[index]
            if precomputed is not None:
                rows.append(self.backend._normalize(precomputed))
                continue
            result = self.backend.extract_from_image(face_bgr, assume_aligned=True)
            if result is not None:
                rows.append(result.embedding)

        if not rows:
            return {"name": "Unknown", "confidence": 0.0, "is_known": False}

        query = self.backend.build_mean_embedding(rows)
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
        return [
            {
                "name": "DemoUser" if self._demo_toggle else "Unknown",
                "confidence": 0.88 if self._demo_toggle else 0.30,
                "bbox": {"x": x, "y": y, "w": boxWidth, "h": boxHeight},
                "is_known": self._demo_toggle,
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

        newEmbeddings: List[np.ndarray] = []
        for index, face_bgr in enumerate(face_images_bgr, start=1):
            if face_bgr is None or face_bgr.size == 0:
                continue
            imagePath = personDir / f"web_{timestamp}_{index}.jpg"
            cv2.imwrite(str(imagePath), face_bgr)
            result = self.backend.extract_from_image(face_bgr, assume_aligned=True)
            if result is not None:
                newEmbeddings.append(result.embedding)
            self.sample_image_paths.setdefault(safeName, imagePath)

        if not newEmbeddings:
            return

        meanEmbedding = self.backend.build_mean_embedding(newEmbeddings)
        if meanEmbedding is None:
            return

        if safeName in self.classNames:
            classIndex = self.classNames.index(safeName)
            existing = self.sampleEmbeddings[self.sampleLabels == classIndex]
            merged = [*existing, *newEmbeddings] if existing.size else newEmbeddings
            proto = self.backend.build_mean_embedding(merged)
            if proto is not None:
                self.prototypes[classIndex] = proto
        else:
            self.classNames.append(safeName)
            self.prototypes = (
                np.vstack([self.prototypes, meanEmbedding])
                if self.prototypes.size
                else meanEmbedding.reshape(1, -1)
            )
            classIndex = len(self.classNames) - 1
            self.labelMap[classIndex] = safeName

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
        self.detection_counts.setdefault(safeName, 0)
        self._persist_index()

    def remove_person(self, name: str) -> None:
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

        self.detection_counts.pop(name, None)
        self.sample_image_paths.pop(name, None)
        self._persist_index()

    def set_threshold(self, value: float) -> None:
        self.threshold = self._resolve_threshold(value)

    def get_known_users(self) -> List[Dict[str, Any]]:
        if not self.labelMap and self.demo_mode:
            return [{"name": "DemoUser", "sample_image_b64": None, "detection_count": 0}]

        users: List[Dict[str, Any]] = []
        names = sorted(set(self.classNames) | set(self.sample_image_paths.keys()))
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
        faceIndexPath.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            faceIndexPath,
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
