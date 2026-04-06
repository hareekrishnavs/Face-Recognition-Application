from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
from insightface.app import FaceAnalysis


MODEL_PACKAGE = "buffalo_l"
DETECTION_SIZE = (640, 640)
EMBEDDING_SIZE = 512


@dataclass
class FaceEmbeddingResult:
    embedding: np.ndarray
    bbox: Optional[tuple[int, int, int, int]] = None
    det_score: float = 0.0
    face_crop_bgr: Optional[np.ndarray] = None


class InsightFaceBackend:
    def __init__(self) -> None:
        self.app = FaceAnalysis(name=MODEL_PACKAGE, providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=0, det_size=DETECTION_SIZE)
        self.recognition = self.app.models["recognition"]

    def detect(self, image_bgr: np.ndarray):
        return self.app.get(image_bgr)

    def extract_from_image(
        self,
        image_bgr: np.ndarray,
        assume_aligned: bool = False,
    ) -> Optional[FaceEmbeddingResult]:
        if image_bgr is None or image_bgr.size == 0:
            return None

        faces = self.detect(image_bgr)
        if faces:
            face = max(faces, key=lambda item: (item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1]))
            x1, y1, x2, y2 = [int(round(v)) for v in face.bbox]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image_bgr.shape[1], x2)
            y2 = min(image_bgr.shape[0], y2)
            crop = image_bgr[y1:y2, x1:x2].copy() if x2 > x1 and y2 > y1 else image_bgr.copy()
            embedding = self._normalize(face.normed_embedding)
            return FaceEmbeddingResult(
                embedding=embedding,
                bbox=(x1, y1, x2 - x1, y2 - y1),
                det_score=float(getattr(face, "det_score", 0.0)),
                face_crop_bgr=crop,
            )

        if not assume_aligned:
            return None

        resized = cv2.resize(image_bgr, (112, 112), interpolation=cv2.INTER_AREA)
        feat = self.recognition.get_feat(resized)[0]
        return FaceEmbeddingResult(embedding=self._normalize(feat), face_crop_bgr=image_bgr.copy())

    def build_mean_embedding(self, embeddings: Iterable[np.ndarray]) -> Optional[np.ndarray]:
        rows = [self._normalize(np.asarray(row, dtype=np.float32)) for row in embeddings if row is not None]
        if not rows:
            return None
        return self._normalize(np.mean(np.stack(rows, axis=0), axis=0))

    def similarity(self, left: np.ndarray, right: np.ndarray) -> float:
        return float(np.dot(self._normalize(left), self._normalize(right)))

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-8:
            return vector
        return vector / norm
