import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import (
    confidenceThresholdCeil,
    confidenceThresholdFloor,
    datasetMetadataPath,
    faceIndexPath,
    labelMapPath,
    modelMetadataPath,
    processedDir,
    trainingSummaryPath,
)
from utils.insightface_backend import EMBEDDING_SIZE, MODEL_PACKAGE, InsightFaceBackend

VALID_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class FaceTrainer:
    def __init__(self) -> None:
        self.backend = InsightFaceBackend()

    def run(self) -> None:
        self.ensureProcessedDataset()

        trainData = self.loadSplitEmbeddings(processedDir / "train", assume_aligned=True)
        valData = self.loadSplitEmbeddings(processedDir / "val", assume_aligned=True)
        testData = self.loadSplitEmbeddings(processedDir / "test", assume_aligned=True)

        classNames = sorted(trainData)
        if not classNames:
            raise ValueError("No classes found in processed train split.")

        labelMap = {index: name for index, name in enumerate(classNames)}
        classToIndex = {name: index for index, name in labelMap.items()}

        prototypes = self.buildPrototypes(trainData, classNames)
        sampleEmbeddings, sampleLabels = self.flattenSamples(trainData, classToIndex)

        threshold, valAccuracy = self.evaluateAndCalibrate(valData, classNames, prototypes)
        testAccuracy = self.evaluateClosedSet(testData, classNames, prototypes)

        self.saveArtifacts(
            classNames=classNames,
            prototypes=prototypes,
            sampleEmbeddings=sampleEmbeddings,
            sampleLabels=sampleLabels,
            threshold=threshold,
        )

        summary = {
            "class_names": classNames,
            "final_val_accuracy": valAccuracy,
            "final_test_accuracy": testAccuracy,
            "unknown_threshold": threshold,
            "inference_mode": "insightface_embeddings",
            "backbone": MODEL_PACKAGE,
            "embedding_size": EMBEDDING_SIZE,
        }
        trainingSummaryPath.parent.mkdir(parents=True, exist_ok=True)
        with trainingSummaryPath.open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2)

        print(f"Classes found: {classNames}")
        print(f"Validation accuracy: {valAccuracy:.2f}%")
        print(f"Test accuracy      : {testAccuracy:.2f}%")
        print(f"Unknown threshold  : {threshold:.3f}")
        print(f"Saved face index   : {faceIndexPath}")

    def ensureProcessedDataset(self) -> None:
        requiredDirs = [processedDir / "train", processedDir / "val", processedDir / "test"]
        missingDirs = [path for path in requiredDirs if not path.exists()]
        if missingDirs or not datasetMetadataPath.exists():
            missingText = ", ".join(str(path) for path in missingDirs) or str(datasetMetadataPath)
            raise FileNotFoundError(
                "Processed dataset is missing. Run `python preprocessing/dataset.py` first. "
                f"Missing: {missingText}"
            )

    def loadSplitEmbeddings(
        self,
        splitDir: Path,
        assume_aligned: bool,
    ) -> Dict[str, List[np.ndarray]]:
        data: Dict[str, List[np.ndarray]] = defaultdict(list)
        for classDir in sorted(path for path in splitDir.iterdir() if path.is_dir()):
            for imagePath in sorted(classDir.iterdir()):
                if not imagePath.is_file() or imagePath.suffix.lower() not in VALID_IMAGE_SUFFIXES:
                    continue
                image = cv2.imread(str(imagePath))
                if image is None:
                    continue
                result = self.backend.extract_from_image(image, assume_aligned=assume_aligned)
                if result is None:
                    continue
                data[classDir.name].append(result.embedding)
        return data

    def buildPrototypes(
        self,
        trainData: Dict[str, List[np.ndarray]],
        classNames: List[str],
    ) -> np.ndarray:
        rows: List[np.ndarray] = []
        for name in classNames:
            proto = self.backend.build_mean_embedding(trainData[name])
            if proto is None:
                raise ValueError(f"No usable embeddings found for class '{name}'.")
            rows.append(proto)
        return np.stack(rows, axis=0).astype(np.float32)

    def flattenSamples(
        self,
        trainData: Dict[str, List[np.ndarray]],
        classToIndex: Dict[str, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        embeddings: List[np.ndarray] = []
        labels: List[int] = []
        for name, rows in trainData.items():
            for row in rows:
                embeddings.append(np.asarray(row, dtype=np.float32))
                labels.append(classToIndex[name])
        return np.stack(embeddings, axis=0).astype(np.float32), np.asarray(labels, dtype=np.int64)

    def evaluateAndCalibrate(
        self,
        splitData: Dict[str, List[np.ndarray]],
        classNames: List[str],
        prototypes: np.ndarray,
    ) -> Tuple[float, float]:
        classToIndex = {name: index for index, name in enumerate(classNames)}
        correct = 0
        total = 0
        positives: List[float] = []
        negatives: List[float] = []

        for className, embeddings in splitData.items():
            trueIndex = classToIndex[className]
            for embedding in embeddings:
                scores = prototypes @ embedding
                predIndex = int(np.argmax(scores))
                correct += int(predIndex == trueIndex)
                total += 1
                positives.append(float(scores[trueIndex]))
                wrongScores = np.delete(scores, trueIndex)
                if wrongScores.size:
                    negatives.append(float(np.max(wrongScores)))

        threshold = self.findBestCutoff(
            np.asarray(positives, dtype=np.float32),
            np.asarray(negatives, dtype=np.float32),
            defaultValue=0.40,
            minimum=confidenceThresholdFloor,
            maximum=confidenceThresholdCeil,
        )
        accuracy = 100.0 * correct / max(1, total)
        return threshold, accuracy

    def evaluateClosedSet(
        self,
        splitData: Dict[str, List[np.ndarray]],
        classNames: List[str],
        prototypes: np.ndarray,
    ) -> float:
        classToIndex = {name: index for index, name in enumerate(classNames)}
        correct = 0
        total = 0
        for className, embeddings in splitData.items():
            trueIndex = classToIndex[className]
            for embedding in embeddings:
                predIndex = int(np.argmax(prototypes @ embedding))
                correct += int(predIndex == trueIndex)
                total += 1
        return 100.0 * correct / max(1, total)

    def findBestCutoff(
        self,
        positiveValues: np.ndarray,
        negativeValues: np.ndarray,
        defaultValue: float,
        minimum: float,
        maximum: float,
    ) -> float:
        if positiveValues.size == 0 or negativeValues.size == 0:
            return defaultValue

        thresholds = np.linspace(minimum, maximum, num=91)
        bestThreshold = defaultValue
        bestScore = -1.0
        for threshold in thresholds:
            truePositiveRate = float(np.mean(positiveValues >= threshold))
            trueNegativeRate = float(np.mean(negativeValues < threshold))
            balancedAccuracy = 0.5 * (truePositiveRate + trueNegativeRate)
            if balancedAccuracy > bestScore:
                bestScore = balancedAccuracy
                bestThreshold = float(threshold)
        return bestThreshold

    def saveArtifacts(
        self,
        classNames: List[str],
        prototypes: np.ndarray,
        sampleEmbeddings: np.ndarray,
        sampleLabels: np.ndarray,
        threshold: float,
    ) -> None:
        faceIndexPath.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            faceIndexPath,
            class_names=np.asarray(classNames),
            prototypes=prototypes.astype(np.float32),
            sample_embeddings=sampleEmbeddings.astype(np.float32),
            sample_labels=sampleLabels.astype(np.int64),
        )

        with labelMapPath.open("w", encoding="utf-8") as file:
            json.dump({index: name for index, name in enumerate(classNames)}, file, indent=2)

        metadata = {
            "inference_mode": "insightface_embeddings",
            "backbone": MODEL_PACKAGE,
            "embedding_size": EMBEDDING_SIZE,
            "unknown_threshold": threshold,
        }
        with modelMetadataPath.open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)


def main() -> None:
    FaceTrainer().run()


if __name__ == "__main__":
    main()
