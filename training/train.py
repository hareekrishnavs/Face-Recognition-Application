import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.dataset import FaceDatasetPreprocessor
from training.dataset import FaceDatasetLoader
from training.model import FaceClassifierCNN
from utils.config import (
    confidenceThresholdCeil,
    confidenceThresholdFloor,
    datasetMetadataPath,
    defaultRecognitionMargin,
    defaultUnknownThreshold,
    deviceName,
    earlyStoppingPatience,
    faceIndexPath,
    imageSize,
    labelMapPath,
    labelSmoothing,
    learningRate,
    modelMetadataPath,
    modelSavePath,
    numEpochs,
    trainingSummaryPath,
    weightDecay,
)
EMBEDDING_SIZE = 512
MODEL_PACKAGE = "buffalo_l"


INSIGHTFACE_TRAINING_SUMMARY_PATH = trainingSummaryPath.with_name(
    "insightface_training_summary.json"
)
INSIGHTFACE_MODEL_METADATA_PATH = modelMetadataPath.with_name(
    "insightface_model_metadata.json"
)


def createInsightFaceBackend():
    try:
        from utils.insightface_backend import (
            EMBEDDING_SIZE as backendEmbeddingSize,
            MODEL_PACKAGE as backendModelPackage,
            InsightFaceBackend,
        )
    except ImportError as exc:
        raise ImportError(
            "InsightFace dependencies are not installed. "
            "Install insightface and onnxruntime before running InsightFace training."
        ) from exc

    global EMBEDDING_SIZE, MODEL_PACKAGE
    EMBEDDING_SIZE = backendEmbeddingSize
    MODEL_PACKAGE = backendModelPackage
    return InsightFaceBackend()


class FaceTrainer:
    def __init__(self) -> None:
        self.device = torch.device(deviceName)
        self.datasetLoader = FaceDatasetLoader()

    def run(self) -> None:
        self.ensureProcessedDataset()

        datasetsMap = self.datasetLoader.createDatasets()
        dataloadersMap = self.datasetLoader.createDataLoaders(datasetsMap)
        classNames = self.datasetLoader.getClassNames(datasetsMap["train"])
        numClasses = self.datasetLoader.getNumClasses(datasetsMap["train"])

        print(f"Classes found: {classNames}")
        print(f"Number of classes: {numClasses}")
        print(f"Using device: {self.device}")
        self.printDatasetSummary(datasetsMap)

        model = FaceClassifierCNN(numClasses=numClasses).to(self.device)
        criterion = nn.CrossEntropyLoss(label_smoothing=labelSmoothing)
        optimizer = optim.AdamW(model.parameters(), lr=learningRate, weight_decay=weightDecay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3,
        )

        bestValLoss = math.inf
        bestValAccuracy = 0.0
        bestThreshold = defaultUnknownThreshold
        bestMargin = defaultRecognitionMargin
        bestState = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        epochsWithoutImprovement = 0

        for epoch in range(numEpochs):
            print(f"\nEpoch {epoch + 1}/{numEpochs}")

            trainLoss, trainAccuracy, _, _, _ = self.runPhase(
                model=model,
                dataloader=dataloadersMap["train"],
                criterion=criterion,
                optimizer=optimizer,
                training=True,
            )
            valLoss, valAccuracy, _, _, valScores = self.runPhase(
                model=model,
                dataloader=dataloadersMap["val"],
                criterion=criterion,
                optimizer=None,
                training=False,
                collectScores=True,
            )

            print(f"Train Loss     : {trainLoss:.4f}")
            print(f"Train Accuracy : {trainAccuracy:.2f}%")
            print(f"Val Loss       : {valLoss:.4f}")
            print(f"Val Accuracy   : {valAccuracy:.2f}%")

            scheduler.step(valLoss)

            if valLoss < bestValLoss:
                bestValLoss = valLoss
                bestValAccuracy = valAccuracy
                bestState = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                bestThreshold, bestMargin = self.estimateUnknownRules(valScores)
                self.saveArtifacts(model, classNames, bestThreshold, bestMargin)
                epochsWithoutImprovement = 0
                print(
                    f"Best model updated. Unknown threshold: {bestThreshold:.3f}, margin: {bestMargin:.3f}"
                )
            else:
                epochsWithoutImprovement += 1

            print(f"Learning Rate  : {optimizer.param_groups[0]['lr']:.6f}")
            if epochsWithoutImprovement >= earlyStoppingPatience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

        model.load_state_dict(bestState)

        valLoss, valAccuracy, valLabels, valPredictions, valScores = self.runPhase(
            model=model,
            dataloader=dataloadersMap["val"],
            criterion=criterion,
            optimizer=None,
            training=False,
            collectScores=True,
        )
        testLoss, testAccuracy, testLabels, testPredictions, _ = self.runPhase(
            model=model,
            dataloader=dataloadersMap["test"],
            criterion=criterion,
            optimizer=None,
            training=False,
            collectScores=False,
        )

        bestThreshold, bestMargin = self.estimateUnknownRules(valScores)
        self.saveArtifacts(model, classNames, bestThreshold, bestMargin)

        self.printClassificationMetrics(valLabels, valPredictions, classNames, "Validation")
        self.printClassificationMetrics(testLabels, testPredictions, classNames, "Test")

        summary = {
            "class_names": classNames,
            "best_val_loss": bestValLoss,
            "best_val_accuracy": bestValAccuracy,
            "final_val_loss": valLoss,
            "final_val_accuracy": valAccuracy,
            "final_test_loss": testLoss,
            "final_test_accuracy": testAccuracy,
            "unknown_threshold": bestThreshold,
            "recognition_margin": bestMargin,
            "inference_mode": "classification",
        }
        trainingSummaryPath.parent.mkdir(parents=True, exist_ok=True)
        with trainingSummaryPath.open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2)

    def ensureProcessedDataset(self) -> None:
        if datasetMetadataPath.exists():
            return
        print("Processed dataset not found. Running preprocessing first.")
        FaceDatasetPreprocessor().run()

    def printDatasetSummary(self, datasetsMap: Dict[str, object]) -> None:
        print(f"Train samples: {len(datasetsMap['train'])}")
        print(f"Val samples: {len(datasetsMap['val'])}")
        print(f"Test samples: {len(datasetsMap['test'])}")

    def runPhase(
        self,
        model: FaceClassifierCNN,
        dataloader,
        criterion,
        optimizer,
        training: bool,
        collectScores: bool = False,
    ):
        model.train(mode=training)
        totalLoss = 0.0
        totalCorrect = 0
        totalSamples = 0
        allLabels: List[int] = []
        allPredictions: List[int] = []
        scoreRows: List[np.ndarray] = []

        for images, labels in dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            if training:
                assert optimizer is not None
                optimizer.zero_grad()

            with torch.set_grad_enabled(training):
                logits = model(images)
                loss = criterion(logits, labels)
                if training:
                    loss.backward()
                    optimizer.step()

            probabilities = F.softmax(logits, dim=1)
            predictions = logits.argmax(dim=1)

            totalLoss += float(loss.item()) * images.size(0)
            totalCorrect += int((predictions == labels).sum().item())
            totalSamples += int(labels.size(0))
            allLabels.extend(labels.detach().cpu().tolist())
            allPredictions.extend(predictions.detach().cpu().tolist())

            if collectScores:
                top2Values, _ = probabilities.topk(k=min(2, probabilities.shape[1]), dim=1)
                maxScores = top2Values[:, 0]
                margins = (
                    top2Values[:, 0] - top2Values[:, 1]
                    if top2Values.shape[1] > 1
                    else top2Values[:, 0]
                )
                correctness = (predictions == labels).float()
                batchRows = torch.stack([maxScores, margins, correctness], dim=1).detach().cpu().numpy()
                scoreRows.extend(batchRows)

        meanLoss = totalLoss / max(1, totalSamples)
        meanAccuracy = 100.0 * totalCorrect / max(1, totalSamples)
        if not collectScores:
            return meanLoss, meanAccuracy, allLabels, allPredictions, np.empty((0, 0))
        return meanLoss, meanAccuracy, allLabels, allPredictions, np.stack(scoreRows, axis=0)

    def estimateUnknownRules(self, scoreRows: np.ndarray) -> Tuple[float, float]:
        if scoreRows.size == 0:
            return defaultUnknownThreshold, defaultRecognitionMargin

        maxScores = scoreRows[:, 0]
        margins = scoreRows[:, 1]
        correctness = scoreRows[:, 2] > 0.5

        correctScores = maxScores[correctness]
        wrongScores = maxScores[~correctness]
        correctMargins = margins[correctness]
        wrongMargins = margins[~correctness]

        threshold = self._findBestCutoff(
            correctScores,
            wrongScores,
            defaultUnknownThreshold,
            minimum=confidenceThresholdFloor,
            maximum=confidenceThresholdCeil,
        )
        margin = self._findBestCutoff(
            correctMargins,
            wrongMargins,
            defaultRecognitionMargin,
            minimum=0.02,
            maximum=0.35,
        )
        return threshold, margin

    def _findBestCutoff(
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

    def printClassificationMetrics(
        self,
        trueLabels: List[int],
        predictedLabels: List[int],
        classNames: List[str],
        splitName: str,
    ) -> None:
        print(f"\n{splitName} classification report:")
        print(
            classification_report(
                trueLabels,
                predictedLabels,
                target_names=classNames,
                digits=4,
                zero_division=0,
            )
        )
        matrix = confusion_matrix(trueLabels, predictedLabels)
        print(f"{splitName} confusion matrix:")
        header = "true\\pred".ljust(16) + " ".join(name[:12].ljust(12) for name in classNames)
        print(header)
        for className, row in zip(classNames, matrix):
            print(f"{className[:12].ljust(16)}" + " ".join(str(value).ljust(12) for value in row))

    def saveArtifacts(
        self,
        model: FaceClassifierCNN,
        classNames: List[str],
        threshold: float,
        margin: float,
    ) -> None:
        modelSavePath.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), modelSavePath)

        with labelMapPath.open("w", encoding="utf-8") as file:
            json.dump({index: name for index, name in enumerate(classNames)}, file, indent=2)

        metadata = {
            "image_size": list(imageSize),
            "class_names": classNames,
            "inference_mode": "classification",
            "unknown_threshold": threshold,
            "recognition_margin": margin,
        }
        with modelMetadataPath.open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)


class InsightFaceEmbeddingTrainer:
    def __init__(self) -> None:
        self.backend = createInsightFaceBackend()
        self.datasetLoader = FaceDatasetLoader()

    def run(self) -> None:
        self.ensureProcessedDataset()

        datasetsMap = self.datasetLoader.createPathDatasets()
        trainData = self.loadSplitEmbeddings(datasetsMap["train"])
        valData = self.loadSplitEmbeddings(datasetsMap["val"])
        testData = self.loadSplitEmbeddings(datasetsMap["test"])

        classNames = self.datasetLoader.getClassNames(datasetsMap["train"])
        classNames = [name for name in classNames if trainData.get(name)]
        if not classNames:
            raise ValueError("No classes with usable embeddings found in processed train split.")

        classToIndex = {name: index for index, name in enumerate(classNames)}
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
        INSIGHTFACE_TRAINING_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with INSIGHTFACE_TRAINING_SUMMARY_PATH.open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2)

        print(f"Classes found: {classNames}")
        print(f"Validation accuracy: {valAccuracy:.2f}%")
        print(f"Test accuracy      : {testAccuracy:.2f}%")
        print(f"Unknown threshold  : {threshold:.3f}")
        print(f"Saved face index   : {faceIndexPath}")

    def ensureProcessedDataset(self) -> None:
        if datasetMetadataPath.exists():
            return
        print("Processed dataset not found. Running preprocessing first.")
        FaceDatasetPreprocessor().run()

    def loadSplitEmbeddings(self, dataset) -> Dict[str, List[np.ndarray]]:
        import cv2

        data: Dict[str, List[np.ndarray]] = defaultdict(list)
        classNames = self.datasetLoader.getClassNames(dataset)
        for imagePath, labelIndex in dataset.samples:
            image = cv2.imread(str(imagePath))
            if image is None:
                continue
            result = self.backend.extract_from_image(image, assume_aligned=True)
            if result is None:
                continue
            data[classNames[labelIndex]].append(result.embedding)
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
            if name not in classToIndex:
                continue
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
            if className not in classToIndex:
                continue
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
            defaultValue=defaultUnknownThreshold,
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
            if className not in classToIndex:
                continue
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

        metadata = {
            "inference_mode": "insightface_embeddings",
            "backbone": MODEL_PACKAGE,
            "embedding_size": EMBEDDING_SIZE,
            "unknown_threshold": threshold,
        }
        with INSIGHTFACE_MODEL_METADATA_PATH.open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)


def main() -> None:
    print("=== Training CNN classifier ===")
    FaceTrainer().run()

    print("\n=== Training InsightFace embedding index ===")
    InsightFaceEmbeddingTrainer().run()


if __name__ == "__main__":
    main()
