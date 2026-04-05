import json
import copy
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix

from training.dataset import FaceDatasetLoader
from training.model import SmallCNNClassifier
from utils.config import (
    deviceName,
    earlyStoppingPatience,
    labelMapPath,
    labelSmoothing,
    learningRate,
    modelSavePath,
    numEpochs,
    weightDecay,
)


class FaceTrainer:
    def __init__(self) -> None:
        self.device = torch.device(deviceName)
        self.datasetLoader = FaceDatasetLoader()

    def run(self) -> None:
        datasetsMap = self.datasetLoader.createDatasets()
        dataloadersMap = self.datasetLoader.createDataLoaders(datasetsMap)

        classNames = self.datasetLoader.getClassNames(datasetsMap["train"])
        numClasses = self.datasetLoader.getNumClasses(datasetsMap["train"])

        print(f"Classes found: {classNames}")
        print(f"Number of classes: {numClasses}")
        print(f"Using device: {self.device}")
        self.printDatasetSummary(datasetsMap)

        model = SmallCNNClassifier(numClasses=numClasses).createModel().to(self.device)
        criterion = nn.CrossEntropyLoss(label_smoothing=labelSmoothing)
        optimizer = optim.Adam(model.parameters(), lr=learningRate, weight_decay=weightDecay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=4,
        )

        bestValLoss = float("inf")
        bestValAccuracy = 0.0
        bestModelState = copy.deepcopy(model.state_dict())
        epochsWithoutImprovement = 0

        for epoch in range(numEpochs):
            print(f"\nEpoch {epoch + 1}/{numEpochs}")

            trainLoss, trainAccuracy = self.trainOneEpoch(
                model=model,
                dataloader=dataloadersMap["train"],
                criterion=criterion,
                optimizer=optimizer,
            )

            valLoss, valAccuracy = self.evaluate(
                model=model,
                dataloader=dataloadersMap["val"],
                criterion=criterion,
            )

            print(f"Train Loss     : {trainLoss:.4f}")
            print(f"Train Accuracy : {trainAccuracy:.2f}%")
            print(f"Val Loss       : {valLoss:.4f}")
            print(f"Val Accuracy   : {valAccuracy:.2f}%")

            scheduler.step(valLoss)

            if valLoss < bestValLoss:
                bestValLoss = valLoss
                bestValAccuracy = valAccuracy
                bestModelState = copy.deepcopy(model.state_dict())
                self.saveModel(model, classNames)
                epochsWithoutImprovement = 0
                print("Best model updated.")
            else:
                epochsWithoutImprovement += 1

            currentLearningRate = optimizer.param_groups[0]["lr"]
            print(f"Learning Rate  : {currentLearningRate:.6f}")

            if epochsWithoutImprovement >= earlyStoppingPatience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

        model.load_state_dict(bestModelState)

        valLoss, valAccuracy, valLabels, valPredictions = self.evaluate(
            model=model,
            dataloader=dataloadersMap["val"],
            criterion=criterion,
            returnPredictions=True,
        )
        testLoss, testAccuracy, testLabels, testPredictions = self.evaluate(
            model=model,
            dataloader=dataloadersMap["test"],
            criterion=criterion,
            returnPredictions=True,
        )

        print("\nTraining completed.")
        print(f"Best validation loss: {bestValLoss:.4f}")
        print(f"Best validation accuracy: {bestValAccuracy:.2f}%")
        print(f"Validation loss: {valLoss:.4f}")
        print(f"Validation accuracy: {valAccuracy:.2f}%")
        print(f"Test loss: {testLoss:.4f}")
        print(f"Test accuracy: {testAccuracy:.2f}%")
        self.printClassificationMetrics(valLabels, valPredictions, classNames, splitName="Validation")
        self.printClassificationMetrics(testLabels, testPredictions, classNames, splitName="Test")

    def printDatasetSummary(self, datasetsMap: Dict[str, object]) -> None:
        print(f"Train samples: {len(datasetsMap['train'])}")
        print(f"Val samples: {len(datasetsMap['val'])}")
        print(f"Test samples: {len(datasetsMap['test'])}")

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
            rowValues = " ".join(str(value).ljust(12) for value in row)
            print(f"{className[:12].ljust(16)}{rowValues}")

    def trainOneEpoch(
        self,
        model: nn.Module,
        dataloader,
        criterion,
        optimizer,
    ) -> tuple:
        model.train()

        runningLoss = 0.0
        correctPredictions = 0
        totalSamples = 0

        for images, labels in dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            runningLoss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)
            correctPredictions += (predictions == labels).sum().item()
            totalSamples += labels.size(0)

        epochLoss = runningLoss / totalSamples
        epochAccuracy = 100.0 * correctPredictions / totalSamples
        return epochLoss, epochAccuracy

    def evaluate(
        self,
        model: nn.Module,
        dataloader,
        criterion,
        returnPredictions: bool = False,
    ) -> tuple:
        model.eval()

        runningLoss = 0.0
        correctPredictions = 0
        totalSamples = 0
        allLabels: List[int] = []
        allPredictions: List[int] = []

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                runningLoss += loss.item() * images.size(0)
                predictions = outputs.argmax(dim=1)
                correctPredictions += (predictions == labels).sum().item()
                totalSamples += labels.size(0)
                allLabels.extend(labels.cpu().tolist())
                allPredictions.extend(predictions.cpu().tolist())

        epochLoss = runningLoss / totalSamples
        epochAccuracy = 100.0 * correctPredictions / totalSamples

        if returnPredictions:
            return epochLoss, epochAccuracy, allLabels, allPredictions

        return epochLoss, epochAccuracy

    def saveModel(self, model: nn.Module, classNames: List[str]) -> None:
        modelSavePath.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), modelSavePath)

        labelMap = {index: className for index, className in enumerate(classNames)}
        with open(labelMapPath, "w", encoding="utf-8") as file:
            json.dump(labelMap, file, indent=4)


def main() -> None:
    faceTrainer = FaceTrainer()
    faceTrainer.run()


if __name__ == "__main__":
    main()
