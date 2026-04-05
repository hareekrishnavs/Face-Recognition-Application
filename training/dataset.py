from typing import Dict, List, Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils.config import (
    batchSize,
    imageSize,
    numWorkers,
    testDir,
    trainDir,
    valDir,
)


class FaceDatasetLoader:
    def __init__(self) -> None:
        self.trainTransform = transforms.Compose([
            transforms.Resize(imageSize),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.02,
            ),
            transforms.RandomResizedCrop(imageSize, scale=(0.9, 1.0)),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.12)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self.featureTransform = transforms.Compose([
            transforms.Resize(imageSize),
            transforms.CenterCrop(imageSize),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self.evalTransform = transforms.Compose([
            transforms.Resize(imageSize),
            transforms.CenterCrop(imageSize),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def createDatasets(self) -> Dict[str, datasets.ImageFolder]:
        trainDataset = datasets.ImageFolder(trainDir, transform=self.trainTransform)
        valDataset = datasets.ImageFolder(valDir, transform=self.evalTransform)
        testDataset = datasets.ImageFolder(testDir, transform=self.evalTransform)

        return {
            "train": trainDataset,
            "val": valDataset,
            "test": testDataset,
        }

    def createFeatureDatasets(self) -> Dict[str, datasets.ImageFolder]:
        trainDataset = datasets.ImageFolder(trainDir, transform=self.featureTransform)
        valDataset = datasets.ImageFolder(valDir, transform=self.featureTransform)
        testDataset = datasets.ImageFolder(testDir, transform=self.featureTransform)

        return {
            "train": trainDataset,
            "val": valDataset,
            "test": testDataset,
        }

    def createDataLoaders(
        self,
        datasetsMap: Dict[str, datasets.ImageFolder],
    ) -> Dict[str, DataLoader]:
        trainLoader = DataLoader(
            datasetsMap["train"],
            batch_size=batchSize,
            shuffle=True,
            num_workers=numWorkers,
        )

        valLoader = DataLoader(
            datasetsMap["val"],
            batch_size=batchSize,
            shuffle=False,
            num_workers=numWorkers,
        )

        testLoader = DataLoader(
            datasetsMap["test"],
            batch_size=batchSize,
            shuffle=False,
            num_workers=numWorkers,
        )

        return {
            "train": trainLoader,
            "val": valLoader,
            "test": testLoader,
        }

    def getClassNames(self, trainDataset: datasets.ImageFolder) -> List[str]:
        return trainDataset.classes

    def getNumClasses(self, trainDataset: datasets.ImageFolder) -> int:
        return len(trainDataset.classes)
