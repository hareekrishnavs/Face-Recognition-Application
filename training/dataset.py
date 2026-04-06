from typing import Dict, List

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils.config import batchSize, imageSize, numWorkers, splitDirs


class FaceDatasetLoader:
    def __init__(self) -> None:
        self.trainTransform = transforms.Compose(
            [
                transforms.Resize(imageSize),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )
        self.evalTransform = transforms.Compose(
            [
                transforms.Resize(imageSize),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    def createDatasets(self) -> Dict[str, datasets.ImageFolder]:
        return {
            "train": datasets.ImageFolder(splitDirs["train"], transform=self.trainTransform),
            "val": datasets.ImageFolder(splitDirs["val"], transform=self.evalTransform),
            "test": datasets.ImageFolder(splitDirs["test"], transform=self.evalTransform),
        }

    def createDataLoaders(
        self, datasetsMap: Dict[str, datasets.ImageFolder]
    ) -> Dict[str, DataLoader]:
        return {
            "train": DataLoader(
                datasetsMap["train"],
                batch_size=batchSize,
                shuffle=True,
                num_workers=numWorkers,
            ),
            "val": DataLoader(
                datasetsMap["val"],
                batch_size=batchSize,
                shuffle=False,
                num_workers=numWorkers,
            ),
            "test": DataLoader(
                datasetsMap["test"],
                batch_size=batchSize,
                shuffle=False,
                num_workers=numWorkers,
            ),
        }

    def getClassNames(self, dataset: datasets.ImageFolder) -> List[str]:
        return list(dataset.classes)

    def getNumClasses(self, dataset: datasets.ImageFolder) -> int:
        return len(dataset.classes)
