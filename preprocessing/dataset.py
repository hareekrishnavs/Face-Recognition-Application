import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN
from PIL import Image, ImageEnhance
from pillow_heif import register_heif_opener
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from utils.config import (
    augmentationCounts,
    deviceName,
    imageSize,
    mtcnnMargin,
    mtcnnMinFaceSize,
    outputImageFormat,
    processedDir,
    randomSeed,
    rawDir,
    splitRatios,
    supportedExtensions,
    testDir,
    trainDir,
    valDir,
)


register_heif_opener()


@dataclass
class ImageRecord:
    imagePath: Path
    personName: str
    imageNumber: int


@dataclass
class ProcessedImageRecord:
    imageArray: np.ndarray
    imageName: str


class FacePreprocessor:
    def __init__(self) -> None:
        self.setSeeds()
        self.device = self.getDevice()
        self.faceDetector = self.createFaceDetector()

    def setSeeds(self) -> None:
        random.seed(randomSeed)
        np.random.seed(randomSeed)
        torch.manual_seed(randomSeed)

    def getDevice(self) -> torch.device:
        return torch.device(deviceName)

    def createFaceDetector(self) -> MTCNN:
        return MTCNN(keep_all=False, device=self.device, margin=mtcnnMargin, min_face_size=mtcnnMinFaceSize, post_process=False,)

    def run(self) -> None:
        self.prepareOutputDirectories()
        personDirectories = self.getPersonDirectories()

        print("\nStarting preprocessing...")
        print(f"Using device: {self.device}")
        print(f"Raw directory: {rawDir}")
        print(f"Processed directory: {processedDir}")

        for personDirectory in personDirectories:
            self.processPersonDirectory(personDirectory)

        print("\nPreprocessing completed successfully.")

    def processPersonDirectory(self, personDirectory: Path) -> None:
        personName = personDirectory.name
        print("\n" + "=" * 60)
        print(f"Processing person: {personName}")
        print("=" * 60)

        imagePaths = self.getImagePaths(personDirectory)
        imageRecords = self.buildImageRecords(imagePaths, personName)

        if not imageRecords:
            print("No valid images found.")
            return

        processedOriginals = self.processPersonImages(imageRecords)

        if not processedOriginals:
            print("No valid face crops found. Skipping this person.")
            return

        splitMap = self.splitImages(processedOriginals)
        self.printSplitSummary(splitMap, title="Before augmentation")

        splitMap["train"] = self.augmentTrainImages(splitMap["train"])
        self.printAugmentationSummary(splitMap)

        self.saveSplitImages(personName, splitMap)
        self.printSavedSummary(personName, splitMap)

    def prepareOutputDirectories(self) -> None:
        if processedDir.exists():
            shutil.rmtree(processedDir)

        for splitDirectory in [trainDir, valDir, testDir]:
            splitDirectory.mkdir(parents=True, exist_ok=True)

    def getPersonDirectories(self) -> List[Path]:
        if not rawDir.exists():
            raise FileNotFoundError(f"Raw directory not found: {rawDir}")

        return sorted([path for path in rawDir.iterdir() if path.is_dir()])

    def getImagePaths(self, personDirectory: Path) -> List[Path]:
        imagePaths = []

        for imagePath in personDirectory.rglob("*"):
            if imagePath.is_file() and imagePath.suffix.lower() in supportedExtensions:
                imagePaths.append(imagePath)

        return sorted(imagePaths)

    def buildImageRecords(self, imagePaths: List[Path], personName: str) -> List[ImageRecord]:
        imageRecords = []

        for index, imagePath in enumerate(imagePaths, start=1):
            imageRecords.append(ImageRecord(imagePath=imagePath, personName=personName, imageNumber=index,))

        return imageRecords

    def processPersonImages(self, imageRecords: List[ImageRecord]) -> List[ProcessedImageRecord]:
        processedImages = []
        personName = imageRecords[0].personName if imageRecords else "Unknown"

        totalImages = len(imageRecords)
        successCount = 0
        failedCount = 0

        for imageRecord in tqdm(imageRecords, desc=f"Images for {personName}"):
            imageArray = self.convertImageFormat(imageRecord.imagePath)
            if imageArray is None:
                failedCount += 1
                continue

            faceImage = self.detectAndCropFace(imageArray)
            if faceImage is None:
                print(f"No face detected in {imageRecord.imagePath.name}")
                failedCount += 1
                continue

            resizedImage = self.resizeImage(faceImage)
            baseName = self.buildBaseName(imageRecord.imageNumber)

            processedImages.append(ProcessedImageRecord(imageArray=resizedImage,imageName=baseName,))
            successCount += 1

        self.printDetectionSummary(personName=personName, totalImages=totalImages, successCount=successCount, failedCount=failedCount,)

        return processedImages

    def convertImageFormat(self, imagePath: Path) -> Optional[np.ndarray]:
        try:
            pilImage = Image.open(imagePath).convert("RGB")
            imageArray = np.array(pilImage)
            return imageArray
        except Exception as error:
            print(f"Could not read image: {imagePath.name} | Error: {error}")
            return None

    def detectAndCropFace(self, imageArray: np.ndarray) -> Optional[np.ndarray]:
        try:
            boxes, probabilities = self.faceDetector.detect(imageArray)
        except Exception as error:
            print(f"Face detection failed: {error}")
            return None

        if boxes is None or len(boxes) == 0:
            return None

        # Since keep_all=False, boxes is a single box
        box = boxes[0]
        probability = probabilities[0] if probabilities is not None else None

        # Accept any detection
        # if probability is not None and probability < 0.6:
        #     return None

        x1, y1, x2, y2 = box
        height, width = imageArray.shape[:2]

        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(width, int(x2))
        y2 = min(height, int(y2))

        if x2 <= x1 or y2 <= y1:
            return None

        croppedFace = imageArray[y1:y2, x1:x2]

        if croppedFace.size == 0:
            return None

        return croppedFace

    def resizeImage(self, imageArray: np.ndarray) -> np.ndarray:
        return cv2.resize(imageArray, imageSize, interpolation=cv2.INTER_AREA)

    def buildBaseName(self, imageNumber: int) -> str:
        return f"image_{imageNumber:04d}"

    def splitImages(self,processedImages: List[ProcessedImageRecord],) -> Dict[str, List[ProcessedImageRecord]]:
        if len(processedImages) < 5:
            raise ValueError(
                "Need at least 5 valid processed images per person to split into train, val, and test."
            )

        trainImages, tempImages = train_test_split(processedImages, train_size=splitRatios["train"], random_state=randomSeed, shuffle=True,)

        valRatioWithinTemp = splitRatios["val"] / (splitRatios["val"] + splitRatios["test"])

        valImages, testImages = train_test_split(tempImages, train_size=valRatioWithinTemp, random_state=randomSeed, shuffle=True,)

        return {"train": trainImages, "val": valImages, "test": testImages,}

    def augmentTrainImages(self, trainImages: List[ProcessedImageRecord],) -> List[ProcessedImageRecord]:
        augmentedTrainImages = []

        for imageRecord in trainImages:
            augmentedTrainImages.append(imageRecord)

            augmentedVariants = self.augmentData(imageArray=imageRecord.imageArray,baseName=imageRecord.imageName,)
            augmentedTrainImages.extend(augmentedVariants)

        return augmentedTrainImages

    def augmentData(self,imageArray: np.ndarray, baseName: str,) -> List[ProcessedImageRecord]:
        augmentedImages = []
        augmentedImages.extend(self.createNoiseVariants(imageArray, baseName))
        augmentedImages.extend(self.createRotateVariants(imageArray, baseName))
        augmentedImages.extend(self.createFlipVariants(imageArray, baseName))
        augmentedImages.extend(self.createScaleVariants(imageArray, baseName))
        augmentedImages.extend(self.createBrightnessVariants(imageArray, baseName))
        return augmentedImages

    def createNoiseVariants( self, imageArray: np.ndarray, baseName: str,) -> List[ProcessedImageRecord]:
        variants = []

        for index in range(1, augmentationCounts["noise"] + 1):
            noise = np.random.normal(0, 10, imageArray.shape).astype(np.float32)
            noisyImage = np.clip(imageArray.astype(np.float32) + noise, 0, 255).astype(np.uint8)

            variants.append(ProcessedImageRecord(imageArray=noisyImage,imageName=f"{baseName}_noise({index})",))

        return variants

    def createRotateVariants(self, imageArray: np.ndarray, baseName: str,) -> List[ProcessedImageRecord]:
        variants = []
        rotationAngles = [-12, 12]

        for index, angle in enumerate(rotationAngles[:augmentationCounts["rotate"]], start=1):
            height, width = imageArray.shape[:2]
            center = (width // 2, height // 2)

            rotationMatrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotatedImage = cv2.warpAffine(imageArray, rotationMatrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,)

            variants.append(ProcessedImageRecord(imageArray=rotatedImage,imageName=f"{baseName}_rotate({index})",))

        return variants

    def createFlipVariants(self,imageArray: np.ndarray,baseName: str,) -> List[ProcessedImageRecord]:
        variants = []
        flipCodes = [1, 0]

        for index, flipCode in enumerate(flipCodes[:augmentationCounts["flip"]], start=1):
            flippedImage = cv2.flip(imageArray, flipCode)

            variants.append(ProcessedImageRecord(imageArray=flippedImage,imageName=f"{baseName}_flip({index})",))

        return variants

    def createScaleVariants(self,imageArray: np.ndarray,baseName: str,) -> List[ProcessedImageRecord]:
        variants = []
        scaleFactors = [0.90, 1.10]

        for index, scaleFactor in enumerate(scaleFactors[:augmentationCounts["scale"]], start=1):
            scaledImage = self.scaleImageKeepSize(imageArray, scaleFactor)

            variants.append(ProcessedImageRecord(imageArray=scaledImage,imageName=f"{baseName}_scale({index})",))

        return variants

    def createBrightnessVariants(self,imageArray: np.ndarray,baseName: str,) -> List[ProcessedImageRecord]:
        variants = []
        brightnessFactors = [0.75, 1.25]
        pilImage = Image.fromarray(imageArray)

        for index, factor in enumerate(
            brightnessFactors[:augmentationCounts["brightness"]],
            start=1,
        ):
            enhancer = ImageEnhance.Brightness(pilImage)
            adjustedImage = np.array(enhancer.enhance(factor))

            variants.append(ProcessedImageRecord(imageArray=adjustedImage,imageName=f"{baseName}_brightness({index})",))

        return variants

    def scaleImageKeepSize(self, imageArray: np.ndarray, scaleFactor: float) -> np.ndarray:
        height, width = imageArray.shape[:2]

        scaledWidth = max(1, int(width * scaleFactor))
        scaledHeight = max(1, int(height * scaleFactor))

        resizedImage = cv2.resize(
            imageArray,
            (scaledWidth, scaledHeight),
            interpolation=cv2.INTER_LINEAR,
        )

        if scaleFactor >= 1.0:
            startX = (scaledWidth - width) // 2
            startY = (scaledHeight - height) // 2
            croppedImage = resizedImage[startY:startY + height, startX:startX + width]
            return croppedImage

        canvas = np.zeros_like(imageArray)
        startX = (width - scaledWidth) // 2
        startY = (height - scaledHeight) // 2
        canvas[startY:startY + scaledHeight, startX:startX + scaledWidth] = resizedImage
        return canvas

    def saveSplitImages(self,personName: str,splitMap: Dict[str, List[ProcessedImageRecord]],) -> None:
        splitDirectoryMap = {"train": trainDir, "val": valDir, "test": testDir,}

        for splitName, imageRecords in splitMap.items():
            personOutputDir = splitDirectoryMap[splitName] / personName
            personOutputDir.mkdir(parents=True, exist_ok=True)

            for imageRecord in imageRecords:
                outputPath = personOutputDir / f"{imageRecord.imageName}.{outputImageFormat}"
                self.saveImage(imageRecord.imageArray, outputPath)

    def saveImage(self, imageArray: np.ndarray, outputPath: Path) -> None:
        pilImage = Image.fromarray(imageArray)

        formatMap = {"jpg": "JPEG","jpeg": "JPEG","png": "PNG",}

        formatName = formatMap.get(outputImageFormat.lower(), outputImageFormat.upper())
        pilImage.save(outputPath, format=formatName, quality=95)

    def printDetectionSummary(self, personName: str, totalImages: int, successCount: int, failedCount: int,) -> None:
        successRate = (successCount / totalImages * 100.0) if totalImages > 0 else 0.0

        print(f"\nSelection summary for {personName}:")
        print(f"  Total images found   : {totalImages}")
        print(f"  Faces selected       : {successCount}")
        print(f"  Images skipped       : {failedCount}")
        print(f"  Selection rate       : {successRate:.2f}%")

    def printSplitSummary(self, splitMap: Dict[str, List[ProcessedImageRecord]], title: str,) -> None:
        print(f"\n{title}:")
        print(f"  Train originals      : {len(splitMap['train'])}")
        print(f"  Val originals        : {len(splitMap['val'])}")
        print(f"  Test originals       : {len(splitMap['test'])}")

    def printAugmentationSummary(self, splitMap: Dict[str, List[ProcessedImageRecord]],) -> None:
        print("\nAfter augmentation:")
        print(f"  Train total          : {len(splitMap['train'])}")
        print(f"  Val total            : {len(splitMap['val'])}")
        print(f"  Test total           : {len(splitMap['test'])}")

    def printSavedSummary(self, personName: str, splitMap: Dict[str, List[ProcessedImageRecord]],) -> None:
        totalSaved = (
            len(splitMap["train"]) +
            len(splitMap["val"]) +
            len(splitMap["test"])
        )

        print(f"\nSaved output for {personName}:")
        print(f"  Train saved          : {len(splitMap['train'])}")
        print(f"  Val saved            : {len(splitMap['val'])}")
        print(f"  Test saved           : {len(splitMap['test'])}")
        print(f"  Total saved          : {totalSaved}")


def main() -> None:
    facePreprocessor = FacePreprocessor()
    facePreprocessor.run()


if __name__ == "__main__":
    main()