import json
import math
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from pillow_heif import register_heif_opener

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import (
    augmentationPlan,
    datasetMetadataPath,
    faceCropMargin,
    imageSize,
    minFaceSize,
    outputImageFormat,
    processedDir,
    randomSeed,
    rawDir,
    splitDirs,
    splitRatios,
    supportedExtensions,
)

register_heif_opener()


@dataclass(frozen=True)
class ImageRecord:
    personName: str
    sourcePath: Path
    sourceStem: str


@dataclass
class CroppedFaceRecord:
    personName: str
    sourcePath: Path
    sourceStem: str
    faceImage: Image.Image


class FaceDatasetPreprocessor:
    def __init__(self) -> None:
        self.frontFaceDetector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.altFaceDetector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
        )
        self.profileFaceDetector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_profileface.xml"
        )
        self.rng = np.random.default_rng(randomSeed)

    def run(self) -> Dict[str, object]:
        rawRecords = self._collect_records()
        croppedRecords, skippedCounts = self._extract_faces(rawRecords)
        splits = self._create_split_map(croppedRecords)

        if processedDir.exists():
            shutil.rmtree(processedDir)
        processedDir.mkdir(parents=True, exist_ok=True)

        summary: Dict[str, Dict[str, Dict[str, int]]] = {
            "raw": defaultdict(dict),
            "processed": defaultdict(lambda: defaultdict(int)),
        }

        for personName, personRecords in rawRecords.items():
            summary["raw"][personName]["count"] = len(personRecords)

        for splitName, splitRecords in splits.items():
            for record in splitRecords:
                saved = self._save_base_image(splitName, record)
                if saved is None:
                    continue

                personSummary = summary["processed"][record.personName]
                personSummary[f"{splitName}_original"] += 1

                if splitName == "train":
                    augmentedCount = self._save_augmented_images(saved, record.personName)
                    personSummary["train_augmented"] += augmentedCount

        manifest = {
            "image_size": list(imageSize),
            "split_ratios": splitRatios,
            "augmentation_plan": augmentationPlan,
            "splits": {
                splitName: self._count_split_images(splitDir)
                for splitName, splitDir in splitDirs.items()
            },
            "per_person": {
                personName: {
                    "raw_count": summary["raw"][personName].get("count", 0),
                    "cropped_count": len(croppedRecords.get(personName, [])),
                    "skipped_no_face": skippedCounts.get(personName, 0),
                    "train_original": summary["processed"][personName].get("train_original", 0),
                    "train_augmented": summary["processed"][personName].get("train_augmented", 0),
                    "val_count": summary["processed"][personName].get("val_original", 0),
                    "test_count": summary["processed"][personName].get("test_original", 0),
                }
                for personName in sorted(rawRecords)
            },
        }

        datasetMetadataPath.parent.mkdir(parents=True, exist_ok=True)
        with datasetMetadataPath.open("w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2, sort_keys=True)

        return manifest

    def _collect_records(self) -> Dict[str, List[ImageRecord]]:
        if not rawDir.exists():
            raise FileNotFoundError(f"Raw dataset directory not found: {rawDir}")

        records: Dict[str, List[ImageRecord]] = {}

        for personDir in sorted(path for path in rawDir.iterdir() if path.is_dir()):
            personRecords: List[ImageRecord] = []
            for imagePath in sorted(path for path in personDir.iterdir() if path.is_file()):
                if imagePath.suffix.lower() not in supportedExtensions and imagePath.suffix:
                    continue
                personRecords.append(
                    ImageRecord(
                        personName=personDir.name.strip(),
                        sourcePath=imagePath,
                        sourceStem=self._normalise_stem(imagePath),
                    )
                )

            if len(personRecords) < 3:
                raise ValueError(
                    f"Class '{personDir.name}' needs at least 3 images for train/val/test splits."
                )

            records[personDir.name.strip()] = personRecords

        if not records:
            raise ValueError(f"No usable images found in {rawDir}")

        return records

    def _extract_faces(
        self, records: Dict[str, List[ImageRecord]]
    ) -> Tuple[Dict[str, List[CroppedFaceRecord]], Dict[str, int]]:
        croppedRecords: Dict[str, List[CroppedFaceRecord]] = {}
        skippedCounts: Dict[str, int] = {}

        for personName, personRecords in sorted(records.items()):
            croppedPersonRecords: List[CroppedFaceRecord] = []
            skipped = 0

            for record in personRecords:
                image = self._read_image(record.sourcePath)
                if image is None:
                    skipped += 1
                    continue

                faceImage = self._extract_face(image)
                if faceImage is None:
                    skipped += 1
                    continue

                croppedPersonRecords.append(
                    CroppedFaceRecord(
                        personName=record.personName,
                        sourcePath=record.sourcePath,
                        sourceStem=record.sourceStem,
                        faceImage=faceImage,
                    )
                )

            if len(croppedPersonRecords) < 3:
                raise ValueError(
                    f"Class '{personName}' has only {len(croppedPersonRecords)} usable face crops after filtering."
                )

            croppedRecords[personName] = croppedPersonRecords
            skippedCounts[personName] = skipped

        return croppedRecords, skippedCounts

    def _create_split_map(
        self, records: Dict[str, List[CroppedFaceRecord]]
    ) -> Dict[str, List[CroppedFaceRecord]]:
        splitMap: Dict[str, List[CroppedFaceRecord]] = {"train": [], "val": [], "test": []}

        for personName, personRecords in sorted(records.items()):
            shuffled = list(personRecords)
            self.rng.shuffle(shuffled)

            total = len(shuffled)
            testCount = max(1, int(round(total * splitRatios["test"])))
            valCount = max(1, int(round(total * splitRatios["val"])))
            trainCount = total - valCount - testCount

            while trainCount < 2:
                if valCount > testCount and valCount > 1:
                    valCount -= 1
                elif testCount > 1:
                    testCount -= 1
                else:
                    raise ValueError(
                        f"Not enough images for a valid split in class '{personName}'."
                    )
                trainCount = total - valCount - testCount

            splitMap["train"].extend(shuffled[:trainCount])
            splitMap["val"].extend(shuffled[trainCount : trainCount + valCount])
            splitMap["test"].extend(shuffled[trainCount + valCount :])

        return splitMap

    def _save_base_image(self, splitName: str, record: CroppedFaceRecord) -> Optional[Path]:
        destinationDir = splitDirs[splitName] / record.personName
        destinationDir.mkdir(parents=True, exist_ok=True)
        destinationPath = destinationDir / f"{record.sourceStem}.{outputImageFormat}"
        record.faceImage.save(destinationPath, quality=95)
        return destinationPath

    def _save_augmented_images(self, imagePath: Path, personName: str) -> int:
        base = Image.open(imagePath).convert("RGB")
        destinationDir = splitDirs["train"] / personName
        baseStem = imagePath.stem
        savedCount = 0

        augmentedImages = self._generate_augmentations(base)
        for augmentName, image in augmentedImages:
            destinationPath = destinationDir / f"{baseStem}__{augmentName}.{outputImageFormat}"
            image.save(destinationPath, quality=95)
            savedCount += 1

        return savedCount

    def _read_image(self, imagePath: Path) -> Optional[Image.Image]:
        try:
            image = Image.open(imagePath)
            image = ImageOps.exif_transpose(image).convert("RGB")
        except Exception:
            return None
        return image

    def _extract_face(self, image: Image.Image) -> Optional[Image.Image]:
        rgb = np.array(image)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)
        faces = self._detect_faces(equalized)
        if not faces:
            faces = self._detect_faces(gray)
        if not faces:
            return None

        x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
        face = self._crop_with_margin(rgb, x, y, w, h)
        resized = cv2.resize(face, imageSize, interpolation=cv2.INTER_AREA)
        return Image.fromarray(resized)

    def _detect_faces(self, gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
        detections: List[Tuple[int, int, int, int]] = []

        for detector, isProfile in (
            (self.frontFaceDetector, False),
            (self.altFaceDetector, False),
            (self.profileFaceDetector, True),
        ):
            if detector.empty():
                continue

            faces = detector.detectMultiScale(
                gray,
                scaleFactor=1.08,
                minNeighbors=5 if not isProfile else 4,
                minSize=(minFaceSize, minFaceSize),
            )
            detections.extend((int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces)

            if isProfile:
                flipped = cv2.flip(gray, 1)
                flippedFaces = detector.detectMultiScale(
                    flipped,
                    scaleFactor=1.08,
                    minNeighbors=4,
                    minSize=(minFaceSize, minFaceSize),
                )
                width = gray.shape[1]
                for (x, y, w, h) in flippedFaces:
                    detections.append((int(width - x - w), int(y), int(w), int(h)))

        return detections

    def _crop_with_margin(
        self, image: np.ndarray, x: int, y: int, w: int, h: int
    ) -> np.ndarray:
        imgH, imgW = image.shape[:2]
        padX = int(math.ceil(w * faceCropMargin))
        padY = int(math.ceil(h * faceCropMargin))

        x1 = max(0, x - padX)
        y1 = max(0, y - padY)
        x2 = min(imgW, x + w + padX)
        y2 = min(imgH, y + h + padY)
        crop = image[y1:y2, x1:x2]
        return self._square_center_crop(crop)

    def _square_center_crop(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        size = min(height, width)
        startX = max(0, (width - size) // 2)
        startY = max(0, (height - size) // 2)
        return image[startY : startY + size, startX : startX + size]

    def _generate_augmentations(self, image: Image.Image) -> List[Tuple[str, Image.Image]]:
        augmentations: List[Tuple[str, Image.Image]] = []

        operations = {
            "rotate_left": lambda img: img.rotate(-12, resample=Image.BILINEAR, fillcolor=(0, 0, 0)),
            "rotate_right": lambda img: img.rotate(12, resample=Image.BILINEAR, fillcolor=(0, 0, 0)),
            "scale_in": lambda img: self._scale_image(img, scale=1.08),
            "scale_out": lambda img: self._scale_image(img, scale=0.92),
            "gaussian_noise": self._add_gaussian_noise,
            "gaussian_blur": lambda img: img.filter(ImageFilter.GaussianBlur(radius=1.1)),
            "brightness": lambda img: ImageEnhance.Brightness(img).enhance(1.16),
            "contrast": lambda img: ImageEnhance.Contrast(img).enhance(1.18),
            "horizontal_flip": ImageOps.mirror,
        }

        for name, repeats in augmentationPlan.items():
            operation = operations[name]
            for index in range(repeats):
                augmented = operation(image.copy()).resize(imageSize)
                suffix = name if repeats == 1 else f"{name}_{index + 1}"
                augmentations.append((suffix, augmented))

        return augmentations

    def _scale_image(self, image: Image.Image, scale: float) -> Image.Image:
        width, height = image.size
        scaledW = max(8, int(round(width * scale)))
        scaledH = max(8, int(round(height * scale)))
        scaled = image.resize((scaledW, scaledH), resample=Image.BILINEAR)

        if scale >= 1.0:
            left = max(0, (scaledW - width) // 2)
            top = max(0, (scaledH - height) // 2)
            return scaled.crop((left, top, left + width, top + height))

        canvas = Image.new("RGB", (width, height))
        left = max(0, (width - scaledW) // 2)
        top = max(0, (height - scaledH) // 2)
        canvas.paste(scaled, (left, top))
        return canvas

    def _add_gaussian_noise(self, image: Image.Image) -> Image.Image:
        array = np.asarray(image).astype(np.float32)
        noisy = array + self.rng.normal(0.0, 8.0, size=array.shape)
        return Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8))

    def _count_split_images(self, splitDir: Path) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        if not splitDir.exists():
            return counts

        for personDir in sorted(path for path in splitDir.iterdir() if path.is_dir()):
            counts[personDir.name] = len([path for path in personDir.iterdir() if path.is_file()])
        return counts

    def _count_person_images(self, splitName: str, personName: str) -> int:
        personDir = splitDirs[splitName] / personName
        if not personDir.exists():
            return 0
        return len([path for path in personDir.iterdir() if path.is_file()])

    def _normalise_stem(self, imagePath: Path) -> str:
        stem = imagePath.stem or imagePath.name
        cleaned = "".join(char if char.isalnum() else "_" for char in stem).strip("_")
        return cleaned.lower() or "image"


def main() -> None:
    summary = FaceDatasetPreprocessor().run()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
