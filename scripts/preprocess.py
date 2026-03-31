import os
import cv2
import random
import numpy as np
from pathlib import Path
from insightface.app import FaceAnalysis

inputDir = "dataset/jpgRaw"
outputDir = "dataset/processed"

imgSize = (224, 224)

augmentationsConfig = {
    "rotate": True,
    "flip": True,
    "brightness": True,
    "blur": True,
    "noise": True
}

splitConfig = {
    "enabled": True,
    "train": 0.7,
    "val": 0.15,
    "test": 0.15
}

# Initialize RetinaFace (InsightFace)
faceApp = FaceAnalysis(name="buffalo_l")
faceApp.prepare(ctx_id=-1)  # -1 = CPU, 0 = GPU



def detectAlignCrop(image):
    faces = faceApp.get(image)

    if len(faces) == 0:
        return None

    # largest face
    face = max(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
    )

    x1, y1, x2, y2 = map(int, face.bbox)

    faceCrop = image[y1:y2, x1:x2]
    return faceCrop


def resizeAndNormalize(image):
    image = cv2.resize(image, imgSize)
    image = image / 255.0
    return (image * 255).astype("uint8")


def augmentRotate(image):
    results = []
    angles = [15, -15]

    for i, angle in enumerate(angles):
        h, w = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
        rotated = cv2.warpAffine(image, matrix, (w, h))
        results.append((f"rotate{i}", rotated))

    return results


def augmentFlip(image):
    return [("flip", cv2.flip(image, 1))]


def augmentBrightness(image):
    results = []
    for i, beta in enumerate([30, -30]):
        bright = cv2.convertScaleAbs(image, alpha=1, beta=beta)
        results.append((f"brightness{i}", bright))
    return results


def augmentBlur(image):
    return [("blur", cv2.GaussianBlur(image, (5, 5), 0))]


def augmentNoise(image):
    noise = np.random.normal(0, 25, image.shape).astype("uint8")
    noisy = cv2.add(image, noise)
    return [("noise", noisy)]


def applyAugmentations(image):
    results = []

    if augmentationsConfig["rotate"]:
        results.extend(augmentRotate(image))

    if augmentationsConfig["flip"]:
        results.extend(augmentFlip(image))

    if augmentationsConfig["brightness"]:
        results.extend(augmentBrightness(image))

    if augmentationsConfig["blur"]:
        results.extend(augmentBlur(image))

    if augmentationsConfig["noise"]:
        results.extend(augmentNoise(image))

    return results


def getSplitFolder():
    r = random.random()

    if r < splitConfig["train"]:
        return "train"
    elif r < splitConfig["train"] + splitConfig["val"]:
        return "val"
    else:
        return "test"


def processDataset():
    for root, _, files in os.walk(inputDir):
        for file in files:
            if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            inputPath = os.path.join(root, file)

            relativePath = os.path.relpath(root, inputDir)
            personName = Path(relativePath).name

            image = cv2.imread(inputPath)
            if image is None:
                print(f"[ERROR] {inputPath}")
                continue

            face = detectAlignCrop(image)
            if face is None:
                print(f"[NO FACE] {inputPath}")
                continue

            baseImage = resizeAndNormalize(face)

            imagesToSave = [("original", baseImage)]

            # Augmentations
            imagesToSave.extend(applyAugmentations(baseImage))

            for suffix, img in imagesToSave:
                if splitConfig["enabled"]:
                    splitFolder = getSplitFolder()
                    saveDir = os.path.join(outputDir, splitFolder, personName)
                else:
                    saveDir = os.path.join(outputDir, personName)

                os.makedirs(saveDir, exist_ok=True)

                baseName = os.path.splitext(file)[0]
                outputName = f"{baseName}_{suffix}.jpg"

                outputPath = os.path.join(saveDir, outputName)

                cv2.imwrite(outputPath, img)

            print(f"[DONE] {inputPath}")


if __name__ == "__main__":
    processDataset()