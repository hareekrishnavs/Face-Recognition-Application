from pathlib import Path

import torch

projectRoot = Path(__file__).resolve().parent.parent
datasetRoot = projectRoot / "dataset"
rawDir = datasetRoot / "raw"
processedDir = datasetRoot / "processed"
capturedDir = datasetRoot / "captured"

trainDir = processedDir / "train"
valDir = processedDir / "val"
testDir = processedDir / "test"
splitDirs = {
    "train": trainDir,
    "val": valDir,
    "test": testDir,
}

modelsDir = projectRoot / "models"
modelSavePath = modelsDir / "bestModel.pth"
faceIndexPath = modelsDir / "face_index.npz"
labelMapPath = modelsDir / "labelMap.json"
trainingSummaryPath = modelsDir / "training_summary.json"
datasetMetadataPath = processedDir / "metadata.json"
modelMetadataPath = modelsDir / "model_metadata.json"

deviceName = "cuda" if torch.cuda.is_available() else "cpu"

supportedExtensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".heic",
    ".heif",
}

outputImageFormat = "jpg"
imageSize = (160, 160)
hiddenDim = 128
faceCropMargin = 0.28
minFaceSize = 48
randomSeed = 42

splitRatios = {
    "train": 0.80,
    "val": 0.10,
    "test": 0.10,
}

augmentationPlan = {
    "rotate_left": 1,
    "rotate_right": 1,
    "scale_in": 1,
    "scale_out": 1,
    "gaussian_noise": 1,
    "gaussian_blur": 1,
    "brightness": 1,
    "contrast": 1,
    "horizontal_flip": 1,
}

batchSize = 16
numWorkers = 0
learningRate = 3e-4
numEpochs = 35
weightDecay = 1e-4
dropoutRate = 0.35
labelSmoothing = 0.05
earlyStoppingPatience = 8
confidenceThresholdFloor = 0.45
confidenceThresholdCeil = 0.90
defaultUnknownThreshold = 0.65
defaultRecognitionMargin = 0.12
inferenceTopK = 3
