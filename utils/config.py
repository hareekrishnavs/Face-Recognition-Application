from pathlib import Path
import torch

deviceName = "cuda" if torch.cuda.is_available() else "cpu"

projectRoot = Path(__file__).resolve().parent.parent

rawDir = projectRoot /"data"/"raw"
processedDir = projectRoot /"data"/"processed"
capturedNewDir = projectRoot /"data"/"captured_new"

trainDir = processedDir / "train"
valDir = processedDir / "val"
testDir = processedDir / "test"

modelsDir = projectRoot / "models"
modelSavePath = modelsDir / "bestModel.pth"
labelMapPath = modelsDir / "labelMap.json"

supportedExtensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

outputImageFormat = "jpeg"
imageSize = (224, 224)

splitRatios = {
    "train": 0.70,
    "val": 0.20,
    "test": 0.10,
}

augmentationCounts = {
    "noise": 2,
    "rotate": 2,
    "flip": 2,
    "scale": 2,
    "brightness": 2,
}

mtcnnMargin = 20
mtcnnMinFaceSize = 5
randomSeed = 42

retinaFaceThreshold = 0.90

batchSize = 16
learningRate = 3e-4
numEpochs = 50
numWorkers = 0
weightDecay = 5e-4
dropoutRate = 0.5
labelSmoothing = 0.05
earlyStoppingPatience = 10
