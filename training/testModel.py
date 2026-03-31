import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path

trainDir = "dataset/processed/train"
testDir = "dataset/processed/test"

modelPath = Path(__file__).resolve().parent.parent / "models" / "arcface_model.pth"

embeddingSize = 512
threshold = 0.6   
device = "cuda" if torch.cuda.is_available() else "cpu"

class FaceModel(nn.Module):
    def __init__(self, embeddingSize):
        super().__init__()
        baseModel = models.resnet50(pretrained=False)

        self.backbone = nn.Sequential(*list(baseModel.children())[:-1])
        self.embedding = nn.Linear(baseModel.fc.in_features, embeddingSize)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.embedding(x)
        x = F.normalize(x)
        return x


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

trainDataset = datasets.ImageFolder(trainDir, transform=transform)
testDataset = datasets.ImageFolder(testDir, transform=transform)

trainLoader = DataLoader(trainDataset, batch_size=32, shuffle=False)
testLoader = DataLoader(testDataset, batch_size=1, shuffle=False)

classNames = trainDataset.classes

print("Classes:", classNames)

model = FaceModel(embeddingSize).to(device)
model.load_state_dict(torch.load(modelPath, map_location=device))
model.eval()

print("Model loaded!")


print("Building embedding database...")

embeddingDB = {name: [] for name in classNames}

with torch.no_grad():
    for images, labels in trainLoader:
        images = images.to(device)
        labels = labels.to(device)

        embeddings = model(images)

        for emb, label in zip(embeddings, labels):
            name = classNames[label.item()]
            embeddingDB[name].append(emb.cpu())

print("Database built!")

def cosineSimilarity(a, b):
    return torch.dot(a, b).item()


def predict(embedding):
    bestName = "Unknown"
    bestScore = -1

    for name, embList in embeddingDB.items():
        for refEmb in embList:
            score = cosineSimilarity(embedding, refEmb)

            if score > bestScore:
                bestScore = score
                bestName = name

    if bestScore < threshold:
        return "Unknown", bestScore

    return bestName, bestScore

print("\n===== TEST RESULTS =====\n")

with torch.no_grad():
    for idx, (image, label) in enumerate(testLoader):
        image = image.to(device)

        embedding = model(image)[0].cpu()

        predName, confidence = predict(embedding)

        imgPath = testDataset.samples[idx][0]
        imgName = os.path.basename(imgPath)

        trueName = classNames[label.item()]

        print(f"Image: {imgName}")
        print(f"True: {trueName}")
        print(f"Pred: {predName}")
        print(f"Confidence: {confidence:.4f}")
        print("-" * 40)