import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

trainDir = "dataset/processed/train"
valDir = "dataset/processed/val"

batchSize = 32
numEpochs = 10
lr = 1e-4
embeddingSize = 512
device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

trainDataset = datasets.ImageFolder(trainDir, transform=transform)
valDataset = datasets.ImageFolder(valDir, transform=transform)

trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True)
valLoader = DataLoader(valDataset, batch_size=batchSize, shuffle=False)

numClasses = len(trainDataset.classes)

print("Classes:", trainDataset.classes)

class FaceModel(nn.Module):
    def __init__(self, embeddingSize):
        super().__init__()
        baseModel = models.resnet50(pretrained=True)

        self.backbone = nn.Sequential(*list(baseModel.children())[:-1])
        self.embedding = nn.Linear(baseModel.fc.in_features, embeddingSize)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.embedding(x)
        x = F.normalize(x)
        return x


class ArcFaceLoss(nn.Module):
    def __init__(self, embeddingSize, numClasses, s=30.0, m=0.50):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(numClasses, embeddingSize))
        nn.init.xavier_uniform_(self.weight)

        self.s = s
        self.m = m

    def forward(self, embeddings, labels):
        cosine = F.linear(embeddings, F.normalize(self.weight))
        theta = torch.acos(torch.clamp(cosine, -1 + 1e-7, 1 - 1e-7))
        targetLogits = torch.cos(theta + self.m)

        oneHot = F.one_hot(labels, num_classes=cosine.size(1)).float()

        logits = cosine * (1 - oneHot) + targetLogits * oneHot
        logits *= self.s

        loss = F.cross_entropy(logits, labels)
        return loss, logits


model = FaceModel(embeddingSize).to(device)
criterion = ArcFaceLoss(embeddingSize, numClasses).to(device)

optimizer = torch.optim.Adam(
    list(model.parameters()) + list(criterion.parameters()),
    lr=lr
)

def trainEpoch():
    model.train()
    totalLoss = 0

    for images, labels in tqdm(trainLoader):
        images = images.to(device)
        labels = labels.to(device)

        embeddings = model(images)
        loss, _ = criterion(embeddings, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        totalLoss += loss.item()

    return totalLoss / len(trainLoader)


def validate():
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in valLoader:
            images = images.to(device)
            labels = labels.to(device)

            embeddings = model(images)
            _, logits = criterion(embeddings, labels)

            preds = torch.argmax(logits, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


for epoch in range(numEpochs):
    trainLoss = trainEpoch()
    valAcc = validate()

    print(f"Epoch {epoch+1}/{numEpochs}")
    print(f"Train Loss: {trainLoss:.4f}")
    print(f"Val Acc: {valAcc:.4f}")



modelPath = Path(__file__).resolve().parent.parent / "models" / "arcface_model.pth"
torch.save(model.state_dict(), modelPath)

print("Model saved!")