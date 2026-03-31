import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image

trainDir = "dataset/processed/train"
modelPath = "models/arcface_model.pth"

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

model = FaceModel(embeddingSize).to(device)
model.load_state_dict(torch.load(modelPath, map_location=device))
model.eval()

print("Model loaded!")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

trainDataset = datasets.ImageFolder(trainDir, transform=transform)
trainLoader = DataLoader(trainDataset, batch_size=32, shuffle=False)

classNames = trainDataset.classes

embeddingDB = {name: [] for name in classNames}

with torch.no_grad():
    for images, labels in trainLoader:
        images = images.to(device)
        embeddings = model(images)

        for emb, label in zip(embeddings, labels):
            name = classNames[label.item()]
            embeddingDB[name].append(emb.cpu())

print("Embedding DB ready!")

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


faceCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(1)

print("Starting webcam... Press ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        try:

            faceRGB = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            facePIL = Image.fromarray(faceRGB)

            faceInput = transform(facePIL).unsqueeze(0).to(device)

            with torch.no_grad():
                embedding = model(faceInput)[0].cpu()

            name, confidence = predict(embedding)

            label = f"{name} ({confidence:.2f})"

            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        except Exception as e:
            print("Error:", e)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()