"""Recognition model note.

This project now uses InsightFace's pretrained `buffalo_l` package
for face detection, alignment, and 512-dim recognition embeddings.

There is no local trainable PyTorch classifier backbone in the active
pipeline anymore. `training/train.py` now builds an embedding index
from `dataset/processed`, and `app/face_engine.py` performs similarity
matching against that index.
"""

MODEL_BACKBONE = "insightface/buffalo_l"
EMBEDDING_SIZE = 512
