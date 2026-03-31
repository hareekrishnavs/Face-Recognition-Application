import os
import shutil
from PIL import Image
import pillow_heif

INPUT_DIR = "dataset/raw"
OUTPUT_DIR = "dataset/jpgRaw"

pillow_heif.register_heif_opener()


def toJpg(inputRoot, outputRoot):
    for root, _, files in os.walk(inputRoot):
        for file in files:
            inputPath = os.path.join(root, file)

            # Preserve folder structure
            relativePath = os.path.relpath(root, inputRoot)
            outputDir = os.path.join(outputRoot, relativePath)
            os.makedirs(outputDir, exist_ok=True)

            name, ext = os.path.splitext(file)
            ext = ext.lower()

            if ext in [".heic", ".jpg", ".jpeg", ".png"]:
                outputPath = os.path.join(outputDir, name + ".jpg")

                if os.path.exists(outputPath):
                    print(f"[SKIP] Exists: {outputPath}")
                    continue

                try:
                    image = Image.open(inputPath)
                    image = image.convert("RGB")
                    image.save(outputPath, "JPEG", quality=95)

                    print(f"[CONVERTED] {inputPath} → {outputPath}")

                except Exception as e:
                    print(f"[ERROR] {inputPath} | {e}")

            else:
                outputPath = os.path.join(outputDir, file)

                if os.path.exists(outputPath):
                    print(f"[SKIP] Exists: {outputPath}")
                    continue

                try:
                    shutil.copy2(inputPath, outputPath)
                    print(f"[COPIED] {inputPath} → {outputPath}")

                except Exception as e:
                    print(f"[ERROR] Copy failed: {inputPath} | {e}")


if __name__ == "__main__":

    toJpg(INPUT_DIR, OUTPUT_DIR)