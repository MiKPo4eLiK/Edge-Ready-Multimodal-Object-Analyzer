import easyocr
import os
import json
from tqdm import tqdm


reader = easyocr.Reader(["en"], gpu=False)

DATASET_DIR = "data"

results = []


for split in ["train", "val", "test"]:
    split_path = os.path.join(DATASET_DIR, split)

    for label in os.listdir(split_path):
        label_path = os.path.join(split_path, label)

        if not os.path.isdir(label_path):
            continue

        for file in tqdm(os.listdir(label_path), desc=f"{split}/{label}"):
            if file.lower().endswith((".jpg", ".png", ".jpeg")):
                image_path = os.path.join(label_path, file)

                text = reader.readtext(image_path, detail=0)
                full_text = " ".join(text)

                clean_text = full_text.lower().strip()

                if len(clean_text) < 3:
                    clean_text = ""

                results.append({
                    "image_path": image_path,
                    "text": clean_text,
                    "label": label,
                    "split": split
                })

with open("dataset_with_text.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("Done! Total samples:", len(results))
