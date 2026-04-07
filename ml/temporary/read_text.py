import json
import random

with open("dataset_with_text.json", "r", encoding="utf-8") as f:
    data = json.load(f)

samples = [x for x in data if x["text"].strip() != ""]
for s in random.sample(samples, 10):
    print("Label:", s["label"])
    print("Text:", s["text"])
    print("-" * 50)
