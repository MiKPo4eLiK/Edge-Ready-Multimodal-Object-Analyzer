import json

with open("dataset_with_text.json", "r", encoding="utf-8") as f:
    data = json.load(f)

total = len(data)
with_text = len([x for x in data if x["text"].strip() != ""])

print("Total:", total)
print("With text:", with_text)
print("Percent with text:", round(with_text/total*100, 2))
