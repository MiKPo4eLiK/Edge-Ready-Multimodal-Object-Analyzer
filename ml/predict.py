import os
from typing import Any
import argparse

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import pytesseract

from transformers import DistilBertTokenizer
from ml.fusion.fusion_model import MultiModalModel


pytesseract.pytesseract.tesseract_cmd = "tesseract"


DEVICE = "cpu"

CLASSES = [
    "animal_meme",
    "dark_humor_meme",
    "reaction_meme",
    "screenshot_meme",
    "text_meme"
]

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")


MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "training_result",
    "best_fusion_checkpoint.pth"
)


model = MultiModalModel(num_classes=len(CLASSES))

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])

model.to(DEVICE)
model.eval()


def predict_image(file) -> tuple[str, Any]:

    image = Image.open(file).convert("RGB")

    # OCR
    text = pytesseract.image_to_string(image)

    # image tensor
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # text tokens
    tokens = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    input_ids = tokens["input_ids"].to(DEVICE)
    attention_mask = tokens["attention_mask"].to(DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor, input_ids, attention_mask)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    predicted_class = CLASSES[predicted.item()]
    confidence_score = round(confidence.item(), 4)

    return predicted_class, confidence_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Meme classification")
    parser.add_argument("image_path", type=str, help="Path to meme image")

    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print("❌ File not found!")
        exit(1)

    predicted_class, confidence = predict_image(args.image_path)

    print("🔍 Prediction result:")
    print(f"Class: {predicted_class}")
    print(f"Confidence: {confidence:.2f}")
