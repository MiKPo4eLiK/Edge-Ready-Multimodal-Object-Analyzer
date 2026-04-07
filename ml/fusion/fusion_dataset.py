import os
import json
import torch
from torch import Tensor
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from transformers import DistilBertTokenizer


class FusionDataset(Dataset):
    def __init__(self, json_path, split, max_len=128) -> None:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.samples = [x for x in data if x["split"] == split]

        self.labels = sorted(list(set(x["label"] for x in data)))
        self.class_to_idx = {c: i for i, c in enumerate(self.labels)}

        self.tokenizer = DistilBertTokenizer.from_pretrained(
            "distilbert-base-uncased"
        )

        self.max_len = max_len

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.base_dir = os.path.dirname(json_path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx) -> dict[str, Tensor]:
        sample = self.samples[idx]

        # image
        image_path = os.path.join(self.base_dir, sample["image_path"])  # type: ignore[attr-defined]
        image_path = os.path.normpath(image_path)
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        # text
        encoding = self.tokenizer(
            sample["text"],  # type: ignore[attr-defined]
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # label
        label = torch.tensor(
            self.class_to_idx[sample["label"]],  # type: ignore[attr-defined]
            dtype=torch.long
        )

        return {
            "image": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": label
        }
