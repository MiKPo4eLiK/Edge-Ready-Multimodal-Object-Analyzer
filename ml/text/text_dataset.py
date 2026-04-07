import json
import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer


class MemeTextDataset(Dataset):
    def __init__(self, json_path, split, max_len=128) -> None:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.samples = [x for x in data if x["split"] == split]

        self.texts = [x["text"] for x in self.samples]
        self.labels = [x["label"] for x in self.samples]

        self.classes = sorted(list(set(self.labels)))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.labels = [self.class_to_idx[l] for l in self.labels]

        self.tokenizer = DistilBertTokenizer.from_pretrained(
            "distilbert-base-uncased"
        )
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx) -> dict[str, Tensor]:
        encoding = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }
