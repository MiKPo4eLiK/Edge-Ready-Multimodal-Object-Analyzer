import torch.nn as nn
from transformers import DistilBertModel


class DistilBertClassifier(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()

        self.bert = DistilBertModel.from_pretrained(
            "distilbert-base-uncased"
        )

        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask) -> object:
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls_output = outputs.last_hidden_state[:, 0]
        x = self.dropout(cls_output)
        logits = self.classifier(x)

        return logits
