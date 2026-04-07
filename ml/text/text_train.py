import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from text_dataset import MemeTextDataset
from text_model import DistilBertClassifier

from sklearn.metrics import confusion_matrix


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
EPOCHS = 15
LR = 3e-5


train_dataset = MemeTextDataset("dataset_with_text.json", split="train")
val_dataset = MemeTextDataset("dataset_with_text.json", split="val")


print("Token lengths (first 10 samples):")
for i in range(10):
    tokens = train_dataset.tokenizer.tokenize(train_dataset.texts[i])
    print(len(tokens))
print("-" * 40)


train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)


model = DistilBertClassifier(num_classes=len(train_dataset.classes)).to(DEVICE)


optimizer = torch.optim.AdamW(model.parameters(), lr=LR)


criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.05)


best_acc = 0
best_preds = None
best_labels = None

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    # validation
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)

    is_best = False
    if acc > best_acc:
        best_acc = acc
        best_preds = all_preds
        best_labels = all_labels
        torch.save(model.state_dict(), "training_result/best_text_model.pt")
        is_best = True

    if is_best:
        print("Best text model saved!")

    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print(f"Train Loss: {avg_loss:.4f}")
    print(f"Validation Accuracy: {acc * 100:.2f}%")
    print("-" * 40)

print(f"\nBest Validation Accuracy: {best_acc * 100:.2f}%")

cm = confusion_matrix(best_labels, best_preds)

print("\nConfusion Matrix (Best Text Model):")
print(cm)
