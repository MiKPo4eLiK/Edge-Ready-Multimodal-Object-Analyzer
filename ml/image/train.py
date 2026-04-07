from collections import Counter
import torch
from torchvision import (
    datasets,
    transforms
)
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.metrics import confusion_matrix
from model import build_model
import random
import numpy as np


#  seed
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# configs
TRAIN_DIR = Path("data/train")
VAL_DIR = Path("data/val")

BATCH_SIZE = 32
EPOCHS = 40
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# transforms
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(300, scale=(0.75, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    transforms.RandomErasing(p=0.25),
])

val_transform = transforms.Compose([
    transforms.Resize(340),
    transforms.CenterCrop(300),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# evaluation function
def evaluate(model_eval, dataloader, device) -> tuple[float, list, list]:
    model_eval.eval()

    correct = 0
    total = 0
    preds_list = []
    labels_list = []

    with torch.no_grad():
        for batch_images, batch_labels in dataloader:
            batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)

            outputs = model_eval(batch_images)

            flipped_images = torch.flip(batch_images, dims=[3])
            outputs_flipped = model_eval(flipped_images)

            outputs = (outputs + outputs_flipped) / 2
            _, predicted = torch.max(outputs, 1)

            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()  # type: ignore[attr-defined]

            preds_list.extend(predicted.cpu().numpy())
            labels_list.extend(batch_labels.cpu().numpy())

    acc = 100 * correct / total
    return acc, preds_list, labels_list


def main() -> None:
    # datasets
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transform)

    # class distribution
    counter = Counter(train_dataset.targets)

    print("Class distribution:")
    for idx, count in counter.items():
        print(train_dataset.classes[idx], ":", count)

    # create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )

    # model
    model = build_model(num_classes=len(train_dataset.classes))
    model = model.to(DEVICE)

    # loss
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.03)

    # optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=1e-4
    )

    # scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=10
    )

    # training loop
    best_accuracy = 0.0
    patience = 10
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        if epoch == 3:
            print("\nUnfreezing backbone...\n")

            for param in model.parameters():
                param.requires_grad = True

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=5e-5,
                weight_decay=1e-4
            )

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=0.5,
                patience=10
            )

        model.train()
        total_loss = 0.0

        for train_images, train_labels in train_loader:
            train_images, train_labels = train_images.to(DEVICE), train_labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(train_images)
            loss = criterion(outputs, train_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        accuracy, _, _ = evaluate(model, val_loader, DEVICE)

        scheduler.step(accuracy)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), "training_result/best_model.pt")
            print("Best model saved!")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print(f"Train Loss: {avg_loss:.4f}")
        print(f"Validation Accuracy: {accuracy:.2f}%")
        print("-" * 40)

        if epochs_no_improve >= patience:
            print("Early stopping triggered!")
            break

    # final evaluation
    model.load_state_dict(torch.load("training_result/best_model.pt", map_location=DEVICE))

    accuracy, all_preds, all_labels = evaluate(model, val_loader, DEVICE)

    cm = confusion_matrix(all_labels, all_preds)

    print("\nConfusion Matrix (Best Model):")
    print(cm)

    print("\nClass names:")
    print(train_dataset.classes)

    print("Training finished 🚀")


if __name__ == "__main__":
    main()
