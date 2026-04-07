import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

from fusion_model import MultiModalModel
from fusion_dataset import FusionDataset

import matplotlib.pyplot as plt


def train() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # datasets
    train_dataset = FusionDataset(
        json_path="dataset_with_text.json",
        split="train"
    )

    val_dataset = FusionDataset(
        json_path="dataset_with_text.json",
        split="val"
    )

    # dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )

    num_classes = len(train_dataset.class_to_idx)

    # model
    model = MultiModalModel(num_classes=num_classes)
    model.to(device)

    # loss
    class_weights = torch.tensor([1.4, 1.0, 1.0, 1.0, 1.1]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4
    )

    # scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    num_epochs = 40
    best_val_loss = float("inf")

    # early stopping parameters
    patience = 10
    patience_counter = 0

    train_losses = []
    val_losses = []
    val_accuracies = []

    start_epoch = 0

    # resume
    if os.path.exists("training_result/last_checkpoint.pth"):
        print("Last checkpoint found. Resuming training...")

        checkpoint = torch.load("training_result/last_checkpoint.pth", map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]
        patience_counter = checkpoint["patience_counter"]

        train_losses = checkpoint["train_losses"]
        val_losses = checkpoint["val_losses"]
        val_accuracies = checkpoint["val_accuracies"]

        print(f"Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, num_epochs):

        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # train
        model.train()
        running_loss = 0.0

        for batch in tqdm(train_loader, desc="Training"):
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            outputs = model(images, input_ids, attention_mask)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)

        train_loss = running_loss / len(train_dataset)

        # validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                images = batch["image"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                outputs = model(images, input_ids, attention_mask)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * labels.size(0)

                _, preds = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (preds == labels).sum().item()  # type: ignore[attr-defined]

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_dataset)
        val_acc = correct / total

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # scheduler
        scheduler.step(val_loss)

        print(
            f"\nTrain Loss: {train_loss:.4f}\n"
            f"Validation Loss: {val_loss:.4f}\n"
            f"Validation Accuracy: {val_acc * 100:.2f}%\n"
            f"Learning Rate: {optimizer.param_groups[0]['lr']}\n"
        )

        # save best + early stopping by val_loss
        if val_loss < best_val_loss:

            best_val_loss = val_loss
            patience_counter = 0

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
            }, "training_result/best_fusion_checkpoint.pth")

            print("Best model saved!")

        else:
            patience_counter += 1
            print(f"Early stopping counter: {patience_counter}/{patience}")

        # save last checkpoint (every epoch)
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "patience_counter": patience_counter,

            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_accuracies": val_accuracies,
        }, "training_result/last_checkpoint.pth")

        print(f"Last checkpoint saved (epoch {epoch + 1})")

        if patience_counter >= patience:
            print("Early stopping triggered!")
            break

    print(f"\nBest Validation Loss: {best_val_loss:.4f}")

    # confusion matrix
    print("\nLoading best model for final evaluation...")

    checkpoint = torch.load("training_result/best_fusion_checkpoint.pth", map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(images, input_ids, attention_mask)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)

    print("\nConfusion Matrix:")

    for i, row in enumerate(cm):
        class_name = list(train_dataset.class_to_idx.keys())[i]
        print(f"{class_name:20} {row}")

    print("\nClass names:")
    print(list(train_dataset.class_to_idx.keys()))

    epochs = range(1, len(train_losses) + 1)

    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()

    plt.savefig("training_result/training_loss.png")
    plt.show()

    plt.figure()
    plt.plot(epochs, val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.legend()

    plt.savefig("training_result/training_accuracy.png")
    plt.show()

    print("Training finished 🚀")


if __name__ == "__main__":
    train()
