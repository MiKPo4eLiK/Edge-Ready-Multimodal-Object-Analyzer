from torchvision import (
    datasets,
    transforms
)
from torch.utils.data import DataLoader
from pathlib import Path

DATA_DIR = Path("data/train")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

images, labels = next(iter(loader))

print("Images shape:", images.shape)
print("Labels:", labels)
print("Classes:", dataset.classes)
