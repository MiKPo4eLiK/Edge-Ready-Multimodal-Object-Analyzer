from pathlib import Path

DATA_DIR = Path("data/train")

def inspect_dataset() -> list[str]:
    classes = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])

    print("📂 Found classes:")
    for i, cls in enumerate(classes):
        images = list((DATA_DIR / cls).glob("*"))
        print(f"{i}: {cls} — {len(images)} images")

    return classes


if __name__ == "__main__":
    inspect_dataset()
