"""
Generate a tiny synthetic cell-image dataset for testing the classifier web UI.
Run: python generate_sample_data.py
"""
import os
from pathlib import Path
from PIL import Image, ImageDraw
import random

random.seed(42)

BASE = Path(__file__).parent / "data"
CONFIG = {
    "x5": {"size": (128, 128), "circles": (3, 6), "radius": (10, 25)},
    "x20": {"size": (256, 256), "circles": (8, 16), "radius": (15, 40)},
}
CLASSES = {
    "class_a": {"bg": (30, 30, 60), "fg": (180, 100, 100)},
    "class_b": {"bg": (30, 60, 30), "fg": (100, 180, 100)},
}
SAMPLES_PER_CLASS = 12


def make_image(size, circles_range, radius_range, bg_color, fg_color):
    img = Image.new("RGB", size, bg_color)
    draw = ImageDraw.Draw(img)
    n = random.randint(*circles_range)
    for _ in range(n):
        r = random.randint(*radius_range)
        x = random.randint(r, size[0] - r)
        y = random.randint(r, size[1] - r)
        color = tuple(min(255, max(0, c + random.randint(-30, 30))) for c in fg_color)
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color, outline=color)
    return img


def main():
    for mag, cfg in CONFIG.items():
        for cls, colors in CLASSES.items():
            folder = BASE / mag / cls
            folder.mkdir(parents=True, exist_ok=True)
            for i in range(SAMPLES_PER_CLASS):
                img = make_image(
                    cfg["size"],
                    cfg["circles"],
                    cfg["radius"],
                    colors["bg"],
                    colors["fg"],
                )
                img.save(folder / f"{cls}_{mag}_{i:03d}.png")
    print(f"Sample dataset created at: {BASE.resolve()}")
    print("Folder structure:")
    for mag in CONFIG:
        for cls in CLASSES:
            print(f"  data/{mag}/{cls}/  -> {SAMPLES_PER_CLASS} images")


if __name__ == "__main__":
    main()
