from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class CaptionSample:
    image: str
    caption: str


def load_caption_rows(captions_file: Path) -> list[CaptionSample]:
    rows: list[CaptionSample] = []
    with captions_file.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image = row["image"].strip()
            caption = row["caption"].strip()
            if image and caption:
                rows.append(CaptionSample(image=image, caption=caption))
    if not rows:
        raise ValueError(f"No caption rows found in {captions_file}")
    return rows


def make_image_caption_map(rows: Iterable[CaptionSample]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for row in rows:
        out.setdefault(row.image, []).append(row.caption)
    return out


def split_images(
    image_names: list[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[str]]:
    if not math.isclose(train_ratio + val_ratio, 0.9, rel_tol=1e-5):
        raise ValueError("Expected train_ratio + val_ratio == 0.9 (test is remaining 0.1)")

    rng = random.Random(seed)
    names = image_names[:]
    rng.shuffle(names)

    n = len(names)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    return {
        "train": names[:n_train],
        "val": names[n_train : n_train + n_val],
        "test": names[n_train + n_val :],
    }
