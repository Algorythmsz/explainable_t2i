from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def l2_normalize(arr: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return arr / denom


def save_index(
    output_dir: Path,
    image_names: list[str],
    image_embeddings: np.ndarray,
    splits: dict[str, list[str]],
    image_to_captions: dict[str, list[str]],
    model_name: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "image_embeddings.npy", image_embeddings)
    with (output_dir / "image_names.json").open("w", encoding="utf-8") as f:
        json.dump(image_names, f, ensure_ascii=False, indent=2)
    with (output_dir / "splits.json").open("w", encoding="utf-8") as f:
        json.dump(splits, f, ensure_ascii=False, indent=2)
    with (output_dir / "image_to_captions.json").open("w", encoding="utf-8") as f:
        json.dump(image_to_captions, f, ensure_ascii=False)

    meta = {
        "model_name": model_name,
        "num_images": len(image_names),
        "embedding_dim": int(image_embeddings.shape[1]),
    }
    with (output_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_index(index_dir: Path):
    image_embeddings = np.load(index_dir / "image_embeddings.npy")
    image_embeddings = l2_normalize(image_embeddings.astype(np.float32))

    with (index_dir / "image_names.json").open("r", encoding="utf-8") as f:
        image_names = json.load(f)
    with (index_dir / "splits.json").open("r", encoding="utf-8") as f:
        splits = json.load(f)
    with (index_dir / "image_to_captions.json").open("r", encoding="utf-8") as f:
        image_to_captions = json.load(f)
    with (index_dir / "meta.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)

    name_to_idx = {name: i for i, name in enumerate(image_names)}
    return image_embeddings, image_names, splits, image_to_captions, meta, name_to_idx
