from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


def load_clip(model_name: str, device: str) -> tuple[CLIPModel, CLIPProcessor]:
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    return model, processor


def batch_iter(items: list, batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def _to_feature_tensor(output: torch.Tensor | object) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output

    pooler_output = getattr(output, "pooler_output", None)
    if isinstance(pooler_output, torch.Tensor):
        return pooler_output

    last_hidden_state = getattr(output, "last_hidden_state", None)
    if isinstance(last_hidden_state, torch.Tensor):
        return last_hidden_state[:, 0, :]

    raise TypeError(f"Unsupported model output type for feature extraction: {type(output)}")


@torch.no_grad()
def encode_images(
    model: CLIPModel,
    processor: CLIPProcessor,
    image_paths: list[Path],
    batch_size: int,
    device: str,
) -> np.ndarray:
    feats: list[np.ndarray] = []
    for batch_paths in tqdm(list(batch_iter(image_paths, batch_size)), desc="Encoding images"):
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        image_features = _to_feature_tensor(model.get_image_features(**inputs))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        feats.append(image_features.cpu().numpy().astype(np.float32))
    return np.concatenate(feats, axis=0)


@torch.no_grad()
def encode_texts(
    model: CLIPModel,
    processor: CLIPProcessor,
    texts: list[str],
    batch_size: int,
    device: str,
) -> np.ndarray:
    feats: list[np.ndarray] = []
    for batch_texts in tqdm(list(batch_iter(texts, batch_size)), desc="Encoding texts"):
        inputs = processor(text=batch_texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        text_features = _to_feature_tensor(model.get_text_features(**inputs))
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        feats.append(text_features.cpu().numpy().astype(np.float32))
    return np.concatenate(feats, axis=0)
