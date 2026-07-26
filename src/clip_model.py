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
    normalize: bool = True,
) -> np.ndarray:
    feats: list[np.ndarray] = []
    for batch_paths in tqdm(list(batch_iter(image_paths, batch_size)), desc="Encoding images"):
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        image_features = _to_feature_tensor(model.get_image_features(**inputs))
        if normalize:
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
    normalize: bool = True,
) -> np.ndarray:
    feats: list[np.ndarray] = []
    for batch_texts in tqdm(list(batch_iter(texts, batch_size)), desc="Encoding texts"):
        inputs = processor(text=batch_texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        text_features = _to_feature_tensor(model.get_text_features(**inputs))
        if normalize:
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        feats.append(text_features.cpu().numpy().astype(np.float32))
    return np.concatenate(feats, axis=0)


@torch.no_grad()
def encode_image_patches(
    model: CLIPModel,
    processor: CLIPProcessor,
    image: Image.Image,
    device: str,
) -> tuple[np.ndarray, int]:
    """Project each vision patch token into the shared CLIP space.

    Returns L2-normalized patch embeddings of shape (num_patches, proj_dim)
    together with the side length of the (square) patch grid. Unlike
    `get_image_features` — which returns only the pooled [CLS] embedding — this
    exposes per-patch embeddings so token-patch alignment can be computed.
    """
    inputs = processor(images=[image], return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    vision_outputs = model.vision_model(pixel_values=pixel_values)
    # Drop the [CLS] token at index 0; keep the spatial patch tokens.
    patch_tokens = vision_outputs.last_hidden_state[:, 1:, :]
    patch_embeds = model.visual_projection(model.vision_model.post_layernorm(patch_tokens))
    patch_embeds = patch_embeds / patch_embeds.norm(dim=-1, keepdim=True)

    patch_embeds = patch_embeds[0].cpu().numpy().astype(np.float32)
    num_patches = patch_embeds.shape[0]
    grid = int(round(num_patches**0.5))
    return patch_embeds, grid


@torch.no_grad()
def encode_text_tokens(
    model: CLIPModel,
    processor: CLIPProcessor,
    text: str,
    device: str,
) -> tuple[np.ndarray, list[str]]:
    """Project each text token into the shared CLIP space.

    Returns L2-normalized token embeddings of shape (seq_len, proj_dim) and the
    matching list of decoded token strings (including the start/end markers).
    """
    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    text_outputs = model.text_model(**inputs)
    token_embeds = model.text_projection(text_outputs.last_hidden_state)
    token_embeds = token_embeds / token_embeds.norm(dim=-1, keepdim=True)

    token_embeds = token_embeds[0].cpu().numpy().astype(np.float32)
    token_ids = inputs["input_ids"][0].tolist()
    tokens = processor.tokenizer.convert_ids_to_tokens(token_ids)
    return token_embeds, tokens
