from __future__ import annotations

import numpy as np
from transformers import CLIPModel, CLIPProcessor

from clip_model import encode_texts


def topk_search(
    text_embeddings: np.ndarray,
    image_embeddings: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    sims = text_embeddings @ image_embeddings.T
    topk_idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    topk_scores = np.take_along_axis(sims, topk_idx, axis=1)

    order = np.argsort(-topk_scores, axis=1)
    topk_idx = np.take_along_axis(topk_idx, order, axis=1)
    topk_scores = np.take_along_axis(topk_scores, order, axis=1)
    return topk_idx, topk_scores


def evaluate_recall(
    model: CLIPModel,
    processor: CLIPProcessor,
    device: str,
    image_embeddings: np.ndarray,
    image_to_idx: dict[str, int],
    split_images: list[str],
    image_to_captions: dict[str, list[str]],
    batch_size: int,
    ks: list[int],
) -> dict[str, float]:
    queries: list[str] = []
    gt_indices: list[int] = []

    for img_name in split_images:
        if img_name not in image_to_captions:
            continue
        for cap in image_to_captions[img_name]:
            queries.append(cap)
            gt_indices.append(image_to_idx[img_name])

    text_emb = encode_texts(model, processor, queries, batch_size, device)
    max_k = max(ks)
    topk_idx, _ = topk_search(text_emb, image_embeddings, max_k)

    gt = np.array(gt_indices)[:, None]
    metrics: dict[str, float] = {}
    for k in ks:
        hit = (topk_idx[:, :k] == gt).any(axis=1)
        metrics[f"R@{k}"] = float(hit.mean())
    return metrics
