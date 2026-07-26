from __future__ import annotations

import argparse
from pathlib import Path

from clip_model import encode_images, encode_texts, load_clip
from dataset_io import load_caption_rows, make_image_caption_map, split_images
from index_io import load_index, save_index
from retrieval import evaluate_recall, topk_search
from utils import resolve_device, seed_everything


def build_index_main(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = resolve_device(args.cpu)

    images_dir = Path(args.images_dir)
    captions_file = Path(args.captions_file)
    output_dir = Path(args.output_dir)

    rows = load_caption_rows(captions_file)
    image_to_captions = make_image_caption_map(rows)

    image_names = sorted(image_to_captions.keys())
    image_paths = [images_dir / name for name in image_names]

    missing = [str(p) for p in image_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} image files are missing. Example: {missing[:3]}"
        )

    splits = split_images(image_names, seed=args.seed)

    model, processor = load_clip(args.model_name, device)
    image_embeddings = encode_images(
        model=model,
        processor=processor,
        image_paths=image_paths,
        batch_size=args.batch_size,
        device=device,
    )

    save_index(
        output_dir=output_dir,
        image_names=image_names,
        image_embeddings=image_embeddings,
        splits=splits,
        image_to_captions=image_to_captions,
        model_name=args.model_name,
    )

    print(f"Saved index to: {output_dir}")
    print(f"Images: {len(image_names)} | Embedding dim: {image_embeddings.shape[1]}")
    print(
        f"Split sizes -> train: {len(splits['train'])}, val: {len(splits['val'])}, test: {len(splits['test'])}"
    )


def eval_main(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = resolve_device(args.cpu)
    index_dir = Path(args.index_dir)

    image_embeddings, _, splits, image_to_captions, meta, name_to_idx = load_index(index_dir)

    model_name = args.model_name if args.model_name else meta["model_name"]
    model, processor = load_clip(model_name, device)

    split_images_list = splits[args.split]
    metrics = evaluate_recall(
        model=model,
        processor=processor,
        device=device,
        image_embeddings=image_embeddings,
        image_to_idx=name_to_idx,
        split_images=split_images_list,
        image_to_captions=image_to_captions,
        batch_size=args.batch_size,
        ks=args.k,
    )

    print(f"Split: {args.split}")
    for k in args.k:
        print(f"R@{k}: {metrics[f'R@{k}']:.4f}")


def topk_main(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = resolve_device(args.cpu)

    index_dir = Path(args.index_dir)
    image_embeddings, image_names, _, _, meta, _ = load_index(index_dir)

    model_name = args.model_name if args.model_name else meta["model_name"]
    model, processor = load_clip(model_name, device)

    text_emb = encode_texts(model, processor, [args.query], args.batch_size, device)
    topk_idx, topk_scores = topk_search(text_emb, image_embeddings, args.k)

    print(f"Query: {args.query}")
    print("Top-k retrieval results:")
    for rank, (idx, score) in enumerate(zip(topk_idx[0], topk_scores[0]), start=1):
        print(f"{rank:2d}. {image_names[int(idx)]} | score={float(score):.4f}")


def serve_main(args: argparse.Namespace) -> None:
    from webapp import run_server

    run_server(
        index_dir=args.index_dir,
        images_dir=args.images_dir,
        host=args.host,
        port=args.port,
        model_name=args.model_name,
        cpu=args.cpu,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CLIP retrieval baseline for Flickr30k")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build-index", help="Build and save image embedding index")
    p_build.add_argument("--images-dir", type=str, required=True)
    p_build.add_argument("--captions-file", type=str, required=True)
    p_build.add_argument("--output-dir", type=str, required=True)
    p_build.add_argument("--model-name", type=str, default="openai/clip-vit-base-patch32")
    p_build.add_argument("--batch-size", type=int, default=64)
    p_build.add_argument("--seed", type=int, default=42)
    p_build.add_argument("--cpu", action="store_true")
    p_build.set_defaults(func=build_index_main)

    p_eval = sub.add_parser("eval", help="Evaluate Recall@K")
    p_eval.add_argument("--index-dir", type=str, required=True)
    p_eval.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    p_eval.add_argument("--k", type=int, nargs="+", default=[1, 5, 10])
    p_eval.add_argument("--batch-size", type=int, default=128)
    p_eval.add_argument("--seed", type=int, default=42)
    p_eval.add_argument("--model-name", type=str, default=None)
    p_eval.add_argument("--cpu", action="store_true")
    p_eval.set_defaults(func=eval_main)

    p_topk = sub.add_parser("topk", help="Retrieve top-k images for a text query")
    p_topk.add_argument("--index-dir", type=str, required=True)
    p_topk.add_argument("--query", type=str, required=True)
    p_topk.add_argument("--k", type=int, default=5)
    p_topk.add_argument("--batch-size", type=int, default=32)
    p_topk.add_argument("--seed", type=int, default=42)
    p_topk.add_argument("--model-name", type=str, default=None)
    p_topk.add_argument("--cpu", action="store_true")
    p_topk.set_defaults(func=topk_main)

    p_serve = sub.add_parser("serve", help="Run the web demo (CLIP explainer + search + negation)")
    p_serve.add_argument("--index-dir", type=str, required=True)
    p_serve.add_argument("--images-dir", type=str, default="data/flickr30k/Images")
    p_serve.add_argument("--host", type=str, default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--model-name", type=str, default=None)
    p_serve.add_argument("--cpu", action="store_true")
    p_serve.set_defaults(func=serve_main)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
