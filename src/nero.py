from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

# Default repo that ships the pretrained adapter (see fetch_adapter).
NERO_REPO_URL = "https://github.com/Algorythmsz/261RCOSE46101"
NERO_CKPT_SUBPATH = "pretrained/nero_lam0.75.pt"

# --------------------------------------------------------------------------- #
# NeRo-CLIP: Negation-Routed adapter for frozen CLIP (Lee, Kwak, Park).
#
# Two pieces are reproduced here from the paper:
#   1. The 11-marker regex router that decides, per caption, whether the
#      negation adapter should fire.
#   2. The rank-8 residual adapter  g(z) = z + W_up · tanh(W_down · z)
#      applied at the final-EOS text embedding. Weights are loaded from a
#      trained checkpoint; until one is provided the adapter is unavailable.
# --------------------------------------------------------------------------- #

# The 11 explicit-negation markers from the paper. "n't" covers contractions
# (isn't, don't, won't, ...); the rest are matched on word boundaries.
NEGATION_MARKERS = [
    "not", "n't", "never", "neither", "no", "none",
    "nothing", "empty", "without", "absent", "missing",
]

_WORD_MARKERS = [m for m in NEGATION_MARKERS if m != "n't"]
_ROUTER_RE = re.compile(
    r"\b(?:" + "|".join(_WORD_MARKERS) + r")\b|n['’]t\b",
    re.IGNORECASE,
)


def route(caption: str) -> tuple[bool, list[str]]:
    """Regex router: does this caption carry explicit negation?

    Returns (is_negated, matched_marker_strings). When True, NeRo-CLIP applies
    the adapter; when False, CLIP's original embedding is used unchanged.
    """
    matches = _ROUTER_RE.findall(caption)
    seen: list[str] = []
    for m in matches:
        low = m.lower()
        if low not in seen:
            seen.append(low)
    return (len(matches) > 0), seen


def fetch_adapter(
    repo_url: str = NERO_REPO_URL,
    subpath: str = NERO_CKPT_SUBPATH,
    cache_dir: str | Path = "artifacts/nero_repo",
) -> str:
    """Clone the NeRo-CLIP repo (shallow) and return the local path to the
    pretrained adapter checkpoint. Reuses the clone if already present.
    """
    cache = Path(cache_dir)
    ckpt = cache / subpath
    if not ckpt.exists():
        if cache.exists():
            shutil.rmtree(cache)
        print(f"Cloning NeRo-CLIP adapter from {repo_url} ...")
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(cache)],
            check=True,
        )
    if not ckpt.exists():
        raise FileNotFoundError(f"'{subpath}' not found in {repo_url}")
    return str(ckpt)


class NeRoAdapter:
    """Rank-r residual adapter: g(z) = z + W_up · tanh(W_down · z).

    W_down: (r, d), W_up: (d, r). Operates on numpy embeddings; the caller is
    responsible for L2-normalizing before computing cosine similarity.
    """

    def __init__(self, w_down: np.ndarray, w_up: np.ndarray):
        self.w_down = np.asarray(w_down, dtype=np.float32)
        self.w_up = np.asarray(w_up, dtype=np.float32)
        self.rank, self.dim = self.w_down.shape

    def apply(self, z: np.ndarray) -> np.ndarray:
        """Apply the residual correction to an embedding (or batch)."""
        single = z.ndim == 1
        z2 = z[None, :] if single else z
        hidden = np.tanh(z2 @ self.w_down.T)  # (N, r)
        out = z2 + hidden @ self.w_up.T  # (N, d)
        return out[0] if single else out

    @classmethod
    def load(cls, path: str | Path) -> "NeRoAdapter":
        """Load adapter weights from a checkpoint.

        Supports a numpy .npz (keys ``w_down``/``w_up``) or a PyTorch
        state_dict (.pt/.pth) — matrix names are matched flexibly so it works
        with the trained NeRo-CLIP checkpoint regardless of exact key naming.
        """
        path = Path(path)
        if path.suffix == ".npz":
            data = np.load(path)
            return cls(data["w_down"], data["w_up"])

        import torch  # local import; only needed for torch checkpoints

        obj = torch.load(path, map_location="cpu")
        state = obj.get("state_dict", obj) if isinstance(obj, dict) else obj

        def find(*needles):
            for key, val in state.items():
                k = key.lower()
                if all(n in k for n in needles):
                    return np.asarray(val.detach().cpu().numpy() if hasattr(val, "detach") else val)
            return None

        w_down = find("down")
        w_up = find("up")
        if w_down is None or w_up is None:
            # Fall back to the two 2-D tensors, ordered by shape (r,d) then (d,r).
            mats = [np.asarray(v.detach().cpu().numpy() if hasattr(v, "detach") else v)
                    for v in state.values() if getattr(v, "ndim", 0) == 2]
            if len(mats) != 2:
                raise ValueError(
                    f"Could not identify w_down/w_up in checkpoint keys: {list(state.keys())}"
                )
            mats.sort(key=lambda m: m.shape[0])  # (r,d) has smaller first dim
            w_down, w_up = mats[0], mats[1]
        return cls(w_down, w_up)
