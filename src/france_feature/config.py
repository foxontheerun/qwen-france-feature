"""Central configuration: model, SAE, and the feature under study.

Pin ``MODEL_REVISION`` / ``SAE_REVISION`` to a commit hash before publishing
numbers, so a re-run is byte-for-byte reproducible.
"""
import torch

MODEL_ID = "Qwen/Qwen3.5-2B-Base"
MODEL_REVISION = None  # e.g. "main" or a pinned commit hash

SAE_REPO = "Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50"
SAE_REVISION = None
TOP_K = 50

# The result: layer 12, feature 20583 encodes the concept "France".
TARGET_LAYER = 12
BEST_FEATURE = 20583

# Layers we load an SAE for (used by the reconstruction sanity check).
RECON_LAYERS = [4, 12, 16, 18, 22]

SEED = 0


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"
