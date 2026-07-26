"""Top-K sparse autoencoder matching the Qwen-Scope checkpoint layout."""
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

from .config import SAE_REPO, SAE_REVISION, TOP_K


class TopKSAE(nn.Module):
    """Frozen top-K SAE: keeps only the ``top_k`` largest post-ReLU activations."""

    def __init__(self, state_dict: dict, top_k: int):
        super().__init__()
        self.W_enc = nn.Parameter(state_dict["W_enc"], requires_grad=False)
        self.b_enc = nn.Parameter(state_dict["b_enc"], requires_grad=False)
        self.W_dec = nn.Parameter(state_dict["W_dec"], requires_grad=False)
        self.b_dec = nn.Parameter(state_dict["b_dec"], requires_grad=False)
        self.top_k = top_k
        self.d_sae, self.d_model = self.W_enc.shape

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = torch.relu(x @ self.W_enc.T + self.b_enc)
        vals, idx = pre.topk(self.top_k, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(-1, idx, vals)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec.T + self.b_dec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def load_sae(layer_idx: int, device: str, dtype: torch.dtype) -> TopKSAE:
    """Download and load the SAE for one layer, cast to the model's dtype.

    Loading in the model's dtype (rather than forcing one) removes every
    dtype-mismatch error when hooks feed activations straight into the SAE.
    """
    path = hf_hub_download(
        repo_id=SAE_REPO, filename=f"layer{layer_idx}.sae.pt", revision=SAE_REVISION
    )
    state_dict = torch.load(path, map_location="cpu")
    return TopKSAE(state_dict, top_k=TOP_K).to(device).to(dtype).eval()
