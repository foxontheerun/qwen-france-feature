"""Load the model + tokenizer and its SAEs in a single, dtype-consistent step."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import MODEL_ID, MODEL_REVISION, get_device
from .sae import load_sae


def load_model(device: str | None = None):
    """Return ``(model, tokenizer, device)``.

    On CPU we use bfloat16 (float16 is poorly supported for CPU matmul); on GPU
    we request float16, but the effective dtype is read back from the weights.
    """
    device = device or get_device()
    dtype = torch.bfloat16 if device == "cpu" else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer, device


def load_saes(model, layers, device: str):
    """Load one SAE per layer, cast to the model's actual dtype."""
    model_dtype = next(model.parameters()).dtype
    return {i: load_sae(i, device=device, dtype=model_dtype) for i in layers}
