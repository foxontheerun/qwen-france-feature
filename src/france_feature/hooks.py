"""Forward-hook factories for capturing and editing residual-stream activations.

Every decoder layer's forward output is a *tuple* whose first element is the
hidden state; hooks must unpack ``output[0]`` and rebuild the tuple, or they
silently corrupt generation.

The ablate/additive hooks preserve the SAE reconstruction *error* ``h - decode(encode(h))``
so that editing one feature leaves the rest of the residual stream untouched.
"""
from contextlib import contextmanager


def make_capture_hook(store: dict, key: str = "h"):
    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        store[key] = h.detach()

    return hook


def make_ablate_hook(sae, feature_id: int):
    """Zero one SAE feature, preserving reconstruction error."""

    def hook(module, inputs, output):
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        rest = output[1:] if is_tuple else ()
        h_f = h.to(sae.W_enc.dtype)
        z = sae.encode(h_f)
        error = h_f - sae.decode(z)
        z[..., feature_id] = 0
        h_new = (sae.decode(z) + error).to(h.dtype)
        return (h_new,) + rest if is_tuple else h_new

    return hook


def make_additive_hook(sae, feature_id: int, value: float):
    """Add a constant to one SAE feature, preserving reconstruction error."""

    def hook(module, inputs, output):
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        rest = output[1:] if is_tuple else ()
        h_f = h.to(sae.W_enc.dtype)
        z = sae.encode(h_f)
        error = h_f - sae.decode(z)
        z[..., feature_id] = z[..., feature_id] + value
        h_new = (sae.decode(z) + error).to(h.dtype)
        return (h_new,) + rest if is_tuple else h_new

    return hook


@contextmanager
def hooked(model, layer_idx: int, hook_fn):
    """Register ``hook_fn`` on one layer for the duration of the block."""
    handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)
    try:
        yield
    finally:
        handle.remove()


def clear_hooks(model):
    """Defensive cleanup: drop any forward hooks left on the decoder layers."""
    for layer in model.model.layers:
        layer._forward_hooks.clear()
