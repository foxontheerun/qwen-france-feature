"""Helpers to turn a prompt into inputs and to read a layer's activations."""
import torch

from .hooks import hooked, make_capture_hook


def to_inputs(tokenizer, prompt: str, device: str, chat: bool = False, thinking: bool = False):
    """Tokenize a prompt, optionally through the chat template.

    ``chat=True`` wraps the prompt as a user turn; on a base model without a
    template it falls back to the raw prompt.
    """
    if chat:
        try:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=thinking,
            )
        except Exception:
            text = prompt
    else:
        text = prompt
    return tokenizer(text, return_tensors="pt").to(device)


def capture(model, tokenizer, text, layer, device, chat=False, thinking=False):
    """Run a forward pass and return ``(hidden_state, inputs)`` at ``layer``.

    ``hidden_state`` has shape ``(1, seq, d_model)``.
    """
    ids = to_inputs(tokenizer, text, device, chat=chat, thinking=thinking)
    store = {}
    with hooked(model, layer, make_capture_hook(store)):
        with torch.no_grad():
            model(**ids)
    return store["h"], ids
