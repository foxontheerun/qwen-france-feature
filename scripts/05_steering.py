"""Additive steering: inject the France feature into neutral generations.

Sweeps the additive strength on story prompts. There is a coherence window
(roughly +2..+4) where the text bends toward France while staying fluent; past
it (+8..+12) the output degrades into fragments. This is a demonstration of
influence, not a clean on/off control.
"""
import _bootstrap  # noqa: F401
import torch

from data.prompts import STORY_PROMPTS
from france_feature import (
    BEST_FEATURE,
    TARGET_LAYER,
    clear_hooks,
    hooked,
    load_model,
    load_saes,
    make_additive_hook,
    to_inputs,
)

VALUES = [2.0, 4.0, 8.0]
MAX_NEW_TOKENS = 30
OUT = _bootstrap.RESULTS / "steering.txt"


def generate(model, tokenizer, device, prompt, sae, value=None):
    ids = to_inputs(tokenizer, prompt, device)
    gen_kwargs = dict(max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                      pad_token_id=tokenizer.eos_token_id)
    if value is None:
        with torch.no_grad():
            out = model.generate(**ids, **gen_kwargs)
    else:
        with hooked(model, TARGET_LAYER, make_additive_hook(sae, BEST_FEATURE, value)):
            with torch.no_grad():
                out = model.generate(**ids, **gen_kwargs)
    return tokenizer.decode(out[0], skip_special_tokens=True)


def main():
    model, tokenizer, device = load_model()
    saes = load_saes(model, [TARGET_LAYER], device)
    sae = saes[TARGET_LAYER]
    clear_hooks(model)

    lines = []
    for prompt in STORY_PROMPTS:
        base = generate(model, tokenizer, device, prompt, sae, value=None)
        block = [f"\nPROMPT:   {prompt!r}", f"BASELINE: {base}"]
        for v in VALUES:
            steered = generate(model, tokenizer, device, prompt, sae, value=v)
            block.append(f"+{v:<4}:    {steered}")
        text = "\n".join(block)
        print(text)
        print("-" * 80)
        lines.append(text)

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
