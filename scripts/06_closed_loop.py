"""Closed-loop steering: a proportional controller on the France feature.

Sensor  -> read the feature's peak activation over the prompt.
Regulator -> add = clip(GAIN * (TARGET - peak), 0, MAX_ADD).
Actuator  -> additive steering during generation, only when add > EPS.

Prompts that already carry France read a high peak and get add ~= 0 (untouched);
prompts that do not get a proportional nudge. A small, honest demo of the
"measure the internal variable, then regulate it" idea — not a robust product.
"""
import _bootstrap  # noqa: F401
import torch

from data.prompts import MIXED_PROMPTS
from france_feature import (
    BEST_FEATURE,
    TARGET_LAYER,
    capture,
    clear_hooks,
    hooked,
    load_model,
    load_saes,
    make_additive_hook,
    to_inputs,
)

TARGET = 0.7    # setpoint: peak >= TARGET is "France already present"
GAIN = 1.0
MAX_ADD = 2.5
EPS = 0.3       # below this, do not intervene
MAX_NEW_TOKENS = 50


def france_peak(model, tokenizer, device, sae, prompt):
    h, _ = capture(model, tokenizer, prompt, TARGET_LAYER, device, chat=True)
    z = sae.encode(h.to(sae.W_enc.dtype))
    return z[0, :, BEST_FEATURE].float().max().item()


def decide_add(peak):
    return max(0.0, min(GAIN * (TARGET - peak), MAX_ADD))


def generate(model, tokenizer, device, prompt, sae, value=0.0):
    ids = to_inputs(tokenizer, prompt, device, chat=True)
    gen_kwargs = dict(max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                      pad_token_id=tokenizer.eos_token_id)
    if value > EPS:
        with hooked(model, TARGET_LAYER, make_additive_hook(sae, BEST_FEATURE, value)):
            with torch.no_grad():
                out = model.generate(**ids, **gen_kwargs)
    else:
        with torch.no_grad():
            out = model.generate(**ids, **gen_kwargs)
    return tokenizer.decode(out[0], skip_special_tokens=True)


def main():
    model, tokenizer, device = load_model()
    saes = load_saes(model, [TARGET_LAYER], device)
    sae = saes[TARGET_LAYER]
    clear_hooks(model)

    for prompt in MIXED_PROMPTS:
        peak = france_peak(model, tokenizer, device, sae, prompt)
        add = decide_add(peak)
        base = generate(model, tokenizer, device, prompt, sae, value=0.0)
        out = generate(model, tokenizer, device, prompt, sae, value=add)
        print(f"\nPROMPT:   {prompt!r}")
        print(f"SENSOR: peak France = {peak:.2f}  ->  add = {add:.2f}")
        print(f"BASELINE: {base}")
        print(f"OUTPUT:   {out}")
        print("-" * 100)

    clear_hooks(model)


if __name__ == "__main__":
    main()
