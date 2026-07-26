"""Locate the France feature and check its specificity.

Captures activations at the target layer on "The capital of France is", lists
the top SAE features on the ' France' token, then ablates each one and measures
how much it moves the logit of ' Paris' relative to other city tokens. Feature
20583 stands out: ablating it drops ' Paris' the least while collapsing the
competing cities — it *carries* the France concept rather than suppressing it.
"""
import _bootstrap  # noqa: F401
import torch

from data.prompts import COMPARISON_TOKENS
from france_feature import (
    TARGET_LAYER,
    capture,
    clear_hooks,
    hooked,
    load_model,
    load_saes,
    make_ablate_hook,
)


def single_token_ids(tokenizer, strings):
    ids = {}
    for s in strings:
        enc = tokenizer.encode(s, add_special_tokens=False)
        if len(enc) == 1:
            ids[s] = enc[0]
        else:
            print(f"skip {s!r}: {len(enc)} tokens")
    return ids


def main():
    model, tokenizer, device = load_model()
    saes = load_saes(model, [TARGET_LAYER], device)
    sae = saes[TARGET_LAYER]

    prompt = "The capital of France is"
    h, ids = capture(model, tokenizer, prompt, TARGET_LAYER, device)
    with torch.no_grad():
        baseline_logits = model(**ids).logits[0, -1].float()

    france_id = tokenizer.encode(" France", add_special_tokens=False)[0]
    france_pos = (ids["input_ids"][0] == france_id).nonzero(as_tuple=True)[0].item()

    z = sae.encode(h.to(sae.W_enc.dtype))
    top_vals, top_ids = z[0, france_pos].topk(20)
    print(f"Top-20 features on ' France' (pos {france_pos}):")
    for fid, val in zip(top_ids.tolist(), top_vals.tolist()):
        print(f"  feature {fid:>5d}: {val:.3f}")

    token_ids = single_token_ids(tokenizer, COMPARISON_TOKENS)
    baseline_vals = {s: baseline_logits[t].item() for s, t in token_ids.items()}

    header = f"\n{'feat':>6} | " + " | ".join(f"Δ{s:<7}" for s in token_ids) + " | spec"
    print(header)
    print("-" * len(header))
    for fid in top_ids.tolist():
        clear_hooks(model)
        with hooked(model, TARGET_LAYER, make_ablate_hook(sae, fid)):
            with torch.no_grad():
                new_logits = model(**ids).logits[0, -1].float()
        deltas = {s: baseline_vals[s] - new_logits[t].item() for s, t in token_ids.items()}
        others = [deltas[s] for s in deltas if s != " Paris"]
        spec = deltas[" Paris"] - max(others)
        line = (
            f"{fid:>6} | "
            + " | ".join(f"{deltas[s]:+7.3f}" for s in token_ids)
            + f" | {spec:+6.3f}"
        )
        print(line)


if __name__ == "__main__":
    main()
