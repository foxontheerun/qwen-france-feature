"""Ablate the France feature on every layer at once.

Ablating feature 20583 at layer 12 alone barely moves ' Paris': the model
re-derives the answer further down. This asks the obvious follow-up — what if
the concept is removed everywhere?

Feature ids are per-SAE: 20583 only means "France" in the layer-12 dictionary.
So for each layer we first *find* that layer's France feature(s) by contrast
(max activation on France prompts minus the same templates about other
countries), then zero all of them simultaneously, preserving each layer's SAE
reconstruction error.

Conditions reported:
  baseline       no intervention
  L12 only       the original single-layer ablation
  all layers     France feature(s) zeroed on every layer
  random ctrl    the same number of random features zeroed on every layer
                 (separates "France removed" from "model damaged")
  cumulative     layers 0..k ablated, k growing: where does Paris die?
"""
import argparse
from contextlib import ExitStack

import _bootstrap  # noqa: F401
import torch

from _bootstrap import RESULTS
from data.prompts import SELECT_CONTROL, SELECT_FRANCE
from france_feature import (
    BEST_FEATURE,
    SEED,
    TARGET_LAYER,
    clear_hooks,
    contrast_scores,
    find_concept_features,
    load_model,
    load_saes,
    multi_hooked,
    score_targets,
    to_inputs,
)

OUT = RESULTS / "multilayer_ablation.txt"

# (prompt, target) — we score the probability of the target's first token.
EVAL = [
    ("The capital of France is", " Paris"),
    ("Die Hauptstadt von Frankreich ist", " Paris"),
    ("Столица Франции —", " Париж"),
    ("The Eiffel Tower is located in the city of", " Paris"),
    ("The capital of Germany is", " Berlin"),  # control: should survive
]
GEN_PROMPTS = [p for p, _ in EVAL[:4]]
MAX_NEW_TOKENS = 15


def score(model, tokenizer, device, ctx):
    return score_targets(model, tokenizer, device, EVAL, ctx)


def generate(model, tokenizer, device, ctx):
    texts = []
    with ctx:
        for prompt in GEN_PROMPTS:
            ids = to_inputs(tokenizer, prompt, device)
            with torch.no_grad():
                o = model.generate(**ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                   pad_token_id=tokenizer.eos_token_id)
            texts.append(tokenizer.decode(o[0][ids["input_ids"].shape[1]:],
                                          skip_special_tokens=True))
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-n", type=int, default=1, help="France features per layer")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    model, tokenizer, device = load_model()
    layers = list(range(len(model.model.layers)))
    saes = load_saes(model, layers, device)
    clear_hooks(model)

    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    log(f"France features per layer (top-n={args.top_n}, score = min_FR - max_ctrl):")
    plan = {}
    for L in layers:
        scores = contrast_scores(model, tokenizer, saes[L], L, device,
                                 SELECT_FRANCE, SELECT_CONTROL)
        found = find_concept_features(scores, args.top_n)
        if found:
            plan[L] = [fid for fid, _ in found]
        desc = ", ".join(f"{fid} ({v:+.2f})" for fid, v in found) or "— none passes contrast"
        log(f"  layer {L:>2}: {desc}")
    if TARGET_LAYER in plan:
        hit = BEST_FEATURE in plan[TARGET_LAYER]
        log(f"  sanity: layer {TARGET_LAYER} recovers {BEST_FEATURE}? {'yes' if hit else 'NO'}")

    g = torch.Generator().manual_seed(SEED)
    rand_plan = {
        L: torch.randint(0, saes[L].d_sae, (len(f),), generator=g).tolist()
        for L, f in plan.items()
    }

    conditions = [
        ("baseline", lambda: ExitStack()),
        ("L12 only", lambda: multi_hooked(model, saes, {TARGET_LAYER: [BEST_FEATURE]})),
        ("all layers", lambda: multi_hooked(model, saes, plan)),
        ("random ctrl", lambda: multi_hooked(model, saes, rand_plan)),
    ]

    log("\nP(target) [rank] top-1 next token")
    header = f"{'condition':<12}" + "".join(f" | {p[:24]:<36}" for p, _ in EVAL)
    log(header)
    log("-" * len(header))
    for name, make in conditions:
        res = score(model, tokenizer, device, make())
        log(f"{name:<12}" + "".join(
            f" | {p:6.3f} [{r:>5}] {t!r:<18}" for p, r, t in res))

    log(f"\nCumulative: ablate France features on layers 0..k  ({EVAL[0][0]!r} -> ' Paris')")
    for k in layers:
        sub = {L: f for L, f in plan.items() if L <= k}
        p, r, t = score(model, tokenizer, device, multi_hooked(model, saes, sub))[0]
        log(f"  0..{k:>2}: P={p:6.3f}  rank={r:>5}  top={t!r}")

    log("\nGreedy generations")
    for name, make in conditions:
        if name == "L12 only":
            continue
        log(f"[{name}]")
        for prompt, text in zip(GEN_PROMPTS, generate(model, tokenizer, device, make())):
            log(f"  {prompt!r} -> {text!r}")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
