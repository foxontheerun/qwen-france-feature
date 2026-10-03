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
from france_feature import (
    BEST_FEATURE,
    SEED,
    TARGET_LAYER,
    capture,
    clear_hooks,
    hooked,
    load_model,
    load_saes,
    make_ablate_hook,
    to_inputs,
)

OUT = RESULTS / "multilayer_ablation.txt"

FRANCE_PROMPTS = [
    "The capital of France is",
    "I spent the summer traveling across France",
    "Die Hauptstadt von Frankreich ist",
    "Столица Франции —",
    "フランスの首都は",
]
CONTROL_PROMPTS = [
    "The capital of Germany is",
    "I spent the summer traveling across Japan",
    "Die Hauptstadt von Italien ist",
    "Столица России —",
    "スペインの首都は",
    "The capital of Spain is",
]

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


def max_acts(model, tokenizer, sae, layer, prompts, device):
    """Per-prompt max-over-positions SAE activation, shape (n_prompts, d_sae)."""
    rows = []
    for p in prompts:
        h, _ = capture(model, tokenizer, p, layer, device)
        z = sae.encode(h.to(sae.W_enc.dtype))[0]
        rows.append(z.max(dim=0).values.float())
    return torch.stack(rows)


def find_france_features(model, tokenizer, sae, layer, device, top_n):
    """Features that fire on every France prompt and not on the controls."""
    fr = max_acts(model, tokenizer, sae, layer, FRANCE_PROMPTS, device)
    ctrl = max_acts(model, tokenizer, sae, layer, CONTROL_PROMPTS, device)
    # min over France prompts: must fire across languages, not on one of them
    score = fr.min(dim=0).values - ctrl.max(dim=0).values
    vals, ids = score.topk(top_n)
    return [(i, v) for i, v in zip(ids.tolist(), vals.tolist()) if v > 0]


def multi_hooked(model, saes, plan):
    """Ablate ``plan[layer] = [feature ids]`` on all listed layers at once."""
    stack = ExitStack()
    for layer, fids in plan.items():
        for fid in fids:
            stack.enter_context(hooked(model, layer, make_ablate_hook(saes[layer], fid)))
    return stack


def score(model, tokenizer, device, ctx):
    """P(target first token) and its rank for every EVAL pair."""
    out = []
    with ctx:
        for prompt, target in EVAL:
            ids = to_inputs(tokenizer, prompt, device)
            tid = tokenizer.encode(target, add_special_tokens=False)[0]
            with torch.no_grad():
                logits = model(**ids).logits[0, -1].float()
            probs = logits.softmax(-1)
            rank = int((logits > logits[tid]).sum().item()) + 1
            top = tokenizer.decode(logits.argmax().item())
            out.append((probs[tid].item(), rank, top))
    return out


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
        found = find_france_features(model, tokenizer, saes[L], L, device, args.top_n)
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
