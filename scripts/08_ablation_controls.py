"""Stress-test the multi-layer ablation (07) against its two weak spots.

1. Weak control. 07's random features are almost never active on these
   prompts, so zeroing them is a no-op. Here every France feature gets a
   *matched* control: a non-France feature (contrast score <= 0) on the same
   layer whose removed vector is about as large on the same prompts —
   strength = max-over-positions activation x decoder-column norm, averaged
   over the evaluation prompts, within 0.5x..2x of the France feature's.
   Repeated over several random draws.
2. Circularity. 07 scored three of the very prompts used to pick the
   features. Here features are still selected on SELECT_* prompts, but
   scored only on HELDOUT_* prompts (new templates, French and Japanese,
   cultural markers) plus held-out controls about other countries.

Also reports mean next-token loss on country-free text, so "concept removed"
can be told apart from "model damaged".

Writes results/ablation_controls.json (for figures) and a .txt log.
"""
import argparse
import json
from contextlib import ExitStack

import _bootstrap  # noqa: F401
import torch

from _bootstrap import RESULTS
from data.prompts import (
    HELDOUT_CONTROL,
    HELDOUT_FRANCE,
    NEUTRAL_TEXT,
    SELECT_CONTROL,
    SELECT_FRANCE,
)
from france_feature import (
    SEED,
    clear_hooks,
    contrast_scores,
    find_concept_features,
    load_model,
    load_saes,
    matched_pool,
    max_acts,
    mean_nll,
    multi_hooked,
    score_targets,
    to_inputs,
)

OUT_TXT = RESULTS / "ablation_controls.txt"
OUT_JSON = RESULTS / "ablation_controls.json"

PAIRS = HELDOUT_FRANCE + HELDOUT_CONTROL
N_FR = len(HELDOUT_FRANCE)
MAX_NEW_TOKENS = 10


def summarize(base, res, idx):
    """Mean P ratio and how many targets lost top-1, over prompts in ``idx``.

    Only prompts the model gets right at baseline (rank 1) are counted —
    ablation cannot "remove" an answer the model never gave.
    """
    keep = [i for i in idx if base[i][1] == 1]
    if not keep:
        return {"n": 0, "mean_ratio": None, "lost_top1": 0}
    ratio = sum(res[i][0] / base[i][0] for i in keep) / len(keep)
    lost = sum(res[i][1] > 1 for i in keep)
    return {"n": len(keep), "mean_ratio": ratio, "lost_top1": lost}


def generate(model, tokenizer, device, prompts, ctx):
    texts = []
    with ctx:
        for prompt in prompts:
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
    ap.add_argument("--draws", type=int, default=10, help="matched-control draws")
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

    fr_prompts = [p for p, _ in HELDOUT_FRANCE]
    plan, pools, layer_info = {}, {}, []
    log(f"Per layer: France feature (selected on SELECT_*), its strength on held-out "
        f"France prompts, matched-control pool size")
    for L in layers:
        sae = saes[L]
        scores = contrast_scores(model, tokenizer, sae, L, device, SELECT_FRANCE, SELECT_CONTROL)
        found = find_concept_features(scores, args.top_n)
        if not found:
            log(f"  layer {L:>2}: no feature passes contrast")
            continue
        d_norm = sae.W_dec.float().norm(dim=0)
        strength = (max_acts(model, tokenizer, sae, L, fr_prompts, device) * d_norm).mean(dim=0)
        plan[L] = [fid for fid, _ in found]
        pools[L] = []
        for fid, sc in found:
            pool, band = matched_pool(scores, strength, fid, 0.5, 2.0), "0.5-2x"
            if not pool:
                pool, band = matched_pool(scores, strength, fid, 0.25, 4.0), "0.25-4x"
            pools[L].append(pool)
            log(f"  layer {L:>2}: feature {fid:>5} (contrast {sc:+.2f}, strength "
                f"{strength[fid].item():.3f})  pool {len(pool)} [{band}]")
            layer_info.append({"layer": L, "feature": fid, "contrast": sc,
                               "strength": strength[fid].item(), "pool": len(pool), "band": band})

    g = torch.Generator().manual_seed(SEED)
    draws = []
    for _ in range(args.draws):
        d = {}
        for L, plist in pools.items():
            picks = []
            for pool in plist:
                if pool:
                    picks.append(pool[torch.randint(len(pool), (1,), generator=g).item()])
            if picks:
                d[L] = picks
        draws.append(d)

    log("\nScoring held-out prompts (none used for feature selection)...")
    base = score_targets(model, tokenizer, device, PAIRS, ExitStack())
    fr = score_targets(model, tokenizer, device, PAIRS, multi_hooked(model, saes, plan))
    matched = [score_targets(model, tokenizer, device, PAIRS, multi_hooked(model, saes, d))
               for d in draws]

    nll = {
        "baseline": mean_nll(model, tokenizer, device, NEUTRAL_TEXT, ExitStack()),
        "france": mean_nll(model, tokenizer, device, NEUTRAL_TEXT, multi_hooked(model, saes, plan)),
        "matched": [mean_nll(model, tokenizer, device, NEUTRAL_TEXT, multi_hooked(model, saes, d))
                    for d in draws],
    }

    log(f"\nP(target) [rank] top-1 | matched control: mean P (min..max), "
        f"lost top-1 in k/{args.draws} draws ('-' = wrong at baseline)")
    rows = []
    for i, (prompt, target) in enumerate(PAIRS):
        group = "FRANCE" if i < N_FR else "CONTROL"
        mp = [m[i][0] for m in matched]
        b, f = base[i], fr[i]
        # "lost" only means something if the model had the answer to begin with
        lost = f"{sum(m[i][1] > 1 for m in matched)}/{len(mp)}" if b[1] == 1 else "-"
        log(f"{group:<7} {prompt[:44]:<44} ->{target!r:<10} "
            f"base {b[0]:.3f} [{b[1]:>5}] {b[2]!r:<10} | "
            f"France-abl {f[0]:.3f} [{f[1]:>5}] {f[2]!r:<10} | "
            f"matched {sum(mp)/len(mp):.3f} ({min(mp):.3f}..{max(mp):.3f}) lost {lost}")
        rows.append({"group": group, "prompt": prompt, "target": target,
                     "base": b, "france": f,
                     "matched": [list(m[i]) for m in matched]})

    fr_idx, ctrl_idx = range(N_FR), range(N_FR, len(PAIRS))
    summary = {
        "france_prompts": {
            "france_ablation": summarize(base, fr, fr_idx),
            "matched": [summarize(base, m, fr_idx) for m in matched],
        },
        "control_prompts": {
            "france_ablation": summarize(base, fr, ctrl_idx),
            "matched": [summarize(base, m, ctrl_idx) for m in matched],
        },
    }

    def fmt(s):
        if not s["n"]:
            return "n/a (no prompt correct at baseline)"
        return f"P kept {s['mean_ratio']*100:5.1f}% of baseline, lost top-1 {s['lost_top1']}/{s['n']}"

    log("\nSummary (only prompts the model answers correctly at baseline)")
    for name, key in (("France prompts ", "france_prompts"), ("Control prompts", "control_prompts")):
        s = summary[key]
        log(f"  {name} | France ablation: {fmt(s['france_ablation'])}")
        ratios = [m["mean_ratio"] for m in s["matched"] if m["mean_ratio"] is not None]
        if ratios:
            lost = [m["lost_top1"] for m in s["matched"]]
            log(f"  {name} | matched control: P kept {min(ratios)*100:.1f}..{max(ratios)*100:.1f}% "
                f"over draws, lost top-1 {min(lost)}..{max(lost)}/{s['matched'][0]['n']}")

    m = nll["matched"]
    log(f"\nMean next-token loss on country-free text: baseline {nll['baseline']:.3f} | "
        f"France ablation {nll['france']:.3f} | matched {sum(m)/len(m):.3f} ({min(m):.3f}..{max(m):.3f})")

    log("\nGreedy generations on held-out France prompts")
    gb = generate(model, tokenizer, device, fr_prompts, ExitStack())
    gf = generate(model, tokenizer, device, fr_prompts, multi_hooked(model, saes, plan))
    gens = []
    for p, a, b in zip(fr_prompts, gb, gf):
        log(f"  {p!r}\n     base:   {a!r}\n     France: {b!r}")
        gens.append({"prompt": p, "base": a, "france": b})

    OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
    OUT_JSON.write_text(json.dumps({
        "top_n": args.top_n, "draws": args.draws, "layers": layer_info,
        "plan": {str(k): v for k, v in plan.items()},
        "matched_plans": [{str(k): v for k, v in d.items()} for d in draws],
        "rows": rows, "summary": summary, "nll": nll, "generations": gens,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nsaved {OUT_TXT}\nsaved {OUT_JSON}")


if __name__ == "__main__":
    main()
