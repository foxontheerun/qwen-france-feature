"""Robustness suite for the multi-layer ablation: the weak spots left after 08.

A. Double dissociation. The same procedure finds a Germany and a Japan feature
   on every layer. Ablating each concept is scored on every concept's
   held-out prompts: a specific ablation hurts its own row only (3x3 matrix).
   This also tests that the method is general, not something special about
   France.
B. Layer bands. France features ablated on 0-7, 8-15, 16-23, 0-19 and 20-23
   only. If the late layers alone carried the effect, the features there could
   be "emit Paris" output features rather than the concept.
C. Statistics. Activity-matched controls (as in 08) over many draws give a
   null distribution for the specificity gap
       gap = P kept on France prompts - P kept on other-country prompts,
   with an empirical p-value for the France ablation, plus bootstrap CIs over
   prompts.

Only prompts the model answers correctly at baseline (rank 1) are counted.
Writes results/robustness.json (for figures) and a .txt log.
"""
import argparse
import json
import random
from contextlib import ExitStack

import _bootstrap  # noqa: F401
import torch

from _bootstrap import RESULTS
from data.prompts import CONCEPTS, NEUTRAL_TEXT
from france_feature import (
    SEED,
    clear_hooks,
    contrast_scores_all_layers,
    find_concept_features,
    load_model,
    load_saes,
    matched_pool,
    max_acts_all_layers,
    mean_nll,
    multi_hooked,
    score_targets,
)

OUT_TXT = RESULTS / "robustness.txt"
OUT_JSON = RESULTS / "robustness.json"

FOCUS = "France"
BANDS = {
    "0-7": range(0, 8),
    "8-15": range(8, 16),
    "16-23": range(16, 24),
    "0-19": range(0, 20),
    "20-23": range(20, 24),
    "all": range(0, 24),
}


def kept(base, res, idx):
    """P kept (ablated / baseline) and lost top-1 over prompts correct at baseline."""
    keep = [i for i in idx if base[i][1] == 1]
    ratios = [res[i][0] / base[i][0] for i in keep]
    return {
        "n": len(keep),
        "mean": sum(ratios) / len(ratios) if ratios else None,
        "lost": sum(res[i][1] > 1 for i in keep),
        "ratios": ratios,
    }


def gap(k_own, k_other):
    if k_own["mean"] is None or k_other["mean"] is None:
        return None
    return k_own["mean"] - k_other["mean"]


def bootstrap_ci(xs, rng, n=2000):
    if not xs:
        return None
    means = sorted(sum(rng.choice(xs) for _ in xs) / len(xs) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def pct(x):
    return "  n/a " if x is None else f"{x * 100:5.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-n", type=int, default=1, help="concept features per layer")
    ap.add_argument("--draws", type=int, default=30, help="matched-control draws")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    model, tokenizer, device = load_model()
    layers = list(range(len(model.model.layers)))
    saes = load_saes(model, layers, device)
    clear_hooks(model)

    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    names = list(CONCEPTS)
    pairs, idx = [], {}
    for c in names:
        start = len(pairs)
        pairs += CONCEPTS[c]["heldout"]
        idx[c] = range(start, len(pairs))
    other_idx = [i for c in names if c != FOCUS for i in idx[c]]

    # --- feature selection, per concept, on its own SELECT prompts only
    plans, scores = {}, {}
    for c in names:
        scores[c] = contrast_scores_all_layers(model, tokenizer, saes, device,
                                               CONCEPTS[c]["select"], CONCEPTS[c]["control"])
        plans[c] = {}
        for L in layers:
            found = find_concept_features(scores[c][L], args.top_n)
            if found:
                plans[c][L] = [fid for fid, _ in found]

    log("Concept feature per layer (contrast score); '-' = none passes")
    log("layer | " + " | ".join(f"{c:<16}" for c in names))
    for L in layers:
        cells = []
        for c in names:
            fids = plans[c].get(L)
            cells.append(f"{','.join(map(str, fids)):<7} ({scores[c][L][fids[0]].item():+.2f})"
                         if fids else "-")
        shared = {f for c in names for f in plans[c].get(L, [])}
        dup = sum(len(plans[c].get(L, [])) for c in names) - len(shared)
        log(f"  {L:>2}  | " + " | ".join(f"{x:<16}" for x in cells) + ("  <- shared id" if dup else ""))

    base = score_targets(model, tokenizer, device, pairs, ExitStack())
    log("\nBaseline: prompts answered correctly (rank 1) per concept: "
        + ", ".join(f"{c} {sum(base[i][1] == 1 for i in idx[c])}/{len(idx[c])}" for c in names))

    # --- A. double dissociation
    log("\nA. Double dissociation: P kept on each concept's held-out prompts [lost top-1 / n]")
    log(f"{'ablate':<10}| " + " | ".join(f"{c:<18}" for c in names) + " | loss on neutral text")
    nll_base = mean_nll(model, tokenizer, device, NEUTRAL_TEXT, ExitStack())
    matrix = {}
    for a in names:
        res = score_targets(model, tokenizer, device, pairs, multi_hooked(model, saes, plans[a]))
        nll = mean_nll(model, tokenizer, device, NEUTRAL_TEXT, multi_hooked(model, saes, plans[a]))
        row = {t: kept(base, res, idx[t]) for t in names}
        matrix[a] = {"kept": row, "nll": nll, "scores": res}
        log(f"{a:<10}| " + " | ".join(
            f"{pct(row[t]['mean'])} [{row[t]['lost']:>2}/{row[t]['n']:<2}]     " for t in names)
            + f" | {nll:.3f} (baseline {nll_base:.3f})")

    # --- B. layer bands for the focus concept
    log(f"\nB. {FOCUS} features ablated on a band of layers only")
    log(f"{'layers':<7}| {FOCUS + ' prompts':<20} | {'other countries':<20} | gap")
    bands = {}
    for name, band in BANDS.items():
        sub = {L: f for L, f in plans[FOCUS].items() if L in band}
        res = score_targets(model, tokenizer, device, pairs, multi_hooked(model, saes, sub))
        k_own, k_oth = kept(base, res, idx[FOCUS]), kept(base, res, other_idx)
        g = gap(k_own, k_oth)
        bands[name] = {"own": k_own, "other": k_oth, "gap": g}
        log(f"{name:<7}| {pct(k_own['mean'])} [{k_own['lost']:>2}/{k_own['n']:<2}]       "
            f"| {pct(k_oth['mean'])} [{k_oth['lost']:>2}/{k_oth['n']:<2}]       "
            f"| {'n/a' if g is None else f'{g * 100:+.1f} pp'}")

    # --- C. activity-matched null distribution for the focus concept
    focus_prompts = [p for p, _ in CONCEPTS[FOCUS]["heldout"]]
    acts = max_acts_all_layers(model, tokenizer, saes, focus_prompts, device)
    pools = {}
    for L, fids in plans[FOCUS].items():
        d_norm = saes[L].W_dec.float().norm(dim=0)
        strength = (acts[L] * d_norm).mean(dim=0)
        pools[L] = []
        for fid in fids:
            pool = matched_pool(scores[FOCUS][L], strength, fid, 0.5, 2.0)
            pools[L].append(pool or matched_pool(scores[FOCUS][L], strength, fid, 0.25, 4.0))

    g_torch = torch.Generator().manual_seed(SEED)
    draws = []
    for _ in range(args.draws):
        plan = {}
        for L, plist in pools.items():
            picks = [pool[torch.randint(len(pool), (1,), generator=g_torch).item()]
                     for pool in plist if pool]
            if picks:
                plan[L] = picks
        res = score_targets(model, tokenizer, device, pairs, multi_hooked(model, saes, plan))
        nll = mean_nll(model, tokenizer, device, NEUTRAL_TEXT, multi_hooked(model, saes, plan))
        k_own, k_oth = kept(base, res, idx[FOCUS]), kept(base, res, other_idx)
        draws.append({"plan": {str(k): v for k, v in plan.items()},
                      "own": k_own, "other": k_oth, "gap": gap(k_own, k_oth), "nll": nll})

    fa = bands["all"]
    obs = fa["gap"]
    null = [d["gap"] for d in draws if d["gap"] is not None]
    p_val = (1 + sum(g <= obs for g in null)) / (len(null) + 1) if obs is not None and null else None
    ci_own = bootstrap_ci(fa["own"]["ratios"], rng)
    ci_oth = bootstrap_ci(fa["other"]["ratios"], rng)

    log(f"\nC. {FOCUS} ablation (all layers) vs {len(null)} activity-matched control draws")
    log(f"  {FOCUS} ablation: {FOCUS} P kept {pct(fa['own']['mean'])} "
        f"(95% CI {pct(ci_own[0]) if ci_own else 'n/a'}..{pct(ci_own[1]) if ci_own else 'n/a'}), "
        f"other countries {pct(fa['other']['mean'])} "
        f"(95% CI {pct(ci_oth[0]) if ci_oth else 'n/a'}..{pct(ci_oth[1]) if ci_oth else 'n/a'})")
    if null:
        s = sorted(null)
        log(f"  specificity gap: France ablation {obs * 100:+.1f} pp | matched draws "
            f"min {s[0] * 100:+.1f}, median {s[len(s) // 2] * 100:+.1f}, max {s[-1] * 100:+.1f} pp")
        log(f"  empirical p (draw gap <= observed): {p_val:.3f}  "
            f"(floor with {len(null)} draws: {1 / (len(null) + 1):.3f})")
        nlls = [d["nll"] for d in draws]
        log(f"  loss on neutral text: matched draws {min(nlls):.3f}..{max(nlls):.3f}")
    log("  per draw: " + "  ".join(
        f"[{pct(d['own']['mean']).strip()} / {pct(d['other']['mean']).strip()}]" for d in draws))

    OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
    OUT_JSON.write_text(json.dumps({
        "top_n": args.top_n, "draws_n": args.draws, "concepts": names,
        "plans": {c: {str(L): f for L, f in p.items()} for c, p in plans.items()},
        "pairs": pairs, "idx": {c: list(r) for c, r in idx.items()}, "baseline": base,
        "nll_baseline": nll_base,
        "matrix": matrix, "bands": bands,
        "matched": {"draws": draws, "observed_gap": obs, "p_value": p_val,
                    "ci_own": ci_own, "ci_other": ci_oth},
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nsaved {OUT_TXT}\nsaved {OUT_JSON}")


if __name__ == "__main__":
    main()
