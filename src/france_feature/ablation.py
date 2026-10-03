"""Multi-layer ablation helpers: find a concept feature per layer, zero it, score.

Feature ids are per-SAE — 20583 means "France" only in the layer-12 dictionary —
so the concept feature has to be located separately on every layer.
"""
from contextlib import ExitStack

import torch

from .activations import capture, to_inputs
from .hooks import hooked, make_ablate_hook, make_capture_hook


def max_acts(model, tokenizer, sae, layer, prompts, device):
    """Per-prompt max-over-positions SAE activation, shape ``(n_prompts, d_sae)``."""
    rows = []
    for p in prompts:
        h, _ = capture(model, tokenizer, p, layer, device)
        z = sae.encode(h.to(sae.W_enc.dtype))[0]
        rows.append(z.max(dim=0).values.float())
    return torch.stack(rows)


def contrast_scores(model, tokenizer, sae, layer, device, concept_prompts, control_prompts):
    """``min`` over concept prompts minus ``max`` over controls, per feature.

    The min makes a feature earn its score on every concept prompt (every
    language), not on one of them.
    """
    fr = max_acts(model, tokenizer, sae, layer, concept_prompts, device)
    ctrl = max_acts(model, tokenizer, sae, layer, control_prompts, device)
    return fr.min(dim=0).values - ctrl.max(dim=0).values


def find_concept_features(scores, top_n):
    """Top-``top_n`` features with a positive contrast score: ``[(id, score)]``."""
    vals, ids = scores.topk(top_n)
    return [(i, v) for i, v in zip(ids.tolist(), vals.tolist()) if v > 0]


def multi_hooked(model, saes, plan):
    """Ablate ``plan[layer] = [feature ids]`` on all listed layers at once."""
    stack = ExitStack()
    for layer, fids in plan.items():
        for fid in fids:
            stack.enter_context(hooked(model, layer, make_ablate_hook(saes[layer], fid)))
    return stack


def score_targets(model, tokenizer, device, pairs, ctx):
    """For each ``(prompt, target)``: P(target's first token), its rank, top-1 token."""
    out = []
    with ctx:
        for prompt, target in pairs:
            ids = to_inputs(tokenizer, prompt, device)
            tid = tokenizer.encode(target, add_special_tokens=False)[0]
            with torch.no_grad():
                logits = model(**ids).logits[0, -1].float()
            prob = logits.softmax(-1)[tid].item()
            rank = int((logits > logits[tid]).sum().item()) + 1
            out.append((prob, rank, tokenizer.decode(logits.argmax().item())))
    return out


def mean_nll(model, tokenizer, device, texts, ctx):
    """Mean next-token negative log-likelihood over ``texts``."""
    total, count = 0.0, 0
    with ctx:
        for t in texts:
            ids = to_inputs(tokenizer, t, device)
            with torch.no_grad():
                logits = model(**ids).logits[0, :-1].float()
            target = ids["input_ids"][0, 1:]
            nll = torch.nn.functional.cross_entropy(logits, target, reduction="sum")
            total += nll.item()
            count += target.numel()
    return total / count


def max_acts_all_layers(model, tokenizer, saes, prompts, device):
    """Like ``max_acts`` for every layer in ``saes``, one forward pass per prompt.

    Returns ``{layer: (n_prompts, d_sae)}``.
    """
    rows = {L: [] for L in saes}
    for p in prompts:
        store = {}
        with ExitStack() as stack:
            for L in saes:
                stack.enter_context(hooked(model, L, make_capture_hook(store, key=L)))
            ids = to_inputs(tokenizer, p, device)
            with torch.no_grad():
                model(**ids)
        for L, sae in saes.items():
            z = sae.encode(store[L].to(sae.W_enc.dtype))[0]
            rows[L].append(z.max(dim=0).values.float())
    return {L: torch.stack(r) for L, r in rows.items()}


def contrast_scores_all_layers(model, tokenizer, saes, device, concept_prompts, control_prompts):
    """``contrast_scores`` for every layer, sharing forward passes: ``{layer: scores}``."""
    fr = max_acts_all_layers(model, tokenizer, saes, concept_prompts, device)
    ctrl = max_acts_all_layers(model, tokenizer, saes, control_prompts, device)
    return {L: fr[L].min(dim=0).values - ctrl[L].max(dim=0).values for L in saes}


def matched_pool(scores, strength, fid, lo, hi):
    """Non-concept features whose strength is within ``lo..hi`` x feature ``fid``'s.

    ``strength`` is max-over-positions activation x decoder-column norm, i.e. the
    size of the vector an ablation actually removes; ``scores <= 0`` excludes
    anything selective for the concept.
    """
    s = strength[fid]
    ok = (scores <= 0) & (strength >= lo * s) & (strength <= hi * s)
    ok[fid] = False
    return ok.nonzero(as_tuple=True)[0].tolist()
