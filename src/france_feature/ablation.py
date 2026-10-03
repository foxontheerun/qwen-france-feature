"""Multi-layer ablation helpers: find a concept feature per layer, zero it, score.

Feature ids are per-SAE — 20583 means "France" only in the layer-12 dictionary —
so the concept feature has to be located separately on every layer.
"""
from contextlib import ExitStack

import torch

from .activations import capture, to_inputs
from .hooks import hooked, make_ablate_hook


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
