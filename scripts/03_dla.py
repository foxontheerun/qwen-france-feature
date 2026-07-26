"""Direct logit attribution for the France feature.

Projects the feature's decoder direction through the unembedding to read which
output tokens it promotes or suppresses — independent of any prompt. Saves a
bar chart to figures/dla_france.png and prints a rank table for named tokens.
"""
import _bootstrap  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np

from data.prompts import DLA_CHECK_TOKENS
from france_feature import BEST_FEATURE, TARGET_LAYER, load_model, load_saes

FIG = _bootstrap.FIGURES / "dla_france.png"


def main():
    model, tokenizer, device = load_model()
    saes = load_saes(model, [TARGET_LAYER], device)
    sae = saes[TARGET_LAYER]

    w_dec_dir = sae.W_dec[:, BEST_FEATURE].float()      # (d_model,)
    w_u = model.lm_head.weight.float()                  # (vocab, d_model)
    direct = (w_u @ w_dec_dir).detach().cpu().numpy()   # effect on each logit

    top_promoted = np.argsort(direct)[-25:][::-1]
    top_suppressed = np.argsort(direct)[:25]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 9))
    for ax, tokens, title, color in [
        (ax1, top_promoted, f"Top-25 PROMOTED\nfeature {BEST_FEATURE}", "#2e7d32"),
        (ax2, top_suppressed, f"Top-25 SUPPRESSED\nfeature {BEST_FEATURE}", "#c62828"),
    ]:
        labels = [repr(tokenizer.decode([int(t)]))[1:-1][:25] for t in tokens]
        values = [direct[int(t)] for t in tokens]
        ax.barh(labels, values, color=color)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("DLA effect on logit")
        ax.invert_yaxis()
        ax.axvline(0, color="black", linewidth=0.5)
    plt.tight_layout()
    fig.savefig(FIG, dpi=150, bbox_inches="tight")
    print(f"saved {FIG}")

    order = np.argsort(-direct)
    print(f"\n{'token':>12} | {'DLA':>9} | {'rank / vocab':>18}")
    print("-" * 48)
    for s in DLA_CHECK_TOKENS:
        enc = tokenizer.encode(s, add_special_tokens=False)
        if len(enc) == 1:
            tid = enc[0]
            rank = int(np.where(order == tid)[0][0])
            print(f"  {s!r:>10} | {direct[tid]:+8.4f} | {rank:>8} / {len(direct)}")
        else:
            print(f"  {s!r:>10} | (multiple tokens)")


if __name__ == "__main__":
    main()
