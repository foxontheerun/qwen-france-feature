"""SAE sanity check: reconstruction quality per layer.

Loads the SAEs and reports L0 (active features) and relative MSE on a single
prompt, confirming the checkpoints are wired up correctly before we trust any
downstream feature.
"""
import _bootstrap  # noqa: F401
import torch

from france_feature import RECON_LAYERS, capture, load_model, load_saes


def main():
    model, tokenizer, device = load_model()
    print(f"device={device}  dtype={next(model.parameters()).dtype}")
    print(f"hidden_size={model.config.hidden_size}  layers={model.config.num_hidden_layers}")

    saes = load_saes(model, RECON_LAYERS, device)

    prompt = "The capital of France is"
    print(f"\nprompt: {prompt!r}\n")
    print(f"{'layer':>5} {'L0':>6} {'MSE':>10} {'rel.MSE':>9}")
    print("-" * 34)
    for layer in RECON_LAYERS:
        h, _ = capture(model, tokenizer, prompt, layer, device)
        sae = saes[layer]
        x = h.to(sae.W_enc.dtype)
        with torch.no_grad():
            z = sae.encode(x)
            x_hat = sae.decode(z)
        mse = (x.float() - x_hat.float()).pow(2).mean().item()
        var = x.float().pow(2).mean().item()
        l0 = (z != 0).float().sum(dim=-1).mean().item()
        print(f"{layer:>5} {l0:>6.1f} {mse:>10.4f} {mse / var:>9.4f}")


if __name__ == "__main__":
    main()
