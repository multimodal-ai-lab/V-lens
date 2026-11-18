"""Fit a tuned lens (Belrose et al. 2023) for a model's text side (the text readouts of Supp. D).

For the hidden state h entering layer l, the tuned lens reads words as head(norm(h + A_l h + b_l)), with one affine
translator (A_l, b_l) per layer, started at zero (so it starts as the logit lens). It is trained by 400 Adam updates
per layer (batch 512 tokens, learning rate 1e-3) so that this distribution matches the model's own final next-word
distribution at the same token (KL divergence, final -> lens), on the same 200 passages as the text J-lens
(configs/prompts/text_lens_passages.txt), tokens from the 5th content token on. Held out: passages 200 to 219, on which
the KL of the tuned lens is reported next to the logit lens's, per layer.

Saved as $VLENS_DATA/lenses/<model>_tuned_lens_n200/tuned.pt {A: [L, d, d] fp16, b: [L, d], meta}; the held-out KL per
layer goes to $VLENS_OUT/fit_<model>_tuned_lens/numbers.json.

    python scripts/fit_tuned_lens.py --model bagel        # also show_o2, sensenova, hunyuan
"""

import argparse
import json

import torch

from vlens import prompts, results
from vlens.paths import LENS_STORE
from vlens.models import load

N_FIT, N_HELD, SKIP_FIRST = 200, 20, 4
STEPS, BATCH, LR = 400, 512, 1e-3


def head_modules(m, name):
    """The model's final norm and word head, as modules that pass gradients."""
    if name == "bagel":
        return m.net.language_model.model.norm, m.net.language_model.lm_head
    if name == "show_o2":
        return m.net.showo.model.norm, m.net.showo.lm_head
    if name == "hunyuan":
        return m.net.model.ln_f, m.net.lm_head
    return m.lm.model.norm, m.lm.lm_head


def collect(m, texts):
    """Hidden states entering every layer and the last layer's output, for the band of content tokens."""
    hs, last = [], []
    with torch.no_grad():
        for t in texts:
            r = m.read(t)
            idx = [p for p in range(r.n_tokens) if r.content[p]][SKIP_FIRST:]
            if not idx:
                continue
            hs.append(torch.stack([r.hidden[l][idx] for l in range(len(m.layers))]).to(torch.bfloat16))  # [L, n, d]
            last.append(r.last_output[idx].to(torch.bfloat16))
    return torch.cat(hs, 1), torch.cat(last, 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="bagel")
    args = ap.parse_args()
    m = load(args.model)
    norm, head = head_modules(m, args.model)
    dev = head.weight.device          # where the head sits (HunyuanImage spreads its layers over several GPUs)
    for p in list(norm.parameters()) + list(head.parameters()):
        p.requires_grad_(False)
    vocab = getattr(m, "text_vocab", None)
    pool = (prompts.PROMPTS / "text_lens_passages.txt").read_text().splitlines()
    H, last = collect(m, pool[:N_FIT])
    H_held, last_held = collect(m, pool[N_FIT:N_FIT + N_HELD])
    n_layers, n_tok, d = H.shape
    print(f"{n_tok} fitting tokens, {H_held.shape[1]} held-out tokens, {n_layers} layers, width {d}", flush=True)

    def logprobs(x):
        z = head(norm(x.to(dev, head.weight.dtype))).float()
        return torch.log_softmax(z[..., :vocab] if vocab else z, -1)

    def kl(target_lp, lp):
        return (target_lp.exp() * (target_lp - lp)).sum(-1).mean()

    A_all, b_all, numbers = [], [], dict(model=args.model, by_layer={})
    for l in range(n_layers):
        A = torch.zeros(d, d, device=dev, requires_grad=True)
        b = torch.zeros(d, device=dev, requires_grad=True)
        opt = torch.optim.Adam([A, b], lr=LR)
        for step in range(STEPS):
            i = torch.randint(n_tok, (BATCH,))
            h = H[l, i].to(dev).float()
            with torch.no_grad():
                target = logprobs(last[i].to(dev))
            loss = kl(target, logprobs(h + h @ A.T + b))
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            tot_t, tot_l, n = 0.0, 0.0, 0
            for s in range(0, H_held.shape[1], BATCH):
                h = H_held[l, s:s + BATCH].to(dev).float()
                target = logprobs(last_held[s:s + BATCH].to(dev))
                tot_t += float(kl(target, logprobs(h + h @ A.T + b))) * len(h)
                tot_l += float(kl(target, logprobs(h))) * len(h)
                n += len(h)
        numbers["by_layer"][l] = dict(kl_tuned=tot_t / n, kl_logit=tot_l / n)
        print(f"layer {l:2d}: held-out KL tuned lens {tot_t / n:.3f}, logit lens {tot_l / n:.3f}", flush=True)
        A_all.append(A.detach().half().cpu())
        b_all.append(b.detach().float().cpu())
    out = LENS_STORE / f"{args.model}_tuned_lens_n{N_FIT}"
    out.mkdir(parents=True, exist_ok=True)
    torch.save(dict(A=torch.stack(A_all), b=torch.stack(b_all),
                    meta=dict(model=args.model, n_passages=N_FIT, steps=STEPS, lr=LR, batch=BATCH)), out / "tuned.pt")
    res = results.folder(f"fit_{args.model}_tuned_lens")
    (res / "numbers.json").write_text(json.dumps(numbers, indent=1))
    print("saved", out)


if __name__ == "__main__":
    main()
