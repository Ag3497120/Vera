#!/usr/bin/env python3
"""Functional Matryoshka test of the shared residual-stream PCA basis.

ACTIVATION_PROBE.md established that one shared PCA basis (trace-normalized
pooled covariance over per-layer-centered block outputs) reconstructs every
GPT-2 small residual-stream layer to ~0.30 rel err at r=64, with a
shared/per-layer ratio of ~1.09 flat in rank. PCA prefixes are nested by
construction (top-16 subset of top-32 subset of ...), so the Matryoshka
question is FUNCTIONAL: if the model's residual stream is forced through the
top-r shared subspace at every block output, does prediction degrade
gracefully with r (Matryoshka) or fall off a cliff?

Protocol
--------
1. Fit on wikitext-2-raw-v1 (train), ~100k tokens, seq_len 256 (same recipe
   as activation_shared_probe.py): collect block outputs
   (hidden_states[l+1], 12 layers), per-layer means, per-layer covariances;
   shared basis = eigvecs of trace-normalized pooled covariance, up to r=256.
   Controls: per-layer PCA bases (upper bound), one random orthonormal basis
   (chance floor; nested prefixes of a single 768x256 Q).
2. Patch on held-out wikitext-2-raw-v1 (test): 40 seqs x 256 tok = 10,240
   tokens. For each rank r in {8,16,32,64,128,256}, forward hooks on block
   outputs replace h with mean_l + P_r P_r^T (h - mean_l):
   a) ALL-LAYER: every block output patched (the stereo-cross claim);
   b) SINGLE-LAYER: only layer 6 (all three bases); layers 3 and 9
      shared-basis only.
3. Metrics vs the unpatched model on the same tokens:
   KL(base || patched) mean over positions; next-token top-1 agreement with
   the unpatched model; patched next-token accuracy and perplexity
   (ppl ratio = patched/base).

Pre-registered fork
-------------------
- MATRYOSHKA_FUNCTIONAL: all-layer degradation graceful: top-1 agreement
  >= ~0.85 at r=128 and >= ~0.7 at r=64, monotone improvement with r,
  shared clearly beats random at every r and approaches per-layer PCA.
- CLIFF / NOT_FUNCTIONAL: all-layer agreement < ~0.5 even at r=128, or
  shared ~= random.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------
# corpus (same packing recipe as activation_shared_probe.py)
# --------------------------------------------------------------------------

def build_corpus_texts(split: str, min_chars: int = 200):
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    texts = [t for t in ds["text"] if len(t.strip()) > min_chars]
    return texts, f"wikitext-2-raw-v1({split})"


def tokenize_corpus(texts, tokenizer, seq_len: int, target_tokens: int, seed: int):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(texts))
    seqs, buf = [], []
    need = target_tokens // seq_len
    for i in order:
        ids = tokenizer(texts[int(i)])["input_ids"]
        buf.extend(ids)
        while len(buf) >= seq_len:
            seqs.append(buf[:seq_len])
            buf = buf[seq_len:]
        if len(seqs) >= need:
            break
    return np.asarray(seqs[:need], dtype=np.int64)


# --------------------------------------------------------------------------
# basis fitting
# --------------------------------------------------------------------------

def fit_bases(model, seqs: np.ndarray, n_sample: int, seed: int, r_max: int,
              batch_size: int = 8):
    """Collect block outputs, return (means [L,d], shared basis [d,r_max],
    per-layer bases list of [d,r_max], random orthonormal basis [d,r_max])."""
    n_seq, seq_len = seqs.shape
    total = n_seq * seq_len
    n_sample = min(n_sample, total)
    rng = np.random.default_rng(seed + 1)
    flat_idx = np.sort(rng.choice(total, size=n_sample, replace=False))

    L = model.config.n_layer
    d = model.config.n_embd
    acts = [np.empty((n_sample, d), dtype=np.float32) for _ in range(L)]

    write_pos = 0
    t0 = time.time()
    for b0 in range(0, n_seq, batch_size):
        batch = torch.from_numpy(seqs[b0:b0 + batch_size])
        flat_lo, flat_hi = b0 * seq_len, (b0 + batch.shape[0]) * seq_len
        lo = np.searchsorted(flat_idx, flat_lo)
        hi = np.searchsorted(flat_idx, flat_hi)
        sel = flat_idx[lo:hi] - flat_lo
        if sel.size == 0:
            continue
        sel_t = torch.from_numpy(sel)
        with torch.no_grad():
            out = model(batch, output_hidden_states=True)
        for li in range(L):
            acts[li][write_pos:write_pos + sel.size] = (
                out.hidden_states[li + 1].reshape(-1, d)[sel_t].numpy()
            )
        write_pos += sel.size
        if (b0 // batch_size) % 10 == 0:
            print(f"  [fit] batch {b0 // batch_size + 1}/"
                  f"{(n_seq + batch_size - 1) // batch_size} "
                  f"sampled={write_pos}/{n_sample} {time.time() - t0:.0f}s")
    assert write_pos == n_sample

    means = np.stack([X.mean(axis=0) for X in acts])  # (L, d)
    covs, traces = [], []
    for li, X in enumerate(acts):
        Xc = (X - means[li]).astype(np.float64)
        C = Xc.T @ Xc
        covs.append(C)
        traces.append(float(np.trace(C)))

    # shared basis: trace-normalized pooled covariance (equal layer weight)
    C_pool = sum(C / t for C, t in zip(covs, traces))
    _, V = np.linalg.eigh(C_pool)
    shared = np.ascontiguousarray(V[:, ::-1][:, :r_max]).astype(np.float32)

    per_layer = []
    for C in covs:
        _, Vl = np.linalg.eigh(C)
        per_layer.append(np.ascontiguousarray(Vl[:, ::-1][:, :r_max]).astype(np.float32))

    g = np.random.default_rng(seed + 12345)
    Q, _ = np.linalg.qr(g.standard_normal((d, r_max)))
    random_basis = np.ascontiguousarray(Q).astype(np.float32)

    print(f"[fit] bases done ({time.time() - t0:.0f}s total)")
    return means.astype(np.float32), shared, per_layer, random_basis


# --------------------------------------------------------------------------
# patched evaluation
# --------------------------------------------------------------------------

def make_patch_hook(mean: torch.Tensor, basis: torch.Tensor):
    """Hook replacing block output h with mean + (h-mean) B B^T.
    GPT-2 block returns a tuple; output[0] is hidden states."""
    def hook(_mod, _inp, out):
        h = out[0]
        hc = h - mean
        h_new = mean + (hc @ basis) @ basis.T
        return (h_new,) + tuple(out[1:])
    return hook


class Accum:
    def __init__(self):
        self.kl_sum = 0.0
        self.n_pos = 0
        self.agree_sum = 0.0
        self.nll_sum = 0.0
        self.top1_correct = 0.0
        self.n_tgt = 0

    def update(self, base_logits, patched_logits, tokens):
        # KL(base || patched) and argmax agreement over all positions
        base_lp = F.log_softmax(base_logits, dim=-1)
        pat_lp = F.log_softmax(patched_logits, dim=-1)
        kl = (base_lp.exp() * (base_lp - pat_lp)).sum(-1)
        self.kl_sum += float(kl.sum())
        self.n_pos += kl.numel()
        self.agree_sum += float((base_logits.argmax(-1) == patched_logits.argmax(-1)).sum())
        # next-token loss/accuracy on shifted targets
        tgt = tokens[:, 1:]
        lp = pat_lp[:, :-1]
        self.nll_sum += float(-lp.gather(-1, tgt.unsqueeze(-1)).sum())
        self.top1_correct += float((lp.argmax(-1) == tgt).sum())
        self.n_tgt += tgt.numel()

    def result(self):
        return {
            "kl_mean": self.kl_sum / self.n_pos,
            "top1_agreement": self.agree_sum / self.n_pos,
            "nll": self.nll_sum / self.n_tgt,
            "ppl": float(np.exp(self.nll_sum / self.n_tgt)),
            "top1_accuracy": self.top1_correct / self.n_tgt,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--fit-tokens", type=int, default=100_000)
    ap.add_argument("--n-sample", type=int, default=15_000)
    ap.add_argument("--eval-seqs", type=int, default=40)
    ap.add_argument("--ranks", type=str, default="8,16,32,64,128,256")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=str(HERE / "results_matryoshka.json"))
    args = ap.parse_args()

    ranks = [int(x) for x in args.ranks.split(",")]
    r_max = max(ranks)
    torch.manual_seed(args.seed)

    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    print("[model] loading openai-community/gpt2 ...")
    tok = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
    model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
    model.eval()
    L = model.config.n_layer

    train_texts, train_name = build_corpus_texts("train")
    fit_seqs = tokenize_corpus(train_texts, tok, args.seq_len, args.fit_tokens, args.seed)
    print(f"[fit corpus] {train_name}: {fit_seqs.shape[0]} x {fit_seqs.shape[1]} "
          f"= {fit_seqs.size} tokens")
    means_np, shared_np, perlayer_np, random_np = fit_bases(
        model, fit_seqs, args.n_sample, args.seed, r_max, batch_size=8)

    means = torch.from_numpy(means_np)                    # (L, d)
    shared = torch.from_numpy(shared_np)                  # (d, r_max)
    perlayer = [torch.from_numpy(B) for B in perlayer_np]
    randb = torch.from_numpy(random_np)

    eval_texts, eval_name = build_corpus_texts("test")
    eval_seqs = tokenize_corpus(eval_texts, tok, args.seq_len,
                                args.eval_seqs * args.seq_len, args.seed + 7)
    print(f"[eval corpus] {eval_name}: {eval_seqs.shape[0]} x {eval_seqs.shape[1]} "
          f"= {eval_seqs.size} tokens")

    def basis_for(kind, layer, r):
        if kind == "shared":
            return shared[:, :r]
        if kind == "random":
            return randb[:, :r]
        if kind == "perlayer":
            return perlayer[layer][:, :r]
        raise ValueError(kind)

    # configs: (basis_kind, scope, rank); scope = "all" or a layer index
    configs = []
    for r in ranks:
        for kind in ("shared", "random", "perlayer"):
            configs.append((kind, "all", r))
        for kind in ("shared", "random", "perlayer"):
            configs.append((kind, 6, r))
        for layer in (3, 9):
            configs.append(("shared", layer, r))
    accums = {cfg: Accum() for cfg in configs}
    base_acc = Accum()  # baseline vs itself: gives base ppl/accuracy

    n_seq = eval_seqs.shape[0]
    t0 = time.time()
    for b0 in range(0, n_seq, args.batch_size):
        batch = torch.from_numpy(eval_seqs[b0:b0 + args.batch_size])
        with torch.no_grad():
            base_logits = model(batch).logits.float()
        base_acc.update(base_logits, base_logits, batch)

        for cfg in configs:
            kind, scope, r = cfg
            layers = range(L) if scope == "all" else [scope]
            handles = [
                model.transformer.h[li].register_forward_hook(
                    make_patch_hook(means[li], basis_for(kind, li, r)))
                for li in layers
            ]
            try:
                with torch.no_grad():
                    pat_logits = model(batch).logits.float()
            finally:
                for h in handles:
                    h.remove()
            accums[cfg].update(base_logits, pat_logits, batch)
        print(f"  [eval] batch {b0 // args.batch_size + 1}/"
              f"{(n_seq + args.batch_size - 1) // args.batch_size} "
              f"({time.time() - t0:.0f}s)")

    base_res = base_acc.result()
    print(f"\n[baseline] ppl={base_res['ppl']:.3f} top1_acc={base_res['top1_accuracy']:.4f}")

    results = {
        "model": "openai-community/gpt2",
        "fit_corpus": train_name,
        "fit_tokens": int(fit_seqs.size),
        "n_sample_vectors_per_layer": int(args.n_sample),
        "eval_corpus": eval_name,
        "eval_tokens": int(eval_seqs.size),
        "seq_len": int(args.seq_len),
        "ranks": ranks,
        "patch": "h -> mean_l + P_r P_r^T (h - mean_l) on block outputs (residual stream)",
        "baseline": base_res,
        "configs": {},
    }

    def key(cfg):
        kind, scope, r = cfg
        return f"{kind}|{'all' if scope == 'all' else f'L{scope}'}|r{r}"

    for cfg in configs:
        res = accums[cfg].result()
        res["ppl_ratio"] = res["ppl"] / base_res["ppl"]
        results["configs"][key(cfg)] = res

    for scope_label, scope in [("ALL-LAYER", "all"), ("SINGLE L6", 6),
                               ("SINGLE L3 (shared only)", 3),
                               ("SINGLE L9 (shared only)", 9)]:
        print(f"\n=== {scope_label} patching ===")
        print(f"{'r':>4} | {'KL sh':>8} {'agr sh':>7} {'pplx sh':>8} | "
              f"{'KL rnd':>8} {'agr rnd':>7} {'pplx rnd':>9} | "
              f"{'KL pl':>8} {'agr pl':>7} {'pplx pl':>8}")
        for r in ranks:
            cells = []
            for kind in ("shared", "random", "perlayer"):
                cfg = (kind, scope, r)
                if cfg in accums:
                    m = results["configs"][key(cfg)]
                    cells.append(f"{m['kl_mean']:8.4f} {m['top1_agreement']:7.4f} "
                                 f"{m['ppl_ratio']:8.3f}")
                else:
                    cells.append(" " * 25)
            print(f"{r:>4} | " + " | ".join(cells))

    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[matryoshka] wrote {args.out}")


if __name__ == "__main__":
    main()
