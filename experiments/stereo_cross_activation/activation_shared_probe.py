#!/usr/bin/env python3
"""Part A: cross-layer SHARED structure in GPT-2 ACTIVATION space.

Contrast with experiments/stereo_cross_bridge/ where shared-basis
reconstruction of WEIGHTS lost to per-layer SVD (bridge/SVD 1.05-1.5).

Protocol
--------
1. Run a calibration corpus (wikitext-2, ~100k tokens, seq_len 256)
   through GPT-2 small; collect per-layer activations at 3 sites:
     - resid:      residual stream (block output, hidden_states[l+1]), 768-d
     - mlp_out:    MLP output after c_proj (before residual add), 768-d
     - mlp_hidden: MLP hidden after GELU, 3072-d
   Subsample a fixed random set of token positions (same positions for
   every layer, so CKA is computed on paired samples).
2. Metrics per site:
   a) Linear CKA between all layer pairs (feature-space form,
      column-centered): CKA = ||Xc^T Yc||_F^2 / (||Xc^T Xc||_F ||Yc^T Yc||_F)
   b) Shared-basis reconstruction, rank r in {16,32,64}:
      - per-layer PCA basis (fit on that layer's centered activations)
        -> relative Fro reconstruction error per layer
      - one SHARED PCA basis fit on all layers pooled
        -> relative Fro error per layer with the same shared basis
      - headline ratio = mean_l shared_err_l / per_layer_err_l
      Centering choice: each layer is mean-centered INDEPENDENTLY before
      pooling; the pooled covariance sums per-layer covariances after
      trace-normalization (each layer contributes equal total variance),
      so high-norm late layers do not dominate the shared basis.
      A raw (un-normalized) pooled variant is also reported in the JSON.

Pre-registered prediction: activation space shares much more than weight
space -> shared/per-layer < 1.10 at r=32 and high adjacent-layer CKA.
If ratio > 1.3 (weight-space-level badness), that is evidence AGAINST
the "probe activations to find shared cross structure" direction.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------
# corpus
# --------------------------------------------------------------------------

def build_corpus_texts(min_chars: int = 200):
    """wikitext-2 train docs; fallback to local repo text if datasets fails."""
    try:
        from datasets import load_dataset

        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        texts = [t for t in ds["text"] if len(t.strip()) > min_chars]
        return texts, "wikitext-2-raw-v1(train)"
    except Exception as e:  # network / dataset failure -> local fallback
        print(f"[corpus] datasets failed ({e}); falling back to local files")
        texts = []
        for p in Path(HERE.parents[1]).rglob("*.md"):
            try:
                t = p.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if len(t) > 2000:
                texts.append(t)
        return texts, "local-repo-markdown-fallback"


def tokenize_corpus(texts, tokenizer, seq_len: int, target_tokens: int, seed: int):
    """Pack documents into fixed-length token sequences (no padding)."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(texts))
    seqs = []
    buf = []
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
# activation collection
# --------------------------------------------------------------------------

def collect_activations(model, seqs: np.ndarray, n_sample: int, seed: int,
                        batch_size: int = 8):
    """Return dict site -> list of (n_sample, d) float32 arrays, one per layer."""
    n_seq, seq_len = seqs.shape
    total = n_seq * seq_len
    n_sample = min(n_sample, total)
    rng = np.random.default_rng(seed + 1)
    flat_idx = np.sort(rng.choice(total, size=n_sample, replace=False))

    n_layers = model.config.n_layer
    d_model = model.config.n_embd
    d_ff = 4 * d_model

    store = {
        "resid": [np.empty((n_sample, d_model), dtype=np.float32) for _ in range(n_layers)],
        "mlp_out": [np.empty((n_sample, d_model), dtype=np.float32) for _ in range(n_layers)],
        "mlp_hidden": [np.empty((n_sample, d_ff), dtype=np.float32) for _ in range(n_layers)],
    }

    grabbed = {}

    def make_hook(name):
        def hook(_mod, _inp, out):
            grabbed[name] = out.detach()
        return hook

    handles = []
    for li, block in enumerate(model.transformer.h):
        handles.append(block.mlp.act.register_forward_hook(make_hook(("mlp_hidden", li))))
        handles.append(block.mlp.register_forward_hook(make_hook(("mlp_out", li))))

    write_pos = 0
    t0 = time.time()
    try:
        for b0 in range(0, n_seq, batch_size):
            batch = torch.from_numpy(seqs[b0:b0 + batch_size])
            flat_lo = b0 * seq_len
            flat_hi = (b0 + batch.shape[0]) * seq_len
            lo = np.searchsorted(flat_idx, flat_lo)
            hi = np.searchsorted(flat_idx, flat_hi)
            sel = flat_idx[lo:hi] - flat_lo  # positions within this batch, flattened
            if sel.size == 0:
                continue
            sel_t = torch.from_numpy(sel)
            with torch.no_grad():
                out = model(batch, output_hidden_states=True)
            for li in range(n_layers):
                store["resid"][li][write_pos:write_pos + sel.size] = (
                    out.hidden_states[li + 1].reshape(-1, d_model)[sel_t].numpy()
                )
                store["mlp_out"][li][write_pos:write_pos + sel.size] = (
                    grabbed[("mlp_out", li)].reshape(-1, d_model)[sel_t].numpy()
                )
                store["mlp_hidden"][li][write_pos:write_pos + sel.size] = (
                    grabbed[("mlp_hidden", li)].reshape(-1, d_ff)[sel_t].numpy()
                )
            write_pos += sel.size
            grabbed.clear()
            if (b0 // batch_size) % 10 == 0:
                print(f"  batch {b0 // batch_size + 1}/{(n_seq + batch_size - 1) // batch_size}"
                      f"  sampled={write_pos}/{n_sample}  {time.time() - t0:.0f}s")
    finally:
        for h in handles:
            h.remove()
    assert write_pos == n_sample, (write_pos, n_sample)
    print(f"[collect] done: {n_sample} token vectors/layer, {time.time() - t0:.0f}s")
    return store


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def center_inplace(acts):
    """Mean-center each layer's activation matrix in place (float32)."""
    for X in acts:
        X -= X.mean(axis=0, keepdims=True)


def linear_cka_matrix(acts):
    """Linear CKA between all layer pairs; feature-space form (paired tokens).

    Expects already column-centered float32 arrays; cross-products in
    float32 (accumulated by BLAS in higher precision), norms in float64.
    """
    L = len(acts)
    self_norm = [float(np.linalg.norm(X.T @ X)) for X in acts]
    M = np.eye(L)
    for i in range(L):
        for j in range(i + 1, L):
            cross = float(np.linalg.norm(acts[i].T @ acts[j])) ** 2
            M[i, j] = M[j, i] = cross / (self_norm[i] * self_norm[j] + 1e-30)
    return M


def pca_recon_metrics(acts, ranks):
    """Per-layer PCA vs shared pooled-PCA reconstruction errors.

    Everything runs off per-layer centered covariance matrices C_l = Xc^T Xc,
    so cost is O(L d^2 n + L d^3), fine for d<=3072. Expects pre-centered
    float32 arrays.
    """
    L = len(acts)
    covs, traces = [], []
    for X in acts:
        X64 = X.astype(np.float64)
        C = X64.T @ X64
        del X64
        covs.append(C)
        traces.append(float(np.trace(C)))

    eigvals, eigvecs = [], []
    for C in covs:
        w, V = np.linalg.eigh(C)
        eigvals.append(w[::-1].copy())     # descending
        eigvecs.append(V[:, ::-1].copy())

    # shared bases from pooled covariance: normalized (equal layer weight) + raw
    C_pool_norm = sum(C / t for C, t in zip(covs, traces))
    C_pool_raw = sum(covs)
    _, Vn = np.linalg.eigh(C_pool_norm)
    _, Vr = np.linalg.eigh(C_pool_raw)
    Vn, Vr = Vn[:, ::-1], Vr[:, ::-1]

    def shared_err(Q, C, tr):
        kept = float(np.trace(Q.T @ C @ Q))
        return float(np.sqrt(max(1.0 - kept / (tr + 1e-30), 0.0)))

    out = []
    for r in ranks:
        per_layer = [float(np.sqrt(max(1.0 - eigvals[l][:r].sum() / (traces[l] + 1e-30), 0.0)))
                     for l in range(L)]
        sh_norm = [shared_err(Vn[:, :r], covs[l], traces[l]) for l in range(L)]
        sh_raw = [shared_err(Vr[:, :r], covs[l], traces[l]) for l in range(L)]
        ratios = [s / (p + 1e-12) for s, p in zip(sh_norm, per_layer)]
        ratios_raw = [s / (p + 1e-12) for s, p in zip(sh_raw, per_layer)]
        out.append({
            "rank": r,
            "per_layer_pca_rel_err": per_layer,
            "per_layer_pca_rel_err_mean": float(np.mean(per_layer)),
            "shared_pca_rel_err": sh_norm,
            "shared_pca_rel_err_mean": float(np.mean(sh_norm)),
            "ratio_shared_over_perlayer_per_layer": ratios,
            "ratio_shared_over_perlayer_mean": float(np.mean(ratios)),
            "ratio_shared_over_perlayer_max": float(np.max(ratios)),
            "ratio_of_means": float(np.mean(sh_norm) / (np.mean(per_layer) + 1e-12)),
            "shared_raw_pool_rel_err_mean": float(np.mean(sh_raw)),
            "ratio_raw_pool_mean": float(np.mean(ratios_raw)),
        })
    return out


def summarize_cka(M):
    L = M.shape[0]
    adj = [float(M[i, i + 1]) for i in range(L - 1)]
    off = M[~np.eye(L, dtype=bool)]
    return {
        "adjacent_mean": float(np.mean(adj)),
        "adjacent_min": float(np.min(adj)),
        "adjacent_per_pair": adj,
        "offdiag_mean": float(np.mean(off)),
        "offdiag_min": float(np.min(off)),
        "matrix": [[float(x) for x in row] for row in M],
    }


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--target-tokens", type=int, default=100_000)
    ap.add_argument("--n-sample", type=int, default=15_000)
    ap.add_argument("--ranks", type=str, default="16,32,64")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=str(HERE / "results_activation_probe.json"))
    args = ap.parse_args()

    ranks = [int(x) for x in args.ranks.split(",")]
    torch.manual_seed(args.seed)

    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    print("[model] loading openai-community/gpt2 ...")
    tok = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
    model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
    model.eval()

    texts, corpus_name = build_corpus_texts()
    seqs = tokenize_corpus(texts, tok, args.seq_len, args.target_tokens, args.seed)
    print(f"[corpus] {corpus_name}: {seqs.shape[0]} seqs x {seqs.shape[1]} tok "
          f"= {seqs.size} tokens; sampling {args.n_sample} vectors/layer")

    store = collect_activations(model, seqs, args.n_sample, args.seed, args.batch_size)
    del model

    results = {
        "model": "openai-community/gpt2",
        "corpus": corpus_name,
        "n_sequences": int(seqs.shape[0]),
        "seq_len": int(seqs.shape[1]),
        "total_tokens": int(seqs.size),
        "n_sample_vectors_per_layer": int(args.n_sample),
        "ranks": ranks,
        "centering": "per-layer mean-centering; shared basis from trace-normalized pooled covariance (equal layer weighting); raw pooled variant also reported",
        "sites": {},
    }

    for site in list(store.keys()):
        acts = store.pop(site)
        d = acts[0].shape[1]
        print(f"\n=== site={site} (d={d}, L={len(acts)}) ===")
        center_inplace(acts)
        t0 = time.time()
        M = linear_cka_matrix(acts)
        cka = summarize_cka(M)
        print(f"  CKA adjacent mean={cka['adjacent_mean']:.4f} "
              f"min={cka['adjacent_min']:.4f} offdiag mean={cka['offdiag_mean']:.4f} "
              f"({time.time() - t0:.0f}s)")
        t0 = time.time()
        recon = pca_recon_metrics(acts, ranks)
        for row in recon:
            print(f"  r={row['rank']:3d}  per-layer PCA err={row['per_layer_pca_rel_err_mean']:.4f}  "
                  f"shared err={row['shared_pca_rel_err_mean']:.4f}  "
                  f"ratio={row['ratio_shared_over_perlayer_mean']:.4f} "
                  f"(max {row['ratio_shared_over_perlayer_max']:.4f}, raw-pool {row['ratio_raw_pool_mean']:.4f})")
        print(f"  ({time.time() - t0:.0f}s)")
        results["sites"][site] = {"dim": int(d), "cka": cka, "recon": recon}

    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[activation_probe] wrote {args.out}")


if __name__ == "__main__":
    main()
