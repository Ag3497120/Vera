#!/usr/bin/env python3
"""Can the all-layer shared-basis bottleneck be made functional?

Follow-up to matryoshka_patch.py / MATRYOSHKA.md. There we found that hard
projection of every block output onto the top-r shared PCA basis collapses
GPT-2 small (top-1 agreement 0.169 at r=128, ppl 27x), but a single-layer
projection degrades gracefully, and per-layer PCA collapses identically -->
the failure is the COMPOUNDING of 12 hard projections, not the shared
coordinates. Two phases:

PHASE 1 (training-free diagnostics)
  1. Compounding curve: hard shared projection at r=128 applied to k layers
     spread evenly through the stack, k in {1,2,4,6,12}
     (layers = round((i+1)*12/(k+1)), i=0..k-1; k=1 -> L6, k=2 -> L4,L8).
  2. Soft all-layer variants at effective rank 64 / 128:
     a) residual shrinkage  h -> mean + PP^T hc + alpha (I-PP^T) hc,
        alpha in {0.1,0.25,0.5}.  NOTE: alpha>0 leaks the full residual
        stream, so this is a DIAGNOSTIC, not an honest bottleneck.
     b) Wiener-style spectral soft projection: weight ALL 768 shared-basis
        components by w_i = lambda_i/(lambda_i+sigma2), sigma2 chosen so
        sum_i w_i = r_eff in {64,128}. A fixed linear map -> honest
        bottleneck variant (soft rather than hard rank truncation).

PHASE 2 (bottleneck-aware distillation)
  Insert the hard shared bottleneck (frozen basis+means) at all 12 block
  outputs and fine-tune the model weights on wikitext-2 train with plain LM
  loss (AdamW 5e-5, warmup 50, grad clip 1.0, batch 8 x 256, <=1000 steps,
  early stop on val-ppl plateau). Eval every 100 steps on held-out
  validation slice (ppl through bottleneck + top-1 agreement vs the ORIGINAL
  unpatched GPT-2). Final eval on the same wikitext-2 test set as
  matryoshka_patch.py.

Pre-registered fork (phase 2, r=128)
  CONTAINER_VIABLE:        ppl <= ~1.5x original baseline (<=82) and/or
                           agreement >= 0.6 after short FT.
  CONTAINER_NOT_RECOVERED: ppl still >3x, agreement <0.4.
  Between: state honestly.

Same fit recipe as before: wikitext-2-raw-v1 train, 390 x 256 tok,
15,000 sampled vectors/layer, per-layer means, shared basis = eigvecs of
trace-normalized pooled covariance of per-layer-centered block outputs.
Eval: wikitext-2-raw-v1 test, 40 x 256 tok (identical token sequences to
the prior run: same tokenizer recipe and seeds).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
CACHE = HERE / "bases_cache_soft_distill.npz"


# --------------------------------------------------------------------------
# corpus (identical packing recipe to matryoshka_patch.py)
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
# shared basis fit (same recipe; keeps FULL eigendecomposition for Wiener)
# --------------------------------------------------------------------------

def fit_shared_basis(model, seqs: np.ndarray, n_sample: int, seed: int,
                     device: torch.device, batch_size: int = 8):
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
        batch = torch.from_numpy(seqs[b0:b0 + batch_size]).to(device)
        flat_lo = b0 * seq_len
        flat_hi = (b0 + batch.shape[0]) * seq_len
        lo = np.searchsorted(flat_idx, flat_lo)
        hi = np.searchsorted(flat_idx, flat_hi)
        sel = flat_idx[lo:hi] - flat_lo
        if sel.size == 0:
            continue
        sel_t = torch.from_numpy(sel).to(device)
        with torch.no_grad():
            out = model(batch, output_hidden_states=True)
        for li in range(L):
            acts[li][write_pos:write_pos + sel.size] = (
                out.hidden_states[li + 1].reshape(-1, d)[sel_t].float().cpu().numpy()
            )
        write_pos += sel.size
        if (b0 // batch_size) % 10 == 0:
            print(f"  [fit] batch {b0 // batch_size + 1}/"
                  f"{(n_seq + batch_size - 1) // batch_size} "
                  f"sampled={write_pos}/{n_sample} {time.time() - t0:.0f}s")
    assert write_pos == n_sample

    means = np.stack([X.mean(axis=0) for X in acts])  # (L, d)
    C_pool = np.zeros((d, d), dtype=np.float64)
    for li, X in enumerate(acts):
        Xc = (X - means[li]).astype(np.float64)
        C = Xc.T @ Xc
        C_pool += C / float(np.trace(C))
    eigvals, V = np.linalg.eigh(C_pool)
    eigvals = np.clip(np.ascontiguousarray(eigvals[::-1]), 0.0, None)  # descending
    V = np.ascontiguousarray(V[:, ::-1])                   # (d, d)
    print(f"[fit] shared basis done ({time.time() - t0:.0f}s)")
    return means.astype(np.float32), eigvals.astype(np.float64), V.astype(np.float32)


def get_bases(model, tok, args, device):
    if CACHE.exists() and not args.refit:
        z = np.load(CACHE)
        print(f"[fit] loaded cached bases from {CACHE.name}")
        return z["means"], z["eigvals"], z["V"]
    train_texts, name = build_corpus_texts("train")
    fit_seqs = tokenize_corpus(train_texts, tok, args.seq_len, args.fit_tokens, args.seed)
    print(f"[fit corpus] {name}: {fit_seqs.shape[0]} x {fit_seqs.shape[1]} tokens")
    means, eigvals, V = fit_shared_basis(model, fit_seqs, args.n_sample,
                                         args.seed, device)
    np.savez(CACHE, means=means, eigvals=eigvals, V=V)
    return means, eigvals, V


# --------------------------------------------------------------------------
# patch hooks: h -> mean_l + f(h - mean_l), f a fixed linear map
# --------------------------------------------------------------------------

def make_hook(mean: torch.Tensor, apply_fn):
    def hook(_mod, _inp, out):
        h = out[0]
        hc = h - mean
        h_new = mean + apply_fn(hc)
        return (h_new,) + tuple(out[1:])
    return hook


def hard_fn(P):                       # P: (d, r)
    return lambda hc: (hc @ P) @ P.T


def shrink_fn(P, alpha: float):
    return lambda hc: alpha * hc + (1.0 - alpha) * ((hc @ P) @ P.T)


def wiener_fn(V, w):                  # V: (d, d), w: (d,)
    Vw = V * w                        # columns scaled by weights
    return lambda hc: (hc @ Vw) @ V.T


def solve_wiener_sigma2(eigvals: np.ndarray, target_rank: float):
    """Find sigma2 with sum_i lambda_i/(lambda_i+sigma2) = target_rank."""
    lo, hi = 1e-15, float(eigvals.max()) * 1e6

    def eff_rank(s):
        return float((eigvals / (eigvals + s)).sum())

    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if eff_rank(mid) > target_rank:
            lo = mid
        else:
            hi = mid
    s = math.sqrt(lo * hi)
    return s, eff_rank(s)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

class Accum:
    def __init__(self):
        self.kl_sum = 0.0
        self.n_pos = 0
        self.agree_sum = 0.0
        self.nll_sum = 0.0
        self.top1_correct = 0.0
        self.n_tgt = 0

    def update(self, base_logits, patched_logits, tokens):
        base_lp = F.log_softmax(base_logits, dim=-1)
        pat_lp = F.log_softmax(patched_logits, dim=-1)
        kl = (base_lp.exp() * (base_lp - pat_lp)).sum(-1)
        self.kl_sum += float(kl.sum())
        self.n_pos += kl.numel()
        self.agree_sum += float((base_logits.argmax(-1) == patched_logits.argmax(-1)).sum())
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


def spread_layers(n_layer: int, k: int):
    """k layers spread evenly: round((i+1)*n_layer/(k+1)). k=1->[6], k=2->[4,8]."""
    if k >= n_layer:
        return list(range(n_layer))
    return sorted({int(round((i + 1) * n_layer / (k + 1))) for i in range(k)})


# --------------------------------------------------------------------------
# phase 1
# --------------------------------------------------------------------------

def run_phase1(model, tok, means_t, eigvals, V_t, eval_seqs, args, device):
    L = model.config.n_layer

    # (name, layers, apply_fn factory) -- apply_fn shared across layers
    configs = []

    P128 = V_t[:, :128]
    for k in (1, 2, 4, 6, 12):
        layers = spread_layers(L, k)
        configs.append((f"compound|k{k}|layers{'-'.join(map(str, layers))}|hard_r128",
                        layers, hard_fn(P128)))

    wiener_meta = {}
    for r in (64, 128):
        P = V_t[:, :r]
        for alpha in (0.1, 0.25, 0.5):
            configs.append((f"shrink|all|r{r}|alpha{alpha}",
                            list(range(L)), shrink_fn(P, alpha)))
        sigma2, eff = solve_wiener_sigma2(eigvals, float(r))
        wiener_meta[f"wiener|all|reff{r}"] = {"sigma2": sigma2, "effective_rank": eff}
        w = torch.from_numpy(
            (eigvals / (eigvals + sigma2)).astype(np.float32)).to(device)
        configs.append((f"wiener|all|reff{r}", list(range(L)), wiener_fn(V_t, w)))
        print(f"[wiener] r_eff={r}: sigma2={sigma2:.6g} eff_rank={eff:.2f}")

    accums = {name: Accum() for name, _, _ in configs}
    base_acc = Accum()

    n_seq = eval_seqs.shape[0]
    t0 = time.time()
    for b0 in range(0, n_seq, args.batch_size):
        batch = torch.from_numpy(eval_seqs[b0:b0 + args.batch_size]).to(device)
        with torch.no_grad():
            base_logits = model(batch).logits.float()
        base_acc.update(base_logits, base_logits, batch)
        for name, layers, fn in configs:
            handles = [model.transformer.h[li].register_forward_hook(
                make_hook(means_t[li], fn)) for li in layers]
            try:
                with torch.no_grad():
                    pat_logits = model(batch).logits.float()
            finally:
                for h in handles:
                    h.remove()
            accums[name].update(base_logits, pat_logits, batch)
        print(f"  [p1 eval] batch {b0 // args.batch_size + 1}/"
              f"{(n_seq + args.batch_size - 1) // args.batch_size} "
              f"({time.time() - t0:.0f}s)")

    base_res = base_acc.result()
    out = {"baseline": base_res, "configs": {}, "wiener_meta": wiener_meta}
    print(f"\n[p1 baseline] ppl={base_res['ppl']:.3f} "
          f"top1_acc={base_res['top1_accuracy']:.4f}")
    print(f"\n{'config':<48} {'KL':>8} {'agr':>7} {'ppl':>10} {'ppl_x':>8} {'acc':>7}")
    for name, _, _ in configs:
        res = accums[name].result()
        res["ppl_ratio"] = res["ppl"] / base_res["ppl"]
        out["configs"][name] = res
        print(f"{name:<48} {res['kl_mean']:8.4f} {res['top1_agreement']:7.4f} "
              f"{res['ppl']:10.2f} {res['ppl_ratio']:8.3f} {res['top1_accuracy']:7.4f}")
    return out


# --------------------------------------------------------------------------
# phase 2: bottleneck-aware fine-tuning
# --------------------------------------------------------------------------

def eval_vs_base(model, base_model, seqs, device, batch_size):
    """model has bottleneck hooks permanently attached; base_model is clean."""
    acc = Accum()
    model.eval()
    for b0 in range(0, seqs.shape[0], batch_size):
        batch = torch.from_numpy(seqs[b0:b0 + batch_size]).to(device)
        with torch.no_grad():
            base_logits = base_model(batch).logits.float()
            pat_logits = model(batch).logits.float()
        acc.update(base_logits, pat_logits, batch)
    return acc.result()


def run_phase2(tok, means_np, V_np, eval_seqs, args, device, rank: int):
    from transformers import GPT2LMHeadModel

    print(f"\n===== PHASE 2: distill with hard shared bottleneck r={rank} =====")
    model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").to(device)
    base_model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").to(device)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad_(False)
    L = model.config.n_layer

    means_t = torch.from_numpy(means_np).to(device)
    P = torch.from_numpy(V_np[:, :rank].copy()).to(device)  # frozen buffer
    hooks = [model.transformer.h[li].register_forward_hook(
        make_hook(means_t[li], hard_fn(P))) for li in range(L)]

    # data
    train_texts, _ = build_corpus_texts("train")
    need_tokens = args.max_steps * args.train_batch * args.seq_len
    train_seqs = tokenize_corpus(train_texts, tok, args.seq_len,
                                 need_tokens, args.seed + 31)
    print(f"[p2 data] train seqs available: {train_seqs.shape[0]} "
          f"(need {need_tokens // args.seq_len}; will cycle if short)")
    val_texts, _ = build_corpus_texts("validation")
    val_seqs = tokenize_corpus(val_texts, tok, args.seq_len,
                               args.val_seqs * args.seq_len, args.seed + 101)
    print(f"[p2 data] val slice: {val_seqs.shape[0]} x {args.seq_len}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        return args.lr

    rng = np.random.default_rng(args.seed + 71)
    order = rng.permutation(train_seqs.shape[0])
    cursor = 0

    def next_batch():
        nonlocal cursor, order
        if cursor + args.train_batch > order.size:
            order = rng.permutation(train_seqs.shape[0])
            cursor = 0
        idx = order[cursor:cursor + args.train_batch]
        cursor += args.train_batch
        return torch.from_numpy(train_seqs[idx]).to(device)

    history = []
    r0 = eval_vs_base(model, base_model, val_seqs, device, args.batch_size)
    history.append({"step": 0, **r0})
    print(f"[p2 r{rank}] step 0 (untrained): val ppl={r0['ppl']:.1f} "
          f"agr={r0['top1_agreement']:.4f}")

    best_ppl = r0["ppl"]
    stall = 0
    t0 = time.time()
    step_done = 0
    loss_win = []
    for step in range(args.max_steps):
        model.train()
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        batch = next_batch()
        out = model(batch, labels=batch, use_cache=False)
        opt.zero_grad(set_to_none=True)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        loss_win.append(float(out.loss))
        step_done = step + 1
        if step_done % 20 == 0:
            print(f"  [p2 r{rank}] step {step_done} loss={np.mean(loss_win):.4f} "
                  f"({time.time() - t0:.0f}s)")
            loss_win = []
        if step_done % args.eval_every == 0:
            res = eval_vs_base(model, base_model, val_seqs, device, args.batch_size)
            history.append({"step": step_done, **res})
            print(f"[p2 r{rank}] step {step_done}: val ppl={res['ppl']:.2f} "
                  f"agr={res['top1_agreement']:.4f} KL={res['kl_mean']:.4f} "
                  f"({time.time() - t0:.0f}s)")
            if res["ppl"] < best_ppl * 0.99:
                best_ppl = res["ppl"]
                stall = 0
            else:
                stall += 1
                best_ppl = min(best_ppl, res["ppl"])
            if stall >= 2 and step_done >= args.min_steps:
                print(f"[p2 r{rank}] early stop at step {step_done} (plateau)")
                break
    wall = time.time() - t0

    # steps to 95% of the (val-ppl) improvement, at eval granularity
    ppl0 = history[0]["ppl"]
    ppl_final = min(h["ppl"] for h in history)
    thresh = ppl_final + 0.05 * (ppl0 - ppl_final)
    steps95 = next(h["step"] for h in history if h["ppl"] <= thresh)

    print(f"[p2 r{rank}] final test eval ...")
    test_res = eval_vs_base(model, base_model, eval_seqs, device, args.batch_size)

    for h in hooks:
        h.remove()
    del model
    return {
        "rank": rank,
        "train": {
            "steps": step_done,
            "wall_time_s": wall,
            "lr": args.lr,
            "warmup": args.warmup,
            "batch": [args.train_batch, args.seq_len],
            "eval_every": args.eval_every,
            "history_val": history,
            "steps_to_95pct_improvement": steps95,
            "val_ppl_start": ppl0,
            "val_ppl_best": ppl_final,
        },
        "final_test": test_res,
    }


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["1", "2", "all"], default="all")
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--fit-tokens", type=int, default=100_000)
    ap.add_argument("--n-sample", type=int, default=15_000)
    ap.add_argument("--eval-seqs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=4)   # eval batch
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto")
    # phase 2
    ap.add_argument("--ranks2", type=str, default="128,64")
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--min-steps", type=int, default=300)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--train-batch", type=int, default=8)
    ap.add_argument("--val-seqs", type=int, default=16)
    ap.add_argument("--refit", action="store_true")
    ap.add_argument("--out", type=str, default=str(HERE / "results_soft_distill.json"))
    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[device] {device}")
    torch.manual_seed(args.seed)

    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    tok = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
    model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").to(device)
    model.eval()

    means_np, eigvals, V_np = get_bases(model, tok, args, device)
    means_t = torch.from_numpy(means_np).to(device)
    V_t = torch.from_numpy(V_np).to(device)

    eval_texts, eval_name = build_corpus_texts("test")
    eval_seqs = tokenize_corpus(eval_texts, tok, args.seq_len,
                                args.eval_seqs * args.seq_len, args.seed + 7)
    print(f"[eval corpus] {eval_name}: {eval_seqs.shape[0]} x {eval_seqs.shape[1]}")

    out_path = Path(args.out)
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    results.setdefault("meta", {
        "model": "openai-community/gpt2",
        "fit_corpus": "wikitext-2-raw-v1(train)",
        "eval_corpus": eval_name,
        "eval_tokens": int(eval_seqs.size),
        "seq_len": args.seq_len,
        "device": str(device),
        "patch": "h -> mean_l + f(h - mean_l) on block outputs; f per config",
    })

    if args.phase in ("1", "all"):
        results["phase1"] = run_phase1(model, tok, means_t, eigvals, V_t,
                                       eval_seqs, args, device)
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"[phase1] wrote {out_path}")

    if args.phase in ("2", "all"):
        del model
        results.setdefault("phase2", {})
        for rank in [int(x) for x in args.ranks2.split(",")]:
            results["phase2"][f"r{rank}"] = run_phase2(
                tok, means_np, V_np, eval_seqs, args, device, rank)
            out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(f"[phase2 r{rank}] wrote {out_path}")


if __name__ == "__main__":
    main()
