#!/usr/bin/env python3
"""Step 5 — Cross-model join probe on the shared r=256 stereo-cross basis.

Model A: existing GPT-2-small Matryoshka student (preferred) or r256 KL
         student, frozen shared P from bases_cache_soft_distill.npz,
         GPT-2 per-layer means (12 layers).
Model B: DistilGPT2 (6 layers, 768-d) distilled into the SAME frozen P with
         DistilGPT2-specific per-layer means (P never refit). Same-width
         cross-architecture join — nontrivial because depth differs.

Protocol:
  1. Fit B means on wikitext-2 train; measure residual energy captured by
     GPT-2's P (explained-variance diagnostic). Optionally learn a small
     768→768 residual affine adapter if --adapter.
  2. Distill B: KL + 0.1 LM, ≤2000 steps, MPS, hard bottleneck all block
     outputs through frozen P.
  3. Solo evals A/B on standard 10,240 wikitext-2 test tokens.
  4. Forward-only joins:
       a) A→B coord swap at early/mid/late sites (asymmetric layer map)
       b) B→A reverse
       c) Controls: random coords, shuffled-batch same-model coords
       d) Optional hub-only stitch (top-16 / top-32 RESPONSE_MAP dims)
  5. Fork: JOIN_VIABLE / JOIN_WEAK / JOIN_FAIL / JOIN_BLOCKED

Outputs: results_cross_model_join.json, student_b_distilgpt2.pt (local),
         CROSS_MODEL_JOIN.md (written separately), cross_model_join_run.log.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
CACHE = HERE / "bases_cache_soft_distill.npz"
STUDENT_A_CANDIDATES = [
    HERE / "matryoshka_student.pt",
    HERE / "kl_distill_student_r256.pt",
]
STUDENT_B_CKPT = HERE / "student_b_distilgpt2.pt"
MEANS_B_CACHE = HERE / "means_b_distilgpt2.npz"

MODEL_A_ID = "openai-community/gpt2"
MODEL_B_ID = "distilbert/distilgpt2"

LM_WEIGHT = 0.1
TEMPERATURE = 1.0
BASELINE_PPL_A = 54.466  # GPT-2 small on this eval pack
RANK = 256

# RESPONSE_MAP top hub dims (all-layer ablation KL order)
HUB16 = [0, 10, 4, 3, 5, 11, 15, 9, 1, 14, 2, 17, 16, 34, 18, 8]
HUB32 = HUB16 + [13, 23, 32, 29, 12, 26, 20, 6, 35, 30, 31, 39, 41, 22, 25, 19]

# Asymmetric join sites: relative depth early / mid / late
# A = GPT-2 (12 blocks 0..11), B = DistilGPT2 (6 blocks 0..5)
JOIN_SITES = [
    {"name": "early", "L_a": 2, "L_b": 1},
    {"name": "mid", "L_a": 5, "L_b": 2},
    {"name": "late", "L_a": 10, "L_b": 4},
]


# --------------------------------------------------------------------------
# corpus (identical packing recipe to all prior runs)
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
# bottleneck / coord hooks
# --------------------------------------------------------------------------

def make_hard_hook(mean: torch.Tensor, P: torch.Tensor,
                   adapter: Optional[nn.Module] = None):
    def hook(_mod, _inp, out):
        h = out[0]
        hc = h - mean
        if adapter is not None:
            hc = adapter(hc)
        h_new = mean + (hc @ P) @ P.T
        return (h_new,) + tuple(out[1:])
    return hook


def attach_hard(model, means_t: torch.Tensor, P: torch.Tensor,
                adapter: Optional[nn.Module] = None):
    return [
        model.transformer.h[li].register_forward_hook(
            make_hard_hook(means_t[li], P, adapter))
        for li in range(model.config.n_layer)
    ]


class ResidualAffine(nn.Module):
    """Small 768→768 affine on centered residual (identity init)."""

    def __init__(self, d: int):
        super().__init__()
        self.linear = nn.Linear(d, d, bias=True)
        nn.init.eye_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, hc):
        # hc: (B, T, d) or (N, d)
        return self.linear(hc)


class JoinState:
    """Mutable state read by join hooks during a forward pass."""

    __slots__ = (
        "mode",          # "passthrough" | "capture" | "inject"
        "layer",         # which layer index the mode applies to
        "coords",        # captured or to-inject (B, T, r)
        "hub_mask",      # optional (r,) bool — if set, blend donor hub dims
        "own_coords",    # receiver's own coords at inject layer (for hub blend)
        "rng",           # for random control
        "control",       # None | "random" | "shuffle"
    )

    def __init__(self):
        self.mode = "passthrough"
        self.layer = -1
        self.coords = None
        self.hub_mask = None
        self.own_coords = None
        self.rng = None
        self.control = None


def make_join_hook(li: int, mean: torch.Tensor, P: torch.Tensor, state: JoinState,
                   adapter: Optional[nn.Module] = None):
    def hook(_mod, _inp, out):
        h = out[0]
        hc = h - mean
        if adapter is not None:
            hc = adapter(hc)
        # default hard bottleneck
        c = hc @ P  # (B, T, r)
        h_recon = mean + c @ P.T

        if state.mode == "capture" and li == state.layer:
            state.coords = c.detach()
            return (h_recon,) + tuple(out[1:])

        if state.mode == "inject" and li == state.layer:
            donor = state.coords
            if donor is None:
                raise RuntimeError("inject with no donor coords")
            # shape align
            if donor.shape != c.shape:
                # truncate/pad time if needed (shouldn't happen same pack)
                t = min(donor.shape[1], c.shape[1])
                donor = donor[:, :t]
                c = c[:, :t]
                h = h[:, :t]
                mean_b = mean
                # rebuild with truncated
                hc = h - mean_b
                if adapter is not None:
                    hc = adapter(hc)
                c = hc @ P

            inj = donor
            if state.control == "shuffle":
                # shuffle batch axis
                perm = torch.randperm(inj.shape[0], device=inj.device)
                inj = inj[perm]
            elif state.control == "random":
                # match per-dim std of donor
                std = inj.float().std(dim=(0, 1), keepdim=True).clamp_min(1e-6)
                inj = torch.randn_like(inj) * std

            if state.hub_mask is not None:
                mask = state.hub_mask.to(dtype=c.dtype, device=c.device)
                # keep receiver's own non-hub coords
                inj = inj * mask + c * (1.0 - mask)

            h_new = mean + inj @ P.T
            return (h_new,) + tuple(out[1:])

        return (h_recon,) + tuple(out[1:])
    return hook


def attach_join(model, means_t: torch.Tensor, P: torch.Tensor, state: JoinState,
                adapter: Optional[nn.Module] = None):
    return [
        model.transformer.h[li].register_forward_hook(
            make_join_hook(li, means_t[li], P, state, adapter))
        for li in range(model.config.n_layer)
    ]


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
        self.agree_sum += float(
            (base_logits.argmax(-1) == patched_logits.argmax(-1)).sum())
        tgt = tokens[:, 1:]
        lp = pat_lp[:, :-1]
        self.nll_sum += float(-lp.gather(-1, tgt.unsqueeze(-1)).sum())
        self.top1_correct += float((lp.argmax(-1) == tgt).sum())
        self.n_tgt += tgt.numel()

    def result(self):
        return {
            "kl_mean": self.kl_sum / max(self.n_pos, 1),
            "top1_agreement": self.agree_sum / max(self.n_pos, 1),
            "nll": self.nll_sum / max(self.n_tgt, 1),
            "ppl": float(np.exp(min(self.nll_sum / max(self.n_tgt, 1), 40.0))),
            "top1_accuracy": self.top1_correct / max(self.n_tgt, 1),
        }


def eval_vs_teacher(student, teacher, seqs, device, batch_size):
    acc = Accum()
    student.eval()
    for b0 in range(0, seqs.shape[0], batch_size):
        batch = torch.from_numpy(seqs[b0:b0 + batch_size]).to(device)
        with torch.no_grad():
            t_logits = teacher(batch).logits.float()
            s_logits = student(batch).logits.float()
        acc.update(t_logits, s_logits, batch)
    return acc.result()


def eval_own_ppl(model, seqs, device, batch_size):
    """NLL/ppl of model alone (no teacher)."""
    nll_sum = 0.0
    n_tgt = 0
    model.eval()
    for b0 in range(0, seqs.shape[0], batch_size):
        batch = torch.from_numpy(seqs[b0:b0 + batch_size]).to(device)
        with torch.no_grad():
            out = model(batch, labels=batch)
        # out.loss is mean over tokens
        n_tok = batch[:, 1:].numel()
        nll_sum += float(out.loss) * n_tok
        n_tgt += n_tok
    nll = nll_sum / max(n_tgt, 1)
    return {"nll": nll, "ppl": float(np.exp(min(nll, 40.0)))}


# --------------------------------------------------------------------------
# fit B means + explained variance of GPT-2 P on DistilGPT2 residuals
# --------------------------------------------------------------------------

def fit_means_and_ev(model, seqs, P_np, n_sample, seed, device, batch_size=8):
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
    model.eval()
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
            print(f"  [means_b] batch {b0 // batch_size + 1} "
                  f"sampled={write_pos}/{n_sample} {time.time() - t0:.0f}s",
                  flush=True)
    assert write_pos == n_sample
    means = np.stack([X.mean(axis=0) for X in acts]).astype(np.float32)

    # explained variance of centered residuals by columns of P
    P = P_np.astype(np.float64)
    ev_per_layer = []
    for li, X in enumerate(acts):
        Xc = (X - means[li]).astype(np.float64)
        total_var = float((Xc ** 2).sum())
        proj = Xc @ P
        recon = proj @ P.T
        kept = float((recon ** 2).sum())
        # also Frobenius: ||Xc||_F^2 vs ||P P^T Xc||_F^2
        ev = kept / max(total_var, 1e-12)
        # residual energy after projection
        resid = Xc - recon
        resid_frac = float((resid ** 2).sum()) / max(total_var, 1e-12)
        ev_per_layer.append({
            "layer": li,
            "explained_var_frac": ev,
            "residual_frac": resid_frac,
            "rms": float(np.sqrt((Xc ** 2).mean())),
        })
    mean_ev = float(np.mean([e["explained_var_frac"] for e in ev_per_layer]))
    print(f"[means_b] done ({time.time() - t0:.0f}s); "
          f"mean explained-var by GPT-2 P: {mean_ev:.4f}", flush=True)
    return means, ev_per_layer, mean_ev


# --------------------------------------------------------------------------
# distill B
# --------------------------------------------------------------------------

def distill_b(student, teacher, train_seqs, val_seqs, means_t, P, device,
              args, adapter=None):
    hooks = attach_hard(student, means_t, P, adapter)
    params = list(student.parameters())
    if adapter is not None:
        params += list(adapter.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)

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
    r0 = eval_vs_teacher(student, teacher, val_seqs, device, args.batch_size)
    history.append({"step": 0, **r0})
    print(f"[distill_b] step 0: val KL={r0['kl_mean']:.4f} "
          f"agr={r0['top1_agreement']:.4f} ppl={r0['ppl']:.1f}", flush=True)

    t0 = time.time()
    step_done = 0
    loss_win, kl_win = [], []
    V = student.config.vocab_size
    for step in range(args.max_steps):
        student.train()
        if adapter is not None:
            adapter.train()
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        batch = next_batch()
        with torch.no_grad():
            t_logits = teacher(batch).logits
            t_probs = F.softmax(t_logits / TEMPERATURE, dim=-1)
            t_ent = float(-(t_probs * torch.log(t_probs.clamp_min(1e-12)))
                          .sum(-1).mean())
        out = student(batch, labels=batch, use_cache=False)
        soft_ce = F.cross_entropy(out.logits.view(-1, V), t_probs.view(-1, V))
        loss = soft_ce + LM_WEIGHT * out.loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        loss_win.append(float(loss.detach()))
        kl_win.append(float(soft_ce.detach()) - t_ent)
        step_done = step + 1
        if step_done % 20 == 0:
            print(f"  [distill_b] step {step_done} loss={np.mean(loss_win):.4f} "
                  f"trainKL={np.mean(kl_win):.4f} ({time.time() - t0:.0f}s)",
                  flush=True)
            loss_win, kl_win = [], []
        if step_done % args.eval_every == 0:
            res = eval_vs_teacher(student, teacher, val_seqs, device,
                                  args.batch_size)
            history.append({"step": step_done, **res})
            print(f"[distill_b] step {step_done}: val KL={res['kl_mean']:.4f} "
                  f"agr={res['top1_agreement']:.4f} ppl={res['ppl']:.2f} "
                  f"acc={res['top1_accuracy']:.4f} ({time.time() - t0:.0f}s)",
                  flush=True)
            w = args.plateau_window
            if step_done >= args.min_steps and len(history) > w:
                best_kl_now = min(h["kl_mean"] for h in history)
                best_agr_now = max(h["top1_agreement"] for h in history)
                past = history[:-w]
                best_kl_past = min(h["kl_mean"] for h in past)
                best_agr_past = max(h["top1_agreement"] for h in past)
                kl_impr = (best_kl_past - best_kl_now) / max(best_kl_past, 1e-9)
                agr_impr = (best_agr_now - best_agr_past) / max(best_agr_past, 1e-9)
                if kl_impr < 0.01 and agr_impr < 0.01:
                    print(f"[distill_b] early stop at step {step_done} "
                          f"(plateau KL impr {kl_impr:.4f}, agr {agr_impr:.4f})",
                          flush=True)
                    break
    wall = time.time() - t0
    for h in hooks:
        h.remove()
    return history, step_done, wall


# --------------------------------------------------------------------------
# join evaluation
# --------------------------------------------------------------------------

def capture_coords(model, means_t, P, layer, seqs, device, batch_size,
                   adapter=None):
    """Return list of coord tensors (one per batch) and concat to (N,T,r)."""
    state = JoinState()
    state.mode = "capture"
    state.layer = layer
    hooks = attach_join(model, means_t, P, state, adapter)
    chunks = []
    model.eval()
    for b0 in range(0, seqs.shape[0], batch_size):
        batch = torch.from_numpy(seqs[b0:b0 + batch_size]).to(device)
        state.coords = None
        with torch.no_grad():
            _ = model(batch)
        if state.coords is None:
            raise RuntimeError(f"no coords captured at layer {layer}")
        chunks.append(state.coords.cpu())
    for h in hooks:
        h.remove()
    return torch.cat(chunks, dim=0)


def eval_joined(receiver, teacher, means_r, P, inject_layer, donor_coords,
                seqs, device, batch_size, control=None, hub_mask=None,
                adapter=None):
    state = JoinState()
    state.mode = "inject"
    state.layer = inject_layer
    state.control = control
    state.hub_mask = hub_mask
    hooks = attach_join(receiver, means_r, P, state, adapter)
    acc = Accum()
    receiver.eval()
    cursor = 0
    for b0 in range(0, seqs.shape[0], batch_size):
        batch = torch.from_numpy(seqs[b0:b0 + batch_size]).to(device)
        n = batch.shape[0]
        state.coords = donor_coords[cursor:cursor + n].to(device)
        cursor += n
        with torch.no_grad():
            t_logits = teacher(batch).logits.float()
            s_logits = receiver(batch).logits.float()
        acc.update(t_logits, s_logits, batch)
    for h in hooks:
        h.remove()
    # also own ppl via Accum already
    return acc.result()


def hub_mask_tensor(dims, rank, device):
    m = torch.zeros(rank, device=device)
    m[torch.tensor(dims, device=device)] = 1.0
    return m


# --------------------------------------------------------------------------

def resolve_student_a():
    for p in STUDENT_A_CANDIDATES:
        if p.exists():
            return p
    raise SystemExit("missing student A checkpoint "
                     "(matryoshka_student.pt or kl_distill_student_r256.pt)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=str, default="all",
                    choices=["all", "distill_b", "join", "probe_p"])
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--eval-seqs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--min-steps", type=int, default=600)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--plateau-window", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--train-batch", type=int, default=8)
    ap.add_argument("--val-seqs", type=int, default=16)
    ap.add_argument("--fit-tokens", type=int, default=100_000)
    ap.add_argument("--n-sample", type=int, default=50_000)
    ap.add_argument("--adapter", action="store_true",
                    help="learn 768→768 residual affine during B distill")
    ap.add_argument("--skip-distill", action="store_true",
                    help="load existing student_b checkpoint")
    ap.add_argument("--ev-block-threshold", type=float, default=0.15,
                    help="mean explained-var below this → JOIN_BLOCKED warning")
    ap.add_argument("--out", type=str,
                    default=str(HERE / "results_cross_model_join.json"))
    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[device] {device}", flush=True)
    torch.manual_seed(args.seed)
    wall0 = time.time()

    if not CACHE.exists():
        raise SystemExit(f"missing {CACHE}")
    z = np.load(CACHE)
    means_a_np, V_np = z["means"], z["V"]
    assert V_np.shape[0] == 768 and V_np.shape[1] >= RANK
    P_np = V_np[:, :RANK].copy()
    print(f"[basis] frozen GPT-2 shared P from {CACHE.name} "
          f"(V {V_np.shape}; r={RANK}); NOT refit for model B", flush=True)

    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    # DistilGPT2 shares GPT-2 tokenizer
    tok = GPT2TokenizerFast.from_pretrained(MODEL_A_ID)
    eval_texts, eval_name = build_corpus_texts("test")
    eval_seqs = tokenize_corpus(eval_texts, tok, args.seq_len,
                                args.eval_seqs * args.seq_len, args.seed + 7)
    print(f"[eval corpus] {eval_name}: {eval_seqs.shape[0]} x {eval_seqs.shape[1]}",
          flush=True)

    results = {
        "meta": {
            "step": 5,
            "model_a": MODEL_A_ID,
            "model_b": MODEL_B_ID,
            "model_b_choice_reason": (
                "DistilGPT2: 768-d (compatible with existing P), 6 layers "
                "(asymmetric depth vs GPT-2's 12) — real cross-architecture "
                "join at personal scale. GPT-2 medium is 1024-d (incompatible)."
            ),
            "basis": CACHE.name,
            "rank": RANK,
            "eval_corpus": eval_name,
            "eval_tokens": int(eval_seqs.size),
            "seq_len": args.seq_len,
            "device": str(device),
            "loss_b": f"forward KL T={TEMPERATURE} + {LM_WEIGHT}*LM",
            "join_sites": JOIN_SITES,
            "hub16": HUB16,
            "hub32": HUB32,
            "fork": {
                "JOIN_VIABLE": "real cross-model coords clearly beat random/"
                               "shuffle; joined ppl finite and not >>10x solo",
                "JOIN_WEAK": "beats random slightly but near-collapse vs solo",
                "JOIN_FAIL": "no better than random; shared P does not transfer",
                "JOIN_BLOCKED": "DistilGPT2 residuals do not live near GPT-2 P "
                                "(low explained var / distill collapse)",
            },
        },
    }

    # ---- Model B: means + EV diagnostic ----
    print(f"[model_b] loading teacher {MODEL_B_ID} ...", flush=True)
    teacher_b = GPT2LMHeadModel.from_pretrained(MODEL_B_ID).to(device)
    teacher_b.eval()
    for p in teacher_b.parameters():
        p.requires_grad_(False)
    assert teacher_b.config.n_embd == 768, teacher_b.config.n_embd
    print(f"[model_b] n_layer={teacher_b.config.n_layer} "
          f"n_embd={teacher_b.config.n_embd}", flush=True)

    if MEANS_B_CACHE.exists() and args.phase != "probe_p":
        zb = np.load(MEANS_B_CACHE)
        means_b_np = zb["means"]
        ev_per_layer = zb["ev_per_layer"].tolist() if "ev_per_layer" in zb else None
        mean_ev = float(zb["mean_ev"]) if "mean_ev" in zb else None
        print(f"[means_b] loaded {MEANS_B_CACHE.name} mean_ev={mean_ev}", flush=True)
        if ev_per_layer is None:
            # recompute EV quickly if missing
            train_texts, _ = build_corpus_texts("train")
            fit_seqs = tokenize_corpus(train_texts, tok, args.seq_len,
                                       args.fit_tokens, args.seed)
            means_b_np, ev_per_layer, mean_ev = fit_means_and_ev(
                teacher_b, fit_seqs, P_np, args.n_sample, args.seed, device,
                batch_size=args.batch_size)
            np.savez(MEANS_B_CACHE, means=means_b_np, mean_ev=np.float64(mean_ev))
    else:
        train_texts, _ = build_corpus_texts("train")
        fit_seqs = tokenize_corpus(train_texts, tok, args.seq_len,
                                   args.fit_tokens, args.seed)
        print(f"[means_b] fitting DistilGPT2 means on {fit_seqs.shape[0]} seqs ...",
              flush=True)
        means_b_np, ev_per_layer, mean_ev = fit_means_and_ev(
            teacher_b, fit_seqs, P_np, args.n_sample, args.seed, device,
            batch_size=args.batch_size)
        np.savez(MEANS_B_CACHE, means=means_b_np, mean_ev=np.float64(mean_ev))
        # also store JSON-serializable EV alongside results
        print(f"[means_b] saved {MEANS_B_CACHE.name}", flush=True)

    results["basis_transfer"] = {
        "mean_explained_var_frac": mean_ev,
        "per_layer": ev_per_layer,
        "ev_block_threshold": args.ev_block_threshold,
        "note": "fraction of DistilGPT2 residual energy in span(GPT-2 P); "
                "P was fit on GPT-2, not DistilGPT2",
    }
    blocked_hint = mean_ev is not None and mean_ev < args.ev_block_threshold
    if blocked_hint:
        print(f"[warn] mean EV {mean_ev:.4f} < {args.ev_block_threshold} "
              f"— residuals poorly aligned with GPT-2 P (JOIN_BLOCKED risk)",
              flush=True)

    if args.phase == "probe_p":
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"[probe_p] wrote {args.out}")
        return

    # ---- student B ----
    adapter = ResidualAffine(768).to(device) if args.adapter else None
    student_b = GPT2LMHeadModel.from_pretrained(MODEL_B_ID).to(device)
    means_b_t = torch.from_numpy(means_b_np).to(device)
    P = torch.from_numpy(P_np).to(device)

    if args.skip_distill or (STUDENT_B_CKPT.exists() and args.phase == "join"):
        if not STUDENT_B_CKPT.exists():
            raise SystemExit(f"missing {STUDENT_B_CKPT}")
        blob = torch.load(STUDENT_B_CKPT, map_location=device, weights_only=False)
        if isinstance(blob, dict) and "model" in blob:
            student_b.load_state_dict(blob["model"])
            if adapter is not None and "adapter" in blob:
                adapter.load_state_dict(blob["adapter"])
            results["student_b_train"] = blob.get("train_meta", {"loaded": True})
        else:
            student_b.load_state_dict(blob)
            results["student_b_train"] = {"loaded": True, "ckpt": STUDENT_B_CKPT.name}
        print(f"[student_b] loaded {STUDENT_B_CKPT.name}", flush=True)
    elif args.phase in ("all", "distill_b"):
        # cold start DistilGPT2 + hard P bottleneck
        print("[distill_b] step-0 (cold + hard P) ...", flush=True)
        hooks0 = attach_hard(student_b, means_b_t, P, adapter)
        step0 = eval_vs_teacher(student_b, teacher_b, eval_seqs, device,
                                args.batch_size)
        for h in hooks0:
            h.remove()
        print(f"[distill_b] STEP-0 TEST: KL={step0['kl_mean']:.4f} "
              f"agr={step0['top1_agreement']:.4f} ppl={step0['ppl']:.2f}",
              flush=True)

        train_texts, _ = build_corpus_texts("train")
        need_tokens = args.max_steps * args.train_batch * args.seq_len
        train_seqs = tokenize_corpus(train_texts, tok, args.seq_len,
                                     need_tokens, args.seed + 31)
        val_texts, _ = build_corpus_texts("validation")
        val_seqs = tokenize_corpus(val_texts, tok, args.seq_len,
                                   args.val_seqs * args.seq_len, args.seed + 101)
        print(f"[data] train seqs={train_seqs.shape[0]} val={val_seqs.shape[0]}",
              flush=True)

        history, steps, wall = distill_b(
            student_b, teacher_b, train_seqs, val_seqs, means_b_t, P, device,
            args, adapter)
        # final test with hooks
        hooks_f = attach_hard(student_b, means_b_t, P, adapter)
        test_b = eval_vs_teacher(student_b, teacher_b, eval_seqs, device,
                                 args.batch_size)
        for h in hooks_f:
            h.remove()
        print(f"[distill_b] TEST: KL={test_b['kl_mean']:.4f} "
              f"agr={test_b['top1_agreement']:.4f} ppl={test_b['ppl']:.2f}",
              flush=True)

        save_blob = {
            "model": student_b.state_dict(),
            "adapter": adapter.state_dict() if adapter is not None else None,
            "means_b": means_b_np,
            "train_meta": {
                "steps": steps,
                "wall_time_s": wall,
                "lr": args.lr,
                "adapter": bool(adapter),
                "step0_test": step0,
                "final_test": test_b,
                "history_val": history,
            },
        }
        torch.save(save_blob, STUDENT_B_CKPT)
        print(f"[distill_b] saved {STUDENT_B_CKPT.name}", flush=True)
        results["student_b_train"] = {
            "steps": steps,
            "wall_time_s": wall,
            "lr": args.lr,
            "adapter": bool(adapter),
            "step0_test": step0,
            "final_test": test_b,
            "history_val": history,
        }
        if args.phase == "distill_b":
            results["meta"]["wall_time_s"] = time.time() - wall0
            Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(f"[distill_b] wrote {args.out}")
            return

    # ---- student A ----
    ckpt_a = resolve_student_a()
    print(f"[student_a] loading {ckpt_a.name} ...", flush=True)
    student_a = GPT2LMHeadModel.from_pretrained(MODEL_A_ID).to(device)
    sd = torch.load(ckpt_a, map_location=device, weights_only=True)
    missing, unexpected = student_a.load_state_dict(sd, strict=False)
    print(f"[student_a] missing={list(missing)} unexpected={list(unexpected)}",
          flush=True)
    teacher_a = GPT2LMHeadModel.from_pretrained(MODEL_A_ID).to(device)
    teacher_a.eval()
    for p in teacher_a.parameters():
        p.requires_grad_(False)
    means_a_t = torch.from_numpy(means_a_np).to(device)
    assert means_a_np.shape[0] == student_a.config.n_layer

    # attach hard bottlenecks for solo eval
    hooks_a = attach_hard(student_a, means_a_t, P, None)
    hooks_b = attach_hard(student_b, means_b_t, P, adapter)

    print("[solo] evaluating A vs teacher A ...", flush=True)
    solo_a = eval_vs_teacher(student_a, teacher_a, eval_seqs, device,
                             args.batch_size)
    print(f"[solo A] KL={solo_a['kl_mean']:.4f} agr={solo_a['top1_agreement']:.4f} "
          f"ppl={solo_a['ppl']:.2f}", flush=True)
    print("[solo] evaluating B vs teacher B ...", flush=True)
    solo_b = eval_vs_teacher(student_b, teacher_b, eval_seqs, device,
                             args.batch_size)
    print(f"[solo B] KL={solo_b['kl_mean']:.4f} agr={solo_b['top1_agreement']:.4f} "
          f"ppl={solo_b['ppl']:.2f}", flush=True)

    # teacher B baseline ppl (no bottleneck)
    teacher_b_ppl = eval_own_ppl(teacher_b, eval_seqs, device, args.batch_size)
    teacher_a_ppl = eval_own_ppl(teacher_a, eval_seqs, device, args.batch_size)
    results["solo"] = {
        "student_a_ckpt": ckpt_a.name,
        "student_a_vs_teacher_a": solo_a,
        "student_b_vs_teacher_b": solo_b,
        "teacher_a_own_ppl": teacher_a_ppl,
        "teacher_b_own_ppl": teacher_b_ppl,
        "baseline_ppl_a_ref": BASELINE_PPL_A,
    }

    for h in hooks_a:
        h.remove()
    for h in hooks_b:
        h.remove()

    # ---- join experiments ----
    print("[join] capturing donor coords + evaluating joins ...", flush=True)
    t_join = time.time()
    join_results = {"A_to_B": {}, "B_to_A": {}}

    # Pre-capture all needed donor coords per site
    donor_cache = {}
    for site in JOIN_SITES:
        key_a = ("A", site["L_a"])
        key_b = ("B", site["L_b"])
        if key_a not in donor_cache:
            print(f"  [capture] A layer {site['L_a']} ...", flush=True)
            donor_cache[key_a] = capture_coords(
                student_a, means_a_t, P, site["L_a"], eval_seqs, device,
                args.batch_size, None)
        if key_b not in donor_cache:
            print(f"  [capture] B layer {site['L_b']} ...", flush=True)
            donor_cache[key_b] = capture_coords(
                student_b, means_b_t, P, site["L_b"], eval_seqs, device,
                args.batch_size, adapter)

    # Also same-model self coords for shuffle controls at inject layers
    for site in JOIN_SITES:
        key_b_self = ("B_self", site["L_b"])
        key_a_self = ("A_self", site["L_a"])
        if key_b_self not in donor_cache:
            donor_cache[key_b_self] = donor_cache[("B", site["L_b"])]
        if key_a_self not in donor_cache:
            donor_cache[key_a_self] = donor_cache[("A", site["L_a"])]

    hub_masks = {
        "all256": None,
        "hub16": hub_mask_tensor(HUB16, RANK, device),
        "hub32": hub_mask_tensor(HUB32, RANK, device),
    }

    def run_direction(direction, donor_key_fn, recv_model, recv_means, recv_teacher,
                      inject_key, recv_adapter):
        out = {}
        for site in JOIN_SITES:
            name = site["name"]
            L_inj = site[inject_key]
            donor_coords = donor_cache[donor_key_fn(site)]
            # same-model coords for shuffle (receiver's own layer)
            if direction == "A_to_B":
                self_coords = donor_cache[("B", site["L_b"])]
            else:
                self_coords = donor_cache[("A", site["L_a"])]

            site_out = {}
            # real cross-model
            for hub_name, hmask in hub_masks.items():
                tag = f"real_{hub_name}"
                print(f"  [{direction}] {name} L_inj={L_inj} {tag} ...", flush=True)
                res = eval_joined(
                    recv_model, recv_teacher, recv_means, P, L_inj, donor_coords,
                    eval_seqs, device, args.batch_size, control=None,
                    hub_mask=hmask, adapter=recv_adapter)
                site_out[tag] = res
                print(f"    ppl={res['ppl']:.2f} agr={res['top1_agreement']:.4f} "
                      f"KL={res['kl_mean']:.4f}", flush=True)

            # controls (full 256)
            print(f"  [{direction}] {name} random ...", flush=True)
            site_out["random"] = eval_joined(
                recv_model, recv_teacher, recv_means, P, L_inj, donor_coords,
                eval_seqs, device, args.batch_size, control="random",
                hub_mask=None, adapter=recv_adapter)
            print(f"    ppl={site_out['random']['ppl']:.2f} "
                  f"agr={site_out['random']['top1_agreement']:.4f}", flush=True)

            print(f"  [{direction}] {name} shuffle_self ...", flush=True)
            site_out["shuffle_self"] = eval_joined(
                recv_model, recv_teacher, recv_means, P, L_inj, self_coords,
                eval_seqs, device, args.batch_size, control="shuffle",
                hub_mask=None, adapter=recv_adapter)
            print(f"    ppl={site_out['shuffle_self']['ppl']:.2f} "
                  f"agr={site_out['shuffle_self']['top1_agreement']:.4f}",
                  flush=True)

            # identity control: inject receiver's own unshuffled coords
            # (should ≈ solo; sanity)
            print(f"  [{direction}] {name} identity_self ...", flush=True)
            site_out["identity_self"] = eval_joined(
                recv_model, recv_teacher, recv_means, P, L_inj, self_coords,
                eval_seqs, device, args.batch_size, control=None,
                hub_mask=None, adapter=recv_adapter)
            print(f"    ppl={site_out['identity_self']['ppl']:.2f} "
                  f"agr={site_out['identity_self']['top1_agreement']:.4f}",
                  flush=True)

            out[name] = {
                "L_a": site["L_a"], "L_b": site["L_b"],
                "inject_layer": L_inj, **site_out,
            }
        return out

    join_results["A_to_B"] = run_direction(
        "A_to_B",
        lambda s: ("A", s["L_a"]),
        student_b, means_b_t, teacher_b, "L_b", adapter,
    )
    join_results["B_to_A"] = run_direction(
        "B_to_A",
        lambda s: ("B", s["L_b"]),
        student_a, means_a_t, teacher_a, "L_a", None,
    )
    join_results["wall_time_s"] = time.time() - t_join
    results["joins"] = join_results

    # ---- verdict ----
    def best_real(direction_dict):
        rows = []
        for site_name, site in direction_dict.items():
            if not isinstance(site, dict) or "real_all256" not in site:
                continue
            rows.append((site_name, site))
        return rows

    def gap_vs_controls(site):
        real = site["real_all256"]
        rnd = site["random"]
        shuf = site["shuffle_self"]
        # lower ppl better; higher agr better
        ppl_gap_rand = rnd["ppl"] / max(real["ppl"], 1e-9)
        ppl_gap_shuf = shuf["ppl"] / max(real["ppl"], 1e-9)
        agr_gap_rand = real["top1_agreement"] - rnd["top1_agreement"]
        agr_gap_shuf = real["top1_agreement"] - shuf["top1_agreement"]
        return {
            "real_ppl": real["ppl"],
            "real_agr": real["top1_agreement"],
            "random_ppl": rnd["ppl"],
            "shuffle_ppl": shuf["ppl"],
            "ppl_ratio_random_over_real": ppl_gap_rand,
            "ppl_ratio_shuffle_over_real": ppl_gap_shuf,
            "agr_gap_vs_random": agr_gap_rand,
            "agr_gap_vs_shuffle": agr_gap_shuf,
            "beats_random": (real["ppl"] < rnd["ppl"] * 0.95) or (agr_gap_rand > 0.02),
            "beats_shuffle": (real["ppl"] < shuf["ppl"] * 0.95) or (agr_gap_shuf > 0.02),
        }

    verdict_details = {"A_to_B": {}, "B_to_A": {}}
    beats_any = False
    clear_beats = False
    collapsed = False
    for direction in ("A_to_B", "B_to_A"):
        solo = solo_b if direction == "A_to_B" else solo_a
        for site_name, site in best_real(join_results[direction]):
            g = gap_vs_controls(site)
            g["solo_ppl"] = solo["ppl"]
            g["ppl_over_solo"] = g["real_ppl"] / max(solo["ppl"], 1e-9)
            if g["ppl_over_solo"] > 10.0 or not np.isfinite(g["real_ppl"]):
                collapsed = True
            if g["beats_random"] and g["beats_shuffle"]:
                beats_any = True
                if g["ppl_over_solo"] < 5.0 and (
                        g["agr_gap_vs_random"] > 0.05
                        or g["ppl_ratio_random_over_real"] > 1.5):
                    clear_beats = True
            verdict_details[direction][site_name] = g

    if blocked_hint and (solo_b["ppl"] > 10 * teacher_b_ppl["ppl"]
                         or solo_b["top1_agreement"] < 0.15):
        fork = "JOIN_BLOCKED"
    elif clear_beats and not collapsed:
        fork = "JOIN_VIABLE"
    elif beats_any and not collapsed:
        fork = "JOIN_WEAK"
    elif beats_any and collapsed:
        fork = "JOIN_WEAK"
    else:
        fork = "JOIN_FAIL"

    results["verdict"] = {
        "fork": fork,
        "details": verdict_details,
        "blocked_hint_from_ev": blocked_hint,
        "mean_ev": mean_ev,
    }
    results["meta"]["wall_time_s"] = time.time() - wall0
    results["meta"]["adapter"] = bool(adapter)

    out_path = Path(args.out)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[verdict] {fork}", flush=True)
    print(f"[done] wrote {out_path} wall={results['meta']['wall_time_s']:.0f}s",
          flush=True)


if __name__ == "__main__":
    main()
