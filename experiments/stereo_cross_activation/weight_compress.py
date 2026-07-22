#!/usr/bin/env python3
"""STEP 4 — turn the r=256 activation container into ACTUAL WEIGHT COMPRESSION.

Prior state (KL_DISTILL_R256.md): GPT-2 small student kl_distill_student_r256.pt
runs with a frozen hard shared-PCA bottleneck at all 12 block outputs
(h -> mean_l + P P^T (h - mean_l), P = V[:, :256] from
bases_cache_soft_distill.npz). Test ppl 78.99 (1.45x baseline 54.47),
agreement 0.561. The container so far only constrains ACTIVATIONS; the
weights are still full 768-dim. This script re-expresses the block weights
through the shared 256-dim basis and measures real parameter savings.

PHASE A — analytic input-side compression (near-exact, verified):
  For blocks 1-11 the block input is exactly h = mean_{l-1} + P c (forced by
  the previous block's output hook). LayerNorm is NOT subspace-preserving:
  ln_1(h) = g * (h - mu 1)/sigma + b_ln, and (h - mu 1) lies in
  span(P, mean_{l-1}, 1), so g*(...) lies in span(diag(g) [P | mean | 1])
  — a fixed <=258-dim subspace per layer. Input-truncating c_attn with THAT
  basis (Q_l, orthonormalized) plus a bias fix
  b' = b + b_ln @ (I - Q Q^T) W is EXACT. The naive variant P~ = orth([P, 1])
  (no g, no mean, no bias fix) is evaluated too, to show LN's affine params
  matter. Block 0's input is raw wte+wpe (unprojected) so its c_attn does
  not qualify.

PHASE B — full containment:
  B1: add a mid-block projection h_mid -> mean_mid_l + P P^T (h_mid -
      mean_mid_l) after the attention residual add (fresh mid means from
      calibration data), on top of the existing output projections.
      Zero-shot eval (expect damage).
  B2: re-parameterize each block to operate natively in 256-dim coords c.
      LN handled EXACTLY in coords (chosen over reconstruct-LN-project and
      justified in WEIGHT_COMPRESS.md): for x = mean + P c the centered
      vector is v + w with v = mean - mu(mean) 1 constant and
      w = P c - mu(Pc) 1 in span(P^ = [P | q1]) (GLOBAL 257-dim, q1 = ones
      direction orthonormalized against P). Then
        ln(x) @ W + b = (r + w_coords @ A)/sigma + b_fold,
        A = P^^T diag(g) W   (257 x out, per block),
        r = (v*g) @ W, b_fold = b_ln @ W + b   (per-block vectors),
        sigma^2 = (|v|^2 + 2 v^T w + |w|^2)/768 + eps computed from
        per-block buffers |v|^2, P^^T v and w_coords.
      Residual-side: attn c_proj -> B = W_cp @ P (768x256), mlp c_proj ->
      B2 = W_mcp @ P (3072x256); all means/biases fold into 256-dim d
      vectors. Block 0 hybrid: full c_attn (raw input), everything after
      the attention residual add in coords. P is stored ONCE globally
      (P^ folds away entirely; runtime needs only s = P^^T 1 and
      pbar = P^T 1/768). B2 must match B1 up to float error — correctness
      check.
  B5: healing distillation from original GPT-2 teacher into the compressed
      parameterization (train the factorized matrices + embeddings + ln_f;
      P and the container geometry buffers frozen). Forward KL + 0.1 LM,
      lr 5e-5, batch 8x256, <=2000 steps, eval every 100, plateau stop,
      MPS float32. Final eval on the standard 10,240 test tokens.

Pre-registered fork:
  COMPRESSION_REAL: healed ppl <= ~2x baseline (<=109) at >=2.5x
                    block-weight compression.
  CONTAINER_ONLY:   healed ppl > 3x baseline.
  Between: state honestly.

Artifacts: results_weight_compress.json, WEIGHT_COMPRESS.md,
mid_means_weight_compress.npz, weight_compress_healed.pt.
Prior files untouched.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
CACHE = HERE / "bases_cache_soft_distill.npz"
STUDENT_CKPT = HERE / "kl_distill_student_r256.pt"
MID_MEANS_CACHE = HERE / "mid_means_weight_compress.npz"
HEALED_CKPT = HERE / "weight_compress_healed.pt"
OUT_JSON = HERE / "results_weight_compress.json"

RANK = 256
D = 768
LM_WEIGHT = 0.1
BASELINE_PPL = 54.466


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
# metrics (same Accum as prior runs)
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


def eval_fn_vs_teacher(logits_fn, teacher, seqs, device, batch_size):
    """logits_fn: batch tensor -> logits tensor (model set to eval by caller)."""
    acc = Accum()
    for b0 in range(0, seqs.shape[0], batch_size):
        batch = torch.from_numpy(seqs[b0:b0 + batch_size]).to(device)
        with torch.no_grad():
            t_logits = teacher(batch).logits.float()
            s_logits = logits_fn(batch).float()
        acc.update(t_logits, s_logits, batch)
    return acc.result()


# --------------------------------------------------------------------------
# hooks: output-projection (as prior runs) and mid-block projection
# --------------------------------------------------------------------------

def make_out_hook(mean: torch.Tensor, P: torch.Tensor):
    def hook(_mod, _inp, out):
        h = out[0]
        hc = h - mean
        h_new = mean + (hc @ P) @ P.T
        return (h_new,) + tuple(out[1:])
    return hook


def attach_out_bottleneck(model, means_t: torch.Tensor, P: torch.Tensor):
    return [model.transformer.h[li].register_forward_hook(
        make_out_hook(means_t[li], P)) for li in range(model.config.n_layer)]


def attach_mid_projection(model, mid_means_t: torch.Tensor, P: torch.Tensor):
    """Project h_mid (input of ln_2 AND the residual branch) onto the
    container. GPT2Block saves `residual = hidden_states` before ln_2, so a
    pre-hook on ln_2 alone only affects the MLP branch; the residual-branch
    correction (proj(h_mid) - h_mid) is added to the MLP output so that the
    block computes proj(h_mid) + mlp(ln_2(proj(h_mid))) exactly."""
    deltas = {}
    hooks = []
    for li, block in enumerate(model.transformer.h):
        def make_pre(li, m):
            def pre(_mod, args):
                h = args[0]
                hp = m + ((h - m) @ P) @ P.T
                deltas[li] = hp - h
                return (hp,) + tuple(args[1:])
            return pre

        def make_post(li):
            def post(_mod, _inp, out):
                return out + deltas.pop(li)
            return post

        hooks.append(block.ln_2.register_forward_pre_hook(make_pre(li, mid_means_t[li])))
        hooks.append(block.mlp.register_forward_hook(make_post(li)))
    return hooks


# --------------------------------------------------------------------------
# PHASE A — analytic input-side truncation of c_attn (blocks 1-11)
# --------------------------------------------------------------------------

def phase_a_apply(student, means_np, P_np, honest: bool, device):
    """Truncate c_attn input side for blocks 1-11. Returns (saved, info)."""
    ones = np.ones(D)
    saved, per_layer = {}, []
    for li in range(1, 12):
        blk = student.transformer.h[li]
        W = blk.attn.c_attn.weight.detach().cpu().double().numpy()   # (768, 2304)
        b = blk.attn.c_attn.bias.detach().cpu().double().numpy()
        saved[li] = (W.copy(), b.copy())
        mean_in = means_np[li - 1].astype(np.float64)
        if honest:
            g = blk.ln_1.weight.detach().cpu().double().numpy()
            bln = blk.ln_1.bias.detach().cpu().double().numpy()
            M = g[:, None] * np.column_stack([P_np.astype(np.float64), mean_in, ones])
        else:
            M = np.column_stack([P_np.astype(np.float64), ones])
        sv = np.linalg.svd(M, compute_uv=False)
        eff_rank = int((sv > sv[0] * 1e-10).sum())
        Q, _ = np.linalg.qr(M)
        QtW = Q.T @ W
        W_new = Q @ QtW
        assert np.isfinite(W_new).all()
        rel_discard = float(np.linalg.norm(W - W_new) / np.linalg.norm(W))
        b_new = b
        if honest:
            b_new = b + bln @ W - (bln @ Q) @ QtW
        with torch.no_grad():
            blk.attn.c_attn.weight.copy_(torch.from_numpy(W_new).float().to(device))
            blk.attn.c_attn.bias.copy_(torch.from_numpy(b_new).float().to(device))
        per_layer.append({"layer": li, "eff_rank": eff_rank,
                          "weight_frac_discarded": rel_discard})
    return saved, per_layer


def phase_a_restore(student, saved, device):
    for li, (W, b) in saved.items():
        blk = student.transformer.h[li]
        with torch.no_grad():
            blk.attn.c_attn.weight.copy_(torch.from_numpy(W).float().to(device))
            blk.attn.c_attn.bias.copy_(torch.from_numpy(b).float().to(device))


# --------------------------------------------------------------------------
# mid-block mean calibration
# --------------------------------------------------------------------------

def calibrate_mid_means(student, seqs, device, batch_size=8):
    """Mean of h_mid (= ln_2 input) per layer, with the output-projection
    hooks active (the configuration the student was trained in)."""
    L = student.config.n_layer
    sums = [torch.zeros(D, dtype=torch.float64) for _ in range(L)]  # CPU f64
    count = [0 for _ in range(L)]
    hooks = []
    for li, block in enumerate(student.transformer.h):
        def make_pre(li):
            def pre(_mod, args):
                h = args[0]
                sums[li] += h.reshape(-1, D).sum(0).cpu().double()
                count[li] += h.shape[0] * h.shape[1]
                return None
            return pre
        hooks.append(block.ln_2.register_forward_pre_hook(make_pre(li)))
    student.eval()
    for b0 in range(0, seqs.shape[0], batch_size):
        batch = torch.from_numpy(seqs[b0:b0 + batch_size]).to(device)
        with torch.no_grad():
            student(batch)
    for h in hooks:
        h.remove()
    return np.stack([(sums[li] / count[li]).float().numpy()
                     for li in range(L)])


# --------------------------------------------------------------------------
# PHASE B2 — compressed model operating natively in 256-dim coords
# --------------------------------------------------------------------------

def gelu_new(x):
    return 0.5 * x * (1.0 + torch.tanh(
        math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)))


def ln_coords(c, s, pbar, vhat, vn, A, rvec, bvec, eps):
    """Exact LayerNorm-then-matmul in coordinates.

    c: (B,T,r) coords of x = mean + P c relative to the block's mean.
    Returns (r + w @ A)/sigma + bvec where w = P^^T (P c - mu(Pc) 1)."""
    mu = c @ pbar                                     # (B,T)
    w = F.pad(c, (0, 1)) - mu.unsqueeze(-1) * s       # (B,T,257)
    normsq = vn + 2.0 * (w @ vhat) + (w * w).sum(-1)  # |v + w|^2
    sigma = torch.sqrt(normsq / D + eps).unsqueeze(-1)
    return (rvec + w @ A) / sigma + bvec


def attend(qkv, n_head, dropout_p, training):
    B, T, three_d = qkv.shape
    d = three_d // 3
    q, k, v = qkv.split(d, dim=-1)
    hd = d // n_head
    q = q.view(B, T, n_head, hd).transpose(1, 2)
    k = k.view(B, T, n_head, hd).transpose(1, 2)
    v = v.view(B, T, n_head, hd).transpose(1, 2)
    u = F.scaled_dot_product_attention(
        q, k, v, is_causal=True, dropout_p=dropout_p if training else 0.0)
    return u.transpose(1, 2).reshape(B, T, d)


class CoordBlock(nn.Module):
    """Blocks 1-11: input/output are 256-dim coords (relative to the
    per-block means, which are folded into the d-vectors)."""

    def __init__(self, n_head=12, r=RANK, rhat=RANK + 1, d_ff=4 * D,
                 eps=1e-5, p_drop=0.1):
        super().__init__()
        self.n_head, self.eps = n_head, eps
        self.A1 = nn.Parameter(torch.zeros(rhat, 3 * D))
        self.r1 = nn.Parameter(torch.zeros(3 * D))
        self.b1 = nn.Parameter(torch.zeros(3 * D))
        self.B_attn = nn.Parameter(torch.zeros(D, r))
        self.d_attn = nn.Parameter(torch.zeros(r))
        self.A2 = nn.Parameter(torch.zeros(rhat, d_ff))
        self.r2 = nn.Parameter(torch.zeros(d_ff))
        self.b2 = nn.Parameter(torch.zeros(d_ff))
        self.B_mlp = nn.Parameter(torch.zeros(d_ff, r))
        self.d_mlp = nn.Parameter(torch.zeros(r))
        self.register_buffer("vhat1", torch.zeros(rhat))
        self.register_buffer("vn1", torch.zeros(()))
        self.register_buffer("vhat2", torch.zeros(rhat))
        self.register_buffer("vn2", torch.zeros(()))
        self.attn_drop_p = p_drop
        self.resid_drop = nn.Dropout(p_drop)
        self.mlp_drop = nn.Dropout(p_drop)

    def forward(self, c, s, pbar):
        qkv = ln_coords(c, s, pbar, self.vhat1, self.vn1,
                        self.A1, self.r1, self.b1, self.eps)
        u = attend(qkv, self.n_head, self.attn_drop_p, self.training)
        c = c + self.resid_drop(u @ self.B_attn + self.d_attn)
        y2 = ln_coords(c, s, pbar, self.vhat2, self.vn2,
                       self.A2, self.r2, self.b2, self.eps)
        c = c + self.mlp_drop(gelu_new(y2) @ self.B_mlp + self.d_mlp)
        return c


class Block0(nn.Module):
    """Block 0 hybrid: raw 768-dim input (wte+wpe, honestly uncompressible
    on the c_attn input side), everything after the attention residual add
    in coords."""

    def __init__(self, n_head=12, r=RANK, rhat=RANK + 1, d_ff=4 * D,
                 eps=1e-5, p_drop=0.1):
        super().__init__()
        self.n_head, self.eps = n_head, eps
        self.ln_1 = nn.LayerNorm(D, eps=eps)
        self.c_attn = nn.Linear(D, 3 * D)
        self.B_attn = nn.Parameter(torch.zeros(D, r))
        self.d_attn = nn.Parameter(torch.zeros(r))
        self.A2 = nn.Parameter(torch.zeros(rhat, d_ff))
        self.r2 = nn.Parameter(torch.zeros(d_ff))
        self.b2 = nn.Parameter(torch.zeros(d_ff))
        self.B_mlp = nn.Parameter(torch.zeros(d_ff, r))
        self.d_mlp = nn.Parameter(torch.zeros(r))
        self.register_buffer("vhat2", torch.zeros(rhat))
        self.register_buffer("vn2", torch.zeros(()))
        self.attn_drop_p = p_drop
        self.resid_drop = nn.Dropout(p_drop)
        self.mlp_drop = nn.Dropout(p_drop)

    def forward(self, h, P, s, pbar):
        qkv = self.c_attn(self.ln_1(h))
        u = attend(qkv, self.n_head, self.attn_drop_p, self.training)
        c = h @ P + self.resid_drop(u @ self.B_attn + self.d_attn)
        y2 = ln_coords(c, s, pbar, self.vhat2, self.vn2,
                       self.A2, self.r2, self.b2, self.eps)
        c = c + self.mlp_drop(gelu_new(y2) @ self.B_mlp + self.d_mlp)
        return c


class CompressedGPT2(nn.Module):
    def __init__(self, vocab=50257, n_pos=1024, n_layer=12, p_drop=0.1):
        super().__init__()
        self.wte = nn.Embedding(vocab, D)
        self.wpe = nn.Embedding(n_pos, D)
        self.drop = nn.Dropout(p_drop)
        self.block0 = Block0(p_drop=p_drop)
        self.blocks = nn.ModuleList(
            [CoordBlock(p_drop=p_drop) for _ in range(n_layer - 1)])
        self.ln_f = nn.LayerNorm(D, eps=1e-5)
        self.register_buffer("P", torch.zeros(D, RANK))
        self.register_buffer("s", torch.zeros(RANK + 1))
        self.register_buffer("pbar", torch.zeros(RANK))
        self.register_buffer("mean_final", torch.zeros(D))

    def forward(self, ids):
        pos = torch.arange(ids.shape[1], device=ids.device)
        h = self.drop(self.wte(ids) + self.wpe(pos))
        c = self.block0(h, self.P, self.s, self.pbar)
        for blk in self.blocks:
            c = blk(c, self.s, self.pbar)
        h = c @ self.P.T + self.mean_final
        return self.ln_f(h) @ self.wte.weight.T


def build_compressed(student, means_np, mid_means_np, P_np, device):
    """Fold student weights + container geometry into CompressedGPT2.
    All algebra in float64, cast to float32 at the end."""
    P = P_np.astype(np.float64)                        # (768, 256)
    ones = np.ones(D)
    q1 = ones - P @ (P.T @ ones)
    q1_norm = float(np.linalg.norm(q1))
    q1 = q1 / q1_norm
    Phat = np.column_stack([P, q1])                    # (768, 257)
    s = Phat.T @ ones                                  # (257,)
    pbar = (P.T @ ones) / D                            # (256,)

    model = CompressedGPT2()
    sdev = student.transformer

    def t32(x):
        return torch.from_numpy(np.ascontiguousarray(x)).float()

    with torch.no_grad():
        model.wte.weight.copy_(sdev.wte.weight.detach().cpu())
        model.wpe.weight.copy_(sdev.wpe.weight.detach().cpu())
        model.ln_f.weight.copy_(sdev.ln_f.weight.detach().cpu())
        model.ln_f.bias.copy_(sdev.ln_f.bias.detach().cpu())
        model.P.copy_(t32(P))
        model.s.copy_(t32(s))
        model.pbar.copy_(t32(pbar))
        model.mean_final.copy_(t32(means_np[11].astype(np.float64)))

        for li in range(12):
            blk = sdev.h[li]
            W_att = blk.attn.c_attn.weight.detach().cpu().double().numpy()
            b_att = blk.attn.c_attn.bias.detach().cpu().double().numpy()
            W_cp = blk.attn.c_proj.weight.detach().cpu().double().numpy()
            b_cp = blk.attn.c_proj.bias.detach().cpu().double().numpy()
            W_fc = blk.mlp.c_fc.weight.detach().cpu().double().numpy()
            b_fc = blk.mlp.c_fc.bias.detach().cpu().double().numpy()
            W_mcp = blk.mlp.c_proj.weight.detach().cpu().double().numpy()
            b_mcp = blk.mlp.c_proj.bias.detach().cpu().double().numpy()
            g1 = blk.ln_1.weight.detach().cpu().double().numpy()
            bln1 = blk.ln_1.bias.detach().cpu().double().numpy()
            g2 = blk.ln_2.weight.detach().cpu().double().numpy()
            bln2 = blk.ln_2.bias.detach().cpu().double().numpy()
            mean_mid = mid_means_np[li].astype(np.float64)
            mean_out = means_np[li].astype(np.float64)

            if li == 0:
                tgt = model.block0
                tgt.ln_1.weight.copy_(t32(g1))
                tgt.ln_1.bias.copy_(t32(bln1))
                tgt.c_attn.weight.copy_(t32(W_att.T))   # Linear: (out, in)
                tgt.c_attn.bias.copy_(t32(b_att))
                d_attn = (b_cp - mean_mid) @ P          # h @ P carried in fwd
            else:
                tgt = model.blocks[li - 1]
                mean_in = means_np[li - 1].astype(np.float64)
                v1 = mean_in - mean_in.mean() * ones
                tgt.A1.copy_(t32(Phat.T @ (g1[:, None] * W_att)))
                tgt.r1.copy_(t32((v1 * g1) @ W_att))
                tgt.b1.copy_(t32(bln1 @ W_att + b_att))
                tgt.vhat1.copy_(t32(Phat.T @ v1))
                tgt.vn1.copy_(torch.tensor(float(v1 @ v1)))
                d_attn = (b_cp + mean_in - mean_mid) @ P
            tgt.B_attn.copy_(t32(W_cp @ P))
            tgt.d_attn.copy_(t32(d_attn))

            v2 = mean_mid - mean_mid.mean() * ones
            tgt.A2.copy_(t32(Phat.T @ (g2[:, None] * W_fc)))
            tgt.r2.copy_(t32((v2 * g2) @ W_fc))
            tgt.b2.copy_(t32(bln2 @ W_fc + b_fc))
            tgt.vhat2.copy_(t32(Phat.T @ v2))
            tgt.vn2.copy_(torch.tensor(float(v2 @ v2)))
            tgt.B_mlp.copy_(t32(W_mcp @ P))
            tgt.d_mlp.copy_(t32((b_mcp + mean_mid - mean_out) @ P))

    return model.to(device), {"q1_norm_before_normalization": q1_norm}


# --------------------------------------------------------------------------
# parameter accounting
# --------------------------------------------------------------------------

def count_params(model, hf_student):
    def n(t):
        return int(t.numel())

    orig_block = {}
    b = hf_student.transformer.h[0]
    orig_block["c_attn"] = n(b.attn.c_attn.weight) + n(b.attn.c_attn.bias)
    orig_block["attn_c_proj"] = n(b.attn.c_proj.weight) + n(b.attn.c_proj.bias)
    orig_block["c_fc"] = n(b.mlp.c_fc.weight) + n(b.mlp.c_fc.bias)
    orig_block["mlp_c_proj"] = n(b.mlp.c_proj.weight) + n(b.mlp.c_proj.bias)
    orig_block["ln_1"] = n(b.ln_1.weight) + n(b.ln_1.bias)
    orig_block["ln_2"] = n(b.ln_2.weight) + n(b.ln_2.bias)
    orig_block_total = sum(orig_block.values())

    def module_total(m):
        return (sum(n(p) for p in m.parameters())
                + sum(n(bf) for bf in m.buffers()))

    comp_block0 = module_total(model.block0)
    comp_block = module_total(model.blocks[0])
    comp_global = n(model.P) + n(model.s) + n(model.pbar) + n(model.mean_final)

    emb = n(model.wte.weight) + n(model.wpe.weight)
    ln_f = n(model.ln_f.weight) + n(model.ln_f.bias)

    orig_blocks_total = 12 * orig_block_total
    comp_blocks_total = comp_block0 + 11 * comp_block + comp_global
    return {
        "original_per_block": {**orig_block, "total": orig_block_total},
        "compressed_per_block_1_to_11": comp_block,
        "compressed_block0_hybrid": comp_block0,
        "compressed_global_container": comp_global,
        "blocks_total": {
            "original": orig_blocks_total,
            "compressed": comp_blocks_total,
            "ratio": orig_blocks_total / comp_blocks_total,
        },
        "full_model": {
            "original": orig_blocks_total + emb + ln_f,
            "compressed": comp_blocks_total + emb + ln_f,
            "ratio": (orig_blocks_total + emb + ln_f)
                     / (comp_blocks_total + emb + ln_f),
            "embeddings_plus_ln_f": emb + ln_f,
        },
    }


# --------------------------------------------------------------------------
# healing distillation
# --------------------------------------------------------------------------

def heal(model, teacher, train_seqs, val_seqs, test_seqs, device, args):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    vocab = teacher.config.vocab_size

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

    def val_eval():
        model.eval()
        return eval_fn_vs_teacher(lambda b: model(b), teacher, val_seqs,
                                  device, args.batch_size)

    history = [{"step": 0, **val_eval()}]
    print(f"[heal] step 0 (folded, unhealed): val KL={history[0]['kl_mean']:.4f} "
          f"agr={history[0]['top1_agreement']:.4f} ppl={history[0]['ppl']:.1f}",
          flush=True)

    t0 = time.time()
    step_done = 0
    loss_win, kl_win = [], []
    for step in range(args.max_steps):
        model.train()
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        batch = next_batch()
        with torch.no_grad():
            t_logits = teacher(batch).logits
            t_probs = F.softmax(t_logits, dim=-1)
            t_ent = float(-(t_probs * torch.log(t_probs.clamp_min(1e-12)))
                          .sum(-1).mean())
        logits = model(batch)
        soft_ce = F.cross_entropy(logits.view(-1, vocab), t_probs.view(-1, vocab))
        lm = F.cross_entropy(logits[:, :-1].reshape(-1, vocab),
                             batch[:, 1:].reshape(-1))
        loss = soft_ce + LM_WEIGHT * lm
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        loss_win.append(float(loss.detach()))
        kl_win.append(float(soft_ce.detach()) - t_ent)
        step_done = step + 1
        if step_done % 20 == 0:
            print(f"  [heal] step {step_done} loss={np.mean(loss_win):.4f} "
                  f"trainKL={np.mean(kl_win):.4f} ({time.time() - t0:.0f}s)",
                  flush=True)
            loss_win, kl_win = [], []
        if step_done % args.eval_every == 0:
            res = val_eval()
            history.append({"step": step_done, **res})
            print(f"[heal] step {step_done}: val KL={res['kl_mean']:.4f} "
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
                    print(f"[heal] early stop at step {step_done} "
                          f"(plateau: KL impr {kl_impr:.4f}, agr impr "
                          f"{agr_impr:.4f} over last "
                          f"{w * args.eval_every} steps)", flush=True)
                    break
    wall = time.time() - t0

    def steps95(key, better_is_less):
        v0 = history[0][key]
        vbest = (min if better_is_less else max)(h[key] for h in history)
        thresh = vbest + (0.05 if better_is_less else -0.05) * abs(v0 - vbest)
        if better_is_less:
            return next(h["step"] for h in history if h[key] <= thresh)
        return next(h["step"] for h in history if h[key] >= thresh)

    model.eval()
    test_res = eval_fn_vs_teacher(lambda b: model(b), teacher, test_seqs,
                                  device, args.batch_size)
    print(f"[heal] FINAL TEST: KL={test_res['kl_mean']:.4f} "
          f"agr={test_res['top1_agreement']:.4f} ppl={test_res['ppl']:.2f} "
          f"acc={test_res['top1_accuracy']:.4f}", flush=True)
    torch.save(model.state_dict(), HEALED_CKPT)
    print(f"[heal] saved healed compressed model to {HEALED_CKPT.name}")

    return {
        "steps": step_done,
        "wall_time_s": wall,
        "lr": args.lr,
        "warmup": args.warmup,
        "batch": [args.train_batch, args.seq_len],
        "eval_every": args.eval_every,
        "max_steps": args.max_steps,
        "plateau_window_steps": args.plateau_window * args.eval_every,
        "history_val": history,
        "steps_to_95pct_kl": steps95("kl_mean", True),
        "steps_to_95pct_agreement": steps95("top1_agreement", False),
        "steps_to_95pct_ppl": steps95("ppl", True),
        "final_test": test_res,
    }


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["zeroshot", "heal", "all"], default="all")
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
    ap.add_argument("--calib-seqs", type=int, default=200)
    ap.add_argument("--skip-baseline-eval", action="store_true")
    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[device] {device}")
    torch.manual_seed(args.seed)

    z = np.load(CACHE)
    means_np, V_np = z["means"], z["V"]
    P_np = np.ascontiguousarray(V_np[:, :RANK])
    print(f"[basis] {CACHE.name}: V {V_np.shape}, P = V[:, :{RANK}] (frozen)")

    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    tok = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
    test_texts, test_name = build_corpus_texts("test")
    test_seqs = tokenize_corpus(test_texts, tok, args.seq_len,
                                args.eval_seqs * args.seq_len, args.seed + 7)
    print(f"[eval corpus] {test_name}: {test_seqs.shape[0]} x {test_seqs.shape[1]}")

    student = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").to(device)
    sd = torch.load(STUDENT_CKPT, map_location=device, weights_only=True)
    student.load_state_dict(sd, strict=False)
    student.eval()
    print(f"[student] loaded {STUDENT_CKPT.name}")
    teacher = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    means_t = torch.from_numpy(means_np).to(device)
    P = torch.from_numpy(P_np.copy()).to(device)
    out_hooks = attach_out_bottleneck(student, means_t, P)

    results = {}
    if OUT_JSON.exists():
        results = json.loads(OUT_JSON.read_text(encoding="utf-8"))

    results.setdefault("meta", {
        "model": "openai-community/gpt2",
        "student": STUDENT_CKPT.name,
        "basis": CACHE.name,
        "rank": RANK,
        "eval_corpus": test_name,
        "eval_tokens": int(test_seqs.size),
        "seq_len": args.seq_len,
        "device": str(device),
        "baseline_ppl_ref": BASELINE_PPL,
        "r256_student_test_ref": {"kl_mean": 0.6931, "top1_agreement": 0.5611,
                                  "ppl": 78.99},
        "fork": {
            "COMPRESSION_REAL": "healed ppl <= ~2x baseline (<=109) at "
                                ">=2.5x block-weight compression",
            "CONTAINER_ONLY": "healed ppl > 3x baseline (>163)",
        },
        "ln_choice": "LN computed EXACTLY in coords via v/w split "
                     "(v = centered per-block mean, constant; w in global "
                     "span([P | q1])); diag(g), means, biases folded into "
                     "factorized matrices — no reconstruct-LN-project "
                     "roundtrip, no approximation",
    })

    def sfn(b):
        return student(b).logits

    if args.stage in ("zeroshot", "all"):
        if not args.skip_baseline_eval:
            print("[baseline] r=256 student sanity eval ...")
            base = eval_fn_vs_teacher(sfn, teacher, test_seqs, device,
                                      args.batch_size)
            print(f"[baseline] KL={base['kl_mean']:.4f} "
                  f"agr={base['top1_agreement']:.4f} ppl={base['ppl']:.2f}")
            results["baseline_r256_student"] = base

        # ---------------- PHASE A ----------------
        print("[phase A] naive input-side truncation (P~ = orth([P, 1]), "
              "no bias fix) ...")
        saved, info_naive = phase_a_apply(student, means_np, P_np,
                                          honest=False, device=device)
        res_naive = eval_fn_vs_teacher(sfn, teacher, test_seqs, device,
                                       args.batch_size)
        phase_a_restore(student, saved, device)
        print(f"[phase A naive] KL={res_naive['kl_mean']:.4f} "
              f"agr={res_naive['top1_agreement']:.4f} ppl={res_naive['ppl']:.2f}")

        print("[phase A] honest truncation (Q = orth(diag(g)[P|mean|1]), "
              "bias fix) ...")
        saved, info_honest = phase_a_apply(student, means_np, P_np,
                                           honest=True, device=device)
        res_honest = eval_fn_vs_teacher(sfn, teacher, test_seqs, device,
                                        args.batch_size)
        phase_a_restore(student, saved, device)
        print(f"[phase A honest] KL={res_honest['kl_mean']:.4f} "
              f"agr={res_honest['top1_agreement']:.4f} ppl={res_honest['ppl']:.2f}")
        results["phase_a"] = {
            "scope": "c_attn of blocks 1-11 (block 0 input is raw "
                     "embeddings, does not qualify)",
            "naive": {"variant": "W -> Ptilde Ptilde^T W, Ptilde = "
                                 "orth([P, ones]), rank 257, no bias fix",
                      "per_layer": info_naive, "test": res_naive},
            "honest": {"variant": "W -> Q Q^T W, Q = orth(diag(g_ln1) "
                                  "[P | mean_in | ones]), rank 258, bias "
                                  "fix b += b_ln (I - QQ^T) W (exact)",
                       "per_layer": info_honest, "test": res_honest},
        }

        # ---------------- mid means ----------------
        if MID_MEANS_CACHE.exists():
            mid_means_np = np.load(MID_MEANS_CACHE)["mid_means"]
            print(f"[calib] loaded mid means from {MID_MEANS_CACHE.name}")
        else:
            train_texts, _ = build_corpus_texts("train")
            calib_seqs = tokenize_corpus(train_texts, tok, args.seq_len,
                                         args.calib_seqs * args.seq_len,
                                         args.seed + 331)
            print(f"[calib] computing mid-block means on "
                  f"{calib_seqs.shape[0]} x {args.seq_len} train tokens ...")
            mid_means_np = calibrate_mid_means(student, calib_seqs, device)
            np.savez(MID_MEANS_CACHE, mid_means=mid_means_np)
            print(f"[calib] saved {MID_MEANS_CACHE.name}")
        results["mid_means"] = {
            "cache": MID_MEANS_CACHE.name,
            "calib_tokens": args.calib_seqs * args.seq_len,
            "norms": [float(np.linalg.norm(m)) for m in mid_means_np],
        }

        # ---------------- PHASE B1 ----------------
        mid_means_t = torch.from_numpy(mid_means_np).to(device)
        print("[phase B1] mid-block projection added (zero-shot) ...")
        mid_hooks = attach_mid_projection(student, mid_means_t, P)
        res_b1 = eval_fn_vs_teacher(sfn, teacher, test_seqs, device,
                                    args.batch_size)
        print(f"[phase B1] KL={res_b1['kl_mean']:.4f} "
              f"agr={res_b1['top1_agreement']:.4f} ppl={res_b1['ppl']:.2f}")
        results["phase_b1"] = {
            "desc": "student + output projections + mid-block projection "
                    "(fresh calibrated means) at all 12 blocks, zero-shot",
            "test": res_b1,
        }

        # ---------------- PHASE B2 ----------------
        print("[phase B2] building compressed model (float64 folding) ...")
        comp, fold_info = build_compressed(student, means_np, mid_means_np,
                                           P_np, device)
        comp.eval()

        # parity check vs B1 (mid hooks still attached)
        diffs = []
        for b0 in range(0, min(8, test_seqs.shape[0]), args.batch_size):
            batch = torch.from_numpy(test_seqs[b0:b0 + args.batch_size]).to(device)
            with torch.no_grad():
                l_b1 = student(batch).logits.float()
                l_b2 = comp(batch).float()
            diffs.append((l_b1 - l_b2).abs())
        max_diff = float(torch.cat(diffs).max())
        mean_diff = float(torch.cat([d.flatten() for d in diffs]).mean())
        print(f"[parity] B2 vs B1 logits: max|diff|={max_diff:.3e} "
              f"mean|diff|={mean_diff:.3e}")

        res_b2 = eval_fn_vs_teacher(lambda b: comp(b), teacher, test_seqs,
                                    device, args.batch_size)
        print(f"[phase B2] KL={res_b2['kl_mean']:.4f} "
              f"agr={res_b2['top1_agreement']:.4f} ppl={res_b2['ppl']:.2f}")

        for h in mid_hooks:
            h.remove()

        params = count_params(comp, student)
        print(f"[params] blocks: {params['blocks_total']['original']:,} -> "
              f"{params['blocks_total']['compressed']:,} "
              f"({params['blocks_total']['ratio']:.2f}x); full model: "
              f"{params['full_model']['original']:,} -> "
              f"{params['full_model']['compressed']:,} "
              f"({params['full_model']['ratio']:.2f}x)")

        results["phase_b2"] = {
            "desc": "re-parameterized model natively in 256-dim coords; "
                    "LN exact in coords; block 0 hybrid (full c_attn)",
            "fold_info": fold_info,
            "parity_vs_b1": {"max_abs_logit_diff": max_diff,
                             "mean_abs_logit_diff": mean_diff},
            "test": res_b2,
            "params": params,
        }
        OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"[zeroshot] wrote {OUT_JSON.name}")

    if args.stage in ("heal", "all"):
        for h in out_hooks:
            h.remove()
        if "phase_b2" not in results:
            raise SystemExit("run --stage zeroshot first")
        mid_means_np = np.load(MID_MEANS_CACHE)["mid_means"]
        comp, _ = build_compressed(student, means_np, mid_means_np,
                                   P_np, device)
        del student
        if device.type == "mps":
            torch.mps.empty_cache()

        train_texts, _ = build_corpus_texts("train")
        need_tokens = args.max_steps * args.train_batch * args.seq_len
        train_seqs = tokenize_corpus(train_texts, tok, args.seq_len,
                                     need_tokens, args.seed + 31)
        print(f"[data] train seqs: {train_seqs.shape[0]} "
              f"(need {need_tokens // args.seq_len}; cycles if short)")
        val_texts, _ = build_corpus_texts("validation")
        val_seqs = tokenize_corpus(val_texts, tok, args.seq_len,
                                   args.val_seqs * args.seq_len,
                                   args.seed + 101)
        print(f"[data] val slice: {val_seqs.shape[0]} x {args.seq_len}")

        n_train = sum(p.numel() for p in comp.parameters() if p.requires_grad)
        print(f"[heal] trainable params: {n_train:,} (P & geometry frozen "
              f"as buffers)")
        results["healing"] = heal(comp, teacher, train_seqs, val_seqs,
                                  test_seqs, device, args)
        results["healing"]["trainable_params"] = n_train

        healed_ppl = results["healing"]["final_test"]["ppl"]
        ratio = results["phase_b2"]["params"]["blocks_total"]["ratio"]
        if healed_ppl <= 2.0 * BASELINE_PPL and ratio >= 2.5:
            verdict = "COMPRESSION_REAL"
        elif healed_ppl > 3.0 * BASELINE_PPL:
            verdict = "CONTAINER_ONLY"
        else:
            verdict = "BETWEEN"
        results["verdict"] = {
            "fork": verdict,
            "healed_test_ppl": healed_ppl,
            "ppl_x_baseline": healed_ppl / BASELINE_PPL,
            "block_weight_compression": ratio,
            "full_model_compression": results["phase_b2"]["params"]
                                             ["full_model"]["ratio"],
        }
        print(f"[verdict] {verdict}: healed ppl {healed_ppl:.2f} "
              f"({healed_ppl / BASELINE_PPL:.2f}x baseline) at "
              f"{ratio:.2f}x block-weight compression")
        OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"[done] wrote {OUT_JSON.name}")


if __name__ == "__main__":
    main()
