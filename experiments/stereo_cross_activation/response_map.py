#!/usr/bin/env python3
"""Response map of the shared r=256 coordinate system (training-free).

STEP 1 of the exploitation roadmap: flow current through the shared
coordinates of the KL-distilled r=256 student (kl_distill_student_r256.pt,
frozen hard shared-PCA bottleneck at all 12 block outputs,
h -> mean_l + P P^T (h - mean_l), P = V[:, :256] from
bases_cache_soft_distill.npz) and map what responds. Forward passes only,
no training. The question: do the shared dimensions behave as stable,
reusable functional parts (puzzle-piece candidates)?

All interventions act on the shared coordinate z = P^T (h - mean_l)
inside the bottleneck hook, then reconstruct h -> mean_l + P z. With no
intervention this is bit-identical to the student's normal bottleneck
(same op order), so the unablated student is the baseline.

Experiments (eval text = held-out wikitext-2 test, 20 x 256 = 5,120 tokens,
same packing recipe/seed as all prior runs => identical to the first 20 of
the standard 40-seq test slice):

A) Ablation response map.
   A1: for EVERY dim i in 0..255: zero z_i at ALL 12 layers; mean KL
       (unablated student || ablated) + top-1 flip rate. Full coverage.
   A2: for the top-32 dims by A1 impact: zero z_i at ONE layer only,
       all 12 layers => 384-cell dim x layer map, per-position KL kept.
   Summary: impact ranking, Gini / top-k share (hub vs diffuse),
   Spearman(impact, eigenvalue rank).

B) Cross-layer role consistency (same part at different layers?).
   B1 ablation profiles: per-position KL vector (5,120 pos) of "dim i
      ablated at layer l"; same-dim across-layer Spearman corr vs
      different-dim null.
   B2 activation profiles: z_i(l, t) time-series on the same text;
      same-dim across-layer Pearson corr vs different-dim null.
      (Caveat noted in MD: residual stream carries coordinates forward,
      so B2 has an architectural tailwind; B1 is the causal test.)

C) Semantic character of top-16 impact dims.
   Max-activating tokens/contexts (top-20 positions over all layers) and
   mean log-prob shift when adding +/-2 sigma_i to z_i at all layers
   (sigma_i = pooled std over layers & positions). Reported: the ODD part
   (delta_plus - delta_minus)/2 = directional/linear response (top
   boosted/suppressed vocab tokens), sign symmetry cos(d+, -d-), and
   odd_frac = ||odd|| / (||odd|| + ||even||) where even = symmetric damage.

D) Write-transfer mini puzzle test.
   For top-4 impact dims (+2 low-impact controls): take the donor value
   v_i = z_i at its max-|z| position in a DIFFERENT text (wikitext-2
   validation), then on recipient text set z_i = v_i at layer 6 only vs
   at ALL layers. Directionality of the induced log-prob shift:
   ratio ||mean_pos delta|| / mean_pos ||delta|| (1 = same direction
   everywhere, ~0 = chaotic), split-half cosine (reproducibility),
   cos(one-layer shift, all-layer shift), cos(all-layer set shift,
   +2sigma amplification shift from C).

Pre-registered fork:
  PARTS_LIKE: impact structured/non-uniform, B same-dim corr clearly
    above null, amplification coherent => raw shared coords are candidate
    puzzle parts.
  FIELD_LIKE: diffuse impact, layer-idiosyncratic roles, incoherent
    interventions => container valid but PCA axes are not parts.
  Between: state honestly.

Prior files untouched. Output: results_response_map.json.
MPS/CPU, float32, forward-only (torch.no_grad everywhere).
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
CACHE = HERE / "bases_cache_soft_distill.npz"
CKPT = HERE / "kl_distill_student_r256.pt"


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
# intervention hooks on the shared coordinate z
# --------------------------------------------------------------------------

class Intervention:
    """Mutable spec read by every hook at call time.

    mode: None | 'zero' | 'add' | 'set'
    dim:  int (coordinate index in 0..r-1)
    layer: None (all layers) or int (that layer only)
    value: float (for 'add'/'set')
    """

    def __init__(self):
        self.clear()

    def clear(self):
        self.mode = None
        self.dim = -1
        self.layer = None
        self.value = 0.0


class Capture:
    def __init__(self, n_layer: int):
        self.active = False
        self.buf = [[] for _ in range(n_layer)]

    def start(self, n_layer: int):
        self.active = True
        self.buf = [[] for _ in range(n_layer)]

    def stop(self):
        self.active = False

    def stacked(self):
        # (L, total_pos, r)
        return torch.stack([torch.cat(b, dim=0) for b in self.buf], dim=0)


def make_hook(li: int, mean: torch.Tensor, P: torch.Tensor,
              iv: Intervention, cap: Capture):
    def hook(_mod, _inp, out):
        h = out[0]
        z = (h - mean) @ P
        if cap.active:
            cap.buf[li].append(z.detach().reshape(-1, z.shape[-1]).cpu())
        if iv.mode is not None and (iv.layer is None or iv.layer == li):
            if iv.mode == "zero":
                z[..., iv.dim] = 0.0
            elif iv.mode == "add":
                z[..., iv.dim] = z[..., iv.dim] + iv.value
            elif iv.mode == "set":
                z[..., iv.dim] = iv.value
        h_new = mean + z @ P.T
        return (h_new,) + tuple(out[1:])
    return hook


# --------------------------------------------------------------------------
# eval machinery
# --------------------------------------------------------------------------

def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    if ra.std() < 1e-12 or rb.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def pearson(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def gini(x):
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = x.size
    if x.sum() <= 0:
        return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)


class Runner:
    """Holds baseline log-probs on device; runs one intervention config."""

    def __init__(self, model, batches, device):
        self.model = model
        self.batches = batches          # list of (B,S) device tensors
        self.device = device
        self.base_lp = []               # per-batch (B,S,V) log-probs, device
        self.base_argmax = []           # per-batch (B,S) argmax, device
        self.base_lp_sum = None         # (V,) sum over all positions, cpu f64

    def compute_baseline(self):
        s = None
        for b in self.batches:
            with torch.no_grad():
                lp = F.log_softmax(self.model(b).logits.float(), dim=-1)
            self.base_lp.append(lp)
            self.base_argmax.append(lp.argmax(-1))
            part = lp.sum(dim=(0, 1)).cpu().double()
            s = part if s is None else s + part
        self.base_lp_sum = s

    def run(self, keep_per_pos=False, keep_delta=False, batch_subset=None):
        """Returns dict with kl_mean, flip_rate and optional extras.

        keep_per_pos: also return flat per-position KL vector (np.float32).
        keep_delta:   also return log-prob shift stats vs baseline:
                      per-half sum of delta (V,), sum of ||delta||, n_pos.
        batch_subset: list of batch indices to run (default: all).
        """
        idxs = batch_subset if batch_subset is not None else range(len(self.batches))
        kl_sum, flip_sum, n_pos = 0.0, 0.0, 0
        per_pos = [] if keep_per_pos else None
        delta_sum_halves = [None, None]
        delta_norm_sum, delta_n = 0.0, 0
        idxs = list(idxs)
        for k, bi in enumerate(idxs):
            b = self.batches[bi]
            with torch.no_grad():
                lp = F.log_softmax(self.model(b).logits.float(), dim=-1)
                blp = self.base_lp[bi]
                kl = (blp.exp() * (blp - lp)).sum(-1)
                kl_sum += float(kl.sum())
                n_pos += kl.numel()
                flip_sum += float((lp.argmax(-1) != self.base_argmax[bi]).sum())
                if keep_per_pos:
                    per_pos.append(kl.reshape(-1).cpu().numpy().astype(np.float32))
                if keep_delta:
                    d = lp - blp
                    half = 0 if k < (len(idxs) + 1) // 2 else 1
                    ds = d.sum(dim=(0, 1)).cpu().double()
                    if delta_sum_halves[half] is None:
                        delta_sum_halves[half] = ds
                    else:
                        delta_sum_halves[half] += ds
                    delta_norm_sum += float(d.reshape(-1, d.shape[-1])
                                            .norm(dim=-1).sum())
                    delta_n += d.shape[0] * d.shape[1]
        out = {"kl_mean": kl_sum / n_pos, "flip_rate": flip_sum / n_pos}
        if keep_per_pos:
            out["per_pos_kl"] = np.concatenate(per_pos)
        if keep_delta:
            out["delta_halves"] = [h.numpy() if h is not None else None
                                   for h in delta_sum_halves]
            out["delta_norm_sum"] = delta_norm_sum
            out["delta_n"] = delta_n
        return out


def cos(a, b):
    # clamp infs (rare, from saturated log-probs under strong interventions)
    a = np.nan_to_num(np.asarray(a, dtype=np.float64), posinf=1e9, neginf=-1e9)
    b = np.nan_to_num(np.asarray(b, dtype=np.float64), posinf=1e9, neginf=-1e9)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float((a / na) @ (b / nb))


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=256)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--eval-seqs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--per-layer-dims", type=int, default=32)
    ap.add_argument("--semantic-dims", type=int, default=16)
    ap.add_argument("--transfer-dims", type=int, default=4)
    ap.add_argument("--transfer-layer", type=int, default=6)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run to validate plumbing")
    ap.add_argument("--out", type=str,
                    default=str(HERE / "results_response_map.json"))
    args = ap.parse_args()

    if args.smoke:
        args.eval_seqs = 4
        args.per_layer_dims = 3
        args.semantic_dims = 2
        args.transfer_dims = 1

    if args.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[device] {device}")
    torch.manual_seed(args.seed)
    rank = args.rank
    t_start = time.time()

    z = np.load(CACHE)
    means_np, V_np, eigvals = z["means"], z["V"], z["eigvals"]
    print(f"[basis] {CACHE.name}: V {V_np.shape}, using top-{rank} columns")

    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    tok = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
    eval_texts, eval_name = build_corpus_texts("test")
    eval_seqs = tokenize_corpus(eval_texts, tok, args.seq_len,
                                args.eval_seqs * args.seq_len, args.seed + 7)
    print(f"[eval corpus] {eval_name}: {eval_seqs.shape[0]} x {eval_seqs.shape[1]} "
          f"= {eval_seqs.size} tokens (same recipe/seed as prior test evals)")
    donor_texts, donor_name = build_corpus_texts("validation")
    donor_seqs = tokenize_corpus(donor_texts, tok, args.seq_len,
                                 4 * args.seq_len, args.seed + 101)
    print(f"[donor corpus] {donor_name}: {donor_seqs.shape[0]} x {args.seq_len}")

    model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").to(device)
    sd = torch.load(CKPT, map_location=device, weights_only=True)
    model.load_state_dict(sd)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"[student] loaded {CKPT.name}")

    means_t = torch.from_numpy(means_np).to(device)
    P = torch.from_numpy(V_np[:, :rank].copy()).to(device)
    iv = Intervention()
    cap = Capture(model.config.n_layer)
    hooks = [model.transformer.h[li].register_forward_hook(
        make_hook(li, means_t[li], P, iv, cap))
        for li in range(model.config.n_layer)]
    n_layer = model.config.n_layer

    batches = [torch.from_numpy(eval_seqs[b0:b0 + args.batch_size]).to(device)
               for b0 in range(0, eval_seqs.shape[0], args.batch_size)]
    runner = Runner(model, batches, device)

    # ---- baseline pass + activation capture -------------------------------
    print("[baseline] forward with capture ...")
    cap.start(n_layer)
    runner.compute_baseline()
    cap.stop()
    z_all = cap.stacked()                    # (L, n_pos, r) cpu float32
    n_pos_total = z_all.shape[1]
    flat_tokens = eval_seqs.reshape(-1)      # (n_pos,)
    sigma = z_all.reshape(-1, rank).std(dim=0).numpy()   # pooled per-dim std
    mean_abs_z = z_all.abs().mean(dim=(0, 1)).numpy()
    print(f"[baseline] captured z: {tuple(z_all.shape)}; "
          f"sigma[0..3]={np.round(sigma[:4], 3).tolist()}")

    results = {"meta": {
        "model": "openai-community/gpt2",
        "student_ckpt": CKPT.name,
        "basis": CACHE.name,
        "rank": rank,
        "eval_corpus": eval_name,
        "eval_tokens": int(eval_seqs.size),
        "donor_corpus": donor_name,
        "device": str(device),
        "baseline": "unablated distilled student (KL and flips vs itself)",
        "note": "all interventions on z = P^T (h - mean_l) inside the "
                "bottleneck hooks; forward passes only",
    }}

    # =======================================================================
    # A1: all-layer ablation of every dim
    # =======================================================================
    n_dims_a1 = rank if not args.smoke else 8
    print(f"[A1] all-layer ablation of {n_dims_a1} dims ...")
    tA = time.time()
    a1 = []
    for i in range(n_dims_a1):
        iv.mode, iv.dim, iv.layer = "zero", i, None
        r = runner.run()
        iv.clear()
        a1.append({"dim": i, "kl_mean": r["kl_mean"],
                   "flip_rate": r["flip_rate"],
                   "eigval": float(eigvals[i]),
                   "sigma": float(sigma[i]),
                   "mean_abs_z": float(mean_abs_z[i])})
        if (i + 1) % 32 == 0:
            print(f"  [A1] {i + 1}/{n_dims_a1} ({time.time() - tA:.0f}s)",
                  flush=True)
    impacts = np.array([d["kl_mean"] for d in a1])
    order = np.argsort(-impacts)
    ranked = [a1[int(i)] for i in order]
    top_share = {}
    for k in (8, 16, 32, 64):
        if k <= n_dims_a1:
            top_share[f"top{k}_share"] = float(
                impacts[order[:k]].sum() / impacts.sum())
    a1_summary = {
        "n_dims": n_dims_a1,
        "kl_mean_median": float(np.median(impacts)),
        "kl_mean_max": float(impacts.max()),
        "kl_mean_min": float(impacts.min()),
        "gini": gini(impacts),
        **top_share,
        "spearman_impact_vs_eigrank": spearman(impacts, -np.arange(n_dims_a1)),
        "spearman_impact_vs_sigma": spearman(
            impacts, np.array([d["sigma"] for d in a1])),
    }
    print(f"[A1] done in {time.time() - tA:.0f}s; gini={a1_summary['gini']:.3f} "
          f"top8_share={top_share.get('top8_share', 0):.3f} "
          f"max KL={impacts.max():.4f} (dim {ranked[0]['dim']}) "
          f"median={np.median(impacts):.5f}")
    results["A1_all_layer_ablation"] = {"per_dim": a1, "summary": a1_summary,
                                        "ranked_dims": [int(i) for i in order]}

    # =======================================================================
    # A2: per-layer ablation for top dims
    # =======================================================================
    n_top = min(args.per_layer_dims, n_dims_a1)
    top_dims = [int(i) for i in order[:n_top]]
    print(f"[A2] per-layer ablation: {n_top} dims x {n_layer} layers ...")
    tA2 = time.time()
    a2_kl = np.zeros((n_top, n_layer), dtype=np.float64)
    a2_flip = np.zeros((n_top, n_layer), dtype=np.float64)
    per_pos_kl = np.zeros((n_top, n_layer, n_pos_total), dtype=np.float32)
    for di, dim in enumerate(top_dims):
        for li in range(n_layer):
            iv.mode, iv.dim, iv.layer = "zero", dim, li
            r = runner.run(keep_per_pos=True)
            iv.clear()
            a2_kl[di, li] = r["kl_mean"]
            a2_flip[di, li] = r["flip_rate"]
            per_pos_kl[di, li] = r["per_pos_kl"]
        print(f"  [A2] dim {dim} ({di + 1}/{n_top}) "
              f"max-layer KL={a2_kl[di].max():.4f} at l={int(a2_kl[di].argmax())} "
              f"({time.time() - tA2:.0f}s)", flush=True)
    results["A2_per_layer_ablation"] = {
        "dims": top_dims,
        "kl_matrix": a2_kl.round(6).tolist(),
        "flip_matrix": a2_flip.round(6).tolist(),
        "note": "rows = dims (A1 impact order), cols = layers 0..11",
    }
    print(f"[A2] done in {time.time() - tA2:.0f}s")

    # =======================================================================
    # B: cross-layer role consistency
    # =======================================================================
    print("[B] cross-layer consistency ...")
    rng = np.random.default_rng(args.seed + 1)

    # B1: ablation-profile correlations (per-position KL vectors)
    same_corrs, same_corrs_dim = [], {}
    floor = 1e-4
    for di in range(n_top):
        cs = []
        for l1 in range(n_layer):
            for l2 in range(l1 + 1, n_layer):
                if a2_kl[di, l1] < floor or a2_kl[di, l2] < floor:
                    continue
                cs.append(spearman(per_pos_kl[di, l1], per_pos_kl[di, l2]))
        if cs:
            same_corrs_dim[top_dims[di]] = float(np.mean(cs))
            same_corrs.extend(cs)
    null_corrs = []
    n_null = 300 if not args.smoke else 20
    tries = 0
    while len(null_corrs) < n_null and tries < n_null * 20:
        tries += 1
        di, dj = rng.integers(0, n_top, size=2)
        if di == dj:
            continue
        l1, l2 = rng.integers(0, n_layer, size=2)
        if l1 == l2:
            continue
        if a2_kl[di, l1] < floor or a2_kl[dj, l2] < floor:
            continue
        null_corrs.append(spearman(per_pos_kl[di, l1], per_pos_kl[dj, l2]))
    b1 = {
        "same_dim_mean": float(np.mean(same_corrs)) if same_corrs else None,
        "same_dim_std": float(np.std(same_corrs)) if same_corrs else None,
        "same_dim_n_pairs": len(same_corrs),
        "null_mean": float(np.mean(null_corrs)) if null_corrs else None,
        "null_std": float(np.std(null_corrs)) if null_corrs else None,
        "null_n_pairs": len(null_corrs),
        "per_dim_mean": {str(k): round(v, 4) for k, v in same_corrs_dim.items()},
        "kl_floor": floor,
        "metric": "spearman corr of per-position KL vectors",
    }
    print(f"  [B1] ablation profiles: same-dim {b1['same_dim_mean']:.3f} "
          f"+/- {b1['same_dim_std']:.3f} (n={b1['same_dim_n_pairs']}), "
          f"null {b1['null_mean']:.3f} +/- {b1['null_std']:.3f}")

    # B2: activation-profile correlations
    z_np = z_all.numpy()  # (L, n_pos, r)
    act_same, act_same_dim = [], {}
    for dim in top_dims:
        series = z_np[:, :, dim]  # (L, n_pos)
        cm = np.corrcoef(series)
        vals = cm[np.triu_indices(n_layer, k=1)]
        act_same_dim[dim] = float(np.nanmean(vals))
        act_same.extend([float(v) for v in vals if np.isfinite(v)])
    act_null = []
    for _ in range(500 if not args.smoke else 30):
        i, j = rng.integers(0, rank, size=2)
        if i == j:
            continue
        l1, l2 = rng.integers(0, n_layer, size=2)
        if l1 == l2:
            continue
        act_null.append(pearson(z_np[l1, :, i], z_np[l2, :, j]))
    b2 = {
        "same_dim_mean": float(np.mean(act_same)),
        "same_dim_std": float(np.std(act_same)),
        "null_mean": float(np.mean(act_null)),
        "null_std": float(np.std(act_null)),
        "per_dim_mean": {str(k): round(v, 4) for k, v in act_same_dim.items()},
        "metric": "pearson corr of z_i time-series across layers",
        "caveat": "residual stream carries coordinates forward, so high "
                  "values are partly architectural; B1 is the causal test",
    }
    print(f"  [B2] activation profiles: same-dim {b2['same_dim_mean']:.3f}, "
          f"null {b2['null_mean']:.3f}")
    results["B_cross_layer_consistency"] = {"B1_ablation_profiles": b1,
                                            "B2_activation_profiles": b2}

    # =======================================================================
    # C: semantic character of top dims
    # =======================================================================
    n_sem = min(args.semantic_dims, n_top)
    sem_dims = top_dims[:n_sem]
    print(f"[C] semantics of top-{n_sem} dims ...")
    tC = time.time()
    c_out = {}
    for dim in sem_dims:
        series = z_np[:, :, dim]                     # (L, n_pos)
        # max-activating positions (positive side), dedupe by position
        flat = series.max(axis=0)                    # max over layers per pos
        best_layer = series.argmax(axis=0)
        top_pos = np.argsort(-flat)[:60]
        seen, contexts = set(), []
        for p in top_pos:
            if len(contexts) >= 20:
                break
            si, ti = divmod(int(p), args.seq_len)
            key = (si, ti // 4)                      # light dedupe
            if key in seen:
                continue
            seen.add(key)
            lo, hi = max(0, ti - 8), min(args.seq_len, ti + 4)
            ctx = tok.decode(eval_seqs[si, lo:ti].tolist()) + " «" + \
                tok.decode([int(eval_seqs[si, ti])]) + "» " + \
                tok.decode(eval_seqs[si, ti + 1:hi].tolist())
            contexts.append({"z": round(float(flat[p]), 2),
                             "layer": int(best_layer[p]),
                             "ctx": ctx.replace("\n", " ")})
        # amplification +/- 2 sigma at all layers
        deltas = {}
        for sign in (+1, -1):
            iv.mode, iv.dim, iv.layer = "add", dim, None
            iv.value = float(sign * 2.0 * sigma[dim])
            r = runner.run(keep_delta=True)
            iv.clear()
            dsum = sum(h for h in r["delta_halves"] if h is not None)
            deltas[sign] = dsum / r["delta_n"]        # mean log-prob shift (V,)
        dplus, dminus = deltas[+1], deltas[-1]
        # odd part = linear/directional response; even part = symmetric damage
        odd = (dplus - dminus) / 2.0
        even = (dplus + dminus) / 2.0
        boost = np.argsort(-odd)[:12]
        supp = np.argsort(odd)[:12]
        c_out[str(dim)] = {
            "sigma": round(float(sigma[dim]), 3),
            "a1_kl": round(float(impacts[dim]), 5),
            "max_activating": contexts,
            "amp_odd_top_boosted": [
                {"tok": tok.decode([int(t)]), "dlp": round(float(odd[t]), 3)}
                for t in boost],
            "amp_odd_top_suppressed": [
                {"tok": tok.decode([int(t)]), "dlp": round(float(odd[t]), 3)}
                for t in supp],
            "sign_symmetry_cos": round(cos(dplus, -dminus), 3),
            "odd_frac": round(float(np.linalg.norm(odd) /
                                    (np.linalg.norm(odd) +
                                     np.linalg.norm(even) + 1e-12)), 3),
            "amp_plus2s_shift_norm": round(float(np.linalg.norm(dplus)), 3),
        }
        print(f"  [C] dim {dim}: sign-sym cos={c_out[str(dim)]['sign_symmetry_cos']} "
              f"odd_frac={c_out[str(dim)]['odd_frac']} "
              f"({time.time() - tC:.0f}s)", flush=True)
    results["C_semantics"] = c_out
    recip = list(range(min(2, len(batches))))
    # keep +2s shift vectors (on recipient batches) for D comparison
    amp_shift = {int(d): None for d in sem_dims}
    for dim in sem_dims:
        iv.mode, iv.dim, iv.layer = "add", dim, None
        iv.value = float(2.0 * sigma[dim])
        r = runner.run(keep_delta=True, batch_subset=recip)
        iv.clear()
        dsum = sum(h for h in r["delta_halves"] if h is not None)
        amp_shift[dim] = dsum / r["delta_n"]
    print(f"[C] done in {time.time() - tC:.0f}s")

    # =======================================================================
    # D: write-transfer
    # =======================================================================
    print("[D] write-transfer ...")
    tD = time.time()
    # donor capture
    donor_batch = torch.from_numpy(donor_seqs).to(device)
    cap.start(n_layer)
    with torch.no_grad():
        model(donor_batch)
    cap.stop()
    z_donor = cap.stacked().numpy()               # (L, donor_pos, r)

    n_tr = min(args.transfer_dims, n_sem)
    tr_dims = top_dims[:n_tr]
    lo_dims = []
    if not args.smoke and n_dims_a1 == rank:
        lo_dims = [int(order[int(0.8 * rank)]), int(order[int(0.95 * rank)])]
    d_out = {}
    for dim in tr_dims + lo_dims:
        dser = z_donor[:, :, dim]
        p = int(np.abs(dser).max(axis=0).argmax())
        dl = int(np.abs(dser[:, p]).argmax())
        v = float(dser[dl, p])
        si, ti = divmod(p, args.seq_len)
        donor_ctx = tok.decode(
            donor_seqs[si, max(0, ti - 8):ti + 1].tolist()).replace("\n", " ")
        entry = {"donor_value": round(v, 3),
                 "donor_value_sigma": round(v / float(sigma[dim]), 2),
                 "donor_layer_of_max": dl,
                 "donor_ctx": donor_ctx,
                 "is_low_impact_control": dim in lo_dims,
                 "a1_kl": round(float(impacts[dim]), 5)}
        shifts = {}
        for mode_name, layer in (("one_layer", args.transfer_layer),
                                 ("all_layers", None)):
            iv.mode, iv.dim, iv.layer, iv.value = "set", dim, layer, v
            r = runner.run(keep_delta=True, batch_subset=recip)
            iv.clear()
            h0, h1 = r["delta_halves"]
            if h1 is None:                       # single recipient batch
                h1 = h0
            n_half = r["delta_n"] / 2
            dsum = (h0 + h1) / r["delta_n"]
            directionality = float(np.linalg.norm((h0 + h1) / r["delta_n"]) /
                                   (r["delta_norm_sum"] / r["delta_n"]))
            entry[mode_name] = {
                "kl_mean": round(r["kl_mean"], 4),
                "flip_rate": round(r["flip_rate"], 4),
                "directionality": round(directionality, 4),
                "split_half_cos": round(cos(h0 / n_half, h1 / n_half), 4),
            }
            shifts[mode_name] = dsum
        entry["cos_one_vs_all_layers"] = round(
            cos(shifts["one_layer"], shifts["all_layers"]), 4)
        if dim in amp_shift and amp_shift[dim] is not None:
            entry["cos_all_layers_vs_plus2s_amp"] = round(
                cos(shifts["all_layers"], amp_shift[dim]), 4)
        d_out[str(dim)] = entry
        print(f"  [D] dim {dim}: all-layer dir={entry['all_layers']['directionality']} "
              f"split-half cos={entry['all_layers']['split_half_cos']} "
              f"one-vs-all cos={entry['cos_one_vs_all_layers']} "
              f"({time.time() - tD:.0f}s)", flush=True)
    results["D_write_transfer"] = {
        "recipient": "first 2 eval batches (~2,560 tokens)",
        "transfer_layer_for_one_layer_mode": args.transfer_layer,
        "per_dim": d_out,
        "metrics_note": "directionality = ||mean_pos dlp|| / mean_pos ||dlp||; "
                        "split_half_cos = cos of mean shifts on two "
                        "recipient halves; dlp = log-prob shift vs baseline",
    }
    print(f"[D] done in {time.time() - tD:.0f}s")

    for h in hooks:
        h.remove()

    wall = time.time() - t_start
    results["meta"]["wall_time_s"] = round(wall, 1)
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[done] wrote {args.out} (wall {wall:.0f}s)")


if __name__ == "__main__":
    main()
