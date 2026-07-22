#!/usr/bin/env python3
"""Orthogonal Procrustes alignment between adjacent-layer projection weights.

Probe: does W_{L+1} ≈ R* W_L for some orthogonal R*?
Hypothesis: RoPE-related q/k may hide a layer-wise rotational shared structure
that fixed Ub/Vb + C (PCA shared bridge) cannot see.

Primary readout: full-W Procrustes relative residual vs random / null baselines.

Note: left-Procrustes residual is invariant to left-orthogonal transforms of W_L
(including row permutations). Nulls must scramble the right factor (columns)
or entries — not rows alone.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from shared_bridge_vs_svd import find_qwen_dir, load_proj_weights

# Practical fork thresholds (pre-registered style):
# Statistical z alone is meaningless when random residuals concentrate tightly
# around ~1/sqrt(2) for square Gaussians. Require a *large* absolute gap.
CLEAR_RATIO_VS_RANDOM = 0.85  # real_proc / random_mean
CLEAR_ABS_PROC = 0.50         # absolute residual floor for "structure"


def prefer_qwen15_chat() -> Path:
    """Match prior stereo experiments (Qwen1.5-0.5B-Chat), not Qwen2.5."""
    env = os.environ.get("STEREO_CROSS_MODEL_DIR")
    if env and Path(env).is_dir():
        return Path(env)
    hub = Path.home() / ".cache/huggingface/hub"
    name = "models--Qwen--Qwen1.5-0.5B-Chat"
    snaps = hub / name / "snapshots"
    if snaps.is_dir():
        for p in snaps.iterdir():
            if list(p.glob("*.safetensors")):
                return p
    return find_qwen_dir()


def relative_fro_diff(A: np.ndarray, B: np.ndarray) -> float:
    """||A - B||_F / ||A||_F."""
    return float(np.linalg.norm(A - B, "fro") / (np.linalg.norm(A, "fro") + 1e-12))


def orthogonal_procrustes(A: np.ndarray, B: np.ndarray, force_rotation: bool = False):
    """Find R orthogonal minimizing ||A - R B||_F.

    Classic: SVD(A B.T) → R = U V^T.
    If force_rotation and det(R) < 0, flip last column of U (SO(n)).
    Returns (R, residual_rel, det_R).
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    # macOS Accelerate sometimes emits spurious overflow warnings on large GEMMs
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        M = A @ B.T
        U, _, Vt = np.linalg.svd(M, full_matrices=False)
        if force_rotation and U.shape[0] == Vt.shape[0]:
            # det(U V^T) = det(U) det(V^T); flip last col of U if negative
            sign_u, _ = np.linalg.slogdet(U)
            sign_v, _ = np.linalg.slogdet(Vt.T)
            if sign_u * sign_v < 0:
                U = U.copy()
                U[:, -1] *= -1.0
        R = U @ Vt
        if R.shape[0] == R.shape[1]:
            sign, _ = np.linalg.slogdet(R)
            det = float(sign)  # ±1 for orthogonal (up to numerics)
        else:
            det = float("nan")
        rel = relative_fro_diff(A, R @ B)
    return R, rel, det


def match_frobenius_norm(X: np.ndarray, target_norm: float) -> np.ndarray:
    n = np.linalg.norm(X, "fro") + 1e-12
    return X * (target_norm / n)


def _baseline_stats(residuals: list[float], raws: list[float] | None = None) -> dict:
    arr = np.asarray(residuals, dtype=np.float64)
    out = {
        "n_trials": int(arr.size),
        "proc_rel_mean": float(arr.mean()),
        "proc_rel_std": float(arr.std()),
        "proc_rel_p05": float(np.percentile(arr, 5)),
        "proc_rel_p50": float(np.percentile(arr, 50)),
        "proc_rel_p95": float(np.percentile(arr, 95)),
        "proc_rel_min": float(arr.min()),
        "proc_rel_max": float(arr.max()),
    }
    if raws is not None:
        raw_arr = np.asarray(raws, dtype=np.float64)
        out["raw_rel_mean"] = float(raw_arr.mean())
        out["raw_rel_std"] = float(raw_arr.std())
        out["ratio_proc_over_raw_mean"] = float((arr / (raw_arr + 1e-12)).mean())
    return out


def random_baseline(
    shape: tuple[int, int],
    n_trials: int,
    rng: np.random.Generator,
    match_norm: float | None = None,
    force_rotation: bool = False,
) -> dict:
    residuals = []
    raws = []
    for _ in range(n_trials):
        W0 = rng.standard_normal(shape)
        W1 = rng.standard_normal(shape)
        if match_norm is not None:
            W0 = match_frobenius_norm(W0, match_norm)
            W1 = match_frobenius_norm(W1, match_norm)
        raws.append(relative_fro_diff(W1, W0))
        _, rel, _ = orthogonal_procrustes(W1, W0, force_rotation=force_rotation)
        residuals.append(rel)
    out = _baseline_stats(residuals, raws)
    out["match_frobenius_norm"] = match_norm
    out["kind"] = "gaussian_independent"
    return out


def column_shuffle_baseline(
    W_src: np.ndarray,
    W_tgt: np.ndarray,
    n_trials: int,
    rng: np.random.Generator,
    force_rotation: bool = False,
) -> dict:
    """Permute columns of W_src (right factor) — NOT absorbed by left R."""
    residuals = []
    for _ in range(n_trials):
        perm = rng.permutation(W_src.shape[1])
        W_shuf = W_src[:, perm]
        _, rel, _ = orthogonal_procrustes(W_tgt, W_shuf, force_rotation=force_rotation)
        residuals.append(rel)
    out = _baseline_stats(residuals)
    out["kind"] = "column_shuffle"
    return out


def entry_shuffle_baseline(
    W_src: np.ndarray,
    W_tgt: np.ndarray,
    n_trials: int,
    rng: np.random.Generator,
    force_rotation: bool = False,
) -> dict:
    """Independently permute all entries of W_src (destroys row/col structure)."""
    residuals = []
    flat = W_src.reshape(-1)
    for _ in range(n_trials):
        W_shuf = rng.permutation(flat).reshape(W_src.shape)
        _, rel, _ = orthogonal_procrustes(W_tgt, W_shuf, force_rotation=force_rotation)
        residuals.append(rel)
    out = _baseline_stats(residuals)
    out["kind"] = "entry_shuffle"
    return out


def pair_metrics(W_l: np.ndarray, W_lp1: np.ndarray) -> dict:
    raw = relative_fro_diff(W_lp1, W_l)
    _, proc_o, det_o = orthogonal_procrustes(W_lp1, W_l, force_rotation=False)
    _, proc_so, det_so = orthogonal_procrustes(W_lp1, W_l, force_rotation=True)
    return {
        "raw_rel": raw,
        "proc_rel_O": proc_o,
        "proc_rel_SO": proc_so,
        "det_R_O": det_o,
        "det_R_SO": det_so,
        "improvement_raw_minus_proc_O": raw - proc_o,
        "ratio_proc_over_raw_O": proc_o / (raw + 1e-12),
        "improvement_raw_minus_proc_SO": raw - proc_so,
        "ratio_proc_over_raw_SO": proc_so / (raw + 1e-12),
    }


def optional_factor_procrustes(W_l: np.ndarray, W_lp1: np.ndarray) -> dict:
    """Optional: Procrustes on left/right SVD factors (thin SVD)."""
    Ul, _, Vhl = np.linalg.svd(W_l, full_matrices=False)
    Up, _, Vhp = np.linalg.svd(W_lp1, full_matrices=False)
    Vl, Vp = Vhl.T, Vhp.T
    _, u_rel, _ = orthogonal_procrustes(Up, Ul, force_rotation=False)
    _, v_rel, _ = orthogonal_procrustes(Vp, Vl, force_rotation=False)
    return {
        "U_proc_rel_O": u_rel,
        "V_proc_rel_O": v_rel,
    }


def summarize_pairs(pairs: list[dict], key: str = "proc_rel_O") -> dict:
    vals = np.asarray([p[key] for p in pairs], dtype=np.float64)
    raws = np.asarray([p["raw_rel"] for p in pairs], dtype=np.float64)
    return {
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "min": float(vals.min()),
        "max": float(vals.max()),
        "raw_mean": float(raws.mean()),
        "improvement_mean": float((raws - vals).mean()),
        "ratio_mean": float((vals / (raws + 1e-12)).mean()),
    }


def fork_verdict(real_mean: float, rand_base: dict, col_base: dict, entry_base: dict) -> dict:
    """Pre-registered practical fork (not mere statistical z)."""
    rand_mu = rand_base["proc_rel_mean"]
    col_mu = col_base["proc_rel_mean"]
    entry_mu = entry_base["proc_rel_mean"]
    ratio_rand = real_mean / (rand_mu + 1e-12)
    ratio_col = real_mean / (col_mu + 1e-12)
    ratio_entry = real_mean / (entry_mu + 1e-12)
    sd = rand_base["proc_rel_std"] + 1e-12
    z = (real_mean - rand_mu) / sd

    clear = (
        real_mean < CLEAR_ABS_PROC
        and ratio_rand < CLEAR_RATIO_VS_RANDOM
        and ratio_col < CLEAR_RATIO_VS_RANDOM
        and ratio_entry < CLEAR_RATIO_VS_RANDOM
    )
    # mild: below all null means but not past practical thresholds
    mild = (
        real_mean < rand_mu
        and real_mean < col_mu
        and real_mean < entry_mu
        and not clear
    )
    if clear:
        fork = "SUPPORTS_ROTATIONAL_STRUCTURE"
    elif mild:
        fork = "WEAK_SIGNAL"
    else:
        fork = "NO_STRUCTURE_UNDER_PROBE"

    return {
        "fork_verdict": fork,
        "ratio_vs_random": float(ratio_rand),
        "ratio_vs_column_shuffle": float(ratio_col),
        "ratio_vs_entry_shuffle": float(ratio_entry),
        "z_vs_random": float(z),
        "clear_abs_threshold": CLEAR_ABS_PROC,
        "clear_ratio_threshold": CLEAR_RATIO_VS_RANDOM,
    }


def run_module_band(
    model_dir: Path,
    module: str,
    layer_start: int,
    n_layers: int,
    rng: np.random.Generator,
    n_random: int,
    n_shuffle: int,
    do_factors: bool,
) -> dict:
    print(
        f"\n=== module={module} layers=[{layer_start},{layer_start + n_layers}) ==="
    )
    keys, mats = load_proj_weights(
        model_dir, n_layers, module=module, layer_start=layer_start
    )
    shape = tuple(mats[0].shape)
    norms = [float(np.linalg.norm(W, "fro")) for W in mats]
    mean_norm = float(np.mean(norms))

    pairs = []
    for i in range(len(mats) - 1):
        m = pair_metrics(mats[i], mats[i + 1])
        m["layer_a"] = layer_start + i
        m["layer_b"] = layer_start + i + 1
        m["key_a"] = keys[i]
        m["key_b"] = keys[i + 1]
        if do_factors:
            m["factors"] = optional_factor_procrustes(mats[i], mats[i + 1])
        pairs.append(m)
        print(
            f"  L{m['layer_a']}→L{m['layer_b']}: raw={m['raw_rel']:.4f}  "
            f"proc_O={m['proc_rel_O']:.4f}  proc_SO={m['proc_rel_SO']:.4f}  "
            f"ratio={m['ratio_proc_over_raw_O']:.4f}  det_O={m['det_R_O']:+.0f}"
        )

    summary_o = summarize_pairs(pairs, "proc_rel_O")
    summary_so = summarize_pairs(pairs, "proc_rel_SO")
    # Exclude first pair in band (often L0 is special)
    pairs_ex0 = pairs[1:] if len(pairs) > 1 else pairs
    summary_o_ex0 = summarize_pairs(pairs_ex0, "proc_rel_O")

    print(f"  [baseline] random Gaussian n={n_random} ...")
    t0 = time.time()
    rand_base = random_baseline(
        shape, n_random, rng, match_norm=mean_norm, force_rotation=False
    )
    print(
        f"  random proc_rel mean={rand_base['proc_rel_mean']:.4f} "
        f"p05={rand_base['proc_rel_p05']:.4f}  ({time.time() - t0:.1f}s)"
    )

    # Use mid-band pair for nulls (more typical than L0)
    mid = max(0, (len(mats) - 2) // 2)
    print(f"  [baseline] column/entry shuffle on L{layer_start+mid}→L{layer_start+mid+1} n={n_shuffle} ...")
    col_base = column_shuffle_baseline(
        mats[mid], mats[mid + 1], n_shuffle, rng, force_rotation=False
    )
    entry_base = entry_shuffle_baseline(
        mats[mid], mats[mid + 1], n_shuffle, rng, force_rotation=False
    )
    print(
        f"  col-shuffle mean={col_base['proc_rel_mean']:.4f}  "
        f"entry-shuffle mean={entry_base['proc_rel_mean']:.4f}"
    )

    fork = fork_verdict(summary_o["mean"], rand_base, col_base, entry_base)
    fork_ex0 = fork_verdict(summary_o_ex0["mean"], rand_base, col_base, entry_base)

    print(
        f"  summary proc_O mean={summary_o['mean']:.4f}  "
        f"(ex-first-pair={summary_o_ex0['mean']:.4f})  "
        f"vs random ratio={fork['ratio_vs_random']:.3f}  fork={fork['fork_verdict']}"
    )

    return {
        "module": module,
        "layer_start": layer_start,
        "layer_end_exclusive": layer_start + n_layers,
        "n_layers": len(mats),
        "shape": list(shape),
        "keys": keys,
        "frobenius_norms": norms,
        "pairs": pairs,
        "summary_O": summary_o,
        "summary_SO": summary_so,
        "summary_O_exclude_first_pair": summary_o_ex0,
        "random_baseline": rand_base,
        "column_shuffle_baseline": col_base,
        "entry_shuffle_baseline": entry_base,
        "gap": fork,
        "gap_exclude_first_pair": fork_ex0,
        "fork_verdict": fork["fork_verdict"],
        "fork_verdict_exclude_first_pair": fork_ex0["fork_verdict"],
    }


def run_cross_module(
    model_dir: Path,
    layer_start: int,
    n_layers: int,
    src_module: str,
    tgt_modules: list[str],
) -> list[dict]:
    """Procrustes src_module L → tgt_module L+1 (module-specificity control)."""
    print(f"\n=== cross-module: {src_module}_L → tgt_{{L+1}} ===")
    _, src_mats = load_proj_weights(
        model_dir, n_layers, module=src_module, layer_start=layer_start
    )
    out = []
    for tgt in tgt_modules:
        _, tgt_mats = load_proj_weights(
            model_dir, n_layers, module=tgt, layer_start=layer_start
        )
        if src_mats[0].shape != tgt_mats[0].shape:
            print(
                f"  skip {src_module}→{tgt}: shape mismatch "
                f"{src_mats[0].shape} vs {tgt_mats[0].shape}"
            )
            continue
        pairs = []
        for i in range(len(src_mats) - 1):
            m = pair_metrics(src_mats[i], tgt_mats[i + 1])
            m["layer_src"] = layer_start + i
            m["layer_tgt"] = layer_start + i + 1
            pairs.append(m)
        summ = summarize_pairs(pairs, "proc_rel_O")
        print(
            f"  {src_module}_L → {tgt}_{{L+1}}: proc_O mean={summ['mean']:.4f}  "
            f"raw mean={summ['raw_mean']:.4f}  ratio={summ['ratio_mean']:.4f}"
        )
        out.append({
            "src_module": src_module,
            "tgt_module": tgt,
            "pairs": pairs,
            "summary_O": summ,
        })
    return out


def write_markdown(path: Path, payload: dict) -> None:
    lines: list[str] = []
    lines.append("# Orthogonal Procrustes: adjacent-layer weight alignment")
    lines.append("")
    lines.append("## Question")
    lines.append("")
    lines.append(
        "Does adjacent-layer `W_{L+1} ≈ R* W_L` (orthogonal `R*`) hold for "
        "`q_proj` / `k_proj`, revealing a rotational shared structure that "
        "fixed `Ub/Vb + C` cannot see?"
    )
    lines.append("")
    lines.append("## Protocol")
    lines.append("")
    lines.append(f"- model: `{payload['model_dir']}`")
    lines.append(f"- bands: {payload.get('bands_note', '')}")
    lines.append(
        "- modules: "
        + ", ".join(payload.get("modules", []))
        + " (v/down = controls)"
    )
    lines.append(
        "- metric: relative Frobenius "
        "`||W_{L+1} - R W_L||_F / ||W_{L+1}||_F` "
        "(also raw `||W_{L+1}-W_L||_F / ||W_{L+1}||_F`)"
    )
    lines.append(
        "- baselines: Gaussian same-shape (Frobenius-norm matched); "
        "**column-shuffle** and **entry-shuffle** of `W_L` "
        "(row-shuffle is invalid for left-Procrustes — absorbed into `R`); "
        "cross-module `q_L → v/k_{L+1}`"
    )
    lines.append(
        f"- fork (practical): `SUPPORTS` only if proc residual < {CLEAR_ABS_PROC} "
        f"**and** real/null < {CLEAR_RATIO_VS_RANDOM} vs random, column-shuffle, "
        "and entry-shuffle. Tiny z-scores against a concentrated ~0.707 Gaussian "
        "floor do **not** count."
    )
    lines.append("")

    for band in payload["bands"]:
        tag = f"L{band['layer_start']}-{band['layer_end_exclusive'] - 1}"
        lines.append(f"## Band {tag}")
        lines.append("")
        lines.append(
            "| module | raw mean | proc_O mean | ex-first | "
            "random mean | col-shuf | entry-shuf | real/rand | fork |"
        )
        lines.append(
            "|--------|----------|-------------|----------|"
            "-------------|----------|------------|-----------|------|"
        )
        for m in band["modules"]:
            rb = m["random_baseline"]
            cb = m["column_shuffle_baseline"]
            eb = m["entry_shuffle_baseline"]
            lines.append(
                f"| **{m['module']}** | {m['summary_O']['raw_mean']:.4f} | "
                f"{m['summary_O']['mean']:.4f} | "
                f"{m['summary_O_exclude_first_pair']['mean']:.4f} | "
                f"{rb['proc_rel_mean']:.4f} | "
                f"{cb['proc_rel_mean']:.4f} | "
                f"{eb['proc_rel_mean']:.4f} | "
                f"{m['gap']['ratio_vs_random']:.3f} | `{m['fork_verdict']}` |"
            )
        lines.append("")

        lines.append("### Per-pair (proc_O)")
        lines.append("")
        for m in band["modules"]:
            cells = ", ".join(
                f"L{p['layer_a']}→{p['layer_b']}={p['proc_rel_O']:.4f}"
                for p in m["pairs"]
            )
            lines.append(f"- **{m['module']}**: {cells}")
        lines.append("")

        if band.get("cross_module"):
            lines.append("### Cross-module control (`q_L` → tgt `L+1`)")
            lines.append("")
            lines.append("| src → tgt | proc_O mean | raw mean | ratio |")
            lines.append("|-----------|-------------|----------|-------|")
            for c in band["cross_module"]:
                s = c["summary_O"]
                lines.append(
                    f"| {c['src_module']} → {c['tgt_module']} | "
                    f"{s['mean']:.4f} | {s['raw_mean']:.4f} | {s['ratio_mean']:.4f} |"
                )
            lines.append("")

    lines.append("## Verdict")
    lines.append("")
    lines.append(payload["overall_verdict_md"])
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- JSON: `{payload['json_path']}`")
    lines.append("- script: `procrustes_layer_align.py`")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def overall_verdict(bands: list[dict]) -> str:
    """Pre-registered fork text from q/k vs controls + baselines."""
    shallow = bands[0]
    by_mod = {m["module"]: m for m in shallow["modules"]}
    q = by_mod.get("q_proj")
    k = by_mod.get("k_proj")
    v = by_mod.get("v_proj")
    d = by_mod.get("down_proj")

    def line(m):
        if m is None:
            return "(missing)"
        return (
            f"{m['module']}: proc_O={m['summary_O']['mean']:.4f} "
            f"(ex-first={m['summary_O_exclude_first_pair']['mean']:.4f}; "
            f"random≈{m['random_baseline']['proc_rel_mean']:.4f}, "
            f"real/rand={m['gap']['ratio_vs_random']:.3f}) → `{m['fork_verdict']}`"
        )

    ropes = [m for m in (q, k) if m is not None]
    controls = [m for m in (v, d) if m is not None]
    rope_support = all(
        m["fork_verdict"] == "SUPPORTS_ROTATIONAL_STRUCTURE" for m in ropes
    )
    rope_none = all(
        m["fork_verdict"] == "NO_STRUCTURE_UNDER_PROBE" for m in ropes
    )
    ctrl_support = any(
        m["fork_verdict"] == "SUPPORTS_ROTATIONAL_STRUCTURE" for m in controls
    )

    bullets = [f"- {line(m)}" for m in (q, k, v, d) if m is not None]

    # Deep band note
    if len(bands) > 1:
        deep = bands[1]
        dby = {m["module"]: m for m in deep["modules"]}
        bits = []
        for name in ("q_proj", "k_proj", "v_proj", "down_proj"):
            if name in dby:
                mm = dby[name]
                bits.append(
                    f"{name}={mm['summary_O']['mean']:.4f}"
                    f"(rand≈{mm['random_baseline']['proc_rel_mean']:.4f})"
                )
        bullets.append(
            f"- deep L{deep['layer_start']}-{deep['layer_end_exclusive']-1}: "
            + "; ".join(bits)
        )

    if rope_support and not ctrl_support:
        conclusion = (
            "**Fork: SUPPORTS_ROTATIONAL_STRUCTURE (RoPE-tilted).** "
            "Adjacent-layer Procrustes residual for q/k is clearly below nulls; "
            "controls do not show the same gap. Worth redesigning bridge as fixed "
            "basis + per-layer rotation for q/k."
        )
    elif rope_support and ctrl_support:
        conclusion = (
            "**Fork: SUPPORTS_ROTATIONAL_STRUCTURE (not RoPE-specific).** "
            "q/k beat baselines, but so do non-RoPE controls — generic layer-wise "
            "orthogonal relatedness, not a RoPE-phase story alone."
        )
    elif rope_none:
        conclusion = (
            "**Fork: NO_STRUCTURE_UNDER_PROBE.** "
            "Adjacent-layer Procrustes residual for q/k sits at or above the "
            "Gaussian / shuffle null floor (~0.70 for square attn projs; ~1.0 for "
            "rectangular down_proj). \"Uniformly bad shared-bridge\" is **not** "
            "hiding an orthogonal map `W_{L+1}≈R W_L` that this full-W probe can "
            "see. Redesigning as fixed Ub/Vb + per-layer R is **not** motivated."
        )
    else:
        conclusion = (
            "**Fork: WEAK / MIXED.** "
            "No practical clear gap vs nulls. Do not redesign the forge bridge "
            "on this alone."
        )

    cross = shallow.get("cross_module") or []
    if cross and q is not None:
        same = q["summary_O"]["mean"]
        cross_means = ", ".join(
            f"{c['tgt_module']}={c['summary_O']['mean']:.4f}" for c in cross
        )
        bullets.append(
            f"- cross-module q_L→tgt: same-module q={same:.4f}; {cross_means}"
        )

    return conclusion + "\n\n" + "\n".join(bullets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=5)
    ap.add_argument("--layer-start", type=int, default=0)
    ap.add_argument(
        "--also-deep",
        action="store_true",
        help="also run L19–23 (same width) if model has enough layers",
    )
    ap.add_argument(
        "--modules",
        type=str,
        default="q_proj,k_proj,v_proj,down_proj",
        help="comma modules",
    )
    ap.add_argument("--n-random", type=int, default=30)
    ap.add_argument("--n-shuffle", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--factors", action="store_true", help="also Procrustes U/V factors")
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--md", type=str, default="")
    args = ap.parse_args()

    model_dir = prefer_qwen15_chat()
    modules = [m.strip() for m in args.modules.split(",") if m.strip()]
    rng = np.random.default_rng(args.seed)

    print(f"[procrustes] model_dir={model_dir}")
    print(
        f"[procrustes] modules={modules}  n_random={args.n_random}  "
        f"n_shuffle={args.n_shuffle}"
    )

    band_starts = [args.layer_start]
    if args.also_deep:
        band_starts.append(19)

    bands = []
    for ls in band_starts:
        mod_results = []
        for mod in modules:
            mod_results.append(
                run_module_band(
                    model_dir,
                    mod,
                    ls,
                    args.layers,
                    rng,
                    args.n_random,
                    args.n_shuffle,
                    args.factors,
                )
            )
        cross = []
        if "q_proj" in modules:
            tgts = [
                m for m in ("v_proj", "k_proj") if m in modules and m != "q_proj"
            ]
            cross = run_cross_module(model_dir, ls, args.layers, "q_proj", tgts)
        bands.append({
            "layer_start": ls,
            "layer_end_exclusive": ls + args.layers,
            "modules": mod_results,
            "cross_module": cross,
        })

    out = Path(args.out) if args.out else (
        Path(__file__).resolve().parent / "results_procrustes.json"
    )
    md_path = Path(args.md) if args.md else (
        Path(__file__).resolve().parent / "PROCRUSTES_COMPARE.md"
    )

    payload = {
        "model_dir": str(model_dir),
        "modules": modules,
        "n_random": args.n_random,
        "n_shuffle": args.n_shuffle,
        "seed": args.seed,
        "clear_abs_threshold": CLEAR_ABS_PROC,
        "clear_ratio_threshold": CLEAR_RATIO_VS_RANDOM,
        "bands_note": ", ".join(
            f"L{b['layer_start']}-{b['layer_end_exclusive'] - 1}" for b in bands
        ),
        "bands": bands,
        "json_path": str(out.name),
    }
    payload["overall_verdict_md"] = overall_verdict(bands)

    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(md_path, payload)
    print(f"\n[procrustes] wrote {out}")
    print(f"[procrustes] wrote {md_path}")
    print("\n" + payload["overall_verdict_md"])


if __name__ == "__main__":
    main()
