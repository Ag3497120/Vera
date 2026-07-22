#!/usr/bin/env python3
"""Per-attention-head shared U/V bridge vs independent SVD.

Forks the RoPE hypothesis for q_proj / k_proj:
  - Head specialization: high variance of bridge/SVD across heads
  - Uniform RoPE structure: all heads similarly high bridge/SVD

Reuses helpers from shared_bridge_vs_svd.py.
Default: Qwen1.5-0.5B-Chat, layers L0-4, modules q_proj and k_proj.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from shared_bridge_vs_svd import (
    build_shared_bridges,
    find_qwen_dir,
    load_proj_weights,
    param_count_independent,
    param_count_shared,
    reconstruct_bridge,
    reconstruct_svd,
    relative_fro,
    solve_C,
    spectrum_energy,
    svd_factors,
)


def load_attn_config(model_dir: Path) -> dict:
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    hidden = int(cfg["hidden_size"])
    n_q = int(cfg["num_attention_heads"])
    n_kv = int(cfg.get("num_key_value_heads", n_q))
    head_dim = int(cfg.get("head_dim", hidden // n_q))
    return {
        "hidden_size": hidden,
        "num_attention_heads": n_q,
        "num_key_value_heads": n_kv,
        "head_dim": head_dim,
        "num_hidden_layers": int(cfg.get("num_hidden_layers", -1)),
        "model_type": cfg.get("model_type"),
        "is_gqa": n_kv != n_q,
    }


def n_heads_for_module(module: str, attn: dict) -> int:
    if module == "q_proj":
        return attn["num_attention_heads"]
    if module in ("k_proj", "v_proj"):
        return attn["num_key_value_heads"]
    raise ValueError(f"per-head split only for q/k/v_proj, got {module}")


def slice_head(W: np.ndarray, head: int, head_dim: int) -> np.ndarray:
    """HF Linear weight is (out_features, in_features); heads are ROW slices."""
    lo = head * head_dim
    hi = lo + head_dim
    return np.asarray(W[lo:hi, :], dtype=np.float64)


def choose_ranks(requested: list[int], head_dim: int, hidden: int) -> list[int]:
    """Cap rank at min(requested, head_dim-1, hidden-1). Drop impossibles."""
    cap = min(head_dim, hidden) - 1
    if cap < 1:
        raise RuntimeError(f"degenerate head matrix head_dim={head_dim} hidden={hidden}")
    out = []
    for r in requested:
        r_eff = min(int(r), cap)
        if r_eff not in out:
            out.append(r_eff)
    return out


def run_one_head(mats_h: list[np.ndarray], rank: int) -> dict:
    m, n = mats_h[0].shape
    L = len(mats_h)
    Us, Ss, Vs = [], [], []
    svd_rels, energies = [], []
    for W in mats_h:
        U, S, V = svd_factors(W, rank)
        Us.append(U)
        Ss.append(S)
        Vs.append(V)
        svd_rels.append(relative_fro(W, reconstruct_svd(U, S, V)))
        energies.append(spectrum_energy(W, rank))

    Ub, Vb = build_shared_bridges(Us, Vs, rank)
    r_eff = Ub.shape[1]
    bridge_rels, C_offdiag = [], []
    for W in mats_h:
        C = solve_C(Ub, Vb, W)
        bridge_rels.append(relative_fro(W, reconstruct_bridge(Ub, C, Vb)))
        off = C.copy()
        np.fill_diagonal(off, 0.0)
        C_offdiag.append(float(np.linalg.norm(off, "fro") / (np.linalg.norm(C, "fro") + 1e-12)))

    ratio = float(np.mean(bridge_rels) / (np.mean(svd_rels) + 1e-12))
    return {
        "rank": r_eff,
        "shape": [m, n],
        "svd_rel_fro_mean": float(np.mean(svd_rels)),
        "bridge_rel_fro_mean": float(np.mean(bridge_rels)),
        "error_ratio_bridge_over_svd": ratio,
        "C_offdiag_fraction_mean": float(np.mean(C_offdiag)),
        "top_r_energy_mean": float(np.mean(energies)),
        "params_independent_svd": param_count_independent(m, n, r_eff, L),
        "params_shared_bridge": param_count_shared(m, n, r_eff, L),
    }


def summarize_heads(per_head: list[dict]) -> dict:
    ratios = np.array([h["error_ratio_bridge_over_svd"] for h in per_head], dtype=np.float64)
    mean = float(ratios.mean())
    std = float(ratios.std(ddof=0))
    cv = float(std / (mean + 1e-12))
    imin = int(ratios.argmin())
    imax = int(ratios.argmax())
    return {
        "n_heads": len(per_head),
        "ratio_mean": mean,
        "ratio_std": std,
        "ratio_cv": cv,
        "ratio_min": float(ratios[imin]),
        "ratio_max": float(ratios[imax]),
        "head_min": imin,
        "head_max": imax,
        "ratio_per_head": [float(x) for x in ratios],
        "svd_rel_fro_mean_across_heads": float(
            np.mean([h["svd_rel_fro_mean"] for h in per_head])
        ),
        "bridge_rel_fro_mean_across_heads": float(
            np.mean([h["bridge_rel_fro_mean"] for h in per_head])
        ),
    }


def run_module(
    model_dir: Path,
    module: str,
    attn: dict,
    layer_start: int,
    n_layers: int,
    ranks: list[int],
) -> dict:
    n_heads = n_heads_for_module(module, attn)
    head_dim = attn["head_dim"]
    hidden = attn["hidden_size"]
    expected_out = n_heads * head_dim

    print(
        f"\n[per-head] module={module} n_heads={n_heads} head_dim={head_dim} "
        f"GQA={attn['is_gqa']} layers=[{layer_start},{layer_start + n_layers})"
    )
    keys, mats = load_proj_weights(
        model_dir, n_layers, module=module, layer_start=layer_start
    )
    m, n = mats[0].shape
    if m != expected_out or n != hidden:
        raise RuntimeError(
            f"{module} shape {(m, n)} != expected ({expected_out}, {hidden}) "
            f"(n_heads={n_heads}, head_dim={head_dim}, hidden={hidden})"
        )
    print(f"[per-head] full W shape=({m},{n}); per-head=({head_dim},{n})")

    ranks_eff = choose_ranks(ranks, head_dim, n)
    skipped = [r for r in ranks if r not in ranks_eff]
    print(
        f"[per-head] ranks requested={ranks} effective={ranks_eff} "
        f"(cap=min(head_dim,hidden)-1={min(head_dim, n) - 1})"
        + (f" skipped_impossible={skipped}" if skipped else "")
    )
    for r in ranks_eff:
        print(f"  rank {r} = {r / head_dim:.3f} × head_dim")

    runs = []
    for rank in ranks_eff:
        print(f"\n=== {module} rank={rank} ===")
        per_head = []
        for h in range(n_heads):
            mats_h = [slice_head(W, h, head_dim) for W in mats]
            row = run_one_head(mats_h, rank)
            row["head"] = h
            per_head.append(row)
            print(
                f"  head[{h:02d}] bridge/SVD={row['error_ratio_bridge_over_svd']:.4f} "
                f"svd={row['svd_rel_fro_mean']:.4f} bridge={row['bridge_rel_fro_mean']:.4f}"
            )
        summary = summarize_heads(per_head)
        print(
            f"  summary mean={summary['ratio_mean']:.4f} std={summary['ratio_std']:.4f} "
            f"cv={summary['ratio_cv']:.4f} "
            f"min=h{summary['head_min']}({summary['ratio_min']:.4f}) "
            f"max=h{summary['head_max']}({summary['ratio_max']:.4f})"
        )
        runs.append(
            {
                "module": module,
                "layer_start": layer_start,
                "layer_end_exclusive": layer_start + n_layers,
                "rank": rank,
                "rank_as_fraction_of_head_dim": rank / head_dim,
                "layers": len(mats),
                "full_shape": [m, n],
                "per_head_shape": [head_dim, n],
                "n_heads": n_heads,
                "head_dim": head_dim,
                "keys": keys,
                "summary": summary,
                "per_head": per_head,
            }
        )

    return {
        "module": module,
        "layer_start": layer_start,
        "layer_end_exclusive": layer_start + n_layers,
        "attn_config": {
            k: attn[k]
            for k in (
                "hidden_size",
                "num_attention_heads",
                "num_key_value_heads",
                "head_dim",
                "is_gqa",
                "model_type",
            )
        },
        "ranks_requested": ranks,
        "ranks_effective": ranks_eff,
        "rank_cap_note": (
            f"per-head matrix is ({head_dim}×{n}); "
            f"rank capped at {min(head_dim, n) - 1}. "
            f"r=128 impossible when head_dim={head_dim}."
        ),
        "runs": runs,
    }


def fork_verdict(module_payloads: list[dict], mlp_ref: float = 1.08) -> dict:
    """Pre-registered fork at the primary comparable rank (prefer 32)."""
    # Collect ratio_mean / std / cv at each shared rank
    by_rank: dict[int, dict] = {}
    for payload in module_payloads:
        for run in payload["runs"]:
            r = run["rank"]
            s = run["summary"]
            by_rank.setdefault(r, {})[payload["module"]] = s

    primary = 32 if 32 in by_rank else sorted(by_rank)[-1]
    stats = by_rank[primary]
    # Aggregate across modules for specialization signal
    cvs = [stats[m]["ratio_cv"] for m in stats]
    spreads = [stats[m]["ratio_max"] - stats[m]["ratio_min"] for m in stats]
    means = [stats[m]["ratio_mean"] for m in stats]
    mins = [stats[m]["ratio_min"] for m in stats]
    maxs = [stats[m]["ratio_max"] for m in stats]

    # Heuristics (pre-registered in PER_HEAD_COMPARE.md):
    # specialization: high relative dispersion (CV>=0.08) OR clear MLP-like vs extreme split
    # uniform RoPE: all head means elevated vs MLP and low CV (<0.08)
    max_cv = max(cvs)
    max_spread = max(spreads)
    any_mlp_like = any(x <= mlp_ref * 1.05 for x in mins)
    any_extreme = any(x >= mlp_ref * 1.15 for x in maxs)
    all_high = all(x >= mlp_ref * 1.12 for x in means)

    if any_mlp_like and any_extreme and max_cv >= 0.05:
        label = "HEAD_SPECIALIZATION"
        reading = (
            "Some heads ≈ MLP-like while others remain extreme; "
            "favors head functional specialization over uniform RoPE structure."
        )
    elif max_cv >= 0.08:
        label = "HEAD_SPECIALIZATION"
        reading = (
            "High cross-head coefficient of variation in bridge/SVD; "
            "favors head functional specialization over uniform RoPE structure."
        )
    elif all_high and max_cv < 0.08:
        label = "UNIFORM_ROPE"
        reading = (
            "All heads show similarly elevated bridge/SVD (low CV, none MLP-like); "
            "favors RoPE frequency structure over head functional specialization."
        )
    else:
        label = "MIXED"
        reading = (
            "Neither cleanly uniform-high nor strongly specialized; "
            "report numbers and compare to whole-matrix q/k."
        )

    return {
        "primary_rank": primary,
        "mlp_ref_r64_approx": mlp_ref,
        "per_module_at_primary": {
            m: {
                "ratio_mean": stats[m]["ratio_mean"],
                "ratio_std": stats[m]["ratio_std"],
                "ratio_cv": stats[m]["ratio_cv"],
                "ratio_min": stats[m]["ratio_min"],
                "ratio_max": stats[m]["ratio_max"],
                "head_min": stats[m]["head_min"],
                "head_max": stats[m]["head_max"],
            }
            for m in stats
        },
        "max_cv": max_cv,
        "max_spread": max_spread,
        "verdict": label,
        "reading": reading,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=5)
    ap.add_argument("--layer-start", type=int, default=0)
    ap.add_argument(
        "--modules",
        type=str,
        default="q_proj,k_proj",
        help="comma modules, default q_proj,k_proj",
    )
    ap.add_argument(
        "--ranks",
        type=str,
        default="8,16,32",
        help="comma ranks; auto-capped by head_dim (default 8,16,32; 128 impossible for head_dim=64)",
    )
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--mlp-ref", type=float, default=1.08,
                    help="MLP-cluster bridge/SVD reference at r=64 for fork comparison")
    args = ap.parse_args()

    model_dir = find_qwen_dir()
    attn = load_attn_config(model_dir)
    ranks = [int(x) for x in args.ranks.split(",") if x.strip()]
    modules = [m.strip() for m in args.modules.split(",") if m.strip()]

    print(f"[per-head] model_dir={model_dir}")
    print(
        f"[per-head] config hidden={attn['hidden_size']} "
        f"n_q={attn['num_attention_heads']} n_kv={attn['num_key_value_heads']} "
        f"head_dim={attn['head_dim']} GQA={attn['is_gqa']}"
    )

    module_payloads = []
    for module in modules:
        module_payloads.append(
            run_module(
                model_dir,
                module,
                attn,
                args.layer_start,
                args.layers,
                ranks,
            )
        )

    verdict = fork_verdict(module_payloads, mlp_ref=args.mlp_ref)
    print(f"\n[per-head] FORK verdict={verdict['verdict']} @ r={verdict['primary_rank']}")
    print(f"[per-head] {verdict['reading']}")

    tag = f"L{args.layer_start}-{args.layer_start + args.layers - 1}"
    out = Path(args.out) if args.out else (
        Path(__file__).resolve().parent / f"results_per_head_{tag}.json"
    )
    payload = {
        "model_dir": str(model_dir),
        "layer_start": args.layer_start,
        "layer_end_exclusive": args.layer_start + args.layers,
        "attn_config": attn,
        "ranks_requested": ranks,
        "modules": module_payloads,
        "fork": verdict,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[per-head] wrote {out}")


if __name__ == "__main__":
    main()
