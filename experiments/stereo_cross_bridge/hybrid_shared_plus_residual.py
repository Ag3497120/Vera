#!/usr/bin/env python3
"""Hybrid: low-rank shared bridge + per-layer residual SVD.

Tests reading of prior experiment:
  dominant shared subspace (r_share) + layer-specific leftover (r_res).

Compare storage and rel_fro vs:
  A) independent SVD rank (r_share + r_res)
  B) shared-only rank r_share (no residual)
  C) hybrid: W ≈ Ub C Vb.T + Ures diag(Sres) Vres.T
"""
from __future__ import annotations

import argparse
import json
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


def param_count_hybrid(m, n, r_share, r_res, L) -> int:
    # Ub,Vb once + C(r_share^2) per layer + residual SVD per layer
    return (
        m * r_share
        + n * r_share
        + L * (r_share * r_share)
        + L * (m * r_res + r_res + n * r_res)
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=5)
    ap.add_argument("--module", type=str, default="down_proj")
    ap.add_argument("--r-share", type=int, default=64)
    ap.add_argument("--r-res", type=int, default=64)
    ap.add_argument(
        "--grid",
        type=str,
        default="64:64,64:128,128:128",
        help="comma list of r_share:r_res",
    )
    args = ap.parse_args()

    model_dir = find_qwen_dir()
    print(f"[hybrid] model_dir={model_dir} module={args.module}")
    keys, mats = load_proj_weights(model_dir, args.layers, module=args.module)
    m, n = mats[0].shape
    L = len(mats)

    grid = []
    for part in args.grid.split(","):
        a, b = part.strip().split(":")
        grid.append((int(a), int(b)))

    rows = []
    for r_share, r_res in grid:
        r_tot = r_share + r_res
        print(f"\n=== r_share={r_share} r_res={r_res} (tot≈{r_tot}) ===")

        # Independent SVD at total rank budget
        svd_rels = []
        energies = []
        Us, Vs = [], []
        for W in mats:
            U, S, V = svd_factors(W, r_share)  # for bridge init only
            Us.append(U)
            Vs.append(V)
            U2, S2, V2 = svd_factors(W, r_tot)
            svd_rels.append(relative_fro(W, reconstruct_svd(U2, S2, V2)))
            energies.append(spectrum_energy(W, r_tot))

        Ub, Vb = build_shared_bridges(Us, Vs, r_share)

        # Shared-only
        share_rels = []
        # Hybrid
        hyb_rels = []
        res_energy = []  # ||R||/||W||
        for W in mats:
            C = solve_C(Ub, Vb, W)
            W_share = reconstruct_bridge(Ub, C, Vb)
            share_rels.append(relative_fro(W, W_share))
            R = W - W_share
            res_energy.append(float(np.linalg.norm(R, "fro") / (np.linalg.norm(W, "fro") + 1e-12)))
            Ur, Sr, Vr = svd_factors(R, r_res)
            W_hat = W_share + reconstruct_svd(Ur, Sr, Vr)
            hyb_rels.append(relative_fro(W, W_hat))

        p_svd = param_count_independent(m, n, r_tot, L)
        p_share = param_count_shared(m, n, r_share, L)
        p_hyb = param_count_hybrid(m, n, r_share, r_res, L)

        row = {
            "module": args.module,
            "r_share": r_share,
            "r_res": r_res,
            "r_total_budget": r_tot,
            "keys": keys,
            "svd_tot_rel_fro_mean": float(np.mean(svd_rels)),
            "share_only_rel_fro_mean": float(np.mean(share_rels)),
            "hybrid_rel_fro_mean": float(np.mean(hyb_rels)),
            "residual_fro_over_W_mean": float(np.mean(res_energy)),
            "top_r_tot_energy_mean": float(np.mean(energies)),
            "params_svd_tot": p_svd,
            "params_share_only": p_share,
            "params_hybrid": p_hyb,
            "hybrid_vs_svd_error_ratio": float(
                np.mean(hyb_rels) / (np.mean(svd_rels) + 1e-12)
            ),
            "hybrid_vs_svd_param_ratio": p_hyb / p_svd,
            "verdict": (
                "HYBRID_PROMISING"
                if np.mean(hyb_rels) <= 1.15 * np.mean(svd_rels)
                and np.mean(hyb_rels) < 0.55
                and p_hyb < p_svd
                else (
                    "HYBRID_NEAR"
                    if np.mean(hyb_rels) <= 1.25 * np.mean(svd_rels) and p_hyb <= p_svd
                    else "HYBRID_WEAK"
                )
            ),
        }
        rows.append(row)
        print(
            f"  SVD(tot) rel={row['svd_tot_rel_fro_mean']:.4f}  "
            f"share-only={row['share_only_rel_fro_mean']:.4f}  "
            f"hybrid={row['hybrid_rel_fro_mean']:.4f}"
        )
        print(
            f"  residual/W={row['residual_fro_over_W_mean']:.4f}  "
            f"params hyb/svd={row['hybrid_vs_svd_param_ratio']:.3f}  "
            f"err hyb/svd={row['hybrid_vs_svd_error_ratio']:.3f}  "
            f"verdict={row['verdict']}"
        )

    out = Path(__file__).resolve().parent / f"hybrid_results_{args.module}.json"
    out.write_text(
        json.dumps({
            "model_dir": str(model_dir),
            "module": args.module,
            "runs": rows,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"\n[hybrid] wrote {out}")


if __name__ == "__main__":
    main()
