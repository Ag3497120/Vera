#!/usr/bin/env python3
"""Trained shared bridge (joint GD) vs truncated SVD.

Follow-up to shared_bridge_vs_svd.py: same structure
    W_l ~= Ub @ C_l @ Vb.T   (shared Ub/Vb, per-layer C_l, NO orthogonality)
but Ub, Vb, {C_l} are jointly optimized with Adam starting from the
frozen-PCA init of the training-free protocol. Question: does minimal
training close the gap to per-layer truncated SVD?

Readouts:
  a) same-rank error ratio bridge/SVD (step 0 must reproduce prior numbers)
  b) PARAM-MATCHED: SVD rank r_svd with L*(m+n)*r_svd ~= m*r + n*r + L*r^2
  c) steps to reach 95% of total loss improvement, wall-clock time
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from shared_bridge_vs_svd import (
    build_shared_bridges,
    find_qwen_dir,
    load_proj_weights,
    reconstruct_svd,
    relative_fro,
    solve_C,
    svd_factors,
)

HERE = Path(__file__).resolve().parent


def bridge_params(m: int, n: int, r: int, L: int) -> int:
    return m * r + n * r + L * r * r


def svd_params(m: int, n: int, r: int, L: int) -> int:
    return L * (m + n) * r


def svd_rel_errors(mats: list[np.ndarray], rank: int) -> list[float]:
    rels = []
    for W in mats:
        U, S, V = svd_factors(W, rank)
        rels.append(relative_fro(W, reconstruct_svd(U, S, V)))
    return rels


def bridge_rel_errors(
    Ws: torch.Tensor, Ub: torch.Tensor, Vb: torch.Tensor, Cs: torch.Tensor
) -> list[float]:
    with torch.no_grad():
        rec = torch.einsum("mr,lrs,ns->lmn", Ub, Cs, Vb)
        num = torch.linalg.norm((Ws - rec).flatten(1), dim=1)
        den = torch.linalg.norm(Ws.flatten(1), dim=1)
        return (num / den).tolist()


def train_bridge(
    mats: list[np.ndarray],
    rank: int,
    steps: int,
    lr: float,
    device: torch.device,
    log_every: int = 100,
    plateau_window: int = 200,
    plateau_rel: float = 1e-5,
):
    """Joint Adam optimization of (Ub, Vb, {C_l}) from the frozen-PCA init."""
    # --- init: exactly the training-free protocol (float64 numpy) ---
    Us, Vs = [], []
    for W in mats:
        U, _, V = svd_factors(W, rank)
        Us.append(U)
        Vs.append(V)
    Ub0, Vb0 = build_shared_bridges(Us, Vs, rank)
    Cs0 = np.stack([solve_C(Ub0, Vb0, W) for W in mats])

    Ws = torch.tensor(np.stack(mats), dtype=torch.float32, device=device)
    Ub = torch.tensor(Ub0, dtype=torch.float32, device=device, requires_grad=True)
    Vb = torch.tensor(Vb0, dtype=torch.float32, device=device, requires_grad=True)
    Cs = torch.tensor(Cs0, dtype=torch.float32, device=device, requires_grad=True)

    den2 = (Ws ** 2).flatten(1).sum(dim=1)  # per-layer ||W||_F^2

    def loss_fn() -> torch.Tensor:
        rec = torch.einsum("mr,lrs,ns->lmn", Ub, Cs, Vb)
        num2 = ((Ws - rec) ** 2).flatten(1).sum(dim=1)
        return (num2 / den2).sum()

    opt = torch.optim.Adam([Ub, Vb, Cs], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr / 10)

    rel0 = bridge_rel_errors(Ws, Ub, Vb, Cs)
    loss_hist: list[float] = []
    log_rows = []
    t0 = time.time()

    for step in range(steps):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()
        sched.step()
        loss_hist.append(float(loss.item()))

        if (step + 1) % log_every == 0:
            rels = bridge_rel_errors(Ws, Ub, Vb, Cs)
            row = {
                "step": step + 1,
                "loss": loss_hist[-1],
                "bridge_rel_fro_mean": float(np.mean(rels)),
                "lr": float(sched.get_last_lr()[0]),
            }
            log_rows.append(row)
            print(f"    step {row['step']:5d}  loss={row['loss']:.6f}  "
                  f"rel_fro_mean={row['bridge_rel_fro_mean']:.6f}  lr={row['lr']:.2e}")
            # plateau early stop: relative loss improvement over the window
            if len(loss_hist) > plateau_window:
                prev = loss_hist[-plateau_window - 1]
                if (prev - loss_hist[-1]) < plateau_rel * abs(loss_hist[-1]):
                    print(f"    early stop at step {step + 1} (plateau)")
                    break

    wall = time.time() - t0
    rel_final = bridge_rel_errors(Ws, Ub, Vb, Cs)

    # steps to reach 95% of total loss improvement
    loss_start = loss_hist[0]
    loss_end = loss_hist[-1]
    target = loss_start - 0.95 * (loss_start - loss_end)
    steps_95 = next(i + 1 for i, v in enumerate(loss_hist) if v <= target)

    return {
        "rel_fro_step0": rel0,
        "rel_fro_final": rel_final,
        "loss_step0": float(loss_hist[0]),
        "loss_final": float(loss_hist[-1]),
        "steps_run": len(loss_hist),
        "steps_to_95pct": int(steps_95),
        "wall_seconds": wall,
        "log": log_rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=5)
    ap.add_argument("--layer-start", type=int, default=0)
    ap.add_argument("--module", type=str, default="down_proj")
    ap.add_argument("--ranks", type=str, default="32,64,128")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    torch.manual_seed(0)

    model_dir = find_qwen_dir()
    print(f"[trained-bridge] model_dir={model_dir} module={args.module} "
          f"layers=[{args.layer_start},{args.layer_start + args.layers}) device={device}")
    keys, mats = load_proj_weights(
        model_dir, args.layers, module=args.module, layer_start=args.layer_start
    )
    m, n = mats[0].shape
    L = len(mats)

    runs = []
    for rank in [int(x) for x in args.ranks.split(",") if x.strip()]:
        print(f"\n=== rank={rank} ===")
        svd_same = svd_rel_errors(mats, rank)
        print(f"  SVD(same rank) rel_fro mean={np.mean(svd_same):.6f}")

        # param-matched SVD rank
        p_bridge = bridge_params(m, n, rank, L)
        r_svd_match = max(1, round(p_bridge / (L * (m + n))))
        p_svd_match = svd_params(m, n, r_svd_match, L)
        svd_matched = svd_rel_errors(mats, r_svd_match)
        print(f"  param-matched SVD: r_svd={r_svd_match} "
              f"(bridge {p_bridge:,} vs svd {p_svd_match:,} params) "
              f"rel_fro mean={np.mean(svd_matched):.6f}")

        tr = train_bridge(mats, rank, args.steps, args.lr, device)

        b0 = float(np.mean(tr["rel_fro_step0"]))
        bf = float(np.mean(tr["rel_fro_final"]))
        ratio0 = b0 / np.mean(svd_same)
        ratiof = bf / np.mean(svd_same)
        pm_win = bf < float(np.mean(svd_matched))
        verdict = (
            "WIN_PARAM_MATCHED" if pm_win
            else ("PARITY_SAME_RANK" if ratiof <= 1.005 else "STILL_LOSES")
        )
        print(f"  step0:  bridge={b0:.6f}  ratio={ratio0:.3f}")
        print(f"  final:  bridge={bf:.6f}  ratio={ratiof:.3f}")
        print(f"  param-matched: bridge {bf:.6f} vs SVD(r={r_svd_match}) "
              f"{np.mean(svd_matched):.6f} -> {'WIN' if pm_win else 'LOSE'}")
        print(f"  steps_to_95pct={tr['steps_to_95pct']}  "
              f"wall={tr['wall_seconds']:.1f}s  verdict={verdict}")

        runs.append({
            "rank": rank,
            "shape": [m, n],
            "layers": L,
            "svd_same_rank_rel_fro_mean": float(np.mean(svd_same)),
            "svd_same_rank_rel_fro_per_layer": svd_same,
            "bridge_rel_fro_mean_step0": b0,
            "bridge_rel_fro_mean_final": bf,
            "bridge_rel_fro_per_layer_step0": tr["rel_fro_step0"],
            "bridge_rel_fro_per_layer_final": tr["rel_fro_final"],
            "error_ratio_step0": float(ratio0),
            "error_ratio_final": float(ratiof),
            "params_bridge": p_bridge,
            "r_svd_param_matched": int(r_svd_match),
            "params_svd_matched": p_svd_match,
            "svd_param_matched_rel_fro_mean": float(np.mean(svd_matched)),
            "param_matched_win": bool(pm_win),
            "steps_run": tr["steps_run"],
            "steps_to_95pct": tr["steps_to_95pct"],
            "wall_seconds": tr["wall_seconds"],
            "loss_step0": tr["loss_step0"],
            "loss_final": tr["loss_final"],
            "train_log": tr["log"],
            "verdict": verdict,
        })

    out = Path(args.out) if args.out else HERE / "results_trained_bridge.json"
    out.write_text(json.dumps({
        "model_dir": str(model_dir),
        "module": args.module,
        "layer_start": args.layer_start,
        "layer_end_exclusive": args.layer_start + args.layers,
        "keys": keys,
        "optimizer": {"name": "adam", "lr": args.lr,
                      "schedule": "cosine to lr/10", "max_steps": args.steps},
        "device": str(device),
        "runs": runs,
    }, indent=2), encoding="utf-8")
    print(f"\n[trained-bridge] wrote {out}")


if __name__ == "__main__":
    main()
