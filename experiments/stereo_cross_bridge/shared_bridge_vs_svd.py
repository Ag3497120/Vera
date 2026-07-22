#!/usr/bin/env python3
"""Shared U/V bridge + per-layer C_valve vs independent SVD.

Small-scale go/no-go for "real" 立体十字:
  W_ℓ ≈ U_bridge @ C_ℓ @ V_bridge.T
vs
  W_ℓ ≈ U_ℓ @ diag(S_ℓ) @ V_ℓ.T  (truncated SVD rank r)

Default: Qwen2.5-0.5B down_proj, first N layers.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def find_qwen_dir() -> Path:
    env = os.environ.get("STEREO_CROSS_MODEL_DIR")
    if env and Path(env).is_dir():
        return Path(env)
    hub = Path.home() / ".cache/huggingface/hub"
    for name in (
        "models--Qwen--Qwen2.5-0.5B-Instruct",
        "models--Qwen--Qwen2.5-0.5B",
        "models--Qwen--Qwen1.5-0.5B-Chat",
    ):
        snaps = hub / name / "snapshots"
        if snaps.is_dir():
            for p in snaps.iterdir():
                if list(p.glob("*.safetensors")):
                    return p
    local = Path("/Users/motonishikoudai/verantyx_v6/Verantyx-God-Mode-Space/qwen_0.5b")
    if local.is_dir() and list(local.glob("*.safetensors")):
        return local
    raise FileNotFoundError("No Qwen 0.5B safetensors found; set STEREO_CROSS_MODEL_DIR")


def load_proj_weights(
    model_dir: Path,
    n_layers: int,
    module: str = "down_proj",
    layer_start: int = 0,
):
    """Load layers.{layer_start .. layer_start+n_layers-1}.{module}.weight

    module examples: down_proj, o_proj, q_proj, k_proj, v_proj, gate_proj, up_proj
    """
    import torch
    from safetensors import safe_open

    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(model_dir)

    needle = f".{module}.weight"
    key_to_file = {}
    for f in files:
        with safe_open(str(f), framework="pt") as sf:
            for k in sf.keys():
                if k.endswith(needle) and ".layers." in k:
                    key_to_file[k] = f

    def layer_idx(k: str) -> int:
        parts = k.split(".")
        for i, p in enumerate(parts):
            if p == "layers":
                return int(parts[i + 1])
        return -1

    lo, hi = layer_start, layer_start + n_layers
    keys = sorted(key_to_file, key=layer_idx)
    keys = [k for k in keys if lo <= layer_idx(k) < hi]
    if len(keys) < 2:
        raise RuntimeError(
            f"Need >=2 {module} layers in [{lo},{hi}), "
            f"available={[layer_idx(k) for k in sorted(key_to_file, key=layer_idx)][:8]}..."
        )

    mats = []
    used = []
    for k in keys:
        with safe_open(str(key_to_file[k]), framework="pt") as sf:
            W = sf.get_tensor(k).to(torch.float32).numpy()
        mats.append(np.asarray(W, dtype=np.float64))
        used.append(k)
        print(f"  loaded {k} {W.shape}")
    return used, mats


# backward-compatible alias
def load_down_projs(model_dir: Path, n_layers: int, layer_start: int = 0):
    return load_proj_weights(
        model_dir, n_layers, module="down_proj", layer_start=layer_start
    )


def svd_factors(W: np.ndarray, rank: int):
    # W: (m, n)  — for down_proj typically (hidden, intermediate)
    U, S, Vh = np.linalg.svd(np.asarray(W, dtype=np.float64), full_matrices=False)
    r = min(rank, int(S.shape[0]))
    return U[:, :r].copy(), S[:r].copy(), Vh[:r, :].T.copy()


def relative_fro(W: np.ndarray, Wh: np.ndarray) -> float:
    num = np.linalg.norm(W - Wh, "fro")
    den = np.linalg.norm(W, "fro") + 1e-12
    return float(num / den)


def build_shared_bridges(Us: list[np.ndarray], Vs: list[np.ndarray], rank: int):
    """PCA-style shared bases from stacked per-layer SVD factors."""
    # Stack columns: [U0|U1|...] then take top-r left singular vectors
    U_stack = np.concatenate(Us, axis=1)  # (m, L*r0)
    V_stack = np.concatenate(Vs, axis=1)
    Qu, _, _ = np.linalg.svd(U_stack, full_matrices=False)
    Qv, _, _ = np.linalg.svd(V_stack, full_matrices=False)
    r = min(rank, Qu.shape[1], Qv.shape[1])
    Ub = Qu[:, :r].astype(np.float64)
    Vb = Qv[:, :r].astype(np.float64)
    # Orthonormal columns already from SVD
    return Ub, Vb


def solve_C(Ub: np.ndarray, Vb: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Optimal Frobenius C for fixed orthonormal-ish Ub,Vb: C = Ub.T @ W @ Vb."""
    return Ub.T @ W @ Vb


def reconstruct_bridge(Ub, C, Vb) -> np.ndarray:
    return Ub @ C @ Vb.T


def reconstruct_svd(U, S, V) -> np.ndarray:
    return (U.astype(np.float64) * S.astype(np.float64)) @ V.astype(np.float64).T


def spectrum_energy(W: np.ndarray, rank: int) -> float:
    """Fraction of Frobenius energy in top-r singular values."""
    S = np.linalg.svd(np.asarray(W, dtype=np.float64), compute_uv=False)
    r = min(rank, S.shape[0])
    return float((S[:r] ** 2).sum() / ((S ** 2).sum() + 1e-12))


def param_count_independent(m, n, r, L) -> int:
    # U,S,V per layer (S as r)
    return L * (m * r + r + n * r)


def param_count_shared(m, n, r, L) -> int:
    # Ub, Vb once + C (r×r) per layer  (no separate S; folded into C)
    return m * r + n * r + L * (r * r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=5)
    ap.add_argument("--layer-start", type=int, default=0,
                    help="first layer index (inclusive), e.g. 19 for deep band")
    ap.add_argument("--module", type=str, default="down_proj",
                    help="weight module: down_proj|o_proj|q_proj|...")
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--ranks", type=str, default="",
                    help="comma ranks to sweep, e.g. 32,64,128 (overrides --rank)")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    model_dir = find_qwen_dir()
    print(
        f"[stereo] model_dir={model_dir} module={args.module} "
        f"layers=[{args.layer_start},{args.layer_start + args.layers})"
    )
    keys, mats = load_proj_weights(
        model_dir, args.layers, module=args.module, layer_start=args.layer_start
    )
    m, n = mats[0].shape
    L = len(mats)
    print(f"[stereo] L={L} shape=({m},{n})")

    ranks = [int(x) for x in args.ranks.split(",") if x.strip()] or [args.rank]
    rows = []

    for rank in ranks:
        print(f"\n=== rank={rank} ===")
        # 1) per-layer SVD
        Us, Ss, Vs = [], [], []
        svd_rels = []
        energies = []
        for W in mats:
            U, S, V = svd_factors(W, rank)
            Us.append(U)
            Ss.append(S)
            Vs.append(V)
            Wh = reconstruct_svd(U, S, V)
            svd_rels.append(relative_fro(W, Wh))
            energies.append(spectrum_energy(W, rank))
        print(f"  independent SVD rel_fro mean={np.mean(svd_rels):.6f} "
              f"max={np.max(svd_rels):.6f}  top{rank}_energy={np.mean(energies):.4f}")

        # 2) shared bridges from stacked U/V
        Ub, Vb = build_shared_bridges(Us, Vs, rank)
        r_eff = Ub.shape[1]

        # 3) LS C_valve per layer
        bridge_rels = []
        C_norms = []
        C_offdiag = []
        for W in mats:
            C = solve_C(Ub, Vb, W)
            Wh = reconstruct_bridge(Ub, C, Vb)
            bridge_rels.append(relative_fro(W, Wh))
            C_norms.append(float(np.linalg.norm(C, "fro")))
            # off-diagonal energy fraction
            off = C.copy()
            np.fill_diagonal(off, 0.0)
            tot = np.linalg.norm(C, "fro") + 1e-12
            C_offdiag.append(float(np.linalg.norm(off, "fro") / tot))

        print(f"  shared bridge+C rel_fro mean={np.mean(bridge_rels):.6f} "
              f"max={np.max(bridge_rels):.6f}")
        print(f"  C offdiag mass mean={np.mean(C_offdiag):.4f} "
              f"(0=diagonal-only)")

        p_ind = param_count_independent(m, n, r_eff, L)
        p_sh = param_count_shared(m, n, r_eff, L)
        print(f"  params independent_SVD={p_ind:,}  shared_bridge={p_sh:,}  "
              f"ratio={p_sh/p_ind:.3f}")

        # dense baseline
        dense = m * n * L
        row = {
            "module": args.module,
            "layer_start": args.layer_start,
            "layer_end_exclusive": args.layer_start + args.layers,
            "rank": r_eff,
            "layers": L,
            "shape": [m, n],
            "keys": keys,
            "svd_rel_fro_mean": float(np.mean(svd_rels)),
            "svd_rel_fro_max": float(np.max(svd_rels)),
            "svd_rel_fro_per_layer": svd_rels,
            "bridge_rel_fro_mean": float(np.mean(bridge_rels)),
            "bridge_rel_fro_max": float(np.max(bridge_rels)),
            "bridge_rel_fro_per_layer": bridge_rels,
            "C_offdiag_fraction_mean": float(np.mean(C_offdiag)),
            "params_independent_svd": p_ind,
            "params_shared_bridge": p_sh,
            "params_dense": dense,
            "compression_vs_dense_svd": p_ind / dense,
            "compression_vs_dense_bridge": p_sh / dense,
            "error_ratio_bridge_over_svd": float(
                np.mean(bridge_rels) / (np.mean(svd_rels) + 1e-12)
            ),
            "top_r_energy_mean": float(np.mean(energies)),
            "verdict": (
                # Need competitive error vs SVD AND absolute quality floor AND fewer params
                "PROMISING"
                if np.mean(bridge_rels) <= 1.25 * np.mean(svd_rels)
                and np.mean(bridge_rels) < 0.35
                and p_sh < p_ind
                else (
                    "COMPRESS_ONLY"
                    if np.mean(bridge_rels) <= 1.25 * np.mean(svd_rels) and p_sh < p_ind
                    else "WEAK_OR_LOSE"
                )
            ),
        }
        rows.append(row)
        print(f"  verdict={row['verdict']}  "
              f"err(bridge)/err(svd)={row['error_ratio_bridge_over_svd']:.3f}")

    tag = f"{args.module}_L{args.layer_start}-{args.layer_start + args.layers - 1}"
    out = Path(args.out) if args.out else (
        Path(__file__).resolve().parent / f"results_{tag}.json"
    )
    out.write_text(json.dumps({
        "model_dir": str(model_dir),
        "module": args.module,
        "layer_start": args.layer_start,
        "layer_end_exclusive": args.layer_start + args.layers,
        "runs": rows,
    }, indent=2), encoding="utf-8")
    print(f"\n[stereo] wrote {out}")


if __name__ == "__main__":
    main()
