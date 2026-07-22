#!/usr/bin/env python3
"""Part B: shared-bridge vs per-layer SVD on GPT-2 small WEIGHTS.

Local reimplementation of the protocol in
experiments/stereo_cross_bridge/shared_bridge_vs_svd.py (that file is
untouched), adapted to GPT-2:

- GPT-2 uses Conv1D: weight stored (in_features, out_features), the
  transpose of nn.Linear. We transpose to (out, in) for consistency.
- attn.c_attn is fused qkv: weight (768, 2304); split output dim into
  thirds -> q, k, v each (768, 768) Conv1D -> transposed to (768, 768).

Modules: q, k, v (from c_attn), o (attn.c_proj), mlp_fc (mlp.c_fc),
mlp_proj (mlp.c_proj). Layers 0-4, ranks 32/64.

Key question (pre-registered fork): GPT-2 has NO RoPE. In Qwen1.5-0.5B,
q/k were the least shareable (bridge/SVD 1.33-1.53 at r=64/128) and the
ATTN_COMPARE.md verdict attributed that to RoPE.
- If GPT-2 q/k are ALSO much worse than its MLP modules -> the RoPE
  explanation weakens; attention-block layer-specificity strengthens.
- If GPT-2 q/k look like its MLP -> RoPE explanation strengthens.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def find_gpt2_dir() -> Path:
    hub = Path.home() / ".cache/huggingface/hub"
    for name in ("models--openai-community--gpt2", "models--gpt2"):
        snaps = hub / name / "snapshots"
        if snaps.is_dir():
            for p in snaps.iterdir():
                if list(p.glob("*.safetensors")):
                    return p
    raise FileNotFoundError("gpt2 safetensors not in HF cache")


def load_gpt2_module_weights(model_dir: Path, module: str, n_layers: int,
                             layer_start: int = 0):
    """Return list of (out, in) float64 weight matrices for one module.

    module in {q, k, v, o, mlp_fc, mlp_proj}
    """
    import torch
    from safetensors import safe_open

    f = next(iter(model_dir.glob("*.safetensors")))
    key_map = {
        "q": "attn.c_attn.weight",
        "k": "attn.c_attn.weight",
        "v": "attn.c_attn.weight",
        "o": "attn.c_proj.weight",
        "mlp_fc": "mlp.c_fc.weight",
        "mlp_proj": "mlp.c_proj.weight",
    }
    suffix = key_map[module]
    mats, keys = [], []
    with safe_open(str(f), framework="pt") as sf:
        for li in range(layer_start, layer_start + n_layers):
            key = f"h.{li}.{suffix}"
            if key not in sf.keys():
                key = f"transformer.{key}"
            W = sf.get_tensor(key).to(torch.float32).numpy().astype(np.float64)
            # Conv1D layout: (in, out)
            if module in ("q", "k", "v"):
                d = W.shape[0]
                i = {"q": 0, "k": 1, "v": 2}[module]
                W = W[:, i * d:(i + 1) * d]
            W = W.T  # -> (out, in), Linear convention
            mats.append(W)
            keys.append(f"{key}[{module}]" if module in ("q", "k", "v") else key)
    return keys, mats


# ---- protocol identical to stereo_cross_bridge/shared_bridge_vs_svd.py ----

def svd_factors(W, rank):
    U, S, Vh = np.linalg.svd(W, full_matrices=False)
    r = min(rank, S.shape[0])
    return U[:, :r].copy(), S[:r].copy(), Vh[:r, :].T.copy()


def relative_fro(W, Wh):
    return float(np.linalg.norm(W - Wh, "fro") / (np.linalg.norm(W, "fro") + 1e-12))


def build_shared_bridges(Us, Vs, rank):
    U_stack = np.concatenate(Us, axis=1)
    V_stack = np.concatenate(Vs, axis=1)
    Qu, _, _ = np.linalg.svd(U_stack, full_matrices=False)
    Qv, _, _ = np.linalg.svd(V_stack, full_matrices=False)
    r = min(rank, Qu.shape[1], Qv.shape[1])
    return Qu[:, :r], Qv[:, :r]


def run_module(mats, keys, ranks):
    m, n = mats[0].shape
    L = len(mats)
    rows = []
    for rank in ranks:
        Us, Ss, Vs, svd_rels = [], [], [], []
        for W in mats:
            U, S, V = svd_factors(W, rank)
            Us.append(U); Ss.append(S); Vs.append(V)
            svd_rels.append(relative_fro(W, (U * S) @ V.T))
        Ub, Vb = build_shared_bridges(Us, Vs, rank)
        bridge_rels = []
        for W in mats:
            C = Ub.T @ W @ Vb
            bridge_rels.append(relative_fro(W, Ub @ C @ Vb.T))
        ratio = float(np.mean(bridge_rels) / (np.mean(svd_rels) + 1e-12))
        rows.append({
            "rank": int(Ub.shape[1]),
            "layers": L,
            "shape": [m, n],
            "keys": keys,
            "svd_rel_fro_mean": float(np.mean(svd_rels)),
            "svd_rel_fro_per_layer": svd_rels,
            "bridge_rel_fro_mean": float(np.mean(bridge_rels)),
            "bridge_rel_fro_per_layer": bridge_rels,
            "error_ratio_bridge_over_svd": ratio,
        })
        print(f"    r={rank:3d}  SVD={np.mean(svd_rels):.4f}  "
              f"bridge={np.mean(bridge_rels):.4f}  ratio={ratio:.3f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=5)
    ap.add_argument("--layer-start", type=int, default=0)
    ap.add_argument("--ranks", type=str, default="32,64")
    ap.add_argument("--out", type=str, default=str(HERE / "results_gpt2_weight_bridge.json"))
    args = ap.parse_args()

    ranks = [int(x) for x in args.ranks.split(",")]
    model_dir = find_gpt2_dir()
    print(f"[gpt2-bridge] model_dir={model_dir} "
          f"layers=[{args.layer_start},{args.layer_start + args.layers}) ranks={ranks}")

    modules = ["q", "k", "v", "o", "mlp_fc", "mlp_proj"]
    all_results = {}
    for module in modules:
        print(f"\n=== module={module} ===")
        keys, mats = load_gpt2_module_weights(
            model_dir, module, args.layers, args.layer_start)
        print(f"    L={len(mats)} shape={mats[0].shape}")
        all_results[module] = run_module(mats, keys, ranks)

    out = Path(args.out)
    out.write_text(json.dumps({
        "model": "openai-community/gpt2",
        "model_dir": str(model_dir),
        "layer_start": args.layer_start,
        "layer_end_exclusive": args.layer_start + args.layers,
        "ranks": ranks,
        "protocol": "reimplementation of stereo_cross_bridge/shared_bridge_vs_svd.py; Conv1D transposed to (out,in); c_attn split into q/k/v thirds",
        "modules": all_results,
    }, indent=2), encoding="utf-8")
    print(f"\n[gpt2-bridge] wrote {out}")


if __name__ == "__main__":
    main()
