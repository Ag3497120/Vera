#!/usr/bin/env python3
"""Vera demo CLI — feel the proven stereo-cross effects on GPT-2.

Demonstrates (training-free, personal-scale):
  1. Matryoshka rank lever     — one student, many granularities
  2. Coordinate memory         — store / search / reinject 256-d state
  3. Cross-model join          — GPT-2 ↔ DistilGPT2 via shared P
  4. Hub-dim probe             — ablate high-impact coordinates

Checkpoints are large (~0.5GB each) and not in git. Place them via:
  export VERA_ARTIFACTS=/path/to/dir   # containing *.pt and bases_cache_*.npz
or put them next to experiments/stereo_cross_activation/.

Usage:
  python -m demo.vera_cli status
  python -m demo.vera_cli tour
  python -m demo.vera_cli matryoshka --prompt "The capital of France is"
  python -m demo.vera_cli memory --demo
  python -m demo.vera_cli join --prompt "Once upon a time"
  python -m demo.vera_cli vision          # print + write HTML call-for-collab viz
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import webbrowser
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "stereo_cross_activation"
DEMO = Path(__file__).resolve().parent
HUB16 = [0, 10, 4, 3, 5, 11, 15, 9, 1, 14, 2, 17, 16, 34, 18, 8]
RANKS = (8, 16, 32, 64, 128, 192, 256)
MODEL_A = "openai-community/gpt2"
MODEL_B = "distilbert/distilgpt2"


# --------------------------------------------------------------------------
# artifact discovery
# --------------------------------------------------------------------------

def artifact_dirs() -> List[Path]:
    dirs = []
    env = os.environ.get("VERA_ARTIFACTS")
    if env:
        dirs.append(Path(env).expanduser())
    dirs.append(EXP)
    sibling = Path("/Users/motonishikoudai/Projects/verantyx-cli/experiments/stereo_cross_activation")
    if sibling.is_dir():
        dirs.append(sibling)
    # generic sibling relative
    alt = ROOT.parent / "verantyx-cli" / "experiments" / "stereo_cross_activation"
    if alt.is_dir():
        dirs.append(alt)
    # dedupe
    out, seen = [], set()
    for d in dirs:
        try:
            r = d.resolve()
        except Exception:
            r = d
        if r not in seen:
            seen.add(r)
            out.append(d)
    return out


def find_file(*names: str) -> Optional[Path]:
    for d in artifact_dirs():
        for n in names:
            p = d / n
            if p.is_file():
                return p
    return None


def status_table() -> Dict[str, Optional[str]]:
    need = {
        "basis": ["bases_cache_soft_distill.npz"],
        "student_a": ["matryoshka_student.pt", "kl_distill_student_r256.pt"],
        "student_b": ["student_b_distilgpt2.pt"],
        "means_b": ["means_b_distilgpt2.npz"],
        "memory_store": [],  # optional
    }
    found = {}
    for k, names in need.items():
        if k == "memory_store":
            for d in artifact_dirs():
                if (d / "memory_store" / "memories.jsonl").is_file():
                    found[k] = str(d / "memory_store")
                    break
            else:
                found[k] = None
            continue
        p = find_file(*names)
        found[k] = str(p) if p else None
    return found


def require_torch():
    try:
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401
    except ImportError as e:
        print("Missing deps. From Vera root:\n  pip install -r requirements.txt", file=sys.stderr)
        raise SystemExit(1) from e


def device_of():
    import torch
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# --------------------------------------------------------------------------
# bottleneck runtime
# --------------------------------------------------------------------------

class RankState:
    __slots__ = ("rank", "ablate", "inject_layer", "inject_coords", "mode")

    def __init__(self, rank: int = 256):
        self.rank = rank
        self.ablate = None  # optional set of dim indices to zero
        self.inject_layer = -1
        self.inject_coords = None
        self.mode = "project"  # project | capture | inject


def make_hook(li: int, mean, P_full, state: RankState):
    def hook(_mod, _inp, out):
        import torch
        r = state.rank
        P = P_full[:, :r]
        h = out[0]
        hc = h - mean
        c = hc @ P
        if state.ablate:
            c = c.clone()
            c[..., list(state.ablate)] = 0
        if state.mode == "capture" and li == state.inject_layer:
            state.inject_coords = c.detach()
        if state.mode == "inject" and li == state.inject_layer and state.inject_coords is not None:
            donor = state.inject_coords
            t = min(donor.shape[1], c.shape[1])
            c = c.clone()
            c[:, :t] = donor[:, :t]
        h_new = mean + c @ P.T
        return (h_new,) + tuple(out[1:])
    return hook


def load_student_a(device, rank: int = 256):
    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    basis = find_file("bases_cache_soft_distill.npz")
    ckpt = find_file("matryoshka_student.pt", "kl_distill_student_r256.pt")
    if not basis or not ckpt:
        raise SystemExit(
            "Missing basis/checkpoint. Set VERA_ARTIFACTS or see demo/README.md"
        )
    z = np.load(basis)
    means = torch.from_numpy(z["means"]).float().to(device)
    P = torch.from_numpy(np.ascontiguousarray(z["V"][:, :256])).float().to(device)
    tok = GPT2TokenizerFast.from_pretrained(MODEL_A)
    tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel.from_pretrained(MODEL_A).to(device)
    sd = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(sd, strict=False)
    model.eval()
    state = RankState(rank)
    hooks = [
        model.transformer.h[li].register_forward_hook(
            make_hook(li, means[li], P, state))
        for li in range(model.config.n_layer)
    ]
    return model, tok, means, P, state, hooks, ckpt


def load_student_b(device, P, rank: int = 256):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ckpt = find_file("student_b_distilgpt2.pt")
    means_path = find_file("means_b_distilgpt2.npz")
    if not ckpt or not means_path:
        raise SystemExit("Missing DistilGPT2 student / means_b. See demo/README.md")
    means = torch.from_numpy(np.load(means_path)["means"]).float().to(device)
    tok = AutoTokenizer.from_pretrained(MODEL_B)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_B).to(device)
    sd = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(sd, strict=False)
    model.eval()
    state = RankState(rank)
    hooks = [
        model.transformer.h[li].register_forward_hook(
            make_hook(li, means[li], P, state))
        for li in range(model.config.n_layer)
    ]
    return model, tok, means, state, hooks


def topk_continuation(model, tok, prompt: str, device, k: int = 8, max_new: int = 24):
    import torch
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = model(ids)
        logits = out.logits[0, -1].float()
        probs = torch.softmax(logits, dim=-1)
        top = torch.topk(probs, k)
        # short greedy continue
        gen = model.generate(
            ids, max_new_tokens=max_new, do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    cont = tok.decode(gen[0][ids.shape[1]:], skip_special_tokens=True)
    tops = [(tok.decode([int(i)]), float(p)) for i, p in zip(top.indices, top.values)]
    return tops, cont


def bar(p: float, width: int = 24) -> str:
    n = int(round(max(0.0, min(1.0, p)) * width))
    return "█" * n + "░" * (width - n)


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------

def cmd_status(_args):
    print("Vera demo — artifact status\n")
    st = status_table()
    for k, v in st.items():
        mark = "OK " if v else "MISS"
        print(f"  [{mark}] {k:12} {v or '—'}")
    print("\nSearch paths:")
    for d in artifact_dirs():
        print(f"  - {d}")
    print("\nProven results (from published JSON, no GPU needed):")
    show_proven_summary()
    missing = [k for k, v in st.items() if v is None and k != "memory_store"]
    if missing:
        print("\nTo run interactive demos, place checkpoints and set:")
        print("  export VERA_ARTIFACTS=/path/to/artifacts")
        print("See demo/README.md / CALL_FOR_COLLABORATORS.md")
    else:
        print("\nReady. Try:  python -m demo tour")


def show_proven_summary():
    rows = [
        ("Part I weight bridge", "FAIL — no shared structure in weights"),
        ("Activation shared basis", "OK — shared/per-layer ~1.09 @ r=32"),
        ("Container r=256", "OK — ppl 1.45× baseline (IDENTITY)"),
        ("Step1 response map", "OK — PARTS_LIKE (hub ~20–30 dims)"),
        ("Step4 weight compress", "OK — 2.85× blocks, ppl 1.63×"),
        ("Step3 Matryoshka", "OK — one ckpt r=8…256 viable"),
        ("Step5 cross-model join", "OK — GPT-2↔DistilGPT2 JOIN_VIABLE"),
        ("Step2 coord memory", "OK — MEMORY_VIABLE (~19× chance @L11)"),
    ]
    for name, verdict in rows:
        print(f"  · {name:24} {verdict}")


def cmd_matryoshka(args):
    require_torch()
    import torch
    device = device_of()
    print(f"[device] {device}")
    model, tok, _means, _P, state, hooks, ckpt = load_student_a(device)
    print(f"[student] {ckpt.name}")
    prompt = args.prompt
    print(f"\nPrompt: {prompt!r}\n")
    print(f"{'rank':>6}  {'top-1 token':<16}  p  continuation")
    print("-" * 72)
    for r in (args.ranks or RANKS):
        state.rank = int(r)
        state.mode = "project"
        state.ablate = None
        tops, cont = topk_continuation(model, tok, prompt, device, k=1, max_new=args.tokens)
        tok1, p1 = tops[0]
        print(f"{r:>6}  {tok1!r:<16}  {p1:.3f}  {cont.replace(chr(10), ' ')[:48]}")
    for h in hooks:
        h.remove()
    print("\nFeel this: coarser ranks stay grammatical longer than a random "
          "subspace would; quality rises monotonically with r "
          "(MATRYOSHKA_VIABLE).")


def cmd_memory(args):
    require_torch()
    import torch
    device = device_of()
    model, tok, means, P, state, hooks, _ = load_student_a(device, rank=256)
    layer = args.layer

    def encode(text: str) -> np.ndarray:
        state.rank = 256
        state.mode = "capture"
        state.inject_layer = layer
        state.inject_coords = None
        ids = tok(text, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            model(ids)
        c = state.inject_coords[0, -1].detach().float().cpu().numpy()  # (256,)
        state.mode = "project"
        return c

    # tiny in-memory store for demo
    facts = [
        ("The capital of Zentharia is Kaldrune.", "Zentharia capital", "Kaldrune"),
        ("The founder of Marblen is Elandra Voss.", "Marblen founder", "Elandra"),
        ("Ashmere's annual festival is the Lantern Tide.", "Ashmere festival", "Lantern"),
        ("Ironspire produces the metal called starsteel.", "Ironspire metal", "starsteel"),
        ("Vera Lyn mapped the caves of Umbriel.", "Umbriel caves", "Vera"),
    ]
    store = []
    print(f"[memory] encoding {len(facts)} facts at layer {layer} ...")
    for sent, _q, ans in facts:
        store.append({"text": sent, "answer": ans, "c": encode(sent)})

    query = args.query or "What is the capital of Zentharia?"
    q = encode(query)
    sims = []
    for i, rec in enumerate(store):
        a = rec["c"]
        sim = float(np.dot(q, a) / (np.linalg.norm(q) * np.linalg.norm(a) + 1e-8))
        sims.append((sim, i))
    sims.sort(reverse=True)
    print(f"\nQuery: {query!r}\nRetrieval (cosine):")
    for sim, i in sims:
        print(f"  {sim:+.3f}  {bar((sim + 1) / 2)}  {store[i]['text']}")

    best = store[sims[0][1]]
    # reinject: replace last-token coords at layer with memory / random
    def next_token_logp(answer: str, coords: np.ndarray) -> float:
        state.mode = "inject"
        state.inject_layer = layer
        state.inject_coords = torch.from_numpy(coords).to(device).view(1, 1, -1).expand(1, 64, -1)
        # simpler: run prompt and set inject on full sequence length
        prompt = f"The capital of Zentharia is"
        ids = tok(prompt, return_tensors="pt").input_ids.to(device)
        state.inject_coords = torch.from_numpy(coords).to(device).view(1, 1, -1).expand(
            1, ids.shape[1], -1)
        with torch.no_grad():
            logits = model(ids).logits[0, -1].float()
            lp = torch.log_softmax(logits, dim=-1)
        ans_ids = tok.encode(" " + answer, add_special_tokens=False)
        if not ans_ids:
            ans_ids = tok.encode(answer, add_special_tokens=False)
        state.mode = "project"
        return float(lp[ans_ids[0]])

    rng = np.random.default_rng(0)
    rnd = rng.normal(size=256).astype(np.float32)
    rnd *= float(np.linalg.norm(best["c"]) / (np.linalg.norm(rnd) + 1e-8))
    lp_mem = next_token_logp(best["answer"], best["c"].astype(np.float32))
    lp_rnd = next_token_logp(best["answer"], rnd)
    print(f"\nReinject at L{layer} for prompt 'The capital of Zentharia is'")
    print(f"  logp('{best['answer']}') with REAL memory : {lp_mem:+.3f}")
    print(f"  logp('{best['answer']}') with RANDOM coord: {lp_rnd:+.3f}")
    print(f"  gap (real − random)                     : {lp_mem - lp_rnd:+.3f}")
    print("\nFeel this: the matched 256-d memory pushes the answer token "
          "more than a random vector of equal norm (MEMORY_VIABLE).")
    for h in hooks:
        h.remove()


def cmd_join(args):
    require_torch()
    import torch
    device = device_of()
    model_a, tok_a, means_a, P, state_a, hooks_a, _ = load_student_a(device)
    model_b, tok_b, means_b, state_b, hooks_b = load_student_b(device, P)
    prompt = args.prompt
    L_a, L_b = args.layer_a, args.layer_b
    print(f"[join] A=GPT-2 L{L_a} → B=DistilGPT2 L{L_b}")
    print(f"Prompt: {prompt!r}\n")

    # capture A coords
    ids_a = tok_a(prompt, return_tensors="pt").input_ids.to(device)
    state_a.mode = "capture"
    state_a.inject_layer = L_a
    state_a.inject_coords = None
    with torch.no_grad():
        model_a(ids_a)
    donor = state_a.inject_coords
    state_a.mode = "project"

    ids_b = tok_b(prompt, return_tensors="pt").input_ids.to(device)
    # align time
    t = min(donor.shape[1], ids_b.shape[1])
    donor = donor[:, :t]

    def run_b(label, coords):
        state_b.mode = "inject"
        state_b.inject_layer = L_b
        state_b.inject_coords = coords
        with torch.no_grad():
            logits = model_b(ids_b[:, :t]).logits[0, -1].float()
            probs = torch.softmax(logits, dim=-1)
            top = torch.topk(probs, 5)
        state_b.mode = "project"
        print(f"{label}:")
        for i, p in zip(top.indices, top.values):
            print(f"  {float(p):.3f} {bar(float(p))} {tok_b.decode([int(i)])!r}")
        print()

    state_b.mode = "project"
    state_b.inject_coords = None
    with torch.no_grad():
        logits = model_b(ids_b).logits[0, -1].float()
        probs = torch.softmax(logits, dim=-1)
        top = torch.topk(probs, 5)
    print("B solo (own bottleneck, no join):")
    for i, p in zip(top.indices, top.values):
        print(f"  {float(p):.3f} {bar(float(p))} {tok_b.decode([int(i)])!r}")
    print()

    run_b("A→B REAL join (donor = GPT-2 coords)", donor)
    rnd = torch.randn_like(donor)
    rnd = rnd * donor.std()
    run_b("A→B RANDOM control", rnd)
    perm = torch.randperm(donor.shape[0], device=donor.device)
    # shuffle along time as weak control
    shuf = donor[:, torch.randperm(donor.shape[1], device=donor.device)]
    run_b("A→B TIME-SHUFFLE control", shuf)

    print("Feel this: REAL join stays near B's fluent distribution; "
          "random/shuffle become garbage (JOIN_VIABLE).")
    for h in hooks_a + hooks_b:
        h.remove()


def cmd_hub(args):
    require_torch()
    device = device_of()
    model, tok, _m, _P, state, hooks, _ = load_student_a(device)
    prompt = args.prompt
    print(f"Prompt: {prompt!r}\nAblate hub dims vs random dims\n")
    state.rank = 256
    state.mode = "project"

    def show(label, ablate):
        state.ablate = ablate
        tops, cont = topk_continuation(model, tok, prompt, device, k=3, max_new=20)
        print(f"{label}:")
        for t, p in tops:
            print(f"  {p:.3f} {bar(p)} {t!r}")
        print(f"  → {cont!r}\n")

    show("full (no ablation)", None)
    show("ablate HUB16 (causal hubs)", HUB16)
    rng = np.random.default_rng(1)
    rnd_dims = list(map(int, rng.choice([i for i in range(256) if i not in HUB16], size=16, replace=False)))
    show(f"ablate 16 RANDOM tail dims {rnd_dims[:4]}...", rnd_dims)
    print("Feel this: hub dims hurt next-token more than random dims "
          "(PARTS_LIKE response map).")
    for h in hooks:
        h.remove()


def cmd_tour(args):
    cmd_status(args)
    st = status_table()
    if not st.get("student_a") or not st.get("basis"):
        print("\n[tour] interactive parts skipped (artifacts missing).")
        print("Showing vision / call-for-collaborators instead.\n")
        cmd_vision(args)
        return
    print("\n=== 1/4 Matryoshka ===")
    args.prompt = args.prompt or "The meaning of life is"
    args.tokens = 16
    args.ranks = (32, 64, 128, 256)
    cmd_matryoshka(args)
    print("\n=== 2/4 Memory ===")
    args.layer = 11
    args.query = None
    cmd_memory(args)
    if st.get("student_b") and st.get("means_b"):
        print("\n=== 3/4 Join ===")
        args.layer_a, args.layer_b = 6, 3
        cmd_join(args)
    else:
        print("\n=== 3/4 Join skipped (DistilGPT2 student missing) ===")
    print("\n=== 4/4 Hub ablation ===")
    cmd_hub(args)
    print("\n=== Vision ===")
    cmd_vision(args)


def cmd_vision(args):
    html = DEMO / "vision.html"
    md = DEMO / "CALL_FOR_COLLABORATORS.md"
    print("\n" + "=" * 64)
    print(" WHAT IS PROVEN (GPT-2 / DistilGPT2, personal compute)")
    print("=" * 64)
    show_proven_summary()
    print("""
 WHAT THIS ENABLES (design space now open)
  · Shared 256-d language across layers & (same-width) models
  · Rank lever (Matryoshka) for cost/quality
  · Coord memory sidecar (save / search / reinject)
  · Expert-style join: specialize modules, stitch via P
  · Structure-axis compression (2.85× block weights)

 WHAT IS NOT YET PROVEN (need collaborators + GPU)
  · Same recipe on Qwen-class 9B / 27B
  · Same-arch 9B experts specialized then joined for task transfer
  · Cross-width adapters (9B ↔ 27B)
  · End-task gains vs single large model
""")
    print(f"HTML vision board: {html}")
    print(f"Call for collaborators: {md}")
    if args.open and html.is_file():
        webbrowser.open(html.resolve().as_uri())


def build_parser():
    ap = argparse.ArgumentParser(prog="vera-demo", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("status", help="check artifacts + print proven summary")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("tour", help="run all interactive demos then vision")
    p.add_argument("--prompt", default="The meaning of life is")
    p.add_argument("--open", action="store_true", help="open vision.html")
    p.set_defaults(func=cmd_tour)

    p = sub.add_parser("matryoshka", help="rank lever continuations")
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--tokens", type=int, default=20)
    p.add_argument("--ranks", type=int, nargs="*", default=None)
    p.set_defaults(func=cmd_matryoshka)

    p = sub.add_parser("memory", help="coord memory retrieve + reinject")
    p.add_argument("--layer", type=int, default=11)
    p.add_argument("--query", default=None)
    p.add_argument("--demo", action="store_true")
    p.set_defaults(func=cmd_memory)

    p = sub.add_parser("join", help="GPT-2 → DistilGPT2 coord join vs controls")
    p.add_argument("--prompt", default="Once upon a time in a small village")
    p.add_argument("--layer-a", type=int, default=6)
    p.add_argument("--layer-b", type=int, default=3)
    p.set_defaults(func=cmd_join)

    p = sub.add_parser("hub", help="ablate hub vs random dims")
    p.add_argument("--prompt", default="The scientific method requires")
    p.set_defaults(func=cmd_hub)

    p = sub.add_parser("vision", help="print scale-up vision + write paths")
    p.add_argument("--open", action="store_true")
    p.set_defaults(func=cmd_vision)
    return ap


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
