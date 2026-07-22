#!/usr/bin/env python3
"""Step 2 — Coordinate memory sidecar (personal-scale demo).

Training-free pipeline on the Matryoshka / r=256 hard-bottleneck student:
  encode → append-only store → nearest-neighbor search → reinject + evaluate.

CLI:
  python coord_memory.py ingest
  python coord_memory.py query --text "..."
  python coord_memory.py reinject-eval
  python coord_memory.py run-all     # ingest + retrieval + reinject battery

Pre-registered fork:
  MEMORY_VIABLE: retrieval clearly above chance AND reinject of real memory
    beats random-memory control on at least one measurable behavioral shift
    (KL gap or factual token boost).
  MEMORY_STORE_ONLY: retrieval works but reinject ≈ random.
  MEMORY_FAIL: retrieval ~ chance.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
CACHE = HERE / "bases_cache_soft_distill.npz"
CKPT_PREFERRED = HERE / "matryoshka_student.pt"
CKPT_FALLBACK = HERE / "kl_distill_student_r256.pt"
STORE_DIR = HERE / "memory_store"
STORE_JSONL = STORE_DIR / "memories.jsonl"
STORE_COORDS = STORE_DIR / "coords_f16.npz"
RESULTS_PATH = HERE / "results_coord_memory.json"

MODEL_ID = "openai-community/gpt2"
RANK = 256
HUB16 = [0, 10, 4, 3, 5, 11, 15, 9, 1, 14, 2, 17, 16, 34, 18, 8]
DEFAULT_LAYERS = (5, 11)  # mid + last
CHANCE_RECALL1 = None  # filled from N at runtime


# --------------------------------------------------------------------------
# synthetic paired factoids (labeled retrieval by construction)
# --------------------------------------------------------------------------

def build_factoids(n: int = 80, seed: int = 0) -> List[dict]:
    """Hand-built / combinatorial synthetic facts with unique keys.

    Each record has a full memory sentence, a cloze query that should retrieve
    it, and a target answer string for reinject logprob measurement.
    """
    rng = np.random.default_rng(seed)
    places = [
        "Zentharia", "Kaldrune", "Marblen", "Orrhaven", "Sylphora",
        "Drakmoor", "Vellora", "Ashmere", "Nymbridge", "Corvance",
        "Lirathen", "Pellwick", "Thornvale", "Umbriel", "Yarrowfen",
        "Brindleport", "Cinderreach", "Duskholm", "Evermere", "Frostglen",
        "Glimmerbay", "Hollowmere", "Ironspire", "Jadewatch", "Kestrelrun",
        "Lumenfall", "Mistwood", "Northwick", "Oakenspire", "Pearlreach",
        "Quillmark", "Ravenholt", "Silverfen", "Tidewatch", "Underbrook",
        "Violetmere", "Windhollow", "Xanthera", "Yellowmere", "Zephyrgate",
    ]
    roles = [
        ("capital", "city", [
            "Kaldrune", "NeoPort", "Silverfen", "Ashmere", "Lumenfall",
            "Ironspire", "Pearlreach", "Tidewatch", "Frostglen", "Jadewatch",
        ]),
        ("founder", "person", [
            "Elandra Voss", "Mirren Holt", "Kael Draven", "Sera Quill",
            "Orrin Vale", "Tessa Brin", "Halden Rook", "Nyra Sol",
            "Corin Ash", "Vera Lyn",
        ]),
        ("annual festival", "event", [
            "Moonfire", "Starfall", "Harvest Veil", "Tidewake",
            "Ember Dance", "Frostsong", "Lanternwake", "Skybridge",
            "Rootwake", "Sunspire",
        ]),
        ("river", "river", [
            "the Silverglass", "the Blackthorn", "the Amberflow",
            "the Quietfen", "the Redwake", "the Palebrook",
            "the Stormglass", "the Softmere", "the Ironfen", "the Goldwake",
        ]),
        ("primary export", "goods", [
            "obsidian glass", "saffron silk", "cedar resin", "pearl salt",
            "moonsteel", "amber tea", "frostgrain", "river jade",
            "suncopper", "mistwool",
        ]),
        ("ruling house", "house", [
            "House Vellorn", "House Ashwick", "House Draven", "House Quill",
            "House Brindle", "House Solmere", "House Rook", "House Vale",
            "House Thorn", "House Lyn",
        ]),
        ("oldest temple", "temple", [
            "the Hall of Embers", "the Quiet Spire", "the Glass Sanctum",
            "the Root Chapel", "the Sky Reliquary", "the Tide Cloister",
            "the Frost Abbey", "the Lantern Keep", "the Amber Vault",
            "the Star Cloister",
        ]),
        ("motto", "motto", [
            "Stand in quiet light", "Forge before dawn", "Hold the fen",
            "Rise with the tide", "Keep the ember", "Walk the glass",
            "Guard the root", "Speak soft truth", "Carry the lantern",
            "Anchor the sky",
        ]),
    ]

    # unique (place, role) pairs
    combos = [(p, r) for p in places for r in roles]
    rng.shuffle(combos)
    combos = combos[:n]

    facts = []
    for i, (place, (role, kind, answers)) in enumerate(combos):
        answer = answers[i % len(answers)]
        if role == "capital":
            text = f"The capital of {place} is {answer}."
            query = f"The capital of {place} is"
        elif role == "founder":
            text = f"The founder of {place} was {answer}."
            query = f"The founder of {place} was"
        elif role == "annual festival":
            text = f"In {place}, the annual festival is called {answer}."
            query = f"In {place}, the annual festival is called"
        elif role == "river":
            text = f"The main river of {place} is {answer}."
            query = f"The main river of {place} is"
        elif role == "primary export":
            text = f"The primary export of {place} is {answer}."
            query = f"The primary export of {place} is"
        elif role == "ruling house":
            text = f"The ruling house of {place} is {answer}."
            query = f"The ruling house of {place} is"
        elif role == "oldest temple":
            text = f"The oldest temple in {place} is {answer}."
            query = f"The oldest temple in {place} is"
        else:  # motto
            text = f'The motto of {place} is "{answer}".'
            query = f'The motto of {place} is "'

        facts.append({
            "id": f"fact_{i:03d}",
            "text": text,
            "query": query,
            "answer": answer,
            "place": place,
            "role": role,
            "kind": kind,
            "label": f"fact_{i:03d}",
        })
    return facts


def build_wikitext_snippets(n: int = 40, seed: int = 1) -> List[dict]:
    """Optional unlabeled store filler from wikitext (short lines)."""
    try:
        from datasets import load_dataset
    except Exception as e:
        print(f"[wikitext] skip ({e})", flush=True)
        return []
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [t.strip() for t in ds["text"] if 40 <= len(t.strip()) <= 220]
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(texts), size=min(n, len(texts)), replace=False)
    out = []
    for j, idx in enumerate(pick):
        out.append({
            "id": f"wiki_{j:03d}",
            "text": texts[int(idx)],
            "query": None,
            "answer": None,
            "label": "wikitext",
            "kind": "wikitext",
        })
    return out


# --------------------------------------------------------------------------
# model / hooks
# --------------------------------------------------------------------------

class MemState:
    __slots__ = (
        "mode",          # passthrough | capture | inject
        "layers_capture",  # set of layer indices to capture
        "inject_layer",
        "inject_coords",   # (r,) or (1,1,r) — memory vector
        "blend",           # alpha in [0,1]: inj = a*mem + (1-a)*own
        "hub_mask",        # optional (r,) float mask
        "captured",        # dict layer -> (T, r) last-batch coords
        "pos",             # "last" | "all"
    )

    def __init__(self):
        self.mode = "passthrough"
        self.layers_capture = set()
        self.inject_layer = -1
        self.inject_coords = None
        self.blend = 1.0
        self.hub_mask = None
        self.captured = {}
        self.pos = "last"


def make_mem_hook(li: int, mean: torch.Tensor, P: torch.Tensor, state: MemState):
    def hook(_mod, _inp, out):
        h = out[0]
        c = (h - mean) @ P  # (B, T, r)
        h_bn = mean + c @ P.T

        if state.mode == "capture" and li in state.layers_capture:
            state.captured[li] = c.detach()

        if state.mode == "inject" and li == state.inject_layer:
            mem = state.inject_coords
            if mem is None:
                raise RuntimeError("inject with no coords")
            if mem.dim() == 1:
                mem = mem.view(1, 1, -1)
            elif mem.dim() == 2:
                mem = mem.view(1, 1, -1)
            mem = mem.to(dtype=c.dtype, device=c.device)
            a = float(state.blend)
            if state.pos == "all":
                own = c
                inj = a * mem.expand_as(own) + (1.0 - a) * own
            else:
                inj = c.clone()
                own_last = c[:, -1:, :]
                blended = a * mem.expand_as(own_last) + (1.0 - a) * own_last
                inj[:, -1:, :] = blended
            if state.hub_mask is not None:
                m = state.hub_mask.to(dtype=c.dtype, device=c.device)
                if state.pos == "all":
                    inj = inj * m + c * (1.0 - m)
                else:
                    inj[:, -1:, :] = (
                        inj[:, -1:, :] * m + c[:, -1:, :] * (1.0 - m)
                    )
            h_new = mean + inj @ P.T
            return (h_new,) + tuple(out[1:])

        return (h_bn,) + tuple(out[1:])
    return hook


def attach_mem(model, means_t, P, state: MemState):
    return [
        model.transformer.h[li].register_forward_hook(
            make_mem_hook(li, means_t[li], P, state))
        for li in range(model.config.n_layer)
    ]


def resolve_ckpt() -> Path:
    if CKPT_PREFERRED.exists():
        return CKPT_PREFERRED
    if CKPT_FALLBACK.exists():
        return CKPT_FALLBACK
    raise SystemExit(f"missing student ckpt ({CKPT_PREFERRED.name} or "
                     f"{CKPT_FALLBACK.name})")


def load_student(device):
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    z = np.load(CACHE)
    means_np, V_np = z["means"], z["V"]
    P_np = V_np[:, :RANK].copy()
    means_t = torch.from_numpy(means_np).to(device)
    P = torch.from_numpy(P_np).to(device)

    tok = GPT2TokenizerFast.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = GPT2LMHeadModel.from_pretrained(MODEL_ID).to(device)
    ckpt = resolve_ckpt()
    sd = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(sd)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    state = MemState()
    hooks = attach_mem(model, means_t, P, state)
    return model, tok, means_t, P, state, hooks, ckpt


# --------------------------------------------------------------------------
# encode / store / search
# --------------------------------------------------------------------------

@torch.no_grad()
def encode_text(model, tok, state: MemState, text: str, layers: Sequence[int],
                device) -> Dict[int, np.ndarray]:
    """Return last-token coords per layer: {layer: (r,) float32}."""
    ids = tok(text, return_tensors="pt", add_special_tokens=True)["input_ids"]
    ids = ids.to(device)
    state.mode = "capture"
    state.layers_capture = set(layers)
    state.captured = {}
    _ = model(ids)
    state.mode = "passthrough"
    out = {}
    for li in layers:
        c = state.captured[li]  # (1, T, r)
        out[li] = c[0, -1, :].float().cpu().numpy()
    return out


def ensure_store():
    STORE_DIR.mkdir(parents=True, exist_ok=True)


def clear_store():
    ensure_store()
    if STORE_JSONL.exists():
        STORE_JSONL.unlink()
    if STORE_COORDS.exists():
        STORE_COORDS.unlink()


def _coord_key(rec_id: str, layer: int) -> str:
    return f"{rec_id}_c_L{layer}"


def save_store(records: List[dict]):
    """Write thin JSONL metadata + compressed float16 coord NPZ."""
    ensure_store()
    arrays = {}
    meta_rows = []
    for rec in records:
        meta = {k: v for k, v in rec.items() if not str(k).startswith("c_L")}
        meta_rows.append(meta)
        for k, v in rec.items():
            if str(k).startswith("c_L"):
                arrays[_coord_key(rec["id"], int(str(k)[3:]))] = (
                    np.asarray(v, dtype=np.float32).astype(np.float16)
                )
    with STORE_JSONL.open("w", encoding="utf-8") as f:
        for m in meta_rows:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    if arrays:
        np.savez_compressed(STORE_COORDS, **arrays)


def load_store() -> List[dict]:
    if not STORE_JSONL.exists():
        return []
    rows = []
    with STORE_JSONL.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    coords = {}
    if STORE_COORDS.exists():
        z = np.load(STORE_COORDS)
        coords = {k: z[k].astype(np.float32) for k in z.files}
    # attach coords; also support legacy JSONL-embedded c_L*
    out = []
    for r in rows:
        rec = dict(r)
        layers = rec.get("layers") or []
        for li in layers:
            key = _coord_key(rec["id"], int(li))
            legacy = f"c_L{li}"
            if key in coords:
                rec[legacy] = coords[key].tolist()
            elif legacy in rec:
                pass  # already embedded
        out.append(rec)
    return out


def append_records(records: List[dict]):
    """Replace-or-extend: load existing, append, rewrite (demo-scale)."""
    existing = load_store()
    # drop coords-less duplicates by id
    by_id = {r["id"]: r for r in existing}
    for r in records:
        by_id[r["id"]] = r
    save_store(list(by_id.values()))


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def l2_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def search(store: List[dict], query_c: np.ndarray, layer: int,
           metric: str = "cosine", top_k: int = 5,
           hub_only: bool = False) -> List[Tuple[float, dict]]:
    """Nearest neighbors among store records at the given layer."""
    scored = []
    q = query_c.copy()
    if hub_only:
        mask = np.zeros_like(q)
        mask[HUB16] = 1.0
        q = q * mask
    for rec in store:
        key = f"c_L{layer}"
        if key not in rec:
            continue
        c = np.asarray(rec[key], dtype=np.float32)
        if hub_only:
            c = c * mask
        if metric == "cosine":
            score = cos_sim(q, c)  # higher better
        else:
            score = -l2_dist(q, c)  # higher better
        scored.append((score, rec))
    scored.sort(key=lambda x: -x[0])
    return scored[:top_k]


# --------------------------------------------------------------------------
# reinject evaluation
# --------------------------------------------------------------------------

@torch.no_grad()
def next_token_logprobs(model, tok, state: MemState, prompt: str, device,
                        inject_layer: Optional[int] = None,
                        inject_c: Optional[np.ndarray] = None,
                        blend: float = 1.0,
                        hub_only: bool = False,
                        pos: str = "last") -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (log_probs over vocab at last position, input_ids)."""
    ids = tok(prompt, return_tensors="pt")["input_ids"].to(device)
    if inject_layer is not None and inject_c is not None:
        state.mode = "inject"
        state.inject_layer = inject_layer
        state.inject_coords = torch.from_numpy(
            np.asarray(inject_c, dtype=np.float32)).to(device)
        state.blend = blend
        state.pos = pos
        if hub_only:
            m = torch.zeros(RANK, device=device)
            m[HUB16] = 1.0
            state.hub_mask = m
        else:
            state.hub_mask = None
    else:
        state.mode = "passthrough"
        state.hub_mask = None

    out = model(ids)
    state.mode = "passthrough"
    state.hub_mask = None
    logits = out.logits[0, -1, :]  # (V,)
    return F.log_softmax(logits.float(), dim=-1), ids


def answer_logprob(logp: torch.Tensor, tok, answer: str) -> Tuple[float, int]:
    """Mean logprob of answer token ids (continuation after prompt)."""
    ids = tok(answer, add_special_tokens=False)["input_ids"]
    if not ids:
        return 0.0, 0
    # only score first token of answer (next-token focus); also return n
    first = ids[0]
    return float(logp[first].item()), len(ids)


def kl_from_to(p_log: torch.Tensor, q_log: torch.Tensor) -> float:
    """KL(softmax(p) || softmax(q)) from log-probs."""
    p = p_log.exp()
    return float((p * (p_log - q_log)).sum().item())


def top1_agree(a_log: torch.Tensor, b_log: torch.Tensor) -> bool:
    return bool(int(a_log.argmax()) == int(b_log.argmax()))


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------

def cmd_ingest(args, model, tok, state, device):
    if args.reset or not STORE_JSONL.exists():
        clear_store()
    else:
        ensure_store()
    facts = build_factoids(n=args.n_facts, seed=args.seed)
    wiki = build_wikitext_snippets(n=args.n_wiki, seed=args.seed + 3) \
        if args.n_wiki > 0 else []
    layers = tuple(args.layers)
    t0 = time.time()
    records = []
    for item in facts + wiki:
        coords = encode_text(model, tok, state, item["text"], layers, device)
        rec = {
            "id": item["id"],
            "text": item["text"],
            "query": item.get("query"),
            "answer": item.get("answer"),
            "label": item.get("label"),
            "kind": item.get("kind"),
            "place": item.get("place"),
            "role": item.get("role"),
            "layers": list(layers),
        }
        for li in layers:
            rec[f"c_L{li}"] = coords[li].astype(np.float32)
        records.append(rec)
        if len(records) % 20 == 0:
            print(f"[ingest] {len(records)} ...", flush=True)
    if args.reset or not STORE_JSONL.exists():
        save_store(records)
    else:
        append_records(records)
    elapsed = time.time() - t0
    n_written = len(records)
    store_bytes = (
        (STORE_JSONL.stat().st_size if STORE_JSONL.exists() else 0)
        + (STORE_COORDS.stat().st_size if STORE_COORDS.exists() else 0)
    )
    print(f"[ingest] wrote {n_written} records to {STORE_DIR.name}/ "
          f"({elapsed:.1f}s, {store_bytes} bytes)", flush=True)
    return {
        "n_written": n_written,
        "n_facts": len(facts),
        "n_wiki": len(wiki),
        "layers": list(layers),
        "wall_s": elapsed,
        "store_path": str(STORE_DIR.relative_to(HERE)),
        "store_bytes": store_bytes,
    }


def cmd_query(args, model, tok, state, device):
    store = load_store()
    if not store:
        raise SystemExit("empty store — run ingest first")
    layers = tuple(args.layers)
    coords = encode_text(model, tok, state, args.text, layers, device)
    layer = args.layer
    hits = search(store, coords[layer], layer, metric=args.metric,
                  top_k=args.top_k, hub_only=args.hub_only)
    print(f"[query] layer=L{layer} metric={args.metric} hub_only={args.hub_only}")
    for rank, (score, rec) in enumerate(hits, 1):
        snip = rec["text"][:80].replace("\n", " ")
        print(f"  #{rank} score={score:.4f} id={rec['id']}  {snip}")
    return {"hits": [{"score": s, "id": r["id"], "text": r["text"]}
                     for s, r in hits]}


def eval_retrieval(store, model, tok, state, device, layers, metric, top_ks,
                   hub_only=False) -> dict:
    """Labeled recall@k on factoids that have query+label."""
    labeled = [r for r in store if r.get("query") and r.get("label")
               and str(r["label"]).startswith("fact_")]
    if not labeled:
        return {"error": "no labeled facts"}
    n = len(labeled)
    n_store = sum(1 for r in store if f"c_L{layers[0]}" in r)
    chance = {f"recall@{k}": min(k, n_store) / max(n_store, 1) for k in top_ks}
    per_layer = {}
    for layer in layers:
        hits_at = {k: 0 for k in top_ks}
        ranks = []
        key = f"c_L{layer}"
        searchable = [r for r in store if key in r]
        n_s = len(searchable)
        for rec in labeled:
            q_c = encode_text(model, tok, state, rec["query"], (layer,),
                              device)[layer]
            scored = search(store, q_c, layer, metric=metric,
                            top_k=max(max(top_ks), n_s), hub_only=hub_only)
            ids = [r["id"] for _, r in scored]
            if rec["id"] in ids:
                rank = ids.index(rec["id"]) + 1
            else:
                rank = n_s + 1
            ranks.append(rank)
            for k in top_ks:
                if rank <= k:
                    hits_at[k] += 1
        layer_chance = {
            f"recall@{k}": min(k, n_s) / max(n_s, 1) for k in top_ks
        }
        per_layer[f"L{layer}"] = {
            "n": n,
            "n_store": n_s,
            "recall": {f"@{k}": hits_at[k] / n for k in top_ks},
            "mean_rank": float(np.mean(ranks)),
            "median_rank": float(np.median(ranks)),
            "chance": layer_chance,
            "above_chance": {
                f"@{k}": (hits_at[k] / n) >= 3.0 * layer_chance[f"recall@{k}"]
                for k in top_ks
            },
        }
        print(f"[retrieval] L{layer} recall@1={hits_at[1]/n:.3f} "
              f"@5={hits_at.get(5,0)/n:.3f} mean_rank={np.mean(ranks):.1f} "
              f"(chance@1={layer_chance['recall@1']:.3f})", flush=True)
    return {"n_labeled": n, "n_store": n_store, "metric": metric,
            "hub_only": hub_only, "per_layer": per_layer, "chance": chance}


def eval_reinject(store, model, tok, state, device, inject_layers,
                  blend: float, hub_only: bool, pos: str, seed: int,
                  max_items: int, mode: str = "retrieved") -> dict:
    """Compare baseline / real / random-memory inject on answer logprob.

    mode:
      retrieved — inject top-1 NN memory for the query (end-to-end)
      oracle    — inject the true matching fact's stored coords (causal ceiling)
    """
    labeled = [r for r in store if r.get("query") and r.get("answer")
               and str(r.get("label", "")).startswith("fact_")]
    rng = np.random.default_rng(seed)
    if max_items and len(labeled) > max_items:
        idx = rng.choice(len(labeled), size=max_items, replace=False)
        labeled = [labeled[i] for i in idx]

    pool = {li: [] for li in inject_layers}
    for r in store:
        for li in inject_layers:
            key = f"c_L{li}"
            if key in r:
                pool[li].append((r["id"], np.asarray(r[key], dtype=np.float32)))

    results = {}
    for li in inject_layers:
        rows = []
        for rec in labeled:
            key = f"c_L{li}"
            if mode == "oracle":
                if key not in rec:
                    continue
                mem_c = np.asarray(rec[key], dtype=np.float32)
                top_rec = rec
                top_score = 1.0
                retrieved_correct = True
            else:
                q_c = encode_text(model, tok, state, rec["query"], (li,),
                                  device)[li]
                hits = search(store, q_c, li, metric="cosine", top_k=1)
                if not hits:
                    continue
                top_score, top_rec = hits[0]
                retrieved_correct = top_rec["id"] == rec["id"]
                mem_c = np.asarray(top_rec[key], dtype=np.float32)

            others = [p for p in pool[li] if p[0] != rec["id"]]
            if not others:
                continue
            rand_id, rand_c = others[int(rng.integers(0, len(others)))]

            base_lp, _ = next_token_logprobs(
                model, tok, state, rec["query"], device)
            real_lp, _ = next_token_logprobs(
                model, tok, state, rec["query"], device,
                inject_layer=li, inject_c=mem_c, blend=blend,
                hub_only=hub_only, pos=pos)
            rand_lp, _ = next_token_logprobs(
                model, tok, state, rec["query"], device,
                inject_layer=li, inject_c=rand_c, blend=blend,
                hub_only=hub_only, pos=pos)

            ans_base, _ = answer_logprob(base_lp, tok, rec["answer"])
            ans_real, _ = answer_logprob(real_lp, tok, rec["answer"])
            ans_rand, _ = answer_logprob(rand_lp, tok, rec["answer"])

            kl_real = kl_from_to(base_lp, real_lp)
            kl_rand = kl_from_to(base_lp, rand_lp)

            rows.append({
                "id": rec["id"],
                "retrieved_id": top_rec["id"],
                "retrieved_correct": retrieved_correct,
                "retrieve_score": float(top_score),
                "answer": rec["answer"],
                "ans_logp_base": ans_base,
                "ans_logp_real": ans_real,
                "ans_logp_rand": ans_rand,
                "delta_real": ans_real - ans_base,
                "delta_rand": ans_rand - ans_base,
                "gap_real_minus_rand": (ans_real - ans_base) - (ans_rand - ans_base),
                "kl_base_to_real": kl_real,
                "kl_base_to_rand": kl_rand,
                "kl_gap_real_minus_rand": kl_real - kl_rand,
                "top1_agree_real": top1_agree(base_lp, real_lp),
                "top1_agree_rand": top1_agree(base_lp, rand_lp),
                "random_id": rand_id,
            })

        if not rows:
            results[f"L{li}"] = {"error": "no rows"}
            continue

        def mean(key):
            return float(np.mean([r[key] for r in rows]))

        win_frac = float(np.mean([
            1.0 if r["gap_real_minus_rand"] > 0 else 0.0 for r in rows]))
        correct_rows = [r for r in rows if r["retrieved_correct"]]
        win_frac_correct = (
            float(np.mean([1.0 if r["gap_real_minus_rand"] > 0 else 0.0
                           for r in correct_rows]))
            if correct_rows else float("nan")
        )
        mean_gap = mean("gap_real_minus_rand")
        mean_kl_gap = mean("kl_gap_real_minus_rand")
        # Factual token boost is the primary behavioral metric. KL magnitude
        # alone is ambiguous (random often moves *more* but undirectedly).
        beats_random = (
            mean_gap > 0.05 and win_frac >= 0.55
        ) or (
            mean_gap > 0.15 and win_frac >= 0.52
        ) or (
            mode == "oracle" and mean_gap > 0.05 and win_frac >= 0.55
        )

        summary = {
            "mode": mode,
            "n": len(rows),
            "n_retrieved_correct": len(correct_rows),
            "retrieve_acc@1": len(correct_rows) / len(rows),
            "mean_delta_real": mean("delta_real"),
            "mean_delta_rand": mean("delta_rand"),
            "mean_gap_real_minus_rand": mean_gap,
            "win_frac_real_gt_rand": win_frac,
            "win_frac_among_correct_retrieve": win_frac_correct,
            "mean_kl_base_to_real": mean("kl_base_to_real"),
            "mean_kl_base_to_rand": mean("kl_base_to_rand"),
            "mean_kl_gap_real_minus_rand": mean_kl_gap,
            "top1_agree_real_rate": mean("top1_agree_real"),
            "top1_agree_rand_rate": mean("top1_agree_rand"),
            "beats_random_control": beats_random,
            "blend": blend,
            "hub_only": hub_only,
            "pos": pos,
            "examples": rows[:8],
        }
        results[f"L{li}"] = summary
        print(
            f"[reinject/{mode}] L{li} n={len(rows)} retrieve@1="
            f"{summary['retrieve_acc@1']:.3f} "
            f"Δans real={summary['mean_delta_real']:+.3f} "
            f"rand={summary['mean_delta_rand']:+.3f} "
            f"gap={mean_gap:+.3f} win={win_frac:.2f} "
            f"beats_rand={beats_random}",
            flush=True,
        )
    return results


def decide_verdict(retrieval: dict, reinject: dict) -> Tuple[str, str]:
    """Apply pre-registered fork.

    Retrieval is 'clearly above chance' if on any layer recall@1 ≥ 3×chance
    or recall@5 ≥ 3×chance@5 (protocol: labeled synthetic facts).
    """
    ret_ok = False
    best_r1 = 0.0
    best_r5 = 0.0
    chance1 = 0.0
    chance5 = 0.0
    for _layer_key, block in retrieval.get("per_layer", {}).items():
        r1 = block["recall"]["@1"]
        r5 = block["recall"]["@5"]
        c1 = block["chance"]["recall@1"]
        c5 = block["chance"]["recall@5"]
        best_r1 = max(best_r1, r1)
        best_r5 = max(best_r5, r5)
        chance1, chance5 = c1, c5
        if r1 >= 3.0 * c1 or r5 >= 3.0 * c5:
            ret_ok = True

    reinject_ok = False
    reinject_notes = []
    for layer_key, block in reinject.items():
        if not isinstance(block, dict):
            continue
        if block.get("beats_random_control"):
            reinject_ok = True
            reinject_notes.append(layer_key)
        # oracle path: injecting the true matching memory
        if block.get("mode") == "oracle" and block.get("beats_random_control"):
            reinject_ok = True
            reinject_notes.append(layer_key)

    if not ret_ok:
        return "MEMORY_FAIL", (
            f"retrieval ~chance (best recall@1={best_r1:.3f}/chance={chance1:.3f}, "
            f"@5={best_r5:.3f}/chance={chance5:.3f})"
        )
    if ret_ok and not reinject_ok:
        return "MEMORY_STORE_ONLY", (
            "retrieval clearly above chance, but reinject of real memory "
            "did not beat random-memory control on answer-logprob / KL gap"
        )
    return "MEMORY_VIABLE", (
        "retrieval above chance AND real memory reinject beats "
        f"random-memory control ({', '.join(reinject_notes) or 'ok'})"
    )


def cmd_run_all(args):
    if args.device == "auto":
        device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[device] {device}", flush=True)
    wall0 = time.time()

    model, tok, means_t, P, state, hooks, ckpt = load_student(device)
    print(f"[student] {ckpt.name}", flush=True)

    # ---- ingest ----
    args.reset = True
    ingest_meta = cmd_ingest(args, model, tok, state, device)
    store = load_store()

    # ---- retrieval ----
    print("[retrieval] evaluating labeled recall ...", flush=True)
    retrieval = eval_retrieval(
        store, model, tok, state, device,
        layers=tuple(args.layers), metric="cosine",
        top_ks=(1, 3, 5), hub_only=False,
    )
    retrieval_hub = eval_retrieval(
        store, model, tok, state, device,
        layers=tuple(args.layers), metric="cosine",
        top_ks=(1, 3, 5), hub_only=True,
    )

    # ---- reinject battery ----
    print("[reinject] evaluating real vs random memory ...", flush=True)
    reinject = {}
    configs = [
        {"name": "retrieved_full_replace_last", "blend": 1.0, "hub_only": False,
         "pos": "last", "mode": "retrieved"},
        {"name": "retrieved_blend05_last", "blend": 0.5, "hub_only": False,
         "pos": "last", "mode": "retrieved"},
        {"name": "retrieved_hub16_replace_last", "blend": 1.0, "hub_only": True,
         "pos": "last", "mode": "retrieved"},
        {"name": "oracle_full_replace_last", "blend": 1.0, "hub_only": False,
         "pos": "last", "mode": "oracle"},
        {"name": "oracle_hub16_replace_last", "blend": 1.0, "hub_only": True,
         "pos": "last", "mode": "oracle"},
    ]
    for cfg in configs:
        print(f"[reinject] config={cfg['name']}", flush=True)
        reinject[cfg["name"]] = eval_reinject(
            store, model, tok, state, device,
            inject_layers=tuple(args.layers),
            blend=cfg["blend"], hub_only=cfg["hub_only"], pos=cfg["pos"],
            seed=args.seed + 11, max_items=args.max_eval, mode=cfg["mode"],
        )

    # pick best reinject config for verdict (any beats_random)
    # flatten for decide: use the config with most beats_random True
    flat_for_verdict = {}
    for cfg_name, by_layer in reinject.items():
        for lk, block in by_layer.items():
            flat_for_verdict[f"{cfg_name}/{lk}"] = block

    verdict, reason = decide_verdict(retrieval, flat_for_verdict)

    wall = time.time() - wall0
    store_bytes = (
        (STORE_JSONL.stat().st_size if STORE_JSONL.exists() else 0)
        + (STORE_COORDS.stat().st_size if STORE_COORDS.exists() else 0)
    )

    results = {
        "meta": {
            "step": 2,
            "student_ckpt": ckpt.name,
            "basis": CACHE.name,
            "rank": RANK,
            "hub16": HUB16,
            "layers": list(args.layers),
            "device": str(device),
            "n_facts": ingest_meta["n_facts"],
            "n_wiki": ingest_meta["n_wiki"],
            "store_path": ingest_meta["store_path"],
            "store_bytes": store_bytes,
            "wall_time_s": wall,
            "training_free": True,
            "fork": {
                "MEMORY_VIABLE": (
                    "retrieval clearly above chance AND reinject of real "
                    "memory beats random-memory control on at least one "
                    "measurable behavioral shift (KL gap or factual token "
                    "boost)"
                ),
                "MEMORY_STORE_ONLY": (
                    "retrieval works but reinject is no better than random "
                    "(coords searchable but not causally useful as memory)"
                ),
                "MEMORY_FAIL": "retrieval ~chance",
            },
        },
        "ingest": ingest_meta,
        "retrieval": retrieval,
        "retrieval_hub16": retrieval_hub,
        "reinject": reinject,
        "verdict": verdict,
        "verdict_reason": reason,
    }

    out_path = Path(args.out)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[verdict] {verdict}: {reason}", flush=True)
    print(f"[wall] {wall:.1f}s  wrote {out_path.name}", flush=True)

    for h in hooks:
        h.remove()
    return results


def main():
    ap = argparse.ArgumentParser(description="Step 2 coordinate memory sidecar")
    ap.add_argument("command", choices=["ingest", "query", "reinject-eval",
                                        "run-all"])
    ap.add_argument("--n-facts", type=int, default=80)
    ap.add_argument("--n-wiki", type=int, default=20)
    ap.add_argument("--layers", type=int, nargs="+", default=list(DEFAULT_LAYERS))
    ap.add_argument("--layer", type=int, default=5,
                    help="query search layer")
    ap.add_argument("--text", type=str, default="")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--metric", type=str, default="cosine",
                    choices=["cosine", "l2"])
    ap.add_argument("--hub-only", action="store_true")
    ap.add_argument("--blend", type=float, default=1.0)
    ap.add_argument("--pos", type=str, default="last",
                    choices=["last", "all"])
    ap.add_argument("--max-eval", type=int, default=80)
    ap.add_argument("--reset", action="store_true",
                    help="clear store before ingest")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--out", type=str, default=str(RESULTS_PATH))
    args = ap.parse_args()

    if args.command == "run-all":
        cmd_run_all(args)
        return

    if args.device == "auto":
        device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model, tok, means_t, P, state, hooks, ckpt = load_student(device)
    print(f"[student] {ckpt.name} on {device}", flush=True)

    if args.command == "ingest":
        args.reset = True if args.reset or not STORE_JSONL.exists() else args.reset
        cmd_ingest(args, model, tok, state, device)
    elif args.command == "query":
        if not args.text:
            raise SystemExit("--text required for query")
        cmd_query(args, model, tok, state, device)
    elif args.command == "reinject-eval":
        store = load_store()
        if not store:
            raise SystemExit("empty store — run ingest first")
        wall0 = time.time()
        retrieval = eval_retrieval(
            store, model, tok, state, device,
            layers=tuple(args.layers), metric="cosine",
            top_ks=(1, 3, 5), hub_only=False,
        )
        reinject = {
            "full_replace_last": eval_reinject(
                store, model, tok, state, device,
                inject_layers=tuple(args.layers),
                blend=args.blend, hub_only=args.hub_only, pos=args.pos,
                seed=args.seed + 11, max_items=args.max_eval,
            )
        }
        flat = {}
        for cfg, by_l in reinject.items():
            for lk, b in by_l.items():
                flat[f"{cfg}/{lk}"] = b
        verdict, reason = decide_verdict(retrieval, flat)
        out = {
            "meta": {
                "step": 2,
                "student_ckpt": ckpt.name,
                "wall_time_s": time.time() - wall0,
                "device": str(device),
            },
            "retrieval": retrieval,
            "reinject": reinject,
            "verdict": verdict,
            "verdict_reason": reason,
        }
        Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"[verdict] {verdict}: {reason}", flush=True)

    for h in hooks:
        h.remove()


if __name__ == "__main__":
    main()
