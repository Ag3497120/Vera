#!/usr/bin/env python3
"""Step 3 — Matryoshka KL distillation over nested bottleneck ranks.

Warm-starts from the r=256 KL-distilled student (kl_distill_student_r256.pt)
with the frozen shared PCA basis (bases_cache_soft_distill.npz). During each
training step, sample a nested rank r from a schedule and project ALL 12
block outputs through P[:, :r] only. After training, evaluate the SAME
checkpoint at every fixed rank with no further training.

Design insight from II-5 / KL_DISTILL_R256.md: training wide then narrowing
beats dedicated narrow-rank training. This run jointly supervises all
granularities so one student is usable at r ∈ {8,16,32,64,128,192,256}.

  student : GPT-2 small from kl_distill_student_r256.pt + hard shared
            bottleneck hooks (mutable nested rank).
  teacher : original GPT-2 small, frozen, no bottleneck.
  loss    : forward KL at T=1 + 0.1 * LM (same as kl_distill_r256.py).

Protocol:
  1. Step-0 TEST: evaluate the warm-start (r256-only) student at EVERY
     nested rank — the "narrow after r256-only training" control.
  2. Distill with per-step random rank. Val curve tracked at r=256.
  3. Final TEST: same Matryoshka checkpoint at every fixed rank.
  4. Optionally re-eval dedicated r=128 student at r=128 for the table.

Pre-registered fork:
  MATRYOSHKA_VIABLE: at r=256, ppl <= ~1.6x baseline (~87) OR not worse
    than prior r256 student by >10%; AND at r=128, beats dedicated r=128
    (ppl 122.8) by a clear margin (e.g. <110) AND/OR beats the r256-only
    nested narrow eval; AND degradation across ranks is monotone/graceful.
  MATRYOSHKA_WEAK: multi-rank training helps little vs r256-only nested;
    small ranks still collapse.
  Between: state honestly.

Outputs: results_matryoshka_distill.json, matryoshka_student.pt,
         matryoshka_distill_run.log (via shell redirect).
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
WARM_CKPT = HERE / "kl_distill_student_r256.pt"
R128_CKPT = HERE / "kl_distill_student_r128.pt"

LM_WEIGHT = 0.1
TEMPERATURE = 1.0
DEFAULT_RANKS = (8, 16, 32, 64, 128, 192, 256)
BASELINE_PPL = 54.466
R128_DEDICATED_REF = {
    "kl_mean": 1.1151,
    "top1_agreement": 0.4703,
    "ppl": 122.83,
    "top1_accuracy": 0.231,
}


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
# mutable-rank bottleneck
# --------------------------------------------------------------------------

class RankState:
    __slots__ = ("rank",)

    def __init__(self, rank: int):
        self.rank = rank


def make_hook(mean: torch.Tensor, P_full: torch.Tensor, rank_state: RankState):
    def hook(_mod, _inp, out):
        r = rank_state.rank
        P = P_full[:, :r]
        h = out[0]
        hc = h - mean
        h_new = mean + (hc @ P) @ P.T
        return (h_new,) + tuple(out[1:])
    return hook


def attach_bottleneck(model, means_t: torch.Tensor, P_full: torch.Tensor,
                      rank_state: RankState):
    return [model.transformer.h[li].register_forward_hook(
        make_hook(means_t[li], P_full, rank_state))
            for li in range(model.config.n_layer)]


def rank_schedule_probs(ranks, schedule: str):
    r = np.asarray(ranks, dtype=np.float64)
    if schedule == "uniform":
        w = np.ones_like(r)
    elif schedule == "linear":
        w = r  # geometric preference toward larger ranks
    elif schedule == "sqrt":
        w = np.sqrt(r)
    else:
        raise ValueError(f"unknown schedule {schedule}")
    w = w / w.sum()
    return w


# --------------------------------------------------------------------------
# metrics
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
        self.agree_sum += float(
            (base_logits.argmax(-1) == patched_logits.argmax(-1)).sum())
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


def eval_vs_teacher(student, teacher, seqs, device, batch_size):
    acc = Accum()
    student.eval()
    for b0 in range(0, seqs.shape[0], batch_size):
        batch = torch.from_numpy(seqs[b0:b0 + batch_size]).to(device)
        with torch.no_grad():
            t_logits = teacher(batch).logits.float()
            s_logits = student(batch).logits.float()
        acc.update(t_logits, s_logits, batch)
    return acc.result()


def eval_at_ranks(student, teacher, seqs, device, batch_size, rank_state,
                  ranks):
    out = {}
    for r in ranks:
        rank_state.rank = int(r)
        res = eval_vs_teacher(student, teacher, seqs, device, batch_size)
        res["ppl_x"] = res["ppl"] / BASELINE_PPL
        out[f"r{r}"] = res
        print(f"  [eval r={r}] KL={res['kl_mean']:.4f} "
              f"agr={res['top1_agreement']:.4f} "
              f"ppl={res['ppl']:.2f} ({res['ppl_x']:.2f}x) "
              f"acc={res['top1_accuracy']:.4f}", flush=True)
    return out


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranks", type=str, default=",".join(map(str, DEFAULT_RANKS)))
    ap.add_argument("--schedule", type=str, default="linear",
                    choices=("uniform", "linear", "sqrt"),
                    help="rank sampling: uniform, or weight ∝ r / sqrt(r)")
    ap.add_argument("--max-rank", type=int, default=256)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--eval-seqs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--max-steps", type=int, default=2500)
    ap.add_argument("--min-steps", type=int, default=600)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--plateau-window", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--train-batch", type=int, default=8)
    ap.add_argument("--val-seqs", type=int, default=16)
    ap.add_argument("--eval-r128-dedicated", action="store_true", default=True)
    ap.add_argument("--no-eval-r128-dedicated", action="store_false",
                    dest="eval_r128_dedicated")
    ap.add_argument("--out", type=str,
                    default=str(HERE / "results_matryoshka_distill.json"))
    ap.add_argument("--ckpt-out", type=str,
                    default=str(HERE / "matryoshka_student.pt"))
    ap.add_argument("--warm-ckpt", type=str, default=str(WARM_CKPT),
                    help="student warm-start checkpoint")
    ap.add_argument("--disable-plateau", action="store_true",
                    help="never early-stop on val@max-rank plateau")
    ap.add_argument("--skip-step0-frontier", action="store_true",
                    help="skip step-0 nested eval (use when resuming)")
    ap.add_argument("--step0-json", type=str, default="",
                    help="reuse step0 frontier from prior results json")
    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[device] {device}", flush=True)
    torch.manual_seed(args.seed)

    ranks = [int(x) for x in args.ranks.split(",") if x.strip()]
    assert max(ranks) <= args.max_rank
    assert args.max_rank in ranks, "max-rank must be in the schedule"
    probs = rank_schedule_probs(ranks, args.schedule)
    print(f"[schedule] {args.schedule}: ranks={ranks}")
    print(f"[schedule] probs="
          + ", ".join(f"r{r}={p:.3f}" for r, p in zip(ranks, probs)),
          flush=True)

    if not CACHE.exists():
        raise SystemExit(f"missing {CACHE} — refusing to refit (protocol)")
    warm_ckpt = Path(args.warm_ckpt)
    if not warm_ckpt.exists():
        raise SystemExit(f"missing warm-start {warm_ckpt}")

    z = np.load(CACHE)
    means_np, V_np = z["means"], z["V"]
    assert V_np.shape[1] >= args.max_rank
    print(f"[basis] loaded {CACHE.name} (V {V_np.shape}; "
          f"nested top-{args.max_rank})", flush=True)

    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    tok = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
    eval_texts, eval_name = build_corpus_texts("test")
    eval_seqs = tokenize_corpus(
        eval_texts, tok, args.seq_len,
        args.eval_seqs * args.seq_len, args.seed + 7)
    print(f"[eval corpus] {eval_name}: "
          f"{eval_seqs.shape[0]} x {eval_seqs.shape[1]}", flush=True)

    student = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").to(device)
    sd = torch.load(warm_ckpt, map_location=device, weights_only=True)
    missing, unexpected = student.load_state_dict(sd, strict=False)
    print(f"[warm start] loaded {warm_ckpt.name} "
          f"(missing={list(missing)}, unexpected={list(unexpected)})",
          flush=True)

    teacher = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    means_t = torch.from_numpy(means_np).to(device)
    P_full = torch.from_numpy(V_np[:, :args.max_rank].copy()).to(device)
    rank_state = RankState(args.max_rank)
    hooks = attach_bottleneck(student, means_t, P_full, rank_state)

    # ---- step-0: r256-only nested frontier (control, before training) ----
    step0_frontier = None
    if args.step0_json:
        prev = json.loads(Path(args.step0_json).read_text())
        step0_frontier = prev["step0_r256_only_nested_frontier"]
        print(f"[step-0] reused frontier from {args.step0_json}", flush=True)
        for k, res in step0_frontier.items():
            print(f"  [reuse {k}] KL={res['kl_mean']:.4f} "
                  f"agr={res['top1_agreement']:.4f} "
                  f"ppl={res['ppl']:.2f}", flush=True)
    elif args.skip_step0_frontier:
        print("[step-0] skipped", flush=True)
        step0_frontier = {f"r{r}": {"kl_mean": None, "top1_agreement": None,
                                    "ppl": None, "ppl_x": None,
                                    "top1_accuracy": None, "nll": None}
                          for r in ranks}
    else:
        print("[step-0] r256-only student nested-rank TEST frontier ...",
              flush=True)
        step0_frontier = eval_at_ranks(
            student, teacher, eval_seqs, device, args.batch_size,
            rank_state, ranks)

    # ---- training data ----
    train_texts, _ = build_corpus_texts("train")
    need_tokens = args.max_steps * args.train_batch * args.seq_len
    train_seqs = tokenize_corpus(
        train_texts, tok, args.seq_len, need_tokens, args.seed + 31)
    print(f"[data] train seqs available: {train_seqs.shape[0]} "
          f"(need {need_tokens // args.seq_len}; will cycle if short)",
          flush=True)
    val_texts, _ = build_corpus_texts("validation")
    val_seqs = tokenize_corpus(
        val_texts, tok, args.seq_len,
        args.val_seqs * args.seq_len, args.seed + 101)
    print(f"[data] val slice: {val_seqs.shape[0]} x {args.seq_len}",
          flush=True)

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.01)

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        return args.lr

    rng = np.random.default_rng(args.seed + 71)
    rank_rng = np.random.default_rng(args.seed + 91)
    order = rng.permutation(train_seqs.shape[0])
    cursor = 0
    rank_counts = {int(r): 0 for r in ranks}

    def next_batch():
        nonlocal cursor, order
        if cursor + args.train_batch > order.size:
            order = rng.permutation(train_seqs.shape[0])
            cursor = 0
        idx = order[cursor:cursor + args.train_batch]
        cursor += args.train_batch
        return torch.from_numpy(train_seqs[idx]).to(device)

    history = []
    rank_state.rank = args.max_rank
    r0 = eval_vs_teacher(student, teacher, val_seqs, device, args.batch_size)
    history.append({"step": 0, "train_rank_mode": "fixed_max", **r0})
    print(f"[matryoshka] step 0 (warm, val@r{args.max_rank}): "
          f"KL={r0['kl_mean']:.4f} agr={r0['top1_agreement']:.4f} "
          f"ppl={r0['ppl']:.1f}", flush=True)

    t0 = time.time()
    step_done = 0
    loss_win, kl_win, rank_win = [], [], []
    V = student.config.vocab_size
    for step in range(args.max_steps):
        student.train()
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        r_sample = int(rank_rng.choice(ranks, p=probs))
        rank_state.rank = r_sample
        rank_counts[r_sample] += 1
        batch = next_batch()
        with torch.no_grad():
            t_logits = teacher(batch).logits
            t_probs = F.softmax(t_logits / TEMPERATURE, dim=-1)
            t_ent = float(
                -(t_probs * torch.log(t_probs.clamp_min(1e-12)))
                .sum(-1).mean())
        out = student(batch, labels=batch, use_cache=False)
        soft_ce = F.cross_entropy(out.logits.view(-1, V), t_probs.view(-1, V))
        loss = soft_ce + LM_WEIGHT * out.loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        loss_win.append(float(loss.detach()))
        kl_win.append(float(soft_ce.detach()) - t_ent)
        rank_win.append(r_sample)
        step_done = step + 1
        if step_done % 20 == 0:
            print(f"  [matryoshka] step {step_done} "
                  f"loss={np.mean(loss_win):.4f} "
                  f"trainKL={np.mean(kl_win):.4f} "
                  f"ranks={rank_win} "
                  f"({time.time() - t0:.0f}s)", flush=True)
            loss_win, kl_win, rank_win = [], [], []
        if step_done % args.eval_every == 0:
            rank_state.rank = args.max_rank
            res = eval_vs_teacher(
                student, teacher, val_seqs, device, args.batch_size)
            history.append({"step": step_done, "val_rank": args.max_rank,
                            **res})
            print(f"[matryoshka] step {step_done}: "
                  f"val@r{args.max_rank} KL={res['kl_mean']:.4f} "
                  f"agr={res['top1_agreement']:.4f} ppl={res['ppl']:.2f} "
                  f"acc={res['top1_accuracy']:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
            w = args.plateau_window
            if step_done >= args.min_steps and len(history) > w:
                best_kl_now = min(h["kl_mean"] for h in history)
                best_agr_now = max(h["top1_agreement"] for h in history)
                past = history[:-w]
                best_kl_past = min(h["kl_mean"] for h in past)
                best_agr_past = max(h["top1_agreement"] for h in past)
                kl_impr = (best_kl_past - best_kl_now) / max(best_kl_past, 1e-9)
                agr_impr = (
                    (best_agr_now - best_agr_past) / max(best_agr_past, 1e-9))
                if (not args.disable_plateau) and kl_impr < 0.01 and agr_impr < 0.01:
                    print(f"[matryoshka] early stop at step {step_done} "
                          f"(plateau: KL impr {kl_impr:.4f}, "
                          f"agr impr {agr_impr:.4f})", flush=True)
                    break
    wall = time.time() - t0

    def steps95(key, better_is_less):
        v0 = history[0][key]
        vbest = (min if better_is_less else max)(h[key] for h in history)
        thresh = vbest + (0.05 if better_is_less else -0.05) * abs(v0 - vbest)
        if better_is_less:
            return next(h["step"] for h in history if h[key] <= thresh)
        return next(h["step"] for h in history if h[key] >= thresh)

    print("[final] Matryoshka student nested-rank TEST frontier ...",
          flush=True)
    final_frontier = eval_at_ranks(
        student, teacher, eval_seqs, device, args.batch_size,
        rank_state, ranks)

    ckpt_path = Path(args.ckpt_out)
    torch.save(student.state_dict(), ckpt_path)
    print(f"[ckpt] saved {ckpt_path.name}", flush=True)

    # dedicated r=128 control (same eval tokens)
    r128_dedicated = dict(R128_DEDICATED_REF)
    r128_dedicated["source"] = "KL_DISTILL.md reference"
    if args.eval_r128_dedicated and R128_CKPT.exists():
        print("[control] re-eval dedicated r=128 student at r=128 ...",
              flush=True)
        for h in hooks:
            h.remove()
        s128 = GPT2LMHeadModel.from_pretrained(
            "openai-community/gpt2").to(device)
        sd128 = torch.load(R128_CKPT, map_location=device, weights_only=True)
        s128.load_state_dict(sd128, strict=False)
        rs128 = RankState(128)
        P128 = torch.from_numpy(V_np[:, :128].copy()).to(device)
        h128 = attach_bottleneck(s128, means_t, P128, rs128)
        res128 = eval_vs_teacher(
            s128, teacher, eval_seqs, device, args.batch_size)
        res128["ppl_x"] = res128["ppl"] / BASELINE_PPL
        for h in h128:
            h.remove()
        r128_dedicated = {**res128, "source": R128_CKPT.name}
        print(f"  [r128 dedicated] KL={res128['kl_mean']:.4f} "
              f"agr={res128['top1_agreement']:.4f} "
              f"ppl={res128['ppl']:.2f}", flush=True)
        hooks = attach_bottleneck(student, means_t, P_full, rank_state)

    f256 = final_frontier[f"r{args.max_rank}"]
    f128 = final_frontier["r128"]
    s0_128 = step0_frontier["r128"]
    s0_192 = step0_frontier.get("r192")
    f192 = final_frontier.get("r192")
    prior_r256_ppl = step0_frontier[f"r{args.max_rank}"]["ppl"]

    results = {
        "meta": {
            "model": "openai-community/gpt2",
            "teacher": "original GPT-2 small, frozen, no bottleneck",
            "basis": CACHE.name,
            "basis_note": "nested columns of shared eigenbasis V; "
                          "same family as kl_distill_r256",
            "warm_start": warm_ckpt.name,
            "eval_corpus": eval_name,
            "eval_tokens": int(eval_seqs.size),
            "seq_len": args.seq_len,
            "device": str(device),
            "loss": f"forward KL (CE vs teacher softmax, T={TEMPERATURE}) "
                    f"+ {LM_WEIGHT} * LM loss",
            "baseline_ppl_ref": BASELINE_PPL,
            "baseline_acc_ref": 0.3102,
            "ranks": ranks,
            "schedule": args.schedule,
            "schedule_probs": {
                f"r{r}": float(p) for r, p in zip(ranks, probs)
            },
            "fork": {
                "MATRYOSHKA_VIABLE": "r256 ppl<=~87 (1.6x) OR not worse than "
                    "prior r256 by >10%; AND r128 beats dedicated 122.8 "
                    "(e.g. <110) AND/OR beats r256-only nested; graceful "
                    "monotone degradation across ranks",
                "MATRYOSHKA_WEAK": "multi-rank helps little vs r256-only "
                    "nested; small ranks still collapse",
            },
        },
        "step0_r256_only_nested_frontier": step0_frontier,
        "train": {
            "steps": step_done,
            "wall_time_s": wall,
            "lr": args.lr,
            "warmup": args.warmup,
            "batch": [args.train_batch, args.seq_len],
            "eval_every": args.eval_every,
            "max_steps": args.max_steps,
            "plateau_window_steps": args.plateau_window * args.eval_every,
            "rank_sample_counts": rank_counts,
            "history_val_at_max_rank": history,
            "steps_to_95pct_kl": steps95("kl_mean", True),
            "steps_to_95pct_agreement": steps95("top1_agreement", False),
            "steps_to_95pct_ppl": steps95("ppl", True),
        },
        "final_matryoshka_frontier": final_frontier,
        "controls": {
            "r128_dedicated": r128_dedicated,
            "prior_r256_at_max": step0_frontier[f"r{args.max_rank}"],
            "prior_r256_nested_r192_from_KL_DISTILL_R256": {
                "kl_mean": 0.9573,
                "top1_agreement": 0.4899,
                "ppl": 106.70,
                "note": "from results_kl_distill_r256.json nested eval",
            },
        },
        "comparisons": {
            "r256_matryoshka_vs_prior": {
                "matryoshka_ppl": f256["ppl"],
                "prior_ppl": prior_r256_ppl,
                "rel_delta": (f256["ppl"] - prior_r256_ppl) / prior_r256_ppl,
            },
            "r128_matryoshka_vs_dedicated": {
                "matryoshka_ppl": f128["ppl"],
                "dedicated_ppl": r128_dedicated["ppl"],
                "delta": f128["ppl"] - r128_dedicated["ppl"],
            },
            "r128_matryoshka_vs_r256only_nested": {
                "matryoshka_ppl": f128["ppl"],
                "r256only_nested_ppl": s0_128["ppl"],
                "delta": f128["ppl"] - s0_128["ppl"],
            },
            "r192_matryoshka_vs_r256only_nested": None if f192 is None else {
                "matryoshka_ppl": f192["ppl"],
                "r256only_nested_ppl": s0_192["ppl"],
                "delta": f192["ppl"] - s0_192["ppl"],
            },
        },
        "checkpoint": ckpt_path.name,
    }
    out_path = Path(args.out)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[wrote] {out_path}", flush=True)
    print(f"[wall] {wall:.0f}s ({wall / 60:.1f} min) for {step_done} steps",
          flush=True)


if __name__ == "__main__":
    main()
