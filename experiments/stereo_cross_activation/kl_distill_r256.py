#!/usr/bin/env python3
"""KL distillation at r=256 with warm start from the r=128 student.

Follow-up to kl_distill_bottleneck.py / KL_DISTILL.md. There, KL-distilling
original GPT-2 into the frozen hard shared-PCA bottleneck (r=128, all 12
block outputs) for 2500 steps reached test ppl 122.8 (2.26x baseline),
agreement 0.470, KL 1.12 — both targets missed (ppl <= 82, agreement >= 0.6).

This run widens the container to r=256. Key structural fact: the PCA bases
are nested (columns of the same eigenbasis V, sorted by eigenvalue), so the
r=256 bottleneck subspace strictly contains the r=128 one. The r=128-trained
student therefore ALREADY satisfies the r=256 constraint exactly — its
activations pass through the wider projector with extra headroom — making it
an ideal warm start.

  student : GPT-2 small, weights loaded from kl_distill_student_r128.pt,
            bottleneck swapped to frozen hard r=256 shared projection
            (same basis family, bases_cache_soft_distill.npz, never refit).
            h -> mean_l + P P^T (h - mean_l), P = V[:, :256].
  teacher : original GPT-2 small, no bottleneck, frozen, eval, no_grad.
  loss    : forward KL at T=1 (CE vs teacher softmax, full vocab) +
            0.1 * plain LM loss.

Protocol order:
  1. STEP-0 TEST EVAL FIRST (before any training) on the standard wikitext-2
     test tokens (40 x 256 = 10,240): the pure effect of widening the
     container around the adapted student.
  2. Distill: AdamW lr 3e-5 (warm start), warmup 50 then constant, batch
     8 x 256, grad clip 1.0, budget <= 3000 steps, eval every 100 on the
     16 x 256 validation slice. Plateau stop (after min-steps): best val KL
     AND best val agreement both improved < 1% (relative) over the last
     400 steps (4 evals).
  3. Final eval on the same test tokens; save kl_distill_student_r256.pt.
  4. Optional --extra-ranks eval (e.g. 192): evaluate the FINAL r=256
     student under a narrower nested bottleneck, no retraining, to map the
     rank-performance frontier.

Pre-registered fork (r=256):
  IDENTITY_AT_256:       agreement >= 0.6 AND/OR ppl <= 1.5x baseline (~82)
                         — container identity-viable at r=256 (1/3 of 768).
  STILL_CAPABILITY_ONLY: agreement stalls < ~0.55 and ppl > ~100 even with
                         warm start — rank is not the binding constraint.
  Between: state honestly.

Prior scripts/results untouched. Output: results_kl_distill_r256.json.
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
WARM_CKPT = HERE / "kl_distill_student_r128.pt"

LM_WEIGHT = 0.1
TEMPERATURE = 1.0


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
# bottleneck hook (identical form to prior runs)
# --------------------------------------------------------------------------

def make_hook(mean: torch.Tensor, P: torch.Tensor):
    def hook(_mod, _inp, out):
        h = out[0]
        hc = h - mean
        h_new = mean + (hc @ P) @ P.T
        return (h_new,) + tuple(out[1:])
    return hook


def attach_bottleneck(model, means_t: torch.Tensor, P: torch.Tensor):
    return [model.transformer.h[li].register_forward_hook(
        make_hook(means_t[li], P)) for li in range(model.config.n_layer)]


# --------------------------------------------------------------------------
# metrics (same Accum as prior runs)
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
        self.agree_sum += float((base_logits.argmax(-1) == patched_logits.argmax(-1)).sum())
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


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=256)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--eval-seqs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=4)   # eval batch
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--min-steps", type=int, default=600)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--plateau-window", type=int, default=4)  # x eval_every steps
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--train-batch", type=int, default=8)
    ap.add_argument("--val-seqs", type=int, default=16)
    ap.add_argument("--extra-ranks", type=str, default="",
                    help="comma-separated nested ranks to eval the FINAL "
                         "student at (no retraining), e.g. 192")
    ap.add_argument("--out", type=str,
                    default=str(HERE / "results_kl_distill_r256.json"))
    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[device] {device}")
    torch.manual_seed(args.seed)
    rank = args.rank

    if not CACHE.exists():
        raise SystemExit(f"missing {CACHE} — refusing to refit (protocol)")
    z = np.load(CACHE)
    means_np, V_np = z["means"], z["V"]
    assert V_np.shape[1] >= rank, f"cache basis only has {V_np.shape[1]} dims"
    print(f"[basis] loaded frozen shared basis from {CACHE.name} "
          f"(V {V_np.shape}; r={rank} = nested top-{rank} columns, "
          f"top-128 subspace identical to the r=128 run by construction)")

    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    tok = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
    eval_texts, eval_name = build_corpus_texts("test")
    eval_seqs = tokenize_corpus(eval_texts, tok, args.seq_len,
                                args.eval_seqs * args.seq_len, args.seed + 7)
    print(f"[eval corpus] {eval_name}: {eval_seqs.shape[0]} x {eval_seqs.shape[1]}")

    # ---- student: r=128-distilled weights, r=256 bottleneck ----
    if not WARM_CKPT.exists():
        raise SystemExit(f"missing warm-start checkpoint {WARM_CKPT}")
    student = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").to(device)
    sd = torch.load(WARM_CKPT, map_location=device, weights_only=True)
    missing, unexpected = student.load_state_dict(sd, strict=False)
    print(f"[warm start] loaded {WARM_CKPT.name} "
          f"(missing={list(missing)}, unexpected={list(unexpected)})")
    teacher = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    means_t = torch.from_numpy(means_np).to(device)
    P = torch.from_numpy(V_np[:, :rank].copy()).to(device)  # frozen
    hooks = attach_bottleneck(student, means_t, P)

    # ---- protocol step 3: step-0 TEST eval before any training ----
    print(f"[r{rank}] step-0 TEST eval (pure container widening) ...")
    step0_test = eval_vs_teacher(student, teacher, eval_seqs, device,
                                 args.batch_size)
    print(f"[r{rank}] STEP-0 TEST: KL={step0_test['kl_mean']:.4f} "
          f"agr={step0_test['top1_agreement']:.4f} "
          f"ppl={step0_test['ppl']:.2f} acc={step0_test['top1_accuracy']:.4f}")

    # ---- training data ----
    train_texts, _ = build_corpus_texts("train")
    need_tokens = args.max_steps * args.train_batch * args.seq_len
    train_seqs = tokenize_corpus(train_texts, tok, args.seq_len,
                                 need_tokens, args.seed + 31)
    print(f"[data] train seqs available: {train_seqs.shape[0]} "
          f"(need {need_tokens // args.seq_len}; will cycle if short)")
    val_texts, _ = build_corpus_texts("validation")
    val_seqs = tokenize_corpus(val_texts, tok, args.seq_len,
                               args.val_seqs * args.seq_len, args.seed + 101)
    print(f"[data] val slice: {val_seqs.shape[0]} x {args.seq_len}")

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.01)

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        return args.lr

    rng = np.random.default_rng(args.seed + 71)
    order = rng.permutation(train_seqs.shape[0])
    cursor = 0

    def next_batch():
        nonlocal cursor, order
        if cursor + args.train_batch > order.size:
            order = rng.permutation(train_seqs.shape[0])
            cursor = 0
        idx = order[cursor:cursor + args.train_batch]
        cursor += args.train_batch
        return torch.from_numpy(train_seqs[idx]).to(device)

    history = []
    r0 = eval_vs_teacher(student, teacher, val_seqs, device, args.batch_size)
    history.append({"step": 0, **r0})
    print(f"[r{rank}] step 0 (warm start): val KL={r0['kl_mean']:.4f} "
          f"agr={r0['top1_agreement']:.4f} ppl={r0['ppl']:.1f}")

    t0 = time.time()
    step_done = 0
    loss_win, kl_win = [], []
    V = student.config.vocab_size
    for step in range(args.max_steps):
        student.train()
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        batch = next_batch()
        with torch.no_grad():
            t_logits = teacher(batch).logits
            t_probs = F.softmax(t_logits / TEMPERATURE, dim=-1)
            t_ent = float(-(t_probs * torch.log(t_probs.clamp_min(1e-12)))
                          .sum(-1).mean())
        out = student(batch, labels=batch, use_cache=False)
        soft_ce = F.cross_entropy(out.logits.view(-1, V), t_probs.view(-1, V))
        loss = soft_ce + LM_WEIGHT * out.loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        loss_win.append(float(loss.detach()))
        kl_win.append(float(soft_ce.detach()) - t_ent)   # KL = CE - H(teacher)
        step_done = step + 1
        if step_done % 20 == 0:
            print(f"  [r{rank}] step {step_done} loss={np.mean(loss_win):.4f} "
                  f"trainKL={np.mean(kl_win):.4f} ({time.time() - t0:.0f}s)",
                  flush=True)
            loss_win, kl_win = [], []
        if step_done % args.eval_every == 0:
            res = eval_vs_teacher(student, teacher, val_seqs, device,
                                  args.batch_size)
            history.append({"step": step_done, **res})
            print(f"[r{rank}] step {step_done}: val KL={res['kl_mean']:.4f} "
                  f"agr={res['top1_agreement']:.4f} ppl={res['ppl']:.2f} "
                  f"acc={res['top1_accuracy']:.4f} ({time.time() - t0:.0f}s)",
                  flush=True)
            w = args.plateau_window
            if step_done >= args.min_steps and len(history) > w:
                best_kl_now = min(h["kl_mean"] for h in history)
                best_agr_now = max(h["top1_agreement"] for h in history)
                past = history[:-w]
                best_kl_past = min(h["kl_mean"] for h in past)
                best_agr_past = max(h["top1_agreement"] for h in past)
                kl_impr = (best_kl_past - best_kl_now) / max(best_kl_past, 1e-9)
                agr_impr = (best_agr_now - best_agr_past) / max(best_agr_past, 1e-9)
                if kl_impr < 0.01 and agr_impr < 0.01:
                    print(f"[r{rank}] early stop at step {step_done} "
                          f"(plateau: KL impr {kl_impr:.4f}, "
                          f"agr impr {agr_impr:.4f} over last "
                          f"{w * args.eval_every} steps)")
                    break
    wall = time.time() - t0

    def steps95(key, better_is_less):
        v0 = history[0][key]
        vbest = (min if better_is_less else max)(h[key] for h in history)
        thresh = vbest + (0.05 if better_is_less else -0.05) * abs(v0 - vbest)
        if better_is_less:
            return next(h["step"] for h in history if h[key] <= thresh)
        return next(h["step"] for h in history if h[key] >= thresh)

    print(f"[r{rank}] final test eval ...")
    test_res = eval_vs_teacher(student, teacher, eval_seqs, device,
                               args.batch_size)
    print(f"[r{rank}] TEST: KL={test_res['kl_mean']:.4f} "
          f"agr={test_res['top1_agreement']:.4f} ppl={test_res['ppl']:.2f} "
          f"acc={test_res['top1_accuracy']:.4f}")

    ckpt = HERE / f"kl_distill_student_r{rank}.pt"
    torch.save(student.state_dict(), ckpt)
    print(f"[r{rank}] saved student checkpoint to {ckpt.name}")

    # ---- optional nested-rank frontier evals of the FINAL student ----
    extra_evals = {}
    extra_ranks = [int(x) for x in args.extra_ranks.split(",") if x.strip()]
    for h in hooks:
        h.remove()
    for er in extra_ranks:
        Pe = torch.from_numpy(V_np[:, :er].copy()).to(device)
        eh = attach_bottleneck(student, means_t, Pe)
        res = eval_vs_teacher(student, teacher, eval_seqs, device,
                              args.batch_size)
        for h in eh:
            h.remove()
        extra_evals[f"r{er}"] = res
        print(f"[frontier] final r{rank} student under nested r={er}: "
              f"KL={res['kl_mean']:.4f} agr={res['top1_agreement']:.4f} "
              f"ppl={res['ppl']:.2f}")

    results = {
        "meta": {
            "model": "openai-community/gpt2",
            "teacher": "original GPT-2 small, frozen, no bottleneck",
            "basis": CACHE.name,
            "basis_note": "nested top-256 columns of the same eigenbasis V; "
                          "top-128 subspace identical to the r=128 run",
            "warm_start": WARM_CKPT.name,
            "eval_corpus": eval_name,
            "eval_tokens": int(eval_seqs.size),
            "seq_len": args.seq_len,
            "device": str(device),
            "loss": f"forward KL (CE vs teacher softmax, T={TEMPERATURE}) "
                    f"+ {LM_WEIGHT} * LM loss",
            "baseline_ppl_ref": 54.466,
            "baseline_acc_ref": 0.3102,
            "targets": "agreement >= 0.6, ppl <= 1.5x baseline (~82)",
            "r128_final_test_ref": {
                "kl_mean": 1.1151, "top1_agreement": 0.4703, "ppl": 122.83,
            },
        },
        f"r{rank}": {
            "rank": rank,
            "warm_start": f"{WARM_CKPT.name} (r=128 KL-distilled student)",
            "loss": f"CE(student, softmax(teacher)) T={TEMPERATURE} "
                    f"+ {LM_WEIGHT} * LM loss",
            "step0_test": step0_test,
            "train": {
                "steps": step_done,
                "wall_time_s": wall,
                "lr": args.lr,
                "warmup": args.warmup,
                "batch": [args.train_batch, args.seq_len],
                "eval_every": args.eval_every,
                "max_steps": args.max_steps,
                "plateau_window_steps": args.plateau_window * args.eval_every,
                "history_val": history,
                "steps_to_95pct_kl": steps95("kl_mean", True),
                "steps_to_95pct_agreement": steps95("top1_agreement", False),
                "steps_to_95pct_ppl": steps95("ppl", True),
            },
            "final_test": test_res,
            "nested_rank_evals_of_final_student": extra_evals,
        },
    }
    out_path = Path(args.out)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[r{rank}] wrote {out_path}")


if __name__ == "__main__":
    main()
