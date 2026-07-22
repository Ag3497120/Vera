#!/usr/bin/env python3
"""KL distillation from original GPT-2 into the frozen-bottleneck student.

Decisive follow-up to soft_and_distill.py / SOFT_DISTILL.md. There, plain LM
fine-tuning through the frozen hard shared-PCA bottleneck (r=128, all 12
block outputs) recovered capability (ppl 27.1x -> 2.51x baseline) but not
identity (top-1 agreement with the original model plateaued ~0.34). Two
caveats: the LM objective never asked for agreement, and the budget (1000
steps) capped the run, not a plateau. This script targets the agreement
clause directly:

  student : GPT-2 small + the SAME frozen r-rank bottleneck on all 12 block
            outputs (basis/means loaded from bases_cache_soft_distill.npz;
            never refit). h -> mean_l + P P^T (h - mean_l).
  teacher : original GPT-2 small, no bottleneck, frozen, eval, no_grad.
  loss    : forward KL at T=1, i.e. CE(student_logits, softmax(teacher_logits))
            with full-vocab soft targets each step, + LM_WEIGHT * plain LM
            loss for stability (LM_WEIGHT = 0.1, reported in results).

WARM-START: soft_and_distill.py never saved the Phase-2 checkpoint (its
run_phase2 ends with `del model`), so the student starts from pretrained
GPT-2 (lr 5e-5 per protocol).

Training: wikitext-2-raw-v1 train, batch 8 x 256, AdamW lr 5e-5, linear
warmup 50 then constant, grad clip 1.0, budget <= 2500 steps. Eval every
100 steps on a held-out validation slice (16 x 256): KL vs teacher, top-1
agreement vs teacher, own ppl/acc. Plateau early stop: best val KL AND best
val agreement both improved < 1% (relative) over the last 300 steps
(3 evals), only after min-steps.

Final eval: SAME wikitext-2 test tokens as matryoshka_patch.py /
soft_and_distill.py (seed recipe identical: 40 x 256 = 10,240 tokens).

Pre-registered fork (r=128):
  IDENTITY_RECOVERED:        agreement >= 0.6 OR ppl <= 1.5x baseline (~82).
  CAPABILITY_ONLY_CONFIRMED: agreement plateaus < ~0.45 despite the KL
                             objective and longer budget.
  Between: state honestly.

Prior scripts/results untouched. Output: results_kl_distill.json.
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

LM_WEIGHT = 0.1   # auxiliary plain-LM loss weight (stability); reported
TEMPERATURE = 1.0


# --------------------------------------------------------------------------
# corpus (identical packing recipe to soft_and_distill.py / matryoshka_patch.py)
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
# bottleneck hook (identical form to soft_and_distill.py)
# --------------------------------------------------------------------------

def make_hook(mean: torch.Tensor, P: torch.Tensor):
    def hook(_mod, _inp, out):
        h = out[0]
        hc = h - mean
        h_new = mean + (hc @ P) @ P.T
        return (h_new,) + tuple(out[1:])
    return hook


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
# KL-distillation training for one rank
# --------------------------------------------------------------------------

def run_distill(tok, means_np, V_np, eval_seqs, args, device, rank: int):
    from transformers import GPT2LMHeadModel

    print(f"\n===== KL distillation, hard shared bottleneck r={rank} =====")
    student = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").to(device)
    teacher = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    L = student.config.n_layer

    means_t = torch.from_numpy(means_np).to(device)
    P = torch.from_numpy(V_np[:, :rank].copy()).to(device)  # frozen
    hooks = [student.transformer.h[li].register_forward_hook(
        make_hook(means_t[li], P)) for li in range(L)]

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
    print(f"[r{rank}] step 0 (untrained): val KL={r0['kl_mean']:.4f} "
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
            # plateau early stop: best KL AND best agreement both improved
            # <1% (relative) over the last plateau-window evals
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

    # steps to 95% of achieved improvement, at eval granularity
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

    if args.save_ckpt:
        ckpt = HERE / f"kl_distill_student_r{rank}.pt"
        torch.save(student.state_dict(), ckpt)
        print(f"[r{rank}] saved student checkpoint to {ckpt.name}")

    for h in hooks:
        h.remove()
    del student, teacher
    if device.type == "mps":
        torch.mps.empty_cache()
    return {
        "rank": rank,
        "warm_start": "none (pretrained GPT-2; no Phase-2 checkpoint was saved)",
        "loss": f"CE(student, softmax(teacher)) T={TEMPERATURE} "
                f"+ {LM_WEIGHT} * LM loss",
        "train": {
            "steps": step_done,
            "wall_time_s": wall,
            "lr": args.lr,
            "warmup": args.warmup,
            "batch": [args.train_batch, args.seq_len],
            "eval_every": args.eval_every,
            "max_steps": args.max_steps,
            "history_val": history,
            "steps_to_95pct_kl": steps95("kl_mean", True),
            "steps_to_95pct_agreement": steps95("top1_agreement", False),
            "steps_to_95pct_ppl": steps95("ppl", True),
        },
        "final_test": test_res,
    }


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranks", type=str, default="128")
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--eval-seqs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=4)   # eval batch
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--max-steps", type=int, default=2500)
    ap.add_argument("--min-steps", type=int, default=600)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--plateau-window", type=int, default=3)  # x eval_every steps
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--train-batch", type=int, default=8)
    ap.add_argument("--val-seqs", type=int, default=16)
    ap.add_argument("--save-ckpt", action="store_true")
    ap.add_argument("--out", type=str, default=str(HERE / "results_kl_distill.json"))
    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[device] {device}")
    torch.manual_seed(args.seed)

    if not CACHE.exists():
        raise SystemExit(f"missing {CACHE} — refusing to refit (protocol)")
    z = np.load(CACHE)
    means_np, V_np = z["means"], z["V"]
    print(f"[basis] loaded frozen shared basis from {CACHE.name}")

    from transformers import GPT2TokenizerFast

    tok = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
    eval_texts, eval_name = build_corpus_texts("test")
    eval_seqs = tokenize_corpus(eval_texts, tok, args.seq_len,
                                args.eval_seqs * args.seq_len, args.seed + 7)
    print(f"[eval corpus] {eval_name}: {eval_seqs.shape[0]} x {eval_seqs.shape[1]}")

    out_path = Path(args.out)
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    results.setdefault("meta", {
        "model": "openai-community/gpt2",
        "teacher": "original GPT-2 small, frozen, no bottleneck",
        "basis": CACHE.name,
        "eval_corpus": eval_name,
        "eval_tokens": int(eval_seqs.size),
        "seq_len": args.seq_len,
        "device": str(device),
        "loss": f"forward KL (CE vs teacher softmax, T={TEMPERATURE}) "
                f"+ {LM_WEIGHT} * LM loss",
        "baseline_ppl_ref": 54.466,
        "baseline_acc_ref": 0.3102,
        "targets": "agreement >= 0.6 OR ppl <= 1.5x baseline (~82)",
    })

    for rank in [int(x) for x in args.ranks.split(",")]:
        results[f"r{rank}"] = run_distill(tok, means_np, V_np, eval_seqs,
                                          args, device, rank)
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"[r{rank}] wrote {out_path}")


if __name__ == "__main__":
    main()
