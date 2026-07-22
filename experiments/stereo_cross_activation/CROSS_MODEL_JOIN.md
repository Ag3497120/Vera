# Step 5 — Cross-model join on the shared r=256 basis (2026-07-23)

Highest-risk step of the exploitation roadmap: distill a **second model**
into the **same frozen shared PCA basis** used by the GPT-2-small container
student, then test whether bottleneck coordinates can be swapped across
models (puzzle “くっつけ” across architectures).

## Model B choice

**Path taken: DistilGPT2 (`distilbert/distilgpt2`) as model B.**

| option | why not / why |
|--------|----------------|
| GPT-2 medium | 1024-d — incompatible with existing 768×256 P without a new basis |
| Second GPT-2-small (diff seed/data) | weaker “same family” test only |
| **DistilGPT2** | **768-d (reuse P), 6 layers vs GPT-2’s 12 — real cross-architecture / asymmetric-depth join at personal scale** |

Student A: `matryoshka_student.pt` (GPT-2 small, r=256 hard bottleneck, all
12 block outputs). Shared P from `bases_cache_soft_distill.npz` — **never
refit**. DistilGPT2 gets its own per-layer means (`means_b_distilgpt2.npz`).
No residual affine adapter was needed.

## Setup

- **P transfer diagnostic**: DistilGPT2 residuals on wikitext-2 train
  (50k tokens) have **mean explained-var 0.928** in span(GPT-2 P)
  (per-layer 0.87–0.97). Far above the JOIN_BLOCKED threshold (0.15).
- **Student B distill**: cold DistilGPT2 + hard `h → mean_l + P Pᵀ (h−mean_l)`
  on all 6 block outputs; loss = forward KL (T=1) + 0.1 LM; AdamW lr 5e-5,
  warmup 50, batch 8×256, **2000 steps**, MPS float32.
- **Join sites** (asymmetric depth map):

  | site | L_A (GPT-2) | L_B (DistilGPT2) |
  |------|-------------|------------------|
  | early | 2 | 1 |
  | mid | 5 | 2 |
  | late | 10 | 4 |

- **Controls**: random coords (matched per-dim σ); shuffled-batch same-model
  coords; identity_self sanity (≈ solo). Optional hub-only stitch using
  RESPONSE_MAP top-16 / top-32 dims.
- Eval: standard 40×256 = **10,240** wikitext-2 test tokens.
- Script `cross_model_join.py`, raw `results_cross_model_join.json`,
  log `cross_model_join_run.log`, checkpoint `student_b_distilgpt2.pt`
  (local only).

### Pre-registered fork

- **JOIN_VIABLE**: real cross-model coords clearly beat random/shuffle;
  joined ppl finite and not ≫10× solo.
- **JOIN_WEAK**: beats random slightly but near-collapse vs solo.
- **JOIN_FAIL**: no better than random.
- **JOIN_BLOCKED**: DistilGPT2 residuals do not live near GPT-2 P /
  distill collapses.

## Solo metrics

| system | vs teacher agr | ppl | note |
|--------|----------------|-----|------|
| Student A (Matryoshka GPT-2) | 0.568 | **77.03** | matches Step 3 frontier |
| Teacher B (DistilGPT2, no BN) | — | 79.73 | own NLL |
| Student B step-0 (cold+P) | 0.290 | 425.3 | |
| **Student B final (2000 steps)** | **0.598** | **95.08** | 1.19× teacher B |

Distill wall **~63 min** (3779 s). Student B is a viable container peer.

## Join tables (real vs controls)

Metrics for the **receiver** vs its teacher. Real = inject donor coords at
the site’s inject layer; all other layers keep the receiver’s own hard
bottleneck.

### A → B (GPT-2 Matryoshka coords → DistilGPT2)

Solo B: ppl 95.08, agr 0.598.

| site | real ppl | real agr | random ppl | shuffle ppl | ppl / solo |
|------|----------|----------|------------|-------------|------------|
| early (L_B=1) | **108.4** | 0.550 | 12238 | 24964 | 1.14× |
| mid (L_B=2) | **101.4** | 0.555 | 8633 | 9910 | 1.07× |
| late (L_B=4) | **95.9** | 0.556 | 5652 | 16831 | **1.01×** |

Hub-only (selected): mid hub16 ppl **97.6** / agr 0.577; late hub16 99.8 /
0.570 — often slightly better agreement than full-256 swap.

### B → A (DistilGPT2 coords → GPT-2 Matryoshka)

Solo A: ppl 77.03, agr 0.568.

| site | real ppl | real agr | random ppl | shuffle ppl | ppl / solo |
|------|----------|----------|------------|-------------|------------|
| early (L_A=2) | **82.7** | 0.548 | 6167 | 33631 | 1.07× |
| mid (L_A=5) | **93.1** | 0.518 | 6177 | 16327 | 1.21× |
| late (L_A=10) | **106.3** | 0.467 | 3457 | 12342 | 1.38× |

Hub-only shines when full swap hurts: mid hub16 **81.0** / 0.556; late hub16
**80.2** / 0.552 — close to solo while full-256 late is 106.

Identity_self controls matched solo ppl exactly (95.08 / 77.03) — inject
path is bit-consistent with the hard bottleneck.

## Verdict: **JOIN_VIABLE**

- Every site, both directions: real ≫ random and real ≫ shuffle
  (ppl ratios 30–400×; agreement gaps +0.29–0.52).
- Joined systems stay **near solo** (worst real/solo = 1.38× at B→A late;
  best A→B late = 1.01×). Language does not collapse.
- P transfer was never blocked (EV 0.93). DistilGPT2 residuals already live
  in the GPT-2 shared dictionary; distillation makes the *function* use that
  dictionary, and the coordinates remain interchangeable.

### Implications for the stereo-cross / puzzle story

1. The shared activation basis is not GPT-2-idiosyncratic — a shallower
   same-width cousin can be projected into it and still language-model.
2. Mid-layer coordinate handoff across models preserves most next-token
   behavior; late full-256 B→A is the weakest cell, and **hub-restricted
   stitch** (Step 1 RESPONSE_MAP) repairs it — puzzle parts matter for joins.
3. Random / shuffle controls destroy the receiver → the win is informational,
   not “any vector in R^256 works.”

## Wall time

| phase | wall |
|-------|------|
| P-probe + means | ~40 s |
| Distill B (2000 steps) | **3779 s (~63 min)** |
| Solo + joins | ~277 s |
| **Total run** | **4057 s (~67.6 min)** |

## Artifacts

- `cross_model_join.py`
- `results_cross_model_join.json`
- `CROSS_MODEL_JOIN.md` (this file)
- `cross_model_join_run.log` (+ `cross_model_join_probe.log`)
- `means_b_distilgpt2.npz` (small; syncable)
- `student_b_distilgpt2.pt` — local only, not pushed
