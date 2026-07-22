# Call for collaborators — scale Vera beyond GPT-2

## What we already proved (personal compute)

On **GPT-2 small / DistilGPT2**, with pre-registered forks and published JSON:

| Result | Verdict | One-line |
|--------|---------|----------|
| Weight-space shared bridge | **FAIL** | No usable shared structure in weights (even with training) |
| Activation-space shared basis | **OK** | Cross-layer shared PCA works (~1.09 shared/per-layer) |
| Functional container r=256 | **OK** | ppl ≈ 1.45× baseline after modest KL distill |
| Response map | **PARTS_LIKE** | ~20–30 hub dims are stable parts |
| Weight compression | **COMPRESSION_REAL** | 2.85× block params, healed ppl 1.63× |
| Matryoshka distill | **MATRYOSHKA_VIABLE** | One checkpoint usable at r=8…256 |
| Cross-model join | **JOIN_VIABLE** | GPT-2 ↔ DistilGPT2 coord swap ≫ random |
| Coord memory | **MEMORY_VIABLE** | Save / NN search / reinject beats random |

**Feel it yourself (GPT-2):** see [`README.md`](./README.md) — `python -m demo tour`

Artifacts & papers: repo root [`README.md`](../README.md), [`REPRODUCE.md`](../REPRODUCE.md).

---

## What we cannot do next without more compute

We do **not** claim 9B joining works yet. The evidence only says:

> If two **same-width** models are distilled into one shared low-d dictionary, mid-layer coords can be exchanged without collapse (on GPT-2-class models).

### Highest-value experiments we want help with

1. **Same-arch 9B ↔ 9B join** (best next step)  
   - Two Qwen-class (or similar) 9B models, **same hidden size**  
   - Specialize A on code (or math), B on general chat  
   - Distill both into one shared P (or shared adapter stack)  
   - **Measure:** does injecting A’s coords into B raise code-token / task metrics vs random coords and vs B alone?

2. **Matryoshka + memory at 9B**  
   - Rank lever + coord memory sidecar under realistic prompts  

3. **Cross-width 9B ↔ 27B** (harder)  
   - Requires learned width adapters — separate study  

4. **Compression stack**  
   - Structure compress ⊕ BitNet/quant on a joined expert  

### What success would unlock (hypothesis, not proven)

- Expert modules trained separately, **hot-swapped** through a shared coordinate API  
- MoE-like *systems* without training one giant MoE from scratch  
- Cheaper multi-model pipelines (coarse coords on device, fine join on server)

---

## What we offer collaborators

- Full protocols, scripts, negative + positive results (this repo)  
- GPT-2 reference implementation + demo CLI  
- Clear pre-registered forks so results stay honest  
- Credit in README / co-authorship discussion for substantial runs  

## What we ask

- GPU time (even a single A100/H100 week helps for 9B distill pilots)  
- Willingness to publish negatives as well as positives  
- Open issues / PRs with logs + `results_*.json`  

**Contact:** open a GitHub issue on [Ag3497120/Vera](https://github.com/Ag3497120/Vera) titled `Scale: <experiment name>`.

---

## Suggested first pilot (minimal 9B protocol)

```text
1. Pick two checkpoints of the SAME arch & hidden size (e.g. base 9B + code-SFT 9B).
2. Fit or reuse a shared residual PCA / dictionary P at rank r (try 256–1024).
3. Distill each model with hard (or soft) projection onto P at every block output
   (mirror experiments/stereo_cross_activation/kl_distill_*.py).
4. Join: capture coords at mid layer from A, inject into B; controls = random & shuffle.
5. Report: ppl/agreement vs teacher, PLUS one task metric (pass@k / exact match).
6. Pre-register: JOIN_VIABLE if real ≫ random AND task metric moves in the expected direction.
```

If you only have compute for **one** run, do experiment (1) with a single mid-layer join and a tiny code eval set. That single number decides whether the GPT-2 story survives scale.
