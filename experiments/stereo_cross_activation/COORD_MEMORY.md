# Step 2 — Coordinate memory sidecar (2026-07-23)

Personal-scale, **training-free** demo: save / search / reinject model
internal states as a sidecar memory in the shared r=256 coordinate system.

Student: `matryoshka_student.pt` (hard bottleneck on all 12 block outputs,
frozen P from `bases_cache_soft_distill.npz`). MPS float32. Wall **~45 s**.

## Protocol (pre-registered)

- **Corpus**: 80 synthetic paired factoids (unique place×attribute) + 20
  short wikitext fillers. Each fact has a cloze query and a target answer.
- **Encode**: last-token bottleneck coords `c ∈ R^256` at mid (**L5**) and
  last (**L11**); optional hub16 subset (RESPONSE_MAP top-16).
- **Store**: append-only demo under `memory_store/` —
  thin `memories.jsonl` + compressed `coords_f16.npz` (~173 KB total).
- **Search**: cosine NN in coord space.
- **Reinject**: replace (or blend) last-token coords at the inject layer;
  controls = random other stored memory. Metrics = answer-token logprob
  gap (real − random) and win-rate; KL vs baseline reported as secondary.
- **Fork**:
  - **MEMORY_VIABLE**: retrieval clearly above chance (≥3×) **and** real
    memory reinject beats random-memory on a measurable shift.
  - **MEMORY_STORE_ONLY**: retrieval works, reinject ≈ random.
  - **MEMORY_FAIL**: retrieval ~ chance.

Script `coord_memory.py` (`ingest` / `query` / `reinject-eval` / `run-all`),
raw `results_coord_memory.json`, log `coord_memory_run.log`.

## Retrieval

Chance@1 with N=100 store ≈ 0.010.

| layer | recall@1 | recall@5 | mean rank | vs chance@1 |
|-------|----------|----------|-----------|-------------|
| L5 | 0.037 | 0.312 | 15.1 | ~3.7× |
| **L11** | **0.188** | **0.487** | **13.9** | **~19×** |

Hub16-only search collapses (recall@1 ≈ 0) — discriminative identity for
these factoids lives outside the RESPONSE_MAP impact hubs (tail dims).
Late-layer coords (L11) are clearly the better memory keys.

## Reinject vs controls

Primary metric: Δ log P(answer first token) under last-token inject,
**real − random**. Random inject also shifts the distribution (often
boosts answer logprob vs baseline), so the gap vs random is the causal
test.

### Retrieved top-1 (end-to-end)

| config | layer | retrieve@1 | Δans real | Δans rand | gap | win | beats? |
|--------|-------|------------|-----------|-----------|-----|-----|--------|
| full replace | L5 | 0.037 | +2.22 | +1.73 | **+0.49** | 0.59 | yes |
| full replace | **L11** | 0.188 | +2.01 | +1.14 | **+0.87** | 0.60 | yes |
| blend 0.5 | L11 | 0.188 | +2.29 | +1.85 | +0.44 | 0.57 | yes |
| hub16 replace | L11 | 0.188 | +1.25 | +1.02 | +0.23 | 0.60 | yes |

### Oracle (true matching memory — causal ceiling)

| config | layer | Δans real | Δans rand | gap | win | beats? |
|--------|-------|-----------|-----------|-----|-----|--------|
| full replace | L5 | +2.29 | +1.73 | +0.55 | 0.61 | yes |
| full replace | **L11** | **+2.77** | +1.14 | **+1.63** | **0.75** | yes |
| hub16 replace | L11 | +1.25 | +1.02 | +0.23 | 0.55 | yes |

KL(baseline → inject) is often *larger* for random than for real — random
moves the next-token distribution more, but undirectedly. Real memory
moves it toward the stored fact’s answer token.

Free-run greedy generation under full last-token replace often collapses
to repeated tokens; that is expected for a hard single-position overwrite.
The quantitative claim uses next-token answer logprob, not open-ended
fluency.

## Verdict: **MEMORY_VIABLE**

- Retrieval at L11 is clearly above chance (recall@1 **0.188** ≈ 19×,
  recall@5 **0.487** ≈ 10×).
- Reinject of real memory beats random-memory control on answer-token
  boost at multiple configs (strongest: oracle L11 gap **+1.63**,
  retrieved L11 gap **+0.87**).
- Coordinates are therefore both **searchable** and **causally useful**
  as a personal-scale memory sidecar — without further training.

### Implications for the stereo-cross story

1. The shared activation dictionary is already a usable memory address
   space: store `c`, retrieve by cosine, reinject at a late layer.
2. Late layers encode more retrievable identity than mid; hub-only keys
   are insufficient for this factoid set (need the full 256 or a
   retrieval-oriented subspace, not the ablation-impact hubs alone).
3. Random stored coords are a strong control — they also perturb logprobs —
   so “any vector in R^256” is not enough; the matched memory is.

## Artifacts

| file | role |
|------|------|
| `coord_memory.py` | CLI pipeline |
| `results_coord_memory.json` | metrics + verdict |
| `COORD_MEMORY.md` | this writeup |
| `memory_store/` | demo DB (jsonl + npz) |
| `coord_memory_run.log` | run log |

No `.pt` checkpoints published to Vera.
