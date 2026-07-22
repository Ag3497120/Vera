# Response map of the shared r=256 coordinate system — GPT-2 small distilled student (2026-07-22)

STEP 1 of the exploitation roadmap: flow current through the shared
coordinates and map what responds. Base system: the KL-distilled student
from `KL_DISTILL_R256.md` (`kl_distill_student_r256.pt`, test ppl 78.99 =
1.45× baseline, agreement 0.561) with its frozen hard shared-PCA bottleneck
at all 12 block outputs, `h → mean_l + P Pᵀ (h − mean_l)`, P = V[:, :256]
from `bases_cache_soft_distill.npz`. Question: do the 256 shared dimensions
behave as stable, reusable functional parts (puzzle-piece candidates)?

## Protocol

- **Training-free**: forward passes only, no gradients. All interventions
  act on the shared coordinate z = Pᵀ(h − mean_l) inside the bottleneck
  hooks, then reconstruct h = mean_l + P z; with no intervention this is
  bit-identical to the student's normal bottleneck, so the unablated
  student is the baseline (KL and top-1 flips measured against it).
- **Eval text**: held-out wikitext-2 test, 20 × 256 = 5,120 tokens, same
  packing recipe/seed as all prior runs (first half of the standard 40-seq
  test slice). Donor text for D: wikitext-2 validation (separate split).
- **Coverage actually run**: A1 ablation of **all 256 dims** (full
  coverage, no sampling needed); A2 per-layer map for the **top-32 dims ×
  12 layers** (384 cells, per-position KL kept); B on those 32 dims; C on
  the top-16; D on the top-4 plus 2 low-impact controls (impact ranks
  ~205 and ~243 of 256).
- Script `response_map.py`, raw `results_response_map.json`, log
  `response_map_run.log`. Prior files untouched. MPS float32.
  **Wall time 1,463 s (24.4 min)** for the full battery.

Pre-registered fork:

- **PARTS_LIKE**: causal impact structured (non-uniform, reproducible),
  high-impact dims keep consistent roles across layers (B clearly above
  null), amplification produces coherent shifts → shared coordinates are
  candidate puzzle parts.
- **FIELD_LIKE**: diffuse impact, layer-idiosyncratic roles, incoherent
  interventions → container valid but raw PCA axes are not parts.
- Between: state honestly.

## A) Ablation response map — strong hub structure

Zeroing coordinate i at all 12 layers simultaneously, all 256 dims:

| statistic | value |
|---|---|
| max KL (dim 0) | 2.993 |
| median KL over 256 dims | 0.0065 |
| Gini of impact distribution | **0.854** |
| top-8 / top-16 / top-32 / top-64 share of total impact | 0.66 / 0.77 / 0.86 / 0.91 |
| dims with KL > 0.1 / > 0.01 | 19 / 89 |
| Spearman(impact, eigenvalue rank) | 0.95 |
| Spearman(impact, activation σ) | 0.99 |

Top-10 dims (all-layer ablation):

| dim | KL | top-1 flip rate | σ_z | note (from C) |
|---|---|---|---|---|
| 0 | 2.993 | 0.748 | 178.2 | outlier/gain axis (huge norm) |
| 10 | 1.124 | 0.505 | 32.1 | late-layer readout |
| 4 | 1.051 | 0.579 | 12.4 | named-entity / formal register |
| 3 | 1.042 | 0.539 | 18.0 | early-layer, damage-dominated |
| 5 | 0.859 | 0.524 | 14.4 | proper nouns / institutions |
| 11 | 0.682 | 0.423 | 16.3 | predicate participles ("deemed/feasible") |
| 15 | 0.348 | 0.356 | 14.7 | wikitext "@-@" hyphen construction |
| 9 | 0.317 | 0.322 | 12.3 | adverbial/discourse connectives |
| 1 | 0.249 | 0.232 | 38.3 | numbers/dates ("number", "about") |
| 14 | 0.239 | 0.311 | 14.7 | "@-@" tokens / names |

Impact is **hub-structured, not diffuse**: 8 of 256 coordinates carry 66%
of the total ablation response; the median dim is ~460× weaker than dim 0.
The tail is not dead (all 256 dims have KL > 0.0025; the 192 dims outside
the top-64 still sum to ~9% of total), but individually the bulk is
near-inert. Impact ranking is almost exactly activation-σ ranking
(ρ = 0.99), and the top-19 impact dims all sit within eigen-index ≤ 34 —
the pooled-PCA eigenvalue order is a good (not perfect: dim 10 has tiny
eigenvalue but rank-2 impact) proxy for causal importance.

### A2: dim × layer map (top-32 dims, per-layer ablation KL)

Condensed heatmap (KL; rows = dims in impact order, selected):

| dim | l0 | l1 | l2 | l3 | l4 | l5 | l6 | l7 | l8 | l9 | l10 | l11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | .00 | .00 | 2.43 | **2.56** | 2.02 | 1.31 | .64 | .40 | .28 | .12 | 1.99 | .22 |
| 10 | .00 | .01 | .01 | .01 | .01 | .01 | .01 | .01 | .01 | .02 | .02 | **1.06** |
| 4 | .07 | .28 | **.42** | .24 | .15 | .12 | .09 | .07 | .04 | .03 | .02 | .02 |
| 3 | .36 | **.96** | .54 | .34 | .22 | .19 | .13 | .10 | .05 | .02 | .01 | .01 |
| 5 | .04 | .23 | **.33** | .17 | .10 | .08 | .06 | .04 | .03 | .02 | .01 | .07 |
| 11 | .00 | .00 | .01 | .01 | .01 | .01 | .01 | .02 | .03 | .04 | .07 | **.67** |
| 1 | .00 | .00 | .00 | .00 | .00 | .00 | .00 | .00 | .00 | .00 | .00 | **.25** |
| 2 | .00 | .03 | .01 | .02 | .04 | .05 | **.08** | .06 | .04 | .02 | .04 | .01 |

Layer marginal (mean KL over the 32 dims): 0.017, 0.050, **0.122, 0.110**,
0.085, 0.061, 0.039, 0.029, 0.022, 0.016, **0.077, 0.135** — bimodal:
early layers 2–4 and the final layer 11 are where the coordinates carry
the most current. Three clear layer archetypes emerge:

- **early-processing dims** (3, 4, 5): peak at layers 1–2, decay
  monotonically — the information is consumed by mid-stack computation;
- **readout dims** (10, 11, 15, 9, 1, 14, 17, 16, …): near-zero
  everywhere except layer 11 — they matter as direct inputs to the LM
  head (23 of the 32 mapped dims peak at l11);
- **whole-stack dims** (0, and mildly 2): active across many layers with
  an extra bump at l10 — global gain/offset-like.

## B) Cross-layer role consistency

**B1 (causal): ablation-profile correlation.** Spearman correlation of
per-position KL vectors (5,120 positions) for "dim i ablated at layer l₁"
vs "at layer l₂", cells with KL < 1e-4 excluded:

| | mean | std | n pairs |
|---|---|---|---|
| same dim, different layers | **0.526** | 0.261 | 2,112 |
| different dim & layer (null) | 0.191 | 0.097 | 300 |

Same-dim consistency is ~3.5 null-SDs above the null and holds for every
one of the 32 dims (per-dim means 0.37–0.65, all above the null mean).
Zeroing the same coordinate at different depths hurts substantially the
same output positions — the coordinate has a layer-stable causal
footprint. (The nonzero null reflects generic "hard tokens get hurt by
everything" structure.)

**B2 (correlational): activation-profile correlation.** Pearson
correlation of z_i(l,·) time-series across layers: same dim **0.808**
(per-dim 0.27–0.93; 30 of 32 above 0.7) vs different-dim null **0.001**.
Caveat: the residual stream carries coordinates forward, so B2 has an
architectural tailwind; B1 is the meaningful causal test, and it agrees.

## C) Semantic character of top-16 dims

Max-activating contexts + the odd (directional) part of the mean log-prob
shift under ±2σ amplification at all layers. Qualitative summary:

| dim | theme | evidence |
|---|---|---|
| 0 | outlier/gain axis, not semantic | σ=178 (5× next); max-acts scattered (dates, mid-word pieces); shift is damage-dominated (odd_frac 0.38) |
| 10 | date/enumeration readout | max-acts on months/"or"/"@" in date ranges; +amp boosts unit suffixes ("gb","hr","k") |
| 4 | formal/proper-noun register | boosts " Theatre"," Orchestra"," Typhoon"," Laboratories"; suppresses conversational ("said","then","always") |
| 3 | early lexical-access | damage-dominated (sign-sym −0.84, odd_frac 0.23): both ±2σ hurt the same way; not a directional lever |
| 5 | institutions/place names | max-acts " Czech"," Meteorological"," Revolutionary"; boosts " Isle"," Patrol"," Laboratories" |
| 11 | predicate participles | boosts " harmed"," warranted"," advisable"," feasible"," deemed"; suppresses city names; max-acts before predicates |
| 15 | wikitext "@-@" hyphenation | max-acts inside "metal @-@ skinned" etc. |
| 9 | adverbial connectives | boosts " among"," partly"," likely"," primarily"," accordingly"; max-acts on "separately","simultaneously" |
| 1 | numeric/date contexts | max-acts "number 13/17", "about 40", "37 – 41" |
| 14 | "@-@" / compound joins | max-acts almost exclusively on "@"/"-" in compounds |
| 2 | sequence-initial position | max-acts at document/segment starts (position-like) |
| 17 | section-break / " " separator | max-acts on blank separators after sentences |
| 16 | awards/nominations contexts | " nomination"," Best Writing" contexts |
| 34 | first-person-plural discourse | max-acts on quoted "we"; boosts " fans"," Adults"," Millennials" |
| 18 | parenthetical/appositive | max-acts on "(", "–", quote-openers; cleanest linear dim (sign-sym 0.77, odd_frac 0.73) |
| 8 | sentence-final punctuation | max-acts on "."; boosts " .", " ,", " ;", " ¶" — end-of-sentence part |

Themes are **coherent and recognizably functional** — punctuation/EOS
(8), position (2), formatting idioms (14, 15, 17), syntax (9, 11, 18),
register/topic (4, 5, 16, 34), numerals (1, 10). Two of the top four
(0, 3) are *not* clean semantic levers: their ±amplification produces
mostly symmetric damage (odd_frac < 0.4), i.e. they behave like
load-bearing infrastructure axes rather than meaning dials. For the
readout-type dims the response is predominantly linear/directional
(odd_frac 0.55–0.73, sign-symmetry cos up to 0.77).

## D) Write-transfer (mini puzzle test)

Donor value = z_i at its max-|z| position in wikitext-2 *validation*
text; written into the recipient (test text, 2,560 tokens) as a constant,
at layer 6 only vs at all layers. Directionality =
‖mean_pos Δlogprob‖ / mean_pos ‖Δlogprob‖; split-half cos = cosine of the
mean shift on two disjoint recipient halves.

| dim | donor (×σ) | all-layers KL | flip | directionality | split-half cos | cos(one-layer, all-layers) |
|---|---|---|---|---|---|---|
| 0 | −18.2 | 3.36 | 0.85 | 0.91 | 0.994 | 0.998 |
| 10 | −5.3 | 5.11 | 0.99 | 0.70 | 0.978 | 0.699 |
| 4 | +3.7 | 2.79 | 0.90 | 0.86 | 0.995 | 0.705 |
| 3 | −3.9 | 2.21 | 0.75 | 0.89 | 0.997 | 0.350 |
| 231 (control) | +6.0 | 0.33 | 0.36 | 0.70 | 0.980 | 0.677 |
| 184 (control) | −5.9 | 0.25 | 0.32 | 0.70 | 0.984 | 0.790 |

The induced shifts are **consistent and directional, not chaotic**: a
single constant written into one coordinate pushes the whole output
distribution in largely the same direction at every position
(directionality 0.70–0.91; a chaotic response would be near 0) and the
direction is reproducible across disjoint recipient halves (split-half
cos 0.98–1.00). Sign is respected: dim 10's negative donor value produces
a shift *anti*-aligned with its +2σ amplification direction (cos −0.58),
while dim 4's positive donor aligns (+0.60) and dim 3's aligns with its
damage direction (+0.92). One-layer vs all-layer writes point the same
way for readout-type dims (cos 0.70–1.00) but diverge for the
early-processing dim 3 (0.35) — consistent with A2: layer 6 is not where
dim 3 does its work. The low-impact controls are equally *coherent* but
~10× weaker at matched σ-multiples — low current, same wiring.

## Verdict: **PARTS_LIKE for the hub, with an honest qualifier for the bulk**

The pre-registered PARTS_LIKE conditions are met on all three legs, but
they are met *by the hub, not by all 256 axes*:

- **Structured, reproducible impact**: Gini 0.854, top-8 share 66%,
  median dim 460× weaker than the top dim — the opposite of diffuse.
- **Layer-stable identities**: same-dim ablation-profile correlation
  0.526 vs 0.191 null (B1, causal), activation-profile 0.81 vs 0.00
  (B2). Coordinates keep their footprint across depths; A2 additionally
  shows each dim has a characteristic depth profile (early-processing /
  final-readout / whole-stack) that is itself part of its identity.
- **Coherent interventions**: top dims have recognizable functional
  themes (EOS punctuation, position, "@-@" formatting, participles,
  connectives, register), and write-transfer produces directional,
  sign-respecting, split-half-reproducible output shifts (directionality
  0.70–0.91).

Qualifiers, stated honestly per the fork's "between" clause:

1. ~19 dims carry KL > 0.1 and ~89 carry > 0.01; the remaining ~170 are
   individually near-inert (though collectively non-trivial and, per D
   controls, coherently wired). Puzzle-piece candidacy is established for
   the top ~20–30 coordinates, not the full 256.
2. The two heaviest axes (0, 3) behave like infrastructure (gain/norm),
   not meaning: amplification is damage-dominated in both signs. Parts
   need not all be semantic — but these are "chassis" parts, not
   interchangeable "feature" parts.
3. B1 consistency is 0.53, clearly above null but far from 1.0 — roles
   are stable, not identical, across depth.

Practical consequence for STEP 2: the shared coordinate system has a
usable, causally addressable hub of ~20–30 stable parts, and raw pooled-
PCA order (or cheaper still, activation σ, ρ = 0.99) finds them without
any causal sweep. A sparse/rotated dictionary is *not* needed to get
candidate parts — but it remains the obvious next move for splitting hub
dims like 0 and 10 that pack several functions (dates + units + gain)
into one axis.

## Artifacts

- `response_map.py` — probe script (smoke-tested, forward-only)
- `results_response_map.json` — all raw numbers (A1 per-dim, A2 384-cell
  matrices, B stats, C contexts/vocab shifts, D transfer metrics)
- `response_map_run.log` — run log
- Runtime: 1,463 s (24.4 min) on MPS, float32, forward passes only
