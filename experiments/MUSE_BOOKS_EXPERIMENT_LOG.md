# MUSE Books — LoRA-BiAL Experiment Log

**Model:** Llama-2-7b-hf | **Data:** Books | **Goal:** Beat SimNPO HM=0.755

HM = harmonic mean of (1-forget_knowmem, 1-verbmem, retain). Gold retrain HM=0.739.

## Results

| ID | Loss | Key Change | fk↓ | vm↓ | rk↑ | ex↓ | HM↑ | Notes |
|----|------|-----------|------|------|------|------|------|-------|
| 01 | GA | baseline, 3ep | 0.180 | 0.163 | 0.164 | 0.021 | — | retain destroyed, INVALID (buggy code) |
| 02 | GA | tight ε=0.05 | 0.000 | 0.004 | 0.000 | 0.008 | 0.000 | total collapse, INVALID |
| 03 | GA | 1ep only | 0.417 | 0.045 | 0.651 | 0.017 | — | good retain but fk too high, INVALID |
| 04 | entropy | 1ep | 0.278 | 0.001 | 0.595 | 0.008 | 0.738 | best HM pre-fix, INVALID |
| 05 | reprOrtho | 1ep | 0.396 | 0.000 | 0.662 | 0.008 | 0.720 | too close to RMU, INVALID |
| 06 | entropy | **FIXED code**, bs=16, 3ep | — | — | — | — | — | killed: L_fgt stagnant at -0.17 |
| 07 | entropy | inner_bs=8 vs outer_bs=16 | — | — | — | — | — | killed: same stagnation as 06 |
| 08 | entropy | inner warmup=10 steps | — | — | — | — | — | killed: L_fgt stagnant even with warmup |
| 09 | **focal_repr_ortho** | new loss: focal-weighted repr similarity | 0.430 | 0.995 | 0.610 | 0.902 | 0.015 | epoch 1 only (killed mid-ep2), LoRA too small to affect generation |
| 10 | repr_orthogonal | plain (no focal), 1ep, **dual fix** | 0.414 | 0.995 | 0.643 | 0.902 | 0.015 | retain improved (dual fix), vm still unchanged |

**INVALID (06-10):** dual update bug — `max(0, r)` zeroed negative residuals, λ stuck at 1.0. Also: shared Adam state (inner/outer), iterators wasted half the data, epoch counting wrong. All fixed.

**INVALID (01-05):** inner loop reused same batch for all K steps; no grad accumulation (effective bs=2 not 16); wrong ALM formula.

## Experiment Notes

### 01-05: Pre-fix runs (all invalid)
Code had 3 critical bugs: (a) inner loop fed same retain batch to all K steps instead of fresh data, (b) no gradient accumulation so effective batch size was 2 instead of 16, (c) ALM used linearized penalty instead of correct quadratic form. Results are directionally informative but not trustworthy.

**Key learnings despite bugs:**
- GA is unbounded, destroys retain over multiple epochs
- entropy_max is bounded, reference-free, works from zero-init LoRA
- reprOrtho works but too similar to existing RMU — weak novelty angle

### 06: First correct run (2026-04-20) — killed early
**Folder:** `muse_Books_LoRABiAL_Ze0_entropy_v2` (deleted)

Fixed all 3 bugs. Config: entropy_max, K=3, bs=16, 3 epochs (105 steps).

**Observation:** L_fgt stagnated at ~-0.17 through 15 steps. Inner loop was too strong — K=3 inner steps at bs=16 each meant inner saw 3x more data per outer step, locking retain so tight the outer couldn't push forgetting. L_ret dropped to 0.005 while L_fgt barely moved.

### 07: Weaker inner loop (2026-04-20) — running
**Folder:** `muse_books_exp_07`

Same as 06 but inner_accumulation_steps=1 (inner_eff_bs=8) vs outer accum=2 (outer_eff_bs=16). Inner is deliberately weaker so outer has room to drive forgetting.

**Rationale:** Inner already sees 3x epochs (K=3). Halving its batch size reduces gradient quality → inner can't lock retain as tightly → outer has more room to push entropy on forget data.

**Result:** Same stagnation as 06. Batch size isn't the issue — inner dominates regardless.

### 08: Outer-only warmup (2026-04-20) — running
**Folder:** `muse_books_exp_08`

inner_warmup_steps=10: outer runs alone for first 10 steps (no inner loop), then K=3 kicks in. Otherwise same as 07 (outer_bs=16, inner_bs=8).

**Rationale:** Zero-init LoRA starts with near-zero retain loss (~0.05). Inner loop has nothing to do but lock params near zero, preventing outer from establishing a forgetting direction. Warmup lets outer move LoRA params into a useful region first, then inner protects retain once there's something to protect.

**Result:** L_fgt still stagnant (-0.15 to -0.27) even during outer-only warmup phase. Problem is entropy_max itself — at lr=3e-5 with LoRA r=16, the loss produces too weak a gradient signal to push entropy meaningfully. The buggy runs "worked" because same-batch repetition created an overfitting signal that doesn't exist with correct diverse batching.

### 09: Focal repr_orthogonal (2026-04-20) — running
**Folder:** `muse_books_exp_09`

New forget loss: `focal_repr_ortho`. Per-sample cosine similarity between forget and retain hidden states, weighted by `sim^γ` (focal). Hard-to-forget samples (still similar to retain) get high weight; already-orthogonal samples get down-weighted.

**Why this should work where entropy_max failed:**
1. Operates in representation space (4096-dim) not vocab space (32k-dim) — much stronger gradient signal
2. Focal weighting concentrates effort on stubborn samples, not wasting gradient on easy ones
3. Different from RMU: RMU pushes toward random vectors; this pushes away from retain representations adaptively

**Config:** focal_repr_ortho, γ=2.0, K=3, outer_bs=16, inner_bs=8, inner_warmup=10, 3 epochs.

**Result (epoch 1 only — killed mid-epoch 2):** fk=0.430, vm=0.995, rk=0.610, ex=0.902, HM=0.015. LoRA weight diff from target was only 0.7-2% Frobenius norm at epoch 1 — too small to change generation behavior. L_fgt was still 0.20 (not converged). By step 55 L_fgt hit 0.000; larger weight changes likely needed. Note: verbmem is hard for all methods (NPO=0.570, BLURNPO=0.809, even Gold retrain=0.145).

**Conclusion:** Premature evaluation — need full 3-epoch run. Re-running as exp_10.

**Novel:** No prior work combines focal weighting with representation-level contrastive loss for LLM unlearning (searched arxiv 2024-2026).

### 11: logit_margin, outer forget-only (2026-04-20) — complete
**Folder:** `muse_books_exp_11`

**Changes from 10:** 6 bug fixes (SGD inner, separate dataloaders, dual update fix, re-clip after implicit, scheduler warmup, epoch counting). Outer step ONLY backprops L_fgt — L_ret computed with no_grad for dual update only. Inner loop is sole retain protector. Loss changed to logit_margin.

**Config:** logit_margin, K=3, outer_bs=16, inner_bs=16, 3 epochs (102 steps), ε=0.70 (but irrelevant — ALM not in gradient).

**Loss dynamics:**
| Step | L_fgt | L_ret | λ | Notes |
|------|-------|-------|---|-------|
| 0 | 26.55 | 0.05 | 0.94 | |
| 15 | 25.18 | 0.04 | 0.00 | λ zeroed (ε=0.70, retain healthy) |
| 25 | 19.83 | 0.07 | 0.00 | forgetting accelerating |
| 30 | 17.80 | 0.14 | 0.00 | retain starting to drift |
| 35 | 13.08 | 0.92 | 0.02 | epoch 2 — retain degrading fast |
| 40 | 9.03 | 4.71 | 1.09 | retain collapsing |
| 45 | 6.71 | 9.46 | 5.09 | catastrophic retain collapse |

**Epoch 1 eval (step 34):** fk=0.242, vm=0.181, rk=0.411, ex=0.062, HM=0.586. Forgetting works — vm dropped from 0.995→0.181 (near gold 0.145). Retain hurt (0.411 vs gold 0.665) because outer has zero retain pressure.

**Final eval (checkpoint-0, 3 epochs):** fk=0.000, vm=0.004, rk=0.000, ex=0.008, HM≈0.000. Total collapse — both forget and retain destroyed.

**Diagnosis:** Without ALM in outer gradient, inner loop (K=3 SGD) can't keep up with outer Adam's aggressive forgetting. The outer has zero retain pressure in its gradient. By epoch 2, forgetting spillover overwhelms inner's ability to repair retain. λ is cosmetic (logged but not in any backward pass).

**Conclusion:** Decoupling outer from retain entirely is too aggressive. Need ALM terms back in outer gradient, but with tighter ε to avoid the old problem (retain gradient dominating forgetting). Next: re-add ALM with ε=0.1-0.2.

### 12: logit_margin + ALM re-added, inequality form (2026-04-20)
**Folder:** `muse_books_exp_12`

**Changes from 11:** ALM back in outer gradient (inequality form: `ρ/2·max(0,r)²`). λ_init=0, ε=0.15. Masking fix on logit_margin. All critic issues resolved.

**Config:** logit_margin, K=3, eff_bs=16, ε=0.15, ρ=0.1, λ_init=0, 3 epochs (killed after e2).

**Loss dynamics:**
| Step | L_fgt | L_ret | r | λ | Notes |
|------|-------|-------|---|---|-------|
| 0 | 26.55 | 0.05 | -0.10 | 0.00 | identical to exp_11 |
| 30 | 17.80 | 0.14 | -0.01 | 0.00 | end e1, retain near ε |
| 35 | 13.07 | 0.92 | +0.77 | 0.17 | ALM kicks in |
| 40 | 9.18 | 3.66 | +3.51 | 1.37 | retain spikes, λ ratchets |
| 50 | 8.44 | 1.37 | +1.22 | 4.27 | ALM pulling retain back |
| 55 | 9.64 | 0.39 | +0.24 | 4.54 | retain recovering |
| 65 | 6.08 | 0.15 | +0.00 | 4.58 | EQUILIBRIUM: L_ret ≈ ε |

**Eval:**
| Epoch | fk↓ | vm↓ | rk↑ | ex↓ | HM↑ |
|-------|------|------|------|------|------|
| 1 | 0.252 | 0.175 | 0.401 | 0.061 | 0.579 |
| 2 | 0.087 | 0.003 | 0.374 | 0.008 | 0.549 |

**Analysis:** ALM works — found equilibrium at L_ret≈ε by step 65 (exp_11 had total collapse). Forgetting is excellent (e2: vm=0.003, fk=0.087 — beats SimNPO). But retain hurt at 0.374 because ALM kicked in too late — damage done during steps 35-50 spike. Epoch 1 is identical to exp_11 e1 (λ=0 throughout).

**Next:** Need ALM active from start. Try λ_init=1.0 with ε=0.15, or higher ρ to react faster.

### 13: logit_margin + ALM from start (λ_init=1, ε=0.15) (2026-04-20)
**Folder:** `muse_books_exp_13`

**Changes from 12:** λ_init=1.0 instead of 0.0. ALM active from step 0.

**Loss dynamics (vs exp_12):**
| Step | L_ret(13) | λ(13) | L_ret(12) | λ(12) | Notes |
|------|-----------|-------|-----------|-------|-------|
| 30 | 0.14 | 0.72 | 0.14 | 0.00 | end e1, λ much higher in 13 |
| 40 | 2.69 | 1.85 | 3.66 | 1.37 | spike smaller in 13 |
| 50 | 0.64 | 3.54 | 1.37 | 4.27 | faster recovery |
| 65 | 0.11 | 3.65 | 0.15 | 4.58 | both at equilibrium |

**Eval:**
| Config | fk↓ | vm↓ | rk↑ | ex↓ | HM↑ |
|--------|------|------|------|------|------|
| Exp13 e1 | 0.256 | 0.179 | 0.413 | 0.063 | 0.589 |
| Exp13 e2 | 0.093 | 0.001 | 0.389 | 0.008 | 0.564 |
| Exp12 e2 | 0.087 | 0.003 | 0.374 | 0.008 | 0.549 |

**Analysis:** λ_init=1 dampened the retain spike (2.69 vs 3.66 peak) and improved rk by +0.015. But retain still drops significantly during the epoch 1→2 transition. The spike at steps 35-45 is inherent to the current architecture: forgetting pressure accumulates over epoch 1 (while retain is easy), then suddenly overwhelms inner loop capacity at the epoch boundary. Forgetting is near-perfect (vm=0.001).

**Bottleneck:** retain_knowmem at ~0.39 vs gold 0.665. Gap is 0.276. Need fundamentally different approach to retain protection — current K=3 inner SGD steps can't keep up.

### 14: focal_logit_margin + ALM (λ=1, ε=0.15) (2026-04-20)
**Folder:** `muse_books_exp_14`

**Changes from 13:** focal_logit_margin (γ=2) instead of plain logit_margin. Per-sample normalization, attention mask applied.

**Loss dynamics vs exp_13:**
| Step | Focal L_fgt | Plain L_fgt | Focal L_ret | Plain L_ret |
|------|-------------|-------------|-------------|-------------|
| 30 | 20.29 | 17.79 | 0.10 | 0.14 |
| 40 | 13.35 | 9.38 | 1.99 | 2.69 |
| 45 | 8.58 | 8.19 | 2.75 | 1.76 |
| 65 | 6.27 | 5.45 | 0.14 | 0.11 |

**Eval:**
| Epoch | fk↓ | vm↓ | rk↑ | ex↓ | HM↑ |
|-------|------|------|------|------|------|
| 1 | 0.290 | 0.338 | 0.443 | 0.184 | 0.546 |
| 2 | 0.112 | 0.001 | 0.391 | 0.008 | 0.567 |

**Analysis:** Focal slows forgetting (L_fgt ~2 higher at epoch boundary), which preserves more retain at epoch 1 (rk=0.443 vs 0.413). But epoch 2 converges to similar point (rk=0.391 vs 0.389). Focal helps marginally but doesn't solve the structural spike issue.

**Summary table — all experiments to date:**
| ID | Loss | Key Config | Best HM | Best rk | Notes |
|----|------|-----------|---------|---------|-------|
| 11 | logit_margin | no ALM | 0.586(e1) | 0.411 | total collapse by e3 |
| 12 | logit_margin | ALM λ=0, ε=0.15 | 0.579(e1) | 0.401 | ALM kicks in late |
| 13 | logit_margin | ALM λ=1, ε=0.15 | 0.589(e1) | 0.413 | best HM so far |
| 14 | focal_logit_margin | ALM λ=1, ε=0.15 | 0.567(e2) | 0.443(e1) | best retain at e1 |

**Target:** SimNPO HM=0.755 (rk=0.714). Gap is primarily in retain.

### 15: logit_margin + cosine lr + adaptive K (2026-04-20)
**Folder:** `muse_books_exp_15`

**Changes from 13:** Cosine lr schedule (warmup 10%), adaptive inner (extra K steps when L_ret > 2ε).

**Key dynamics:** Cosine lr nearly eliminated epoch boundary spike (L_ret=0.149 at step 35 vs 0.882 in exp_13). Adaptive K fired during steps 40-55, capping L_ret at ~2.5 (vs 2.69 uncapped). But cosine decay slowed forgetting significantly.

**Eval:**
| Epoch | fk↓ | vm↓ | rk↑ | ex↓ | HM↑ |
|-------|------|------|------|------|------|
| 1 | 0.347 | 0.812 | **0.539** | 0.645 | 0.318 |
| 2 | 0.137 | 0.013 | 0.363 | 0.010 | 0.536 |

**Analysis:** Best retain ever at epoch 1 (rk=0.539) but forgetting insufficient (vm=0.812). Cosine lr too conservative — outer doesn't push hard enough in epoch 1. Epoch 2 forgetting catches up but retain drops below exp_13. Adaptive K helps cap spikes but can't prevent damage from repeated outer steps.

**Conclusion:** Need middle ground — constant lr but with retain protection earlier. Or: shorter epochs / intermediate checkpoints to find the sweet spot between e1 and e2.
