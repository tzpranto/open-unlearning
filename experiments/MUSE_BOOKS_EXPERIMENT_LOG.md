# MUSE Books — LoRA-BiAL Experiment Log

**Model:** Llama-2-7b-hf | **Data:** Books | **Goal:** Beat SimNPO HM=0.755

HM = harmonic mean of (1-forget_knowmem, 1-verbmem, retain_knowmem). Gold retrain HM=0.739.

**Base config:** `experiment=unlearn/muse/lora_bial_books`, LoRA r=16/α=32, inner K=3 SGD lr=2e-4, outer Adam lr=3e-5, per_device_bs=2, grad_accum=8 (eff_bs=16), gradient checkpointing.

## Results Summary

| ID | Loss | Key Config | fk↓ | vm↓ | rk↑ | ex↓ | HM↑ | Notes |
|----|------|-----------|------|------|------|------|------|-------|
| 11 e1 | logit_margin | no ALM in grad | 0.242 | 0.181 | 0.411 | 0.062 | 0.586 | collapse by e3 |
| 12 e1 | logit_margin | ALM λ=0, ε=0.15 | 0.252 | 0.175 | 0.401 | 0.061 | 0.579 | ALM kicks in late |
| 12 e2 | logit_margin | ALM λ=0, ε=0.15 | 0.087 | 0.003 | 0.374 | 0.008 | 0.549 | |
| 13 e1 | logit_margin | ALM λ=1, ε=0.15 | 0.256 | 0.179 | 0.413 | 0.063 | 0.589 | dampened spike |
| 13 e2 | logit_margin | ALM λ=1, ε=0.15 | 0.093 | 0.001 | 0.389 | 0.008 | 0.564 | |
| 14 e1 | focal_logit_margin | ALM λ=1, ε=0.15 | 0.290 | 0.338 | 0.443 | 0.184 | 0.546 | best e1 rk (logit fam) |
| 14 e2 | focal_logit_margin | ALM λ=1, ε=0.15 | 0.112 | 0.001 | 0.391 | 0.008 | 0.567 | |
| 15 e1 | logit_margin | cosine lr + adaptive K | 0.347 | 0.812 | 0.539 | 0.645 | 0.318 | weak forgetting |
| 15 e2 | logit_margin | cosine lr + adaptive K | 0.137 | 0.013 | 0.363 | 0.010 | 0.536 | |
| **16 e1** | **repr_orthogonal** | ALM λ=1, ε=0.15, 2ep | 0.399 | 0.784 | **0.628** | 0.633 | 0.569 | barely touched forget |
| **16 e2** | **repr_orthogonal** | ALM λ=1, ε=0.15, 2ep | 0.357 | 0.313 | **0.618** | 0.139 | **0.648** | **best HM + rk** |

**Target:** SimNPO HM=0.755 (rk=0.714).

---

## Early Experiments (01-10) — All Invalid

These runs had critical bugs but informed later design decisions.

**Bugs in 01-05:** (a) inner loop reused same batch for all K steps, (b) no gradient accumulation (eff_bs=2 not 16), (c) wrong ALM formula.

**Bugs in 06-10:** dual update zeroed negative residuals (λ stuck at 1.0), shared Adam states between inner/outer, iterators wasted half the data, epoch counting wrong.

| ID | Loss | What we learned |
|----|------|----------------|
| 01-02 | GA | Unbounded loss destroys retain; tight ε collapses everything |
| 03 | GA | 1 epoch has decent retain (0.651) but GA can't be controlled |
| 04 | entropy_max | Best HM pre-fix (0.738) — but result was artifactual |
| 05 | repr_ortho | Good retain (0.662) but too close to RMU — weak novelty |
| 06-08 | entropy_max | Fixed code, but entropy_max gradient too weak for LoRA from zero — L_fgt stagnant across all configs (inner bs, warmup) |
| 09-10 | focal_repr_ortho | Representation-space loss works but LoRA weight changes too small after 1 epoch to affect generation (vm=0.995) |

**Key takeaway:** Entropy_max is dead for zero-init LoRA. Representation losses need more epochs. GA forgets but is uncontrollable. These findings motivated the switch to logit_margin (exp 11+).

---

## Detailed Experiment Notes

### 11: logit_margin, outer forget-only (2026-04-20)
**Folder:** `muse_books_exp_11`
**Reproduce:** `python src/train.py experiment=unlearn/muse/lora_bial_books task_name=muse_books_exp_11 trainer.method_args.epsilon=0.70`

Outer step ONLY backprops L_fgt — L_ret computed with no_grad for dual update only. Inner loop is sole retain protector.

**Diagnosis:** Without ALM in outer gradient, inner loop (K=3 SGD) can't keep up with outer Adam's aggressive forgetting. Total collapse by epoch 3 (fk=0, rk=0). Epoch 1 is the only useful checkpoint.

### 12: logit_margin + ALM re-added, inequality form (2026-04-20)
**Folder:** `muse_books_exp_12`
**Reproduce:** `python src/train.py experiment=unlearn/muse/lora_bial_books task_name=muse_books_exp_12 trainer.method_args.epsilon=0.15 trainer.method_args.lambda_init=0`

ALM back in outer gradient (inequality: `L_fgt + λ·r + ρ/2·max(0,r)²`). λ starts at 0, so first epoch is identical to exp_11. ALM activates at epoch boundary when retain spikes, finds equilibrium L_ret≈ε by step 65.

### 13: logit_margin + ALM from start, λ_init=1 (2026-04-20)
**Folder:** `muse_books_exp_13`
**Reproduce:** `python src/train.py experiment=unlearn/muse/lora_bial_books task_name=muse_books_exp_13 trainer.method_args.epsilon=0.15 trainer.method_args.lambda_init=1.0`

λ_init=1 dampened the epoch boundary retain spike (peak 2.69 vs 3.66 in exp_12). Best HM in logit_margin family.

### 14: focal_logit_margin + ALM (2026-04-20)
**Folder:** `muse_books_exp_14`
**Reproduce:** `python src/train.py experiment=unlearn/muse/lora_bial_books task_name=muse_books_exp_14 trainer.method_args.epsilon=0.15 trainer.method_args.lambda_init=1.0 trainer.method_args.forget_loss_type=focal_logit_margin`

Focal weighting (γ=2) slows forgetting → better epoch 1 retain (0.443) but converges to same rk≈0.39 by epoch 2.

### 15: logit_margin + cosine lr + adaptive K (2026-04-20)
**Folder:** `muse_books_exp_15`
**Reproduce:** `python src/train.py experiment=unlearn/muse/lora_bial_books task_name=muse_books_exp_15 trainer.method_args.epsilon=0.15 trainer.method_args.lambda_init=1.0 trainer.method_args.lr_schedule=cosine trainer.method_args.warmup_fraction=0.1`

Cosine lr nearly eliminated epoch boundary spike but forgetting too slow at epoch 1. Best retain ever (0.539 e1) but useless without forgetting.

### 16: repr_orthogonal + ALM (2026-04-20) — BEST
**Folder:** `muse_books_exp_16`
**Reproduce:** `python src/train.py experiment=unlearn/muse/lora_bial_books task_name=muse_books_exp_16 trainer.method_args.epsilon=0.15 trainer.method_args.lambda_init=1.0 trainer.method_args.forget_loss_type=repr_orthogonal trainer.args.num_train_epochs=2`

**Why:** logit_margin's epoch boundary spike is structural — forgetting in logit space accumulates damage that inner loop can't repair. repr_orthogonal operates in representation space (4096-dim hidden states), pushing forget representations away from retain via cosine similarity. Hypothesis: gentler gradient signal won't overwhelm inner loop.

**Loss dynamics:**
| Step | L_fgt | L_ret | r | λ | Notes |
|------|-------|-------|---|---|-------|
| 0 | 0.449 | 0.052 | -0.098 | 0.990 | |
| 10 | 0.364 | 0.049 | -0.101 | 0.886 | gradual forgetting |
| 20 | 0.317 | 0.052 | -0.098 | 0.783 | |
| 30 | 0.314 | 0.043 | -0.107 | 0.680 | end epoch 1 |
| 34 | 0.301 | 0.049 | -0.101 | 0.641 | **no spike at epoch boundary** |
| 40 | 0.313 | 0.046 | -0.104 | 0.582 | L_ret completely stable |
| 50 | 0.190 | 0.044 | -0.106 | 0.481 | forgetting accelerates |
| 60 | 0.076 | 0.048 | -0.103 | 0.378 | |
| 67 | 0.019 | 0.044 | -0.106 | 0.305 | converged |

**Key finding:** L_ret stayed rock-stable at ~0.05 throughout — **zero epoch boundary spike**. This is fundamentally different from logit_margin (which spiked to 0.92-4.71 at the boundary). λ decreased monotonically from 1.0→0.305 because constraint was always satisfied. No adaptive K needed.

**Bottleneck:** Forgetting is now the weak link. fk=0.357 and vm=0.313 are both higher than logit_margin runs. repr_orthogonal makes representations orthogonal but doesn't directly suppress memorized text generation. Need stronger forgetting signal — per_token_repr_ortho or combined loss.

---

## Key Insights

1. **logit_margin family (11-15):** Strong forgetting (vm→0.001) but structural epoch boundary spike damages retain. Best rk≈0.41-0.44.
2. **repr_orthogonal (16):** Completely stable retain (rk=0.618) but weaker forgetting (vm=0.313). No epoch spike.
3. **The Pareto frontier:** logit_margin forgets hard but hurts retain; repr_orthogonal preserves retain but forgets soft. Need to combine strengths.
4. **Next directions:** per_token_repr_ortho (stronger repr signal), focal variants, or hybrid losses.

---

## Bug Fix: entropy_max padding mask (2026-04-21)

**Code review found:** `_compute_entropy_max_loss` averaged entropy over ALL positions including padding tokens. The `.mean()` included padding positions where the model produces arbitrary logits, diluting the gradient signal on actual forget tokens. This likely contributed to entropy_max's stagnation in exps 06-08 — the "weak gradient signal" diagnosis was partially caused by this bug.

**Fix:** Replaced `.mean()` with masked averaging using `attention_mask`, matching how `_compute_logit_margin_loss` handles it. Entropy is now computed only over valid (non-padding) tokens.

**Implication:** entropy_max deserves a re-run with the same ALM config that works for logit_margin (λ_init=1, ε=0.15, K=3). The loss is bounded, reference-free, and directly targets the output distribution — properties logit_margin lacks.
