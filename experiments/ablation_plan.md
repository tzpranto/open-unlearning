# LoRA-BiAL Ablation Plan

## Goal

Answer one question: **is every piece of LoRA-BiAL load-bearing?**

Every ablation row should show degradation on at least one axis (forgetting, retain, or both)
compared to the full method. If any component can be removed with no drop, a reviewer will
argue it's unnecessary complexity.

---

## Setup

- **Benchmark**: TOFU forget01 (1%), Llama-3.2-1B-Instruct
- **Base config**: `configs/experiment/unlearn/tofu/lora_bial_1b.yaml` with `LoRABiALAdaptive` trainer
- **Seeds**: 42, 123, 456
- **Each ablation changes exactly ONE thing from A0.**

---

## A0: Full Method (Control)

```
trainer=LoRABiALAdaptive, lora_r=8, lora_alpha=16, K=3, eta_theta=3e-5, eta_in=2e-4
epsilon_multiplier=0.85 (auto-ε: ε = 0.85 × mean(initial inner losses))
rho=0.1, lambda_init=1.0, lambda_min=0.1
forget_loss_type=clamped_entropy, clamped_entropy_tau=0.7
gradient_accumulation_steps=4, per_device_train_batch_size=4
num_train_epochs=10, checkpoint_every_epoch=true
```

Auto-epsilon means ε = multiplier × model's initial retain loss. A model with baseline
L_ret=1.8 gets ε≈1.53; one with 0.5 gets ε≈0.43. Adapts to model/dataset automatically.

---

## GROUP 1: Bilevel Structure — "Is the inner loop needed?"

| ID | Name | Override | Expected |
|----|------|----------|----------|
| A1 | K=0 (no inner loop) | `K=0 inner_warmup_steps=999999` | Retain degrades; ALM alone can't recover |
| A2 | K=1 (minimal inner) | `K=1` | Worse retain than K=3, much better than K=0 |
| A3 | K=10 (excessive inner) | `K=10` | Forgetting starved; inner loop over-recovers |

---

## GROUP 2: Forget Loss — "Is clamped entropy the right choice?"

| ID | Name | Override | Expected |
|----|------|----------|----------|
| A4 | Gradient Ascent | `forget_loss_type=ga` | Unbounded gradients → retain collapse |
| A5 | NPO | `forget_loss_type=npo npo_beta=4.0` | Comparable or slightly worse; ref-model overhead |
| A6 | tau=0.3 | `clamped_entropy_tau=0.3` | Under-forgets; tokens stop at 30% H_max |
| A7 | tau=0.5 | `clamped_entropy_tau=0.5` | Moderate; less forgetting than tau=0.7 |
| A8 | tau=0.9 | `clamped_entropy_tau=0.9` | Near-unclamped; starts to over-forget |
| A9 | tau=1.0 (unclamped) | `clamped_entropy_tau=1.0` | Over-forgetting; never self-stabilizes |

---

## GROUP 3: ALM + Dual Update — "Is the constrained optimization needed?"

| ID | Name | Override | Expected |
|----|------|----------|----------|
| A10 | Fixed λ=1.0 | `lambda_init=1.0 lambda_min=1.0 lambda_max=1.0` | No adaptive tightening; under-retains |
| A11 | Fixed λ=5.0 | `lambda_init=5.0 lambda_min=5.0 lambda_max=5.0` | Good retain but forgetting stalls |
| A12 | ρ=0 (no quadratic penalty) | `rho=0.0` | λ never updates; fixed-weight combination |
| A13 | ρ=1.0 (aggressive penalty) | `rho=1.0` | λ escalates fast; forgetting choked |
| A14 | Symmetric dual | `dual_decay_factor=1.0` | Sawtooth λ oscillation; no ratchet |

---

## GROUP 4: Auto-Epsilon — "How sensitive is the constraint threshold?"

Sweep eps_mul ∈ {0.75, 1.0, 1.3, 1.5, 2.0, 3.0}. Control is 0.85.

| ID | Name | Override | Expected |
|----|------|----------|----------|
| A15 | eps_mul=0.75 | `epsilon_multiplier=0.75` | Tight → choked forgetting, strong retain |
| A16 | eps_mul=1.0 | `epsilon_multiplier=1.0` | Mild relaxation |
| A17 | eps_mul=1.3 | `epsilon_multiplier=1.3` | More headroom for forgetting |
| A18 | eps_mul=1.5 | `epsilon_multiplier=1.5` | Retain starts loosening |
| A19 | eps_mul=2.0 | `epsilon_multiplier=2.0` | Loose; retain degrades |
| A20 | eps_mul=3.0 | `epsilon_multiplier=3.0` | Constraint nearly inactive |

---

## GROUP 5: Structural

| ID | Name | Override | Expected |
|----|------|----------|----------|
| A21 | Full-FT (no LoRA) | Disable LoRA, full fine-tune | Catastrophic drift without low-rank regularization |
| A22 | Sanity: GA + no bilevel + no ALM | `forget_loss_type=ga K=0 inner_warmup_steps=999999 rho=0.0 lambda_init=0.0 lambda_min=0.0` | Complete retain collapse (lower bound) |

---

## Priority Tiers

### Tier 1: MUST-HAVE — 12 configs × 3 seeds = 36 runs

| ID | Ablation | Claim it supports |
|----|----------|-------------------|
| A0 | Full method (control) | baseline |
| A1 | K=0 | (a) bilevel is needed |
| A4 | GA forget loss | (b) entropy > GA |
| A5 | NPO forget loss | (b) entropy > NPO |
| A9 | tau=1.0 (unclamped) | (b) clamping is needed |
| A10 | Fixed λ=1.0 | (c) adaptive dual is needed |
| A12 | ρ=0 | (c) augmented Lagrangian is needed |
| A14 | Symmetric dual | (c) asymmetry is needed |
| A15 | eps_mul=0.75 | (d) eps sensitivity — tight end |
| A16 | eps_mul=1.0 | (d) eps sensitivity — near default |
| A19 | eps_mul=2.0 | (d) eps sensitivity — loose end |
| A22 | Sanity (lower bound) | all — full framework is needed |

### Tier 2: STRONGLY RECOMMENDED — 9 configs × 3 seeds = 27 runs

| ID | Ablation | Claim it supports |
|----|----------|-------------------|
| A2 | K=1 | (a) K sensitivity |
| A6 | tau=0.3 | (b) tau sensitivity |
| A7 | tau=0.5 | (b) tau sensitivity |
| A8 | tau=0.9 | (b) tau sensitivity |
| A11 | Fixed λ=5.0 | (c) no fixed λ works |
| A13 | ρ=1.0 | (c) ρ sensitivity |
| A17 | eps_mul=1.3 | (d) eps sensitivity |
| A18 | eps_mul=1.5 | (d) eps sensitivity |
| A20 | eps_mul=3.0 | (d) eps sensitivity — extreme |

### Tier 3: NICE-TO-HAVE (appendix) — 2 configs × 3 seeds = 6 runs

| ID | Ablation | Claim it supports |
|----|----------|-------------------|
| A3 | K=10 | (a) K sensitivity |
| A21 | Full-FT (no LoRA) | structural — LoRA vs full FT |

---

## Estimated Compute

- Each run: ~30–60 min on A100 (1B model, ~110 outer steps)
- Tier 1: 36 runs ≈ 18–36 GPU-hours
- Tier 1+2: 63 runs ≈ 32–63 GPU-hours
- All: 69 runs ≈ 35–69 GPU-hours

---

## Code Changes Required

Only **A14 (symmetric dual)** needs a small code addition:

1. Add `dual_decay_factor: float = 0.1` to `LoRABiAL.__init__()`
2. Use `self.dual_decay_factor` instead of hardcoded `0.1` in `outer_step()` line 354
3. Add `dual_decay_factor: 0.1` to `configs/trainer/LoRABiAL.yaml`

**A21 (full-FT)** needs a flag to skip LoRA wrapping in the trainer.

All other ablations are pure Hydra CLI overrides.

---

## Ablation Map: What Each Experiment Reveals

### Q1: Is the bilevel inner loop needed?

| Experiment | What it shows | Expected outcome |
|------------|---------------|------------------|
| A0 vs A1 (K=0) | Inner loop on/off | A1: retain collapses — ALM alone is reactive, not proactive |
| A0 vs A2 (K=1) | Minimal vs moderate inner | A2: slightly worse retain, showing K=3 is the sweet spot |
| A0 vs A3 (K=10) | Over-recovery | A3: forgetting weakened — inner loop undoes outer progress |

**Takeaway**: K=0 proves bilevel is essential. K sweep shows K=3 balances recovery and forgetting.

### Q2: Is clamped entropy the right forget loss?

| Experiment | What it shows | Expected outcome |
|------------|---------------|------------------|
| A0 vs A4 (GA) | Bounded vs unbounded | A4: unbounded gradients → retain collapse or oscillation |
| A0 vs A5 (NPO) | Entropy vs reference-based | A5: comparable or worse; validates entropy as simpler+better |
| A0 vs A9 (tau=1.0) | Clamped vs unclamped entropy | A9: over-forgetting — no self-stabilization |
| A6–A8 (tau sweep) | Sensitivity of clamp threshold | Goldilocks: 0.3 under-forgets, 0.9–1.0 over-forgets, 0.7 is sweet spot |

**Takeaway**: A4 kills GA. A5 shows entropy beats NPO. A9+tau sweep proves clamping is essential and tau=0.7 is robust.

### Q3: Is the ALM with asymmetric dual update needed?

| Experiment | What it shows | Expected outcome |
|------------|---------------|------------------|
| A0 vs A10 (λ=1) | Adaptive vs fixed-low dual | A10: under-retains — lambda can't grow when needed |
| A0 vs A11 (λ=5) | Adaptive vs fixed-high dual | A11: under-forgets — lambda too strong from start |
| A10+A11 together | No single fixed λ works | Proves the tradeoff is non-stationary |
| A0 vs A12 (ρ=0) | Augmented vs plain Lagrangian | A12: λ never updates — reduces to fixed weighting |
| A0 vs A13 (ρ=1) | ρ sensitivity | A13: λ escalates too fast, forgetting choked |
| A0 vs A14 (symmetric) | Asymmetric vs symmetric dual | A14: sawtooth oscillation — no ratchet effect |

**Takeaway**: A10+A11 prove no fixed λ works. A12 proves augmented penalty is needed for λ to move. A14 proves asymmetry stabilizes the dual.

### Q4: How sensitive is the auto-epsilon threshold?

| Experiment | What it shows | Expected outcome |
|------------|---------------|------------------|
| A15 (0.75) | Too tight | Forgetting choked, strong retain |
| A0 (0.85) | Default | Balanced |
| A16 (1.0) | Mild relaxation | Slightly more forgetting, retain holds |
| A17 (1.3), A18 (1.5) | Moderate relaxation | Retain starts loosening |
| A19 (2.0) | Loose | Retain degrades noticeably |
| A20 (3.0) | Near-inactive | Constraint effectively off |

**Takeaway**: Method is robust in [0.75, 1.3] range. Below 0.75: forgetting choked. Above 2.0: retain collapses. Auto-epsilon removes the need for hand-tuning.

### Q5: Is the full framework needed at all?

| Experiment | What it shows | Expected outcome |
|------------|---------------|------------------|
| A0 vs A22 (GA + no bilevel + no ALM) | Full method vs naive baseline | A22: complete retain collapse — lower bound |
| A0 vs A21 (full-FT, no LoRA) | LoRA vs full fine-tuning | A21: catastrophic parameter drift without low-rank constraint |

**Takeaway**: A22 is the "do nothing right" baseline. A21 justifies LoRA as structural regularization.

---

## GROUP 6: Stressed Ablations — "Exposing each component under pressure"

The initial ablations (A1, A10, A12, A14 on MUSE Books) showed minimal degradation because
A0's params are tuned to be safe — the components rarely activate. To prove each component
is load-bearing, we run under "stressed" params that create the conditions each component
was designed to handle.

**Design principle**: For each component, find params where (a) the control (with component)
still works, and (b) the test (without component) visibly fails. The stress comes from tight
epsilon, aggressive LR, or reduced redundancy from other mechanisms.

---

### A14v2: Asymmetric Dual (stressed)

**Motivation**: Asymmetric decay prevents λ from collapsing after a spike resolves, which
would allow a second retain violation (yo-yo pattern). Under A0 params, ε is so loose that
violations rarely happen, so asymmetry never activates meaningfully.

**Stress design**: Tight ε + high ρ → frequent large λ jumps. Symmetric decay lets λ crash
back down between spikes; asymmetric holds it elevated.

**Shared params (both runs):**
```
epsilon_multiplier=1.5, rho=0.5, eta_theta=5e-5, lambda_init=0.5
lambda_min=0.1, T=250, K=3, clamped_entropy_tau=0.7
```

| | Control | Test |
|---|---|---|
| `dual_decay_factor` | 0.1 (asymmetric) | 1.0 (symmetric) |

**Expected**: Symmetric shows λ oscillation and 2+ retain spikes. Asymmetric converges smoothly.

---

### A1v2: Inner Loop (stressed)

**Motivation**: The inner loop (K>0) provides explicit retain recovery after each outer step.
Under A0 params, the outer LR is so conservative that each step barely damages retain, making
K irrelevant — the ALM alone handles it.

**Stress design**: High outer LR + tight ε + low ρ (so λ can't ramp fast enough to compensate
alone). Without inner recovery, retain spirals because each outer step delivers a large
perturbation that λ-only correction can't fix in one step.

**Shared params (both runs):**
```
epsilon_multiplier=1.5, rho=0.05, eta_theta=8e-5, lambda_init=1.0
T=250, clamped_entropy_tau=0.7, eta_in=2e-4
```

| | Control | Test |
|---|---|---|
| `K` | 3 | 0 |

**Expected**: K=0 shows retain spiral or λ growing unbounded (method stalls). K=3 converges.

---

### A10v2: Adaptive λ (stressed)

**Motivation**: Adaptive λ self-corrects during retain violations — ramping up to penalize
forgetting harder when retain degrades. Under A0 params, K=3 inner loop + loose ε provides
enough retain cushion that λ adaptation is redundant.

**Stress design**: K=1 (minimal inner cushion) + tight ε + high ρ (so adaptive λ is dramatic
when it fires). Fixed λ can't respond to spikes when it's the primary defense mechanism.

**Shared params (both runs):**
```
epsilon_multiplier=1.5, rho=0.4, eta_theta=3e-5, K=1
lambda_min=0.1, T=250, clamped_entropy_tau=0.7
```

| | Control | Test |
|---|---|---|
| λ | adaptive (lambda_init=1.0) | fixed=1.0 (lambda_min=1.0, lambda_max=1.0) |

**Expected**: Fixed λ shows sustained retain degradation or oscillation. Adaptive λ climbs to
~3-5 during spikes then stabilizes.

---

### A12v2: ρ Penalty Augmentation (stressed)

**Motivation**: ρ > 0 serves two roles: (1) quadratic penalty for immediate gradient correction,
(2) drives dual update λ += ρ*r. With ρ=0, λ never adapts — stuck at initial value.

**Stress design**: Very tight ε + aggressive outer LR + low λ_init. With ρ=0, starting λ=0.5
is too low to protect retain and can never increase. With ρ=0.5, λ ramps rapidly to match
violation severity.

**Shared params (both runs):**
```
epsilon_multiplier=1.3, eta_theta=5e-5, K=2, lambda_init=0.5
T=250, clamped_entropy_tau=0.7, eta_in=2e-4
```

| | Control | Test |
|---|---|---|
| `rho` | 0.5 | 0.0 |

**Expected**: ρ=0 shows retain collapse (λ stuck at 0.5, insufficient protection). ρ=0.5
adapts λ and adds quadratic penalty → stable convergence.

---

### Stressed Ablation Compute

- 8 runs (4 pairs) × 1 seed × ~60-90 min each = 8-12 GPU-hours
- Run on MUSE Books, Llama-2-7b-hf (same as initial ablations for comparability)
- If any pair doesn't show clear separation, tighten stress further (lower eps_mul or raise eta_theta)
