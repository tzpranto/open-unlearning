# LoRA-BiAL Ablation Plan

## Intuition: What We Want To See

The ablation answers one core question: **is every piece of LoRA-BiAL load-bearing?**

- **Bilevel (K=0/1/3/10)**: The inner loop is *proactive* retain recovery (fixes retain before
  outer step), while ALM penalties are *reactive* (punish violations after they happen).
  K=0 should show retain degradation that no lambda tuning can fix. K=1 vs K=3 vs K=10
  maps the sweet spot: too few = insufficient recovery, too many = inner loop undoes forgetting.

- **Clamped Entropy (GA/NPO/tau sweep)**: Unlearning losses must be *self-limiting*. GA is
  unbounded — keeps pushing even after tokens are fully forgotten, producing enormous gradients
  that destabilize everything. Clamped entropy is bounded AND self-stabilizing: once a token
  hits tau*H_max entropy, its gradient is exactly zero. The tau sweep shows a Goldilocks pattern:
  tau=1.0 never stabilizes (same problem as GA), tau=0.3 stabilizes too early (under-forgets).

- **ALM + Asymmetric Dual**: No fixed lambda works because the forget-retain tradeoff is
  non-stationary — early you need aggressive forgetting (low lambda), late you need strong
  retain protection (high lambda). Fixed lambda=1.0 under-retains, lambda=5.0 under-forgets.
  Without asymmetry, lambda oscillates in a sawtooth pattern. rho=0 kills both the quadratic
  penalty AND the dual update, reducing the method to a fixed-weight combination.

- **Auto-Epsilon**: "How much retain degradation is acceptable" should be *relative* to the
  starting point. A model with baseline L_ret=1.8 needs a different epsilon than one with 0.5.
  Fixed epsilon either works by luck (hand-tuned for this setup) or fails on other setups.
  The ablation shows auto-epsilon matches or beats hand-tuned on TOFU; its real value is that
  it generalizes to MUSE/different models without per-setup tuning.

**The narrative**: Every row in the ablation table should show degradation on at least one axis
(forgetting, retain, or both) compared to the full method. If any component can be removed
with no drop, a reviewer will argue it's unnecessary complexity.

---

**Benchmark**: TOFU forget01, Llama-3.2-1B-Instruct
**Base config**: `configs/experiment/unlearn/tofu/lora_bial_1b.yaml` with `LoRABiAL` trainer
**Seeds**: 42, 123, 456 (3 seeds for ablations; 5 for final paper numbers if needed)

## Full Method Defaults (the control)

```
trainer=LoRABiALAdaptive, lora_r=8, lora_alpha=16, K=3, eta_theta=3e-5, eta_in=2e-4
epsilon_multiplier=0.85 (auto-epsilon: ε = 0.85 × mean(initial inner losses))
rho=0.1, lambda_init=1.0, lambda_min=0.1
forget_loss_type=clamped_entropy, clamped_entropy_tau=0.7
gradient_accumulation_steps=4, per_device_train_batch_size=4
num_train_epochs=10, checkpoint_every_epoch=true
```

**Note**: The full method uses auto-epsilon (`epsilon_multiplier=0.85`), NOT a fixed epsilon.
The multiplier means: ε = 85% of the model's initial retain loss. This adapts automatically
to the model and dataset — a model with baseline retain loss of 1.8 gets ε≈1.53, while one
with 0.5 gets ε≈0.43. The key insight is that "how much retain degradation is acceptable"
should be relative to the starting point, not an absolute number.

Each ablation changes exactly ONE thing from this control.

---

## GROUP 1: Bilevel Structure (Contribution (a))

### A1. No Inner Loop (K=0) [MUST-HAVE]
- **Isolates**: Whether the bilevel inner/outer decomposition is necessary
- **Expected**: Retain degrades significantly; ALM penalty alone cannot keep L_ret near epsilon without dedicated retain recovery
- **Override**:
  ```
  trainer.method_args.K=0 trainer.method_args.inner_warmup_steps=999999
  ```

### A2. Minimal Inner (K=1)
- **Isolates**: Whether K=3 materially improves over K=1 (minimal bilevel)
- **Expected**: Slightly worse retain than K=3 but much better than K=0
- **Override**:
  ```
  trainer.method_args.K=1
  ```

### A3. Excessive Inner (K=10)
- **Isolates**: Sensitivity to K; whether over-recovering retain starves forgetting
- **Expected**: Forgetting is slower/weaker (higher fgt_Prob, fgt_ROUGE)
- **Override**:
  ```
  trainer.method_args.K=10
  ```

---

## GROUP 2: Clamped Entropy Loss (Contribution (b))

### A4. Gradient Ascent as Forget Loss [MUST-HAVE]
- **Isolates**: Clamped entropy vs simplest baseline (unbounded -CE)
- **Expected**: Retain collapse or oscillation; unbounded gradients overwhelm ALM
- **Override**:
  ```
  trainer.method_args.forget_loss_type=ga
  ```

### A5. NPO as Forget Loss [MUST-HAVE]
- **Isolates**: Clamped entropy vs strong bounded alternative (reference-model-based)
- **Expected**: Comparable or slightly worse due to reference pass overhead / LoRA-disable imperfection
- **Override**:
  ```
  trainer.method_args.forget_loss_type=npo trainer.method_args.npo_beta=4.0
  ```

### A6. Unclamped Entropy (tau=1.0) [MUST-HAVE]
- **Isolates**: The clamping mechanism itself; loss never self-stabilizes
- **Expected**: Over-forgetting that damages retain; keeps pushing already-forgotten tokens
- **Override**:
  ```
  trainer.method_args.clamped_entropy_tau=1.0
  ```

### A7. Aggressive Clamp (tau=0.3)
- **Isolates**: Under-forgetting from premature self-stabilization
- **Expected**: Forgetting too weak; tokens stop at 30% of H_max
- **Override**:
  ```
  trainer.method_args.clamped_entropy_tau=0.3
  ```

### A8. Mid-range tau (tau=0.5)
- **Isolates**: Finer tau sensitivity; robustness characterization
- **Expected**: Moderate performance, less forgetting than tau=0.7
- **Override**:
  ```
  trainer.method_args.clamped_entropy_tau=0.5
  ```

---

## GROUP 3: ALM with Asymmetric Dual Update (Contribution (c))

### A9. Symmetric Dual Update [MUST-HAVE]
- **Isolates**: The 10x asymmetry (fast ratchet up, slow decay down)
- **Expected**: Sawtooth oscillation in L_ret; lambda rises and falls at same rate
- **Code change needed**: Add `dual_decay_factor` param to LoRABiAL (default=0.1), use in outer_step line 354
- **Override**:
  ```
  trainer.method_args.dual_decay_factor=1.0
  ```

### A10. Fixed Lambda=1.0 (no dual update) [MUST-HAVE]
- **Isolates**: Whether learned dual variable is necessary vs fixed weight
- **Expected**: No adaptive constraint tightening; either under-forgets or under-retains
- **Override**:
  ```
  trainer.method_args.lambda_init=1.0 trainer.method_args.lambda_min=1.0 trainer.method_args.lambda_max=1.0
  ```

### A11. Fixed Lambda=5.0 (strong retain, no dual update)
- **Isolates**: Whether a high fixed lambda can substitute for adaptive ALM
- **Expected**: Good retain but forgetting is weak; paired with A10 shows no fixed lambda works
- **Override**:
  ```
  trainer.method_args.lambda_init=5.0 trainer.method_args.lambda_min=5.0 trainer.method_args.lambda_max=5.0
  ```

### A12. No Quadratic Penalty (rho=0) [MUST-HAVE]
- **Isolates**: The augmented term rho/2*max(0,r)^2; reduces to standard Lagrangian
- **Expected**: Lambda never updates (rho*r=0), method is fixed-weight combination
- **Override**:
  ```
  trainer.method_args.rho=0.0
  ```

### A13. Large Penalty (rho=1.0)
- **Isolates**: Sensitivity to rho; aggressive penalty
- **Expected**: Lambda escalates rapidly, forgetting stalls
- **Override**:
  ```
  trainer.method_args.rho=1.0
  ```

---

## GROUP 4: Auto-Epsilon (Contribution (d))

The full method uses `epsilon_multiplier=0.85` (auto-epsilon). These ablations test whether
auto-epsilon is necessary by replacing it with fixed absolute epsilon values.

### A14. Fixed Epsilon (hand-tuned, epsilon=0.15) [MUST-HAVE]
- **Isolates**: Auto-epsilon vs a reasonable hand-tuned fixed value
- **Expected**: May work on this specific setup but the point is: you had to hand-tune it.
  Auto-epsilon works across TOFU/MUSE/different models without per-setup tuning.
- **Override**:
  ```
  trainer.method_args.epsilon_multiplier=0.0 trainer.method_args.epsilon=0.15
  ```

### A15. Fixed Epsilon Too Tight (epsilon=0.05) [MUST-HAVE]
- **Isolates**: What happens when fixed epsilon is too tight
- **Expected**: Under-forgetting; lambda escalates immediately because L_ret > 0.05 is almost always violated
- **Override**:
  ```
  trainer.method_args.epsilon_multiplier=0.0 trainer.method_args.epsilon=0.05
  ```

### A16. Fixed Epsilon Too Loose (epsilon=0.50)
- **Isolates**: What happens when fixed epsilon is too loose
- **Expected**: Good forgetting but retain degrades; constraint rarely activates since L_ret < 0.50 most of the time
- **Override**:
  ```
  trainer.method_args.epsilon_multiplier=0.0 trainer.method_args.epsilon=0.50
  ```

---

## GROUP 5: LoRA Configuration (Structural)

### A17. High Rank (r=64, alpha=128)
- **Isolates**: Whether LoRA's low-rank constraint acts as structural regularizer
- **Expected**: Potential overfit/collapse from higher-dimensional update space
- **Override**:
  ```
  trainer.method_args.lora_r=64 trainer.method_args.lora_alpha=128
  ```

### A18. Very Low Rank (r=2, alpha=4)
- **Isolates**: Minimum capacity needed for effective forgetting
- **Expected**: Under-forgetting from insufficient expressivity
- **Override**:
  ```
  trainer.method_args.lora_r=2 trainer.method_args.lora_alpha=4
  ```

---

## GROUP 6: Sanity Check

### A20. GA + No Bilevel + No ALM (lower bound)
- **Isolates**: Value of entire LoRA-BiAL framework vs simplest possible approach
- **Expected**: Complete retain collapse
- **Override**:
  ```
  trainer.method_args.forget_loss_type=ga trainer.method_args.K=0 trainer.method_args.inner_warmup_steps=999999 trainer.method_args.rho=0.0 trainer.method_args.lambda_init=0.0 trainer.method_args.lambda_min=0.0
  ```

---

## Priority Tiers

### Tier 1: MUST-HAVE (rejection risk without these) -- 10 runs x 3 seeds = 30 runs
| ID | Ablation | Claim |
|----|----------|-------|
| A1 | K=0 (no inner loop) | (a) bilevel |
| A4 | GA forget loss | (b) clamped entropy |
| A5 | NPO forget loss | (b) clamped entropy |
| A6 | tau=1.0 (unclamped) | (b) clamping |
| A9 | Symmetric dual | (c) asymmetric ALM |
| A10 | Fixed lambda=1.0 | (c) adaptive dual |
| A12 | rho=0 (no penalty) | (c) augmented term |
| A14 | Fixed epsilon=0.15 | (d) auto-epsilon vs hand-tuned |
| A15 | Fixed epsilon=0.05 | (d) epsilon sensitivity |
| Full | Control (full method) | baseline |

### Tier 2: STRONGLY RECOMMENDED -- 5 runs x 3 seeds = 15 runs
| ID | Ablation | Claim |
|----|----------|-------|
| A2 | K=1 | (a) K sensitivity |
| A7 | tau=0.3 | (b) tau sensitivity |
| A11 | Fixed lambda=5.0 | (c) paired with A10 |
| A13 | rho=1.0 | (c) rho sensitivity |
| A16 | Fixed epsilon=0.50 | (d) epsilon sensitivity |

### Tier 3: NICE-TO-HAVE (appendix) -- 5 runs x 3 seeds = 15 runs
| ID | Ablation | Claim |
|----|----------|-------|
| A3 | K=10 | (a) K sensitivity |
| A8 | tau=0.5 | (b) tau sensitivity |
| A17 | r=64 | structural |
| A18 | r=2 | structural |
| A20 | sanity check | all |

## Estimated Compute

- Each run: ~30-60 min on A100 (1B model, 10 epochs, ~110 outer steps)
- Tier 1: ~27 runs = 14-27 GPU-hours
- Tier 1+2: ~42 runs = 21-42 GPU-hours
- All: ~57 runs = 29-57 GPU-hours

## Code Changes Required

Only A9 (symmetric dual) needs a small code addition:

1. Add `dual_decay_factor: float = 0.1` parameter to `LoRABiAL.__init__()`
2. Use `self.dual_decay_factor` instead of hardcoded `0.1` in `outer_step()` line 354
3. Add `dual_decay_factor: 0.1` to `configs/trainer/LoRABiAL.yaml`

All other ablations are pure config overrides via Hydra CLI.
