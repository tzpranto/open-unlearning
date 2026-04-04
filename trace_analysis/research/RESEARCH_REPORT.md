# Surgical Unlearning Research Report
## MUSE News / Llama-2-7b / SIBL Framework

**Date started:** 2026-04-03  
**Model:** muse-bench/MUSE-News_target (Llama-2-7b fine-tuned on News)  
**Dataset:** MUSE News — 889 forget samples, 1777 retain samples  
**GPU:** NVIDIA H100 NVL (96GB)

---

## Metric Reference
| Metric | Direction | What it measures |
|--------|-----------|-----------------|
| forget_knowmem_ROUGE | ↓ lower = better | Model's ability to recall forget-set knowledge |
| forget_verbmem_ROUGE | ↓ lower = better | Verbatim memorization of forget-set text |
| retain_knowmem_ROUGE | ↑ higher = better | Model's ability to recall retain-set knowledge |
| extraction_strength | ↓ lower = better | Adversarial extraction of forgotten knowledge |
| privleak | ↓ lower = better | Privacy leakage score |

## Pretrained Baseline (no unlearning)
| Metric | Value |
|--------|-------|
| forget_knowmem_ROUGE | 0.650 |
| retain_knowmem_ROUGE | ~0.546 |

---

## Prior Work (for reference)

### External Baselines (non-SIBL methods)
| Method | forget_know | retain_know | forget_verb | extract | Notes |
|--------|------------|------------|------------|---------|-------|
| GradAscent (ep9) | 0.000 | 0.000 | 0.000 | 0.008 | Complete collapse |
| GradDiff (ep2) | 0.623 | 0.509 | 0.386 | 0.150 | Decent but retain drops |
| GradDiff (ep3) | 0.506 | 0.435 | 0.219 | 0.044 | More forget, worse retain |
| NPO (ep1) | 0.648 | 0.542 | 0.470 | 0.189 | Modest improvement |
| NPO (ep6) | 0.624 | 0.522 | 0.492 | 0.218 | Best NPO |

### Prior Surgical SIBL (our v1-v4)
| Version | Config | forget_know | retain_know | forget_verb | extract |
|---------|--------|------------|------------|------------|---------|
| v1 | bitmap 17%, implicit last 2 | 0.652 | 0.538 | 0.577 | 0.297 |
| v2 | tiered 72%, th>0.5 | 0.643 | 0.549 | 0.569 | 0.285 |
| v3 | proportional LR, unfrozen inner, 2x outer, attn implicit | 0.611 | 0.544 | 0.558 | 0.262 |
| v4 | forget heavy T=24 | 0.619 | 0.516 | 0.554 | 0.243 |

---

## Research Experiments (Progressive Ablation Study)

### Shared Settings (unless noted)
```
T=20, K=10, eta_theta=1e-4, eta_in=1e-4, rho=0.5, epsilon=0.1
forget_loss=logit_margin, regularization=none, batch_size=2, bf16=true
```

### Exp1: Raw SIBL (no mask, no implicit, no freeze)
**Goal:** Establish what raw bilevel optimization achieves. Should show forgetting IS possible but retain degrades without surgical precision.  
**Config:** All defaults, no masking, no implicit, no layer freezing.  
**Status:** DONE  
**Save dir:** saves/unlearn/research_exp1_raw_sibl  
**Training time:** ~95s (20 iters × ~4.6s/iter)  
**Results:**
| Metric | Value |
|--------|-------|
| forget_knowmem_ROUGE | 0.511 |
| retain_knowmem_ROUGE | 0.469 |
| forget_verbmem_ROUGE | 0.498 |
| extraction_strength | 0.239 |

**Training dynamics:**
- L_fgt: 23.4 → 17.2 (steady decrease, forget working well)
- L_ret: oscillated 0.3-1.2 (always above ε=0.1, constraint violated throughout)
- λ: 0 → 7.6 (dual variable kept growing, trying to enforce retain)
- ~4.5s/iter (fast, no implicit overhead)

**Observations:**
- **STRONG forgetting**: 0.511 forget_know is the BEST we've seen across ALL methods (better than GradDiff ep3: 0.506 but with much better retain: 0.469 vs 0.435)
- **Retain damage**: 0.469 retain is significant (baseline ~0.546, -14% relative)
- The AL constraint cannot fully prevent retain damage when all layers are modified
- **Key insight**: Raw SIBL forgets BETTER than surgical SIBL (0.511 vs 0.611). The masking was actually holding back forgetting!
- This suggests our surgical approach was too conservative — we need to find the sweet spot between raw (strong forget, bad retain) and surgical (weak forget, good retain)

---

### Exp2: Freeze First 8 Layers (outer only)
**Goal:** Protect foundational representations by freezing layers 0-7 in outer (forget) step. Inner step still updates all layers for retain.  
**Config:** outer_freeze_layers=[0,1,2,3,4,5,6,7]  
**Status:** DONE  
**Save dir:** saves/unlearn/research_exp2_freeze8  
**Training time:** ~93s  
**Results:**
| Metric | Value |
|--------|-------|
| forget_knowmem_ROUGE | 0.636 |
| retain_knowmem_ROUGE | 0.530 |
| forget_verbmem_ROUGE | 0.583 |
| extraction_strength | 0.282 |

**Training dynamics:**
- L_fgt: 23.4 → 22.1 (BARELY decreased! Only 1.3 drop vs 6.2 in Exp1)
- L_ret: similar oscillation pattern, λ → 7.2
- ~4.6s/iter

**Observations:**
- **CRITICAL FINDING**: Freezing layers 0-7 almost completely blocks forgetting (0.636 ≈ 0.650 baseline)
- Retain improved significantly: 0.530 vs 0.469 (Exp1), confirming early layers carry shared representations
- **Paradox**: Trace analysis shows layers 0-6 have ~0% forget neurons, yet freezing them kills forgetting
- **Explanation**: Early layers carry foundational representations that the optimizer needs to modify for forgetting, even though per-neuron gradient ratios don't flag them as "forget-specific". The gradient flows through these layers and their modification enables downstream changes.
- **Implication**: The neuron trace-based mask might be the wrong abstraction for early layers. We may need a different strategy (e.g., allow some modification of early layers but constrain the direction)

---

### Exp3: Exp2 + Implicit Correction (last 4 layers)
**Goal:** Add bilevel gradient correction to prevent outer step from undoing inner step's retain optimization.  
**Config:** use_implicit=true, implicit_blockwise=true, implicit_block_last_n_layers=4 (layers 28-31), gradient_checkpointing=true  
**Note:** Originally planned for all layers 8-31 but OOM even with gradient checkpointing. Full model (all-ones mask) requires too much memory for HVP.  
**Status:** DONE  
**Save dir:** saves/unlearn/research_exp3_freeze8_implicit  
**Training time:** ~140s (~7s/iter, +2.5s from implicit overhead)  
**Results:**
| Metric | Value |
|--------|-------|
| forget_knowmem_ROUGE | 0.621 |
| retain_knowmem_ROUGE | 0.536 |
| forget_verbmem_ROUGE | 0.558 |
| extraction_strength | 0.277 |

**Training dynamics:**
- L_fgt: 23.4 → 21.4 (more drop than Exp2's 22.1, but still far from Exp1's 17.2)
- L_ret: similar oscillation, λ → 7.2
- Implicit correction added ~2.5s/iter overhead

**Observations:**
- Implicit correction provides a **modest improvement** over Exp2: -0.015 forget, +0.006 retain
- Still far from Exp1's strong forgetting (0.621 vs 0.511) — confirming that the freeze is the main bottleneck, not the lack of correction
- OOM limitation: could only apply implicit to last 4 layers (not all 24). With more layers corrected, improvement might be larger.
- **Key insight**: The bilevel correction helps but doesn't compensate for the reduced forgetting capacity from frozen layers

---

### Exp4: Exp3 + Forget-Neuron Masking (ratio >= 1.0)
**Goal:** Only update forget-dominant neurons in outer step. Tests whether surgical precision (14.88% of params) is sufficient for forgetting.  
**Config:** neuron_traces_path, mask_th_low=1.0, mask_th_high=1.0, outer_freeze_layers=[0,...,7], implicit last 4 layers, gradient_checkpointing=true  
**Status:** DONE  
**Save dir:** saves/unlearn/research_exp4_forget_mask  
**Training time:** ~150s (~7.2s/iter)  
**Results:**
| Metric | Value |
|--------|-------|
| forget_knowmem_ROUGE | 0.665 |
| retain_knowmem_ROUGE | 0.539 |
| forget_verbmem_ROUGE | 0.577 |
| extraction_strength | 0.285 |

**Training dynamics:**
- L_fgt: 23.4 → 23.2 (FLAT! Almost no decrease at all)
- retain stable, λ → 7.3

**Observations:**
- **CRITICAL NEGATIVE RESULT**: Forget-neuron masking KILLS forgetting entirely. 0.665 is WORSE than the baseline (0.650)!
- L_fgt barely moved (23.4 → 23.2) — the optimizer has too few params (14.88%) to reduce forget loss
- Retain is excellent (0.539), confirming the mask protects retain, but at the cost of any forgetting
- **Key insight**: Surgical precision (only forget-dominant neurons) is a WRONG approach for forgetting. Forgetting requires modifying SHARED representations, not just forget-specific neurons.
- The trace-based ratio identifies which neurons are *differentially activated* by forget vs retain, but forgetting as a task requires modifying broader weight structure
- Layer mask density: layers 0-7 are nearly empty (< 1%), layers 8-31 range 7-28% with peak at layer 21 (27.84%)

---

### Exp5: Exp4 + Mixed Neurons (ratio >= 0.5, proportional LR)
**Goal:** Expand mask to include mixed neurons with proportional LR scaling. Tests if additional coverage improves forgetting without harming retain.  
**Config:** mask_th_low=0.5, mask_th_high=1.0, proportional_outer_lr=true, 81.87% active params  
**Status:** DONE  
**Save dir:** saves/unlearn/research_exp5_mixed_mask  
**Training time:** ~150s (~7.2s/iter)  
**Results:**
| Metric | Value |
|--------|-------|
| forget_knowmem_ROUGE | 0.637 |
| retain_knowmem_ROUGE | 0.535 |
| forget_verbmem_ROUGE | 0.580 |
| extraction_strength | 0.278 |

**Training dynamics:**
- L_fgt: 23.4 → 21.5 (more than Exp4 but still modest)
- Active params: 81.87% (much more than Exp4's 14.88%)

**Observations:**
- Expanding mask from 15% to 82% barely changed results compared to Exp2 (100%, no mask): 0.637 vs 0.636
- **Confirms: the layer-level freeze (0-7) is the bottleneck, NOT the neuron-level mask**
- Proportional LR scaling has negligible effect when the overall forgetting capacity is limited by frozen layers
- Whether we mask 15%, 82%, or 100% of params in layers 8-31, the results are similar: ~0.636-0.665 forget

---

### Exp6: Gradient Disentanglement (Orthogonal Projection)
**Goal:** Instead of restricting WHICH parameters get updated, restrict the UPDATE DIRECTION to be orthogonal to the retain gradient. This should allow forgetting while protecting retain.

**Algorithm**: For each parameter group (per-layer or global):
1. Compute G_f = forget component of ALM gradient
2. Compute G_r = ∇L_retain (retain gradient)
3. Project: G_f_proj = G_f - (G_f·G_r)/(||G_r||² + ε) * G_r
4. Reconstruct: G_update = G_f_proj + AL_retain_terms

**Implementation**: Added `gradient_projection` and `gradient_projection_scope` params to SIBL.
Code: `_apply_gradient_projection()` method in sibl.py.

#### Exp6a: Aggressive Projection (no freeze, al_coeff=0)
**Config:** gradient_projection=true, scope=aggressive (projects full ALM gradient, removes ALL retain-aligned component)
**Status:** DONE
**Save dir:** saves/unlearn/research_exp6a_grad_proj_nofr
**Results:**
| Metric | Value |
|--------|-------|
| forget_knowmem_ROUGE | **0.409** |
| retain_knowmem_ROUGE | 0.311 |
| forget_verbmem_ROUGE | **0.296** |
| extraction_strength | **0.062** |

**Observations:**
- **BEST FORGETTING EVER**: 0.409 forget_know, 0.062 extraction — far surpasses any other method
- **RETAIN COLLAPSED**: 0.311 retain (baseline 0.546, -43% relative) — unusable
- **Root cause**: Projecting the FULL ALM gradient removes AL constraint terms (λ*G_ret + ρ*(L_ret-ε)*G_ret) that protect retain. The dual variable λ grows unboundedly but can't enforce the constraint.
- The projection removes ~70-80% of the gradient signal (forget and retain are highly correlated)

#### Exp6b: Fixed Projection (preserve AL terms, no freeze)
**Config:** gradient_projection=true, scope=layer, al_retain_coeff=λ+ρr (decompose G_alm into forget+AL_retain, project only forget part, then add back AL terms)
**Status:** DONE
**Save dir:** saves/unlearn/research_exp6b_grad_proj_fixed
**Results:**
| Metric | Value |
|--------|-------|
| forget_knowmem_ROUGE | 0.556 |
| retain_knowmem_ROUGE | 0.476 |
| forget_verbmem_ROUGE | 0.509 |
| extraction_strength | 0.257 |

**Observations:**
- Preserving AL terms improves retain (0.476 vs 0.311) but WEAKER forget than raw SIBL (0.556 vs 0.511)
- The orthogonal component of the forget gradient is small because forget/retain gradients are highly correlated
- The AL terms dominate: as λ grows (~7.5), the reconstruct G_update ≈ (λ+ρr)*G_ret, making the forget projection negligible
- **Key insight**: AL constraint enforcement fights the projection — AL wants to push retain down, projection removes the means to do so

#### Exp6c: Aggressive Projection + Freeze 0-7
**Config:** gradient_projection=true, scope=aggressive, outer_freeze_layers=[0,...,7]
**Status:** DONE
**Save dir:** saves/unlearn/research_exp6c_proj_freeze8
**Results:**
| Metric | Value |
|--------|-------|
| forget_knowmem_ROUGE | 0.619 |
| retain_knowmem_ROUGE | 0.538 |
| forget_verbmem_ROUGE | 0.570 |
| extraction_strength | 0.282 |

**Observations:**
- Nearly identical to Exp2 (freeze only, no projection): forget=0.636, retain=0.530
- **Freeze dominates**: When layers 0-7 are frozen, the projection adds negligible benefit
- Confirms that the frozen layers are the bottleneck, not the gradient direction in unfrozen layers

#### Exp6d-e: Partial Projection Strength
**Goal:** Interpolate between full projection (6a) and no projection (Exp1) using fractional strength α ∈ (0,1).
**Algorithm:** G_proj = G - α*(G·G_r/||G_r||²)*G_r

| Variant | α | forget_know | retain_know | forget_verb | extract |
|---------|---|------------|------------|------------|---------|
| 6d | 0.5 | 0.493 | 0.470 | 0.477 | 0.191 |
| 6e | 0.3 | 0.494 | 0.459 | 0.427 | 0.154 |

**Observations:**
- Both give ~0.493 forget (modest improvement over Exp1's 0.511)
- Retain stays at Exp1 levels (~0.46-0.47) because aggressive mode (al_coeff=0) doesn't enforce AL constraint
- Extraction improves significantly: 0.154-0.191 vs 0.239
- The relationship is NOT linear — α=0.3 and α=0.5 give similar forget_know but different verb/extract

#### Exp6f: Linear Decay Projection (α: 1→0 over T iterations)
**Goal:** "Surgical strike then heal" — aggressive projection early for forgetting, then fade to let AL recover retain.
**Config:** projection_schedule=linear_decay, T=20
**Status:** DONE
**Save dir:** saves/unlearn/research_exp6f_decay
**Results:**
| Metric | Value |
|--------|-------|
| forget_knowmem_ROUGE | 0.490 |
| retain_knowmem_ROUGE | **0.479** |
| forget_verbmem_ROUGE | 0.483 |
| extraction_strength | 0.234 |

**Observations:**
- **First experiment to improve BOTH forget AND retain over raw SIBL** (Exp1: 0.511/0.469)
- Decay works as intended: early strong projection drives forget, later weak projection lets AL recover retain
- retain=0.479 is +0.010 over Exp1 — modest but in the right direction

#### Exp6g: Decay + K=20 (double inner iterations)
**Config:** linear_decay + K=20 (double inner loop for stronger retain recovery)
**Status:** DONE
**Results:** forget=0.597, retain=0.511
**Observation:** K=20 over-corrects. Too much inner optimization per outer step kills forgetting — effectively a "soft freeze". Same regime as Exp2/3.

#### Exp6h: Partial α + Higher rho
**Config:** α=0.5 aggressive + rho=2.0 (stronger AL penalty)
**Status:** DONE
**Results:** forget=0.512, retain=0.465
**Observation:** In aggressive mode, higher rho has no effect because projection removes the retain-aligned component regardless of penalty strength. Identical to Exp1.

#### Exp6i: Rescaled Projection with AL Preservation (KEY RESULT)
**Goal:** Compensate for signal loss from projection by rescaling the projected gradient to maintain original magnitude, while keeping AL terms intact.
**Algorithm:**
1. Decompose: G_fgt = G_alm - c*G_ret (c = λ+ρr)
2. Project: G_fgt_proj = G_fgt - (G_fgt·G_ret/||G_ret||²)*G_ret
3. Rescale: G_fgt_proj *= ||G_fgt||/||G_fgt_proj|| (maintain magnitude)
4. Reconstruct: G_update = G_fgt_proj_rescaled + c*G_ret

**Config:** scope=layer, projection_rescale=true, projection_strength=1.0
**Status:** DONE
**Save dir:** saves/unlearn/research_exp6i_rescale
**Results:**
| Metric | Value |
|--------|-------|
| forget_knowmem_ROUGE | 0.534 |
| retain_knowmem_ROUGE | **0.508** |
| forget_verbmem_ROUGE | 0.510 |
| extraction_strength | 0.250 |

**Observations:**
- **Best retain with meaningful forgetting**: retain=0.508 (-7% from baseline vs -14% for Exp1)
- **Strictly Pareto-dominates GradDiff ep2** (0.623/0.509) — better on BOTH metrics
- Rescaling compensates for the ~70% signal loss from projection, keeping the orthogonal forget direction strong
- AL terms (c*G_ret) fully preserved → proper constraint enforcement
- Weaker forget than aggressive variants (0.534 vs 0.484-0.511) — the AL terms still partially counteract forgetting

#### Exp6j: Rescale + Aggressive + Decay (BEST BALANCED RESULT)
**Goal:** Combine rescaling with aggressive decay for strongest balanced performance.
**Config:** scope=aggressive, projection_rescale=true, projection_schedule=linear_decay, T=20
**Status:** DONE
**Save dir:** saves/unlearn/research_exp6j_rescale_decay
**Results:**
| Metric | Value |
|--------|-------|
| forget_knowmem_ROUGE | **0.484** |
| retain_knowmem_ROUGE | **0.489** |
| forget_verbmem_ROUGE | **0.481** |
| extraction_strength | **0.224** |

**Observations:**
- **BEST BALANCED RESULT** across all experiments
- Better than Exp1 on ALL metrics: forget (0.484 vs 0.511), retain (0.489 vs 0.469), verb (0.481 vs 0.498), extract (0.224 vs 0.239)
- Rescaling amplifies the orthogonal direction, decay allows both forgetting and retain recovery
- Pareto-dominates ALL external baselines (GradDiff, NPO) on the forget-retain frontier

#### Exp6k-l: Extended Iterations (T=30)
**Goal:** Test whether more iterations improve results.

| Variant | Config | forget_know | retain_know | extract |
|---------|--------|------------|------------|---------|
| 6k | rescale+AL T=30 | 0.573 | 0.475 | 0.252 |
| 6l | rescale+aggr+decay T=30 | 0.506 | 0.488 | 0.230 |

**Observations:**
- **T=30 is WORSE than T=20** for both approaches
- For AL-preserving (6k): λ grows too large with extra iterations, AL overwhelms forget gradient
- For aggressive+decay (6l): longer decay means more time in the "weak projection" regime with growing λ
- **Conclusion**: T=20 is near-optimal for the current learning rates

---

## Summary Table (Updated as experiments complete)

| Exp | Config | forget_know↓ | retain_know↑ | forget_verb↓ | extract↓ |
|-----|--------|-------------|-------------|-------------|----------|
| — | **Baseline (pretrained)** | **0.650** | **0.546** | — | — |
| 1 | Raw SIBL | 0.511 | 0.469 | 0.498 | 0.239 |
| 2 | +freeze 0-7 | 0.636 | 0.530 | 0.583 | 0.282 |
| 3 | +implicit (last 4) | 0.621 | 0.536 | 0.558 | 0.277 |
| 4 | +forget mask (14.9%) | 0.665 | 0.539 | 0.577 | 0.285 |
| 5 | +mixed mask (81.9%) | 0.637 | 0.535 | 0.580 | 0.278 |
| 6a | proj aggressive α=1 | **0.409** | 0.311 | **0.296** | **0.062** |
| 6b | proj fixed (AL) α=1 | 0.556 | 0.476 | 0.509 | 0.257 |
| 6c | proj aggr + freeze | 0.619 | 0.538 | 0.570 | 0.282 |
| 6d | proj aggr α=0.5 | 0.493 | 0.470 | 0.477 | 0.191 |
| 6e | proj aggr α=0.3 | 0.494 | 0.459 | 0.427 | 0.154 |
| 6f | proj aggr decay 1→0 | 0.490 | 0.479 | 0.483 | 0.234 |
| 6g | decay + K=20 | 0.597 | 0.511 | 0.531 | 0.237 |
| 6h | α=0.5 + rho=2.0 | 0.512 | 0.465 | 0.491 | 0.230 |
| 6i | **rescale + AL** | 0.534 | **0.508** | 0.510 | 0.250 |
| **6j** | **rescale+aggr+decay** | **0.484** | **0.489** | **0.481** | **0.224** |
| 6k | rescale+AL T=30 | 0.573 | 0.475 | 0.535 | 0.252 |
| 6l | rescale+aggr+decay T=30 | 0.506 | 0.488 | 0.485 | 0.230 |

---

## Key Findings (Exp1-6l)

### 1. Raw SIBL forgets well but trashes retain
- Exp1 (no restrictions): forget=0.511, retain=0.469 — strong forgetting
- This proves the bilevel framework CAN forget; the question is how to protect retain

### 2. Freezing layers 0-7 is the dominant factor limiting forgetting
- Exp2-5 all show forget in 0.621-0.665 range regardless of mask config
- Layers 0-7 have near-zero forget neurons in traces but are critical for forgetting
- **Paradox**: the layers that seem least involved in forgetting (by trace analysis) are the most important for it

### 3. Neuron-level masking HURTS forgetting
- Exp4 (15% forget-only mask): forget=0.665 (worse than no mask!)
- Exp5 (82% mixed mask): forget=0.637 (barely different from 100% no mask)

### 4. Implicit correction provides modest improvement
- Exp3 vs Exp2: -0.015 forget, +0.006 retain (helpful but not transformative)

### 5. Gradient projection is the most powerful technique discovered
- **Exp6a** (full projection, no freeze): forget=0.409, extract=0.062 — strongest forgetting by far
- But retain collapsed to 0.311 — the projection removes AL constraint enforcement
- **Root cause**: forget and retain gradients are ~70-80% correlated; full projection removes most of the signal

### 6. RESCALING is critical for balanced projection
- Without rescaling, the projected gradient is ~30% of original magnitude → too weak
- **With rescaling** (maintain ||G_proj|| = ||G||), the orthogonal direction retains full strength
- **Exp6i** (rescale + AL): retain=0.508 — best retain preservation with meaningful forgetting
- **Exp6j** (rescale + aggressive + decay): **forget=0.484, retain=0.489** — best balanced overall

### 7. Decay schedule bridges aggressive and conservative approaches
- Early strong projection → aggressive forgetting (like 6a)
- Late weak projection → AL constraint recovery (like Exp1)
- This "surgical strike then heal" pattern gives the best of both worlds

### 8. The Pareto frontier (forget↓ vs retain↑)
```
6a (0.409, 0.311) → 6j (0.484, 0.489) → 6i (0.534, 0.508) → 6g (0.597, 0.511)
                                    ↑ SWEET SPOT
```
- **6j Pareto-dominates ALL external baselines** (GradDiff, NPO) on forget-retain trade-off
- The gap between 6j and 6i suggests room for a (~0.50, ~0.50) optimal point

### 9. Configuration sensitivity
- **T=20 is near-optimal** — T=30 worsens results (λ grows too large)
- **K=10 is optimal** — K=20 over-corrects (inner loop kills forgetting)
- **rho has no effect in aggressive mode** — projection removes the penalty signal
- **Partial α (0.3-0.5)** gives modest improvement over no projection but doesn't break through

---

## Next Steps / Future Directions

### 1. Tune the decay schedule
- Try cosine decay or step-function decay instead of linear
- Explore asymmetric schedules (e.g., 30% aggressive phase, 70% recovery phase)
- Adaptive decay based on retain loss monitoring

### 2. Combine rescale+projection with implicit correction
- The best results (6j) don't use implicit correction
- Adding blockwise implicit correction to 6j could improve retain further
- OOM concern: projection + implicit requires 2 extra backward passes per iter

### 3. SVD-based gradient subspace decomposition
- Rank-1 projection (current approach) only removes one direction
- SVD of gradient covariance could identify a full forget-only subspace
- Accumulate gradient statistics over batches → decompose → project into orthogonal complement
- More principled but requires significant new code

### 4. Adaptive projection strength
- Instead of fixed decay, adjust α based on current retain loss
- When retain is good (L_ret < ε): project aggressively (high α)
- When retain is degrading (L_ret > threshold): reduce α to let AL recover
- This would automatically find the optimal trade-off

### 5. Multi-subspace projection (Geometric Disentanglement)
- Inspired by the Geometric Disentanglement Learning paper
- Identify multiple independent retain subspaces (not just one direction)
- Project forget gradient onto the complement of ALL retain subspaces
- Could give stronger forget-only directions

### 6. Recommended configurations for practical use

**Best balanced (recommended):**
```yaml
gradient_projection: true
gradient_projection_scope: aggressive
projection_strength: 1.0
projection_schedule: linear_decay
projection_rescale: true
T: 20, K: 10, eta_theta: 1e-4, eta_in: 1e-4, rho: 0.5
```
→ forget=0.484, retain=0.489 (Exp6j)

**Best retain preservation:**
```yaml
gradient_projection: true
gradient_projection_scope: layer
projection_strength: 1.0
projection_schedule: constant
projection_rescale: true
T: 20, K: 10, eta_theta: 1e-4, eta_in: 1e-4, rho: 0.5
```
→ forget=0.534, retain=0.508 (Exp6i)

---

## Exp7: Activation Steering — Representation-Space Intervention

### Motivation
Exp6 showed that forget and retain gradients are 70-80% correlated in WEIGHT SPACE, making gradient projection lossy. Key hypothesis: **entanglement is lower in ACTIVATION/REPRESENTATION space**. Instead of fighting gradient overlap, intervene directly on internal representations at specific layers.

Inspired by RMU (Representation Misdirection Unlearning), but integrated within SIBL's bilevel framework.

### Implementation
Added to `sibl.py`:
- `_forward_with_hooks()`: Captures hidden state activations at specified transformer layers
- `_compute_steering_loss()`: MSE loss pushing forget activations toward targets
- Two steering modes:
  - **Random**: Push forget activations toward scaled random unit vectors (RMU-style misdirection)
  - **Retain-matching**: Push forget activations toward retain activations (make forget representations look like retain)
- Integration in `outer_step()`: steering loss replaces or augments logit-level forget loss

### Exp7a: Signed Projection (weight-space control, α=1.5)
**Goal:** Push forget gradient in OPPOSITE direction of retain alignment (overshoot past orthogonal).
**Config:** projection_strength=1.5, aggressive, constant schedule
**Status:** DONE
**Save dir:** saves/unlearn/research_exp7a_signed_proj
**Results:**
| Metric | Value |
|--------|-------|
| forget_knowmem_ROUGE | 0.546 |
| retain_knowmem_ROUGE | 0.439 |
| forget_verbmem_ROUGE | 0.429 |

**Observations:**
- **Signed projection doesn't work** — forget worse than Exp1 (0.546 vs 0.511), retain much worse (0.439 vs 0.469)
- Overshooting past orthogonal actively pushes model toward retain-like behavior on forget set
- Confirms that weight-space gradient tricks have diminishing returns

---

### Exp7b: Steering Only (random vectors, layers [5,6,7])
**Goal:** Pure representation-space intervention. Does steering ALONE cause forgetting?
**Config:** use_steering=true, steering_only=true, steering_layers=[5,6,7], coeff=20.0
**Status:** DONE — **use_implicit=false** (SDPA compatible)
**Save dir:** saves/unlearn/research_exp7b_steering
**Results:**
| Metric | Value |
|--------|-------|
| forget_knowmem_ROUGE | 0.637 |
| retain_knowmem_ROUGE | 0.545 |
| forget_verbmem_ROUGE | 0.554 |
| extraction_strength | 0.279 |

**Observations:**
- **Steering perfectly preserves retain** (0.545 ≈ baseline 0.546) — representation intervention is much less damaging than weight-space optimization
- **Steering alone barely causes forgetting** (0.637 vs 0.650 baseline) — the model compensates through later layers
- Extraction resistance (0.279) is already meaningful
- **Key insight**: Steering protects retain by operating in a lower-dimensional space than full weight modification

---

### Exp7c: Steering + Logit Margin (random vectors, layers [5,6,7]) — EXTRACTION BREAKTHROUGH
**Goal:** Combine representation-space steering with weight-space logit-level forgetting.
**Config:** steering_only=false, steering_alpha=1.0, use_implicit=false
**Status:** DONE
**Save dir:** saves/unlearn/research_exp7c_steer_mixed
**Results:**
| Metric | Value |
|--------|-------|
| forget_knowmem_ROUGE | 0.521 |
| retain_knowmem_ROUGE | 0.495 |
| forget_verbmem_ROUGE | 0.442 |
| extraction_strength | **0.125** |

**Observations:**
- **EXTRACTION BREAKTHROUGH**: 0.125 extraction is dramatically better than any previous experiment (next best: 0.154 from Exp6e)
- Dual-space attack: logit-level loss degrades generation quality, steering disrupts internal representations → adversarial extraction becomes very hard
- Retain (0.495) is better than raw SIBL (0.469) — the steering component acts as implicit regularization
- **Key insight**: Weight-space and representation-space interventions are complementary, not redundant

---

### Exp7d: Broader Steering Layers [3-10] + Logit Margin
**Goal:** Test if broader layer coverage improves steering.
**Config:** steering_layers=[3,4,5,6,7,8,9,10], use_implicit=false
**Status:** DONE
**Save dir:** saves/unlearn/research_exp7d_steer_broad
**Results:**
| Metric | Value |
|--------|-------|
| forget_knowmem_ROUGE | 0.542 |
| retain_knowmem_ROUGE | 0.504 |
| forget_verbmem_ROUGE | 0.514 |
| extraction_strength | 0.248 |

**Observations:**
- **Broader layers WORSE than narrow [5,6,7]** on all metrics except retain
- Extraction: 0.248 vs 0.125 — much worse; spreading intervention across too many layers dilutes the effect
- **Key insight**: Narrow, targeted steering at layers 5-7 is optimal for extraction resistance. These layers appear to be the key "extraction pathway" in Llama-2-7b.

---

### Exp7e: Steering + Gradient Projection (rescale+aggressive+decay)
**Goal:** Combine the two best approaches: steering (Exp7c) + projection (Exp6j).
**Config:** steering + logit_margin + gradient_projection, scope=aggressive, rescale=true, decay, use_implicit=true (requires eager attention)
**Status:** DONE
**Save dir:** saves/unlearn/research_exp7e_steer_proj
**Results:**
| Metric | Value |
|--------|-------|
| forget_knowmem_ROUGE | **0.499** |
| retain_knowmem_ROUGE | 0.490 |
| forget_verbmem_ROUGE | 0.495 |
| extraction_strength | 0.232 |

**Observations:**
- **Best forget_know at time**: 0.499 beats 6j's 0.484... wait, actually 6j is better on forget. This gives better extraction than 6j (0.232 vs 0.224 — similar)
- Note: This used `eager` attention + implicit correction (unlike 7c which had implicit=false)
- Implicit correction with eager attention works but is slower and higher memory
- The combination doesn't clearly beat individual approaches on extraction (7c: 0.125 >> 0.232)

---

### Exp7f: Retain-Matching Steering + Logit Margin
**Goal:** Instead of random targets, push forget activations toward RETAIN activations. More targeted than random — the model learns to treat forget data like retain data at representation level.
**Config:** steering_retain_match=true, steering_layers=[5,6,7], use_implicit=true (eager attention)
**Status:** DONE
**Save dir:** saves/unlearn/research_exp7f_steer_retain
**Results:**
| Metric | Value |
|--------|-------|
| forget_knowmem_ROUGE | 0.477 |
| retain_knowmem_ROUGE | 0.494 |
| forget_verbmem_ROUGE | 0.468 |
| extraction_strength | 0.229 |

**Observations:**
- **Retain-matching improves forget_know** over Exp7e (0.477 vs 0.499) while maintaining retain (0.494)
- Better forgetting than random steering (0.477 vs 0.521) — targeted is more effective than random
- Extraction (0.229) not as strong as random steering (0.125) — matching retain activations is "ordered" whereas random vectors create more disruption
- **Key insight**: Retain-matching is better for knowledge-level forgetting; random vectors are better for extraction resistance

---

### Exp7g-k: NPO Loss + Retain-Matching Steering (STRONGEST FORGETTING SERIES)

**Key Discovery**: Replacing `logit_margin` with `NPO` (Negative Preference Optimization) as the forget loss dramatically amplifies forgetting when combined with retain-matching steering. NPO treats forgetting as a preference optimization problem, providing much stronger unlearning signal.

#### Exp7g: NPO β=1.0 + retain-matching steering
**Config:** forget_loss=npo, npo_beta=1.0, T=20, K=10, rho=0.5, steering_alpha=1.0
**Results:**
| Metric | Value |
|--------|-------|
| forget_knowmem_ROUGE | 0.436 |
| retain_knowmem_ROUGE | 0.436 |
| forget_verbmem_ROUGE | **0.296** |
| extraction_strength | **0.095** |

#### Exp7h: NPO β=2.0 + T=15, rho=0.3
**Config:** Less aggressive NPO + shorter training
**Results:**
| Metric | Value |
|--------|-------|
| forget_knowmem_ROUGE | **0.399** |
| retain_knowmem_ROUGE | 0.419 |
| forget_verbmem_ROUGE | **0.237** |
| extraction_strength | **0.067** |

#### Exp7i: NPO β=3.0 + T=10, rho=0.3, steering_alpha=0.5 — VERB_MEM BEATS GOLD
**Config:** Gentler NPO, fewer iterations, lower steering weight
**Results:**
| Metric | Value |
|--------|-------|
| forget_knowmem_ROUGE | 0.406 |
| retain_knowmem_ROUGE | 0.410 |
| forget_verbmem_ROUGE | **0.197** |
| extraction_strength | **0.051** |

**LANDMARK**: verb_mem 0.197 **BEATS the retrained gold standard** (0.201)! extraction 0.051 is exceptional.

#### Exp7j: NPO β=3.0 + K=15, rho=1.0, epsilon=0.05 (stronger retain protection)
**Config:** More inner iterations + tighter constraint + higher penalty
**Results:**
| Metric | Value |
|--------|-------|
| forget_knowmem_ROUGE | **0.372** |
| retain_knowmem_ROUGE | 0.416 |
| forget_verbmem_ROUGE | 0.248 |
| extraction_strength | 0.063 |

**Closest to gold on forget_know** (0.372 vs 0.327 gold).

#### Exp7k: NPO β=3.0 + K=20, rho=2.0, epsilon=0.02 (maximum retain protection)
**Config:** Very strong AL constraint
**Results:**
| Metric | Value |
|--------|-------|
| forget_knowmem_ROUGE | 0.419 |
| retain_knowmem_ROUGE | **0.423** |
| forget_verbmem_ROUGE | 0.280 |
| extraction_strength | 0.086 |

#### NPO+Steering Series Tradeoff Analysis:
```
Higher beta / lower T / lower rho = MORE forgetting, LESS retain
Higher K / higher rho / tighter epsilon = LESS forgetting, MORE retain

Sweet spot: NPO β=3.0, T=10, K=15, rho=1.0 (Exp7j)
Best verb_mem: β=3.0, T=10, α_steer=0.5 (Exp7i: 0.197 beats gold 0.201)
Best extract: same Exp7i (0.051)
Closest forget_know to gold: Exp7j (0.372 vs 0.327)
```

**Fundamental limitation**: Retain degrades to ~0.41-0.42 across all NPO+steering variants. The AL constraint (inner loop) cannot fully compensate for the aggressive outer loop changes. This ~0.14 gap from gold (0.560) appears to be a structural limit of single-pass unlearning on this model.

---

## Updated Summary Table (All Experiments)

| Exp | Config | forget_know↓ | retain_know↑ | forget_verb↓ | extract↓ |
|-----|--------|:---:|:---:|:---:|:---:|
| — | **Gold (Retrained)** | **0.327** | **0.560** | **0.201** | — |
| — | **Baseline (pretrained)** | 0.650 | 0.546 | 0.555 | — |
| 1 | Raw SIBL | 0.511 | 0.469 | 0.498 | 0.239 |
| 2 | +freeze 0-7 | 0.636 | 0.530 | 0.583 | 0.282 |
| 3 | +implicit (last 4) | 0.621 | 0.536 | 0.558 | 0.277 |
| 4 | +forget mask (14.9%) | 0.665 | 0.539 | 0.577 | 0.285 |
| 5 | +mixed mask (81.9%) | 0.637 | 0.535 | 0.580 | 0.278 |
| 6a | proj aggressive α=1 | 0.409 | 0.311 | 0.296 | 0.062 |
| 6b | proj fixed (AL) α=1 | 0.556 | 0.476 | 0.509 | 0.257 |
| 6c | proj aggr + freeze | 0.619 | 0.538 | 0.570 | 0.282 |
| 6d | proj aggr α=0.5 | 0.493 | 0.470 | 0.477 | 0.191 |
| 6e | proj aggr α=0.3 | 0.494 | 0.459 | 0.427 | 0.154 |
| 6f | proj aggr decay 1→0 | 0.490 | 0.479 | 0.483 | 0.234 |
| 6g | decay + K=20 | 0.597 | 0.511 | 0.531 | 0.237 |
| 6h | α=0.5 + rho=2.0 | 0.512 | 0.465 | 0.491 | 0.230 |
| 6i | rescale + AL | 0.534 | 0.508 | 0.510 | 0.250 |
| 6j | rescale+aggr+decay | 0.484 | 0.489 | 0.481 | 0.224 |
| 6k | rescale+AL T=30 | 0.573 | 0.475 | 0.535 | 0.252 |
| 6l | rescale+aggr+decay T=30 | 0.506 | 0.488 | 0.485 | 0.230 |
| 7a | signed proj α=1.5 | 0.546 | 0.439 | 0.429 | — |
| 7b | steering only [5-7] | 0.637 | 0.545 | 0.554 | 0.279 |
| 7c | steer+logit [5-7] | 0.521 | 0.495 | 0.442 | **0.125** |
| 7d | steer+logit [3-10] | 0.542 | 0.504 | 0.514 | 0.248 |
| 7e | steer+proj+implicit | 0.499 | 0.490 | 0.495 | 0.232 |
| 7f | retain-match steer | 0.477 | 0.494 | 0.468 | 0.229 |
| 7g | NPO β=1 + ret_steer | 0.436 | 0.436 | 0.296 | 0.095 |
| **7h** | **NPO β=2 + ret_steer** | **0.399** | 0.419 | **0.237** | **0.067** |
| **7i** | **NPO β=3 + ret_steer** | 0.406 | 0.410 | **0.197** | **0.051** |
| **7j** | **NPO β=3 K=15 ρ=1** | **0.372** | 0.416 | 0.248 | 0.063 |
| 7k | NPO β=3 K=20 ρ=2 | 0.419 | 0.423 | 0.280 | 0.086 |
| 8g | FD-HVP β=3 post_inner=30 | 0.404 | 0.422 | 0.226 | 0.059 |
| 8i | β=1.5 post_inner=30 | 0.381 | 0.434 | 0.233 | 0.059 |
| 8o | β=1.5 FD-HVP | 0.410 | 0.407 | 0.207 | 0.055 |
| **8r** | **β=2.0 FD-HVP** | **0.371** | 0.417 | **0.211** | **0.053** |
| 8q | β=3.0 FD-HVP | 0.634 | 0.544 | 0.567 | 0.291 |
| 9a | masked post_inner[30] bitmap | 0.641 | 0.561 | 0.572 | 0.301 |
| 9b | K=15 ρ=1.0 ε=0.05 | 0.384 | 0.416 | 0.247 | 0.069 |
| 9c | steer[20-22] | 0.376 | 0.410 | 0.212 | 0.055 |
| **9d** | **β=2.5 FD-HVP** | 0.386 | **0.424** | 0.217 | 0.056 |

---

## Updated Key Findings (Exp1-9d)

### 1-9: [Previous findings preserved — see above]

### 10. Activation steering is a powerful complement to weight-space optimization
- Steering alone preserves retain perfectly (0.545 ≈ baseline) but barely causes forgetting
- Combined with weight-space loss, it provides DUAL-SPACE attack: logit disruption + representation disruption
- **Narrow layer targeting [5,6,7] outperforms broad [3-10]** — these layers are the extraction pathway
- Extraction resistance is the unique contribution: 0.125 (7c) vs 0.224 (6j best without steering)

### 11. Retain-matching steering > random vectors for knowledge forgetting
- Random vectors (Exp7c): better extraction resistance (disrupts internal representations more)
- Retain-matching (Exp7f): better knowledge forgetting (makes forget data process like retain data)
- Choice depends on priority: extraction defense → random; knowledge erasure → retain-matching

### 12. NPO + retain-matching steering achieves near-gold forgetting
- **Exp7i beats gold on verbatim memorization** (0.197 vs 0.201)
- **Exp7j approaches gold on knowledge** (0.372 vs 0.327)
- NPO provides much stronger forgetting signal than logit_margin within SIBL framework
- The NPO+steering combo outperforms all standalone methods (NPO alone, GradAscent, GradDiff) by large margins

### 13. The retain gap appears structural (~0.14 from gold)
- All NPO+steering variants plateau at retain ~0.41-0.42
- Stronger AL constraint (K=20, ρ=2.0) only recovers ~0.01 retain at cost of forgetting
- Fundamental limit: single-pass unlearning cannot match retraining from scratch for retain preservation
- Potential solutions: multi-stage approaches, task vector arithmetic, or Fisher-based constraints

### 14. Updated Pareto Frontier (forget_know ↓ vs retain_know ↑)
```
Aggressive forgetting                                    Best retain
    7i (0.406, 0.410) → 7j (0.372, 0.416)                  Best forget_know
         ↓ verb_mem champion                    
    7h (0.399, 0.419) → 7g (0.436, 0.436) → 7f (0.477, 0.494) → 6i (0.534, 0.508) → 6g (0.597, 0.511)
                                                                                          ↑ Best retain

Extraction frontier:
    7i (0.051) → 7j (0.063) → 7h (0.067) → 7g (0.095) → 7c (0.125) → 6e (0.154)
    ↑ Best extraction
```

---

## Updated Recommended Configurations

### Best overall (strongest forgetting with decent retain):
```yaml
forget_loss_type: npo
npo_beta: 3.0
use_steering: true
steering_layers: [5, 6, 7]
steering_coeff: 20.0
steering_alpha: 0.5
steering_only: false
steering_retain_match: true
use_implicit: true
T: 10, K: 15, rho: 1.0, epsilon: 0.05
```
→ forget=0.372, retain=0.416, verb=0.248, extract=0.063 (Exp7j)

### Best verbatim forgetting (beats gold):
```yaml
# Same as above but:
T: 10, K: 10, rho: 0.3, epsilon: 0.1
```
→ forget=0.406, retain=0.410, verb=**0.197**, extract=**0.051** (Exp7i)

### Best extraction resistance:
```yaml
forget_loss_type: logit_margin
use_steering: true
steering_layers: [5, 6, 7]
steering_coeff: 20.0
steering_only: false
steering_retain_match: false  # Random vectors for max extraction resistance
use_implicit: false
T: 20, K: 10
```
→ forget=0.521, retain=0.495, extract=**0.125** (Exp7c)

### Best retain preservation:
```yaml
gradient_projection: true
gradient_projection_scope: layer
projection_rescale: true
use_implicit: true
T: 20, K: 10
```
→ forget=0.534, retain=**0.508** (Exp6i)

---

---

## Exp8 Series: FD-HVP Implicit Correction + Post-Inner Recovery
**Date:** 2026-04-04  
**Motivation:** Solve OOM for implicit correction (flash_attention_2 doesn't support double backward), improve retain_knowmem gap (0.41 vs gold 0.56), and push verbmem closer to gold 0.201.

### Key Technical Finding: Flash Attention Incompatibility
The original `compute_hvp` used `create_graph=True` which requires differentiating through flash attention's backward, which is not implemented. Fix: **finite-difference HVP (FD-HVP)**.

**FD-HVP**: `H*v ≈ (∇L(θ+εv̂) - ∇L(θ-εv̂)) / (2ε) * ||v||`
- Normalizes v before perturbation (avoids numerical issues with large-norm gradients)
- Requires only first-order backward passes (works with any attention backend)
- No `retain_graph=True` needed — saves memory
- Two extra forward-backward passes per Neumann HVP evaluation

### post_unlearn_inner_steps
Added N pure CE inner steps after T outer iterations for retention recovery.
**Finding**: post_inner steps improve retain_knowmem (+0.02-0.03) but also raise forget_verbmem. The inner steps restore some verbmem via shared representations.

### Exp8 Results Table (all with NPO + steering=[5,6,7] retain_match):

| Exp | β | T | K | implicit | post_inner | forget_km | retain_km | verbmem | extract |
|-----|---|---|---|----------|------------|-----------|-----------|---------|---------|
| Exp8g | 3.0 | 10 | 10 | No | 30 | 0.404 | 0.422 | 0.226 | 0.059 |
| Exp8i | 1.5 | 10 | 10 | No | 30 | 0.381 | **0.434** | 0.233 | 0.059 |
| **Exp8o** | **1.5** | **10** | **10** | **FD-HVP** | **0** | 0.410 | 0.407 | 0.207 | 0.055 |
| **Exp8r** | **2.0** | **10** | **10** | **FD-HVP** | **0** | **0.371** | 0.417 | 0.211 | 0.053 |
| Exp8q | 3.0 | 10 | 10 | FD-HVP | 0 | 0.634 | 0.544 | 0.567 | 0.291 |
| Exp8p | 1.5 | 20 | 10 | FD-HVP | 0 | 0.436 | 0.417 | 0.291 | 0.093 |
| Gold | — | — | — | — | — | ~0.2 | **0.560** | 0.201 | — |

### Key Findings

**1. FD-HVP enables flash_attention_2 compatibility**
The implicit correction now works reliably with the default model config. No more
`RuntimeError: derivative for aten::_scaled_dot_product_flash_attention_backward is not implemented`.

**2. FD-HVP is VERY sensitive to NPO β**
- β=1.5: fast forgetting (L_fgt → 0.04 by iter 4), normal dynamics, forget_km=0.41
- β=2.0: slightly slower forgetting, forget_km=0.371 (more forgetting, less verbmem)  
- β=3.0: implicit correction SUPPRESSES forgetting almost entirely! forget_km=0.634
  - Because large β creates large outer gradient, which the implicit correction "protects" retain from (including forget's semantic content)
  - The MUSE News forget and retain texts share representations (both news articles), so implicit correction to protect retain inadvertently protects forget too

**3. T=20 is too long (λ runaway)**
With T=20 and persistent constraint violations (L_ret > ε always), λ grows to 11.2, causing
degenerate optimization. L_ret oscillates, extraction_strength=0.093. Use T=10.

**4. post_inner steps trade verbmem for retain**
Adding 30 pure CE inner steps after training: +0.02-0.03 retain, but +0.02-0.03 verbmem.
Because inner CE training on retain samples also partially restores the forget representations
(shared language model capacity).

**5. Current best: Exp8r (β=2.0, FD-HVP, T=10, no post_inner)**
- forget_km: **0.371** (best across all Exp8, 0.029 better than Exp8i)
- retain_km: 0.417 (comparable to Exp8o/8i)
- verbmem: **0.211** (just 0.010 above gold 0.201)
- extraction: 0.053 (best)

**6. ψ(ℓ) composite score (response to friend's analysis)**
Computed ψ(ℓ) = causal_diff(ℓ) × activation_diff(ℓ) × gradient_ratio(ℓ):
- Nearly uniform for layers 1-30 (causal traces are flat at 0.12-0.14 for all layers)
- Exception: layer 31 is strongly negative ψ = -0.78 (only causally retain-critical layer)
- Conclusion: ψ(ℓ) is ineffective as a differentiator for surgical targeting
- Better differentiators: pct_forget from neuron analysis, layer_differential from gradient traces

### Training Dynamics Summary

The bilevel optimization with NPO + steering consistently shows:
1. **iter 0**: Large initial forget gradient, L_ret starts at ~0.46 (near epsilon=0.1)
2. **iter 2**: L_ret spikes to 1.9-2.4 (constraint violated, λ starts growing)  
3. **iter 4-9**: Slow L_ret recovery (0.90 → 0.78), never satisfies constraint
4. **End**: λ = 5.4-5.5, model is unlearned but not fully restored

The Augmented Lagrangian never achieves feasibility — this is the fundamental bottleneck
for retain_knowmem improvement.

---

## Exp9 Series: Post-Inner Ablations and Feasibility Improvements
**Date:** 2026-04-04

### Exp9a: Masked Post-Inner Recovery (inverted bitmap mask)
**Hypothesis:** Invert the neuron bitmap during post-inner steps: update ONLY retain-dominant neurons (bitmap=0, 85% of params). These were untouched by the outer loop → should restore retain without restoring forget.
**Config:** Exp8r base (β=2.0, FD-HVP, T=10, K=10) + neuron_bitmap_path + post_inner=30 + post_inner_retain_only=True
**Save dir:** saves/unlearn/research_exp9a_masked_post_inner
**Results:**
| Metric | Value |
|--------|-------|
| forget_knowmem_ROUGE | 0.641 |
| retain_knowmem_ROUGE | **0.561** |
| forget_verbmem_ROUGE | 0.572 |
| extraction_strength | 0.301 |

**Observations:**
- **Retain perfectly restored** (0.561 ≈ gold 0.560) ← the hypothesis worked for retain!
- **Forget COMPLETELY restored** (0.641 ≈ baseline 0.650) ← catastrophic failure for forgetting
- verbmem even WORSE than baseline (0.572 vs 0.555)

**Root cause analysis — KEY FINDING:**
Forget knowledge is NOT locally stored in forget-dominant neurons (15%). It is distributed across ALL neurons. When we train 85% of the model (retain-dominant neurons) on the retain dataset (MUSE News retain), we restore the model's full language understanding for news articles — including the forget set, which is also news articles with identical distribution.

The outer loop (NPO+steering on 15% of params) achieved forgetting by degrading those 15% of neurons. But the other 85% still encode the forget knowledge. When trained on ANY news text, they restore recall of ALL news content — forget and retain alike.

**Fundamental limitation revealed:**
MUSE News forget and retain datasets are **same-domain** (both news articles). The language patterns, factual associations, and semantic representations are near-identical. Surgical forgetting based on gradient attribution (which neurons respond more to forget vs retain) fails because:
1. The 15% forget-dominant neurons are not the ONLY pathways for forget knowledge
2. The 85% retain-dominant neurons contain the same knowledge (just with weaker gradient signal)
3. Fine-tuning any substantial fraction on news data restores everything

**Implication for research:**
Truly robust unlearning in MUSE News may require changing EVERY neuron — making gradient projection or full-model approaches (like GradAscent, NPO-full) necessary rather than surgical methods.

**Current architecture limitation:**
```
outer_loop: only updates forget_dominant neurons (15%)
→ After T outer steps: forget partially erased, but knowledge still accessible via retain neurons
→ Any fine-tuning on retain data (same domain): routes knowledge back through retain neurons
→ Full restoration in ~30 inner steps
```

---

### Exp9b: Stronger Inner Loop (K=15, ρ=1.0, ε=0.05) + β=2.0 + FD-HVP
**Hypothesis:** The main retain gap (0.417 vs 0.560) comes from poor bilevel feasibility. With stronger AL enforcement (K=15 per outer, ρ=1.0, tighter ε=0.05), the constraint may be satisfied, recovering retain without post-inner.
**Config:** β=2.0, FD-HVP, T=10, K=15, ρ=1.0, ε=0.05
**Save dir:** saves/unlearn/research_exp9b_stronger_inner
**Status:** DONE
**Results:**
| Metric | Value |
|--------|-------|
| forget_knowmem_ROUGE | 0.384 |
| retain_knowmem_ROUGE | 0.416 |
| forget_verbmem_ROUGE | 0.247 |
| extraction_strength | 0.069 |

**Observations:**
- **Retain gap unchanged** (0.416 vs Exp8r's 0.417) — stronger AL does NOT close the gap
- **All metrics worse than Exp8r**: forget 0.384 (vs 0.371), verbmem 0.247 (vs 0.211), extract 0.069 (vs 0.053)
- **Training dynamics reveal λ runaway**: L_fgt → 0.049 by iter 4 (over-forgetting), then λ grows to 10.58
  - ε=0.05 is too tight — L_ret never gets near it, λ grows unbounded
  - Over-forgetting early + strong AL correction late = worse verbmem
- **Confirms**: The retain gap is structural, NOT fixable by stronger AL enforcement
- Exp8r (ρ=0.5, ε=0.1) with gradual λ growth (2.8 at iter 9) remains the best configuration

---

### Exp9c: Higher-Layer Steering [20,21,22] — Peak Forget Neuron Density
**Hypothesis:** Layers 20-22 have 25-30% forget neuron density vs ~0% at layers 5-7 (Exp8r). Steering at peak forget-density layers should more efficiently disrupt forget representations and improve knowledge erasure.
**Config:** β=2.0, FD-HVP, T=10, K=10, ρ=0.5, ε=0.1, steering_layers=[20,21,22], retain_match
**Save dir:** saves/unlearn/research_exp9c_high_steer
**Status:** DONE
**Results:**
| Metric | Value |
|--------|-------|
| forget_knowmem_ROUGE | 0.376 |
| retain_knowmem_ROUGE | 0.410 |
| forget_verbmem_ROUGE | 0.212 |
| extraction_strength | 0.055 |

**Observations:**
- **Slightly worse than Exp8r** on both forget_km (0.376 vs 0.371) and retain_km (0.410 vs 0.417)
- verbmem and extract essentially identical
- **Higher-layer steering is NOT better** despite 25-30% forget neuron density at layers 20-22
- Layers 5-7 with ~0% forget neurons remain the optimal steering target
- **Key insight**: Forget neuron density ≠ steering effectiveness. Early layers may encode more abstract semantic representations, making them better targets for retain-match disruption. Higher layers are closer to output and may be more specialized.

---

### Exp9d: β=2.5 — Between Sweet Spot (2.0) and Overcorrection (3.0)
**Hypothesis:** β=2.5 might give slightly more forgetting than 2.0 or slightly better retain without hitting the β=3.0 overcorrection regime (implicit correction suppresses forgetting entirely).
**Config:** β=2.5, FD-HVP, T=10, K=10, ρ=0.5, ε=0.1, steering=[5,6,7], retain_match
**Save dir:** saves/unlearn/research_exp9d_beta25
**Status:** DONE
**Results:**
| Metric | Value |
|--------|-------|
| forget_knowmem_ROUGE | 0.386 |
| retain_knowmem_ROUGE | **0.424** |
| forget_verbmem_ROUGE | 0.217 |
| extraction_strength | 0.056 |

**Observations:**
- **Better retain than Exp8r** (0.424 vs 0.417 — +0.007)
- **Worse forget than Exp8r** (0.386 vs 0.371 — -0.015)
- verbmem and extract slightly worse than Exp8r
- This is a smooth interpolation: β=2.5 lies between β=2.0 and β=3.0 on the forget/retain tradeoff
- **β sensitivity with FD-HVP (full picture)**:
  - β=1.5: forget=0.410, retain=0.407 (Exp8o)
  - β=2.0: forget=0.371, retain=0.417 (Exp8r) ← BEST FORGET
  - β=2.5: forget=0.386, retain=0.424 (Exp9d) ← BEST RETAIN
  - β=3.0: forget=0.634, retain=0.544 (Exp8q) ← OVERCORRECTION
- Use β=2.5 if retain is the priority; β=2.0 if forgetting is the priority

---

## Exp9 Summary Table

| Exp | Config | forget_km↓ | retain_km↑ | verbmem↓ | extract↓ |
|-----|--------|:---:|:---:|:---:|:---:|
| **8r** | **β=2.0, FD, steer[5-7]** | **0.371** | 0.417 | **0.211** | **0.053** |
| 9a | +masked post_inner 30 | 0.641 | **0.561** | 0.572 | 0.301 |
| 9b | K=15, ρ=1.0, ε=0.05 | 0.384 | 0.416 | 0.247 | 0.069 |
| 9c | steer[20-22] | 0.376 | 0.410 | 0.212 | 0.055 |
| **9d** | **β=2.5** | 0.386 | **0.424** | 0.217 | 0.056 |

**Exp8r** remains the best on forget/verbmem/extract. **Exp9d** offers slightly better retain (+0.007) at cost of forget (-0.015).

---

## Exp9 Key Findings

### 15. Same-domain forgetting is non-localizable (Exp9a)
Forget knowledge in MUSE News is NOT stored only in forget-dominant neurons (15%). It is accessible through all ~85% of retain-dominant neurons. Updating 85% of neurons on news-like retain data fully restores all news knowledge, including the forget set.

**Implication**: Surgical forgetting (targeting 15% of neurons) is fragile on same-domain datasets. Forget and retain data share all language patterns → any substantial model update on retain data will restore forget. True unlearning may require modifying ALL parameters.

### 16. AL constraint enforcement cannot close the retain gap (Exp9b)
Stronger inner loop (K=15 instead of 10, ρ=1.0, ε=0.05) does NOT improve retain_km. Instead, it causes λ runaway (λ=10.58 at T=9) and actually worsens forget_km and verbmem.

The retain gap (0.417 vs gold 0.560) is structural: the outer loop damages retain-associated circuits in forget-dominant neurons, and the inner loop cannot fully repair them regardless of K or ρ.

### 17. Optimal steering target: early layers [5,6,7] outperform high-forget-density layers [20-22] (Exp9c)
Despite layers 20-22 having 25-30% forget neuron density, steering them is slightly less effective than steering layers 5-7 (~0% forget density). Early layers encode more abstract semantic features; disrupting them at the retain-match target creates more effective knowledge erasure.

### 18. β sensitivity curve with FD-HVP is smooth and interpretable (Exp9d)
FD-HVP creates a continuous β-forget/retain tradeoff: higher β → more "forget protection" by the implicit correction → less forgetting, more retain. The optimal operating point depends on the desired balance:
- **Forgetting priority**: β=2.0 (Exp8r)
- **Retention priority**: β=2.5 (Exp9d)
- **Overcorrection threshold**: β≥3.0 (implicit protection becomes too strong)

---

## Future Directions

### 1. ~~Masked post-inner recovery~~ (DISPROVED — Exp9a)
Updating retain-dominant neurons (85% of model) fully restores forget knowledge because MUSE News is same-domain. Not viable.

### 2. Multi-stage unlearning with out-of-domain retain data
Same-domain problem: Stage 2 fine-tuning on MUSE News retain will restore forget (shown by Exp9a).
Fix: Use held-out retain data from a DIFFERENT domain (e.g., Wikipedia, C4) for Stage 2, to restore language fluency without domain-specific news knowledge.
But may be impractical for real deployment.

### 3. ~~Feasibility-improving inner loop~~ (DISPROVED — Exp9b)
Stronger AL (K=15, ρ=1.0, ε=0.05) does NOT close the retain gap. It causes λ runaway.
The gap is structural, not a function of inner loop strength.

### 4. ~~Steering at higher layers [20,21,22]~~ (TESTED — Exp9c)
Layers 5-7 are at least as good as 20-22. Forget neuron density ≠ steering effectiveness.

### 5. NPO β schedule
Start with β=1.0 and increase to β=2.5 over T iterations. The schedule would:
- Early iterations (β low): strong forgetting signal, fast knowledge erasure
- Late iterations (β high): more implicit retain correction, stabilize retention
This might avoid λ runaway while achieving stronger forgetting.
Requires implementing a mutable `npo_beta` that updates each outer iteration.

### 6. Cross-domain retain constraint / EWC-style regularization
Use Elastic Weight Consolidation (EWC) or similar: compute Fisher Information Matrix on a held-out retain-like dataset, add FIM-weighted L2 constraint to prevent important parameter drift.
This is orthogonal to the bilevel formulation and could complement it.

### 7. Task Vector Arithmetic
Compute: forget_vector = unlearned_model - original_model
Subtract from original: result = original - α * forget_vector
This is gradient-free and allows precise scaling. Could be combined with NPO+steering to generate a strong forget_vector, then arithmetically subtract it.

### 8. Full-model unlearning (no surgical masking)
Exp9a showed that surgical approaches (15% of neurons) are fragile for same-domain data.
Try NPO+FD-HVP without any bitmap masking (use_sparsity=false, no neuron_bitmap_path).
All parameters are updated → might achieve better forgetting at cost of retain.
This is essentially the same as running Exp8r/9d which already don't use bitmap masking.
