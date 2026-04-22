# MUSE News — LoRA-BiAL Experiment Log

**Model:** Llama-2-7b-hf | **Data:** News | **Goal:** Beat PDU HM=0.554

HM = harmonic mean of (1-forget_knowmem, 1-verbmem, retain_knowmem). Gold retrain HM=0.660.

**Base config:** `experiment=unlearn/muse/lora_bial_news`, LoRA r=16/α=32, inner K=3 SGD lr=2e-4, outer Adam lr=3e-5, per_device_bs=2, grad_accum=8 (eff_bs=16), gradient checkpointing.

## Results Summary

| ID | Loss | Key Config | fk↓ | vm↓ | rk↑ | ex↓ | HM↑ | Notes |
|----|------|-----------|------|------|------|------|------|-------|
| 02 | clamped_entropy | ε=0.70, λ_init=1.0, 4ep | 0.577 | 0.490 | 0.542 | — | 0.486 | loose ε, weak forget |
| 03 | clamped_entropy | ε=1.00, λ_init=1.0, 4ep | 0.537 | 0.501 | 0.523 | — | 0.494 | |
| 04 | clamped_entropy | auto-ε 3×, ρ=0.05, λ_max=10, 4ep | 0.528 | 0.388 | 0.501 | 0.130 | 0.522 | best so far |
| 05 | clamped_entropy+PCGrad | auto-ε 3×, ρ=0.05, λ_max=10, 4ep | — | — | — | — | — | identical to exp_04, PCGrad no effect |
| 06 | logit_margin | ε=0.15, λ_init=1.0, 4ep | 0.593 | 0.508 | 0.512 | 0.246 | — | ε too tight for News (baseline L_ret=0.79), λ→10.15, forgetting suppressed |

**Target:** PDU HM=0.554 (fk=0.489, vm=0.042, rk=0.414).

---

## Key Diagnostics

### Problem: Clamped entropy doesn't reduce verbmem on News
- Training L_fgt converges (6.6→0.6) but eval vm=0.388 (terrible)
- Clamped entropy maximizes distribution entropy but doesn't disrupt argmax — model still generates memorized text via greedy decoding
- On Books this worked because book text is unique; on News, factual knowledge is redundantly encoded
- PDU uses logit_margin `(max-mean)²` which directly attacks argmax → vm=0.042

### Problem: PCGrad is a no-op
- Gradient projection doesn't help because the damage to retain comes from applying g_fgt itself (parameter-space coupling), not from g_ret opposing g_fgt
- Trajectories were numerically identical with and without PCGrad

### Problem: ε=0.15 is impossible for News
- Books baseline L_ret=0.052, so ε=0.15 gives 3× headroom
- News baseline L_ret=0.794, so ε=0.15 is violated from step 0
- λ ratchets to maximum, suppressing all forgetting

---

## Next Experiments (Bucket List)

### Priority 1: τ sweep on MUSE News
- Current τ=0.7 clamps gradient to zero once token entropy reaches 70% of H_max
- Entity tokens (names, dates) carrying factual knowledge may need pushing to 90-95% of H_max
- Try τ=0.85, 0.90, 0.95 with auto-ε and same ALM config as exp_04
- **Hypothesis:** Higher τ forces harder forgetting on discriminatory tokens, reducing both fk and vm

### Priority 2: Entropy-weighted token loss
- Weight per-token loss by current entropy H — high-entropy tokens (entities, facts) get amplified
- Low-entropy tokens ("the", "was") already uncertain, get downweighted
- Implementation: ~3 lines in compute_clamped_entropy_loss
- **Hypothesis:** Focuses gradient on knowledge-bearing tokens instead of wasting it on common words
- Related work: ETW (Koh et al., ACL 2026) validates entropy weighting on TOFU, but not on MUSE and not in bilevel setting

### Priority 3: logit_margin with auto-ε on News
- logit_margin directly attacks argmax (what verbmem measures)
- Need auto-ε (multiplier=1.5-3.0) since ε=0.15 is impossible for News
- Our ALM should give better retain than PDU's simple approach

---

## TOFU 1B Validation Experiments

Running on Llama-3.2-1B-Instruct, TOFU forget01, to validate recipes quickly before 7B MUSE runs.

| ID | Config | forget_quality | model_utility | Notes |
|----|--------|---------------|--------------|-------|
| exp_01 | clamped_entropy, ε=0.15, eta_theta=3e-5, 100 steps | 0.029 | 0.601 | over-forgot (L_fgt→0.04), MIA detectable |
| exp_02 | clamped_entropy, ε=0.15, eta_theta=1e-5, 100 steps | ⏳ | ⏳ | gentler LR, checkpoints at 25/50/75/100 |
