# Surgical SIBL Ablation Report
## MUSE News / Llama-2-7b / DS-BiAL Method

**Last updated:** 2026-04-08
**Data pipeline:** Fixed in commit 655f367 — each outer step sees eff_bs=32 of diverse forget samples (proper epoch iteration). Old experiments used T=10 steps on 1 sample each.

---

## Setup

- Model: `muse-bench/MUSE-News_target` (Llama-2-7b fine-tuned on News)
- Forget: 889 raw samples, chunked to ~800 1024-token sequences → 25 steps/epoch at accum=32
- Retain: 1777 raw samples, randomly sampled per step
- Training: 3 epochs, eff_bs=32 (bs=1, accum=32), 75 total outer steps
- **CRITICAL**: Must pass `trainer.args.num_train_epochs=3` explicitly; default.yaml has 10 epochs → would run 250 steps

## Metric Reference
| Metric | Direction | Gold Target |
|--------|-----------|--------|
| forget_knowmem_ROUGE ↓ | lower = better | 0.328 |
| retain_knowmem_ROUGE ↑ | higher = better | 0.560 |
| forget_verbmem_ROUGE ↓ | lower = better | 0.202 |
| extraction_strength ↓ | lower = better | 0.024 |

---

## Baseline Reference
| Method | fk↓ | rk↑ | fv↓ | ex↓ | Notes |
|--------|-----|-----|-----|-----|-------|
| Target (no unlearn) | 0.644 | 0.555 | 0.579 | 0.295 | Pretrained |
| Gold (retrain) | **0.328** | **0.560** | **0.202** | **0.024** | Oracle |
| GradAscent | 0.003 | 0.008 | 0.049 | 0.008 | Collapse |
| GradDiff | 0.330 | 0.247 | 0.005 | 0.008 | ≈fk gold, bad retain |
| NPO | 0.517 | 0.420 | 0.357 | 0.096 | — |
| SimNPO | 0.584 | 0.470 | 0.258 | 0.052 | — |
| DS-BiAL old (broken data, T=10, 1 sample) | 0.286 | 0.234 | 0.148 | 0.024 | Good forget, bad retain |
| Exp8r (old broken, NPO+steering) | 0.371 | 0.417 | 0.211 | 0.053 | Best balanced old result |

---

## KEY FINDINGS (accumulated 2026-04-08)

### Finding 1: ε constraint was permanently violated at ε=0.1
- L_ret at initialization ≈ 0.9; ε=0.1 → r = +0.8 always violated
- λ grows unboundedly (31+ by step 75) → outer step 95% retain, 5% forget
- **Fix: ε=0.8** (retain only penalized if it exceeds 0.8 above baseline)

### Finding 2: logit_margin has a hard ceiling at fk≈0.52
- Confirmed across A1a, C1, C2, C4 (all ε variants, K variants, post_inner variants)
- logit_margin = (max_logit - mean_logit) reduction — doesn't reduce ROUGE on memorized text
- **Cannot beat gold with logit_margin. Need NPO or activation steering.**

### Finding 3: NPO collapses at ε=0.1, works at ε=0.8
- At ε=0.1: λ-runaway forces NPO to L_fgt→0 at step 3 (DPO ratio saturates)
- At ε=0.8: NPO is stable (old Exp8r: fk=0.286 on broken data)
- **NPO+steering IS viable with relaxed epsilon**

### Finding 4: Gradient projection kills forgetting for same-domain data
- MUSE News = same-domain (85% of neurons encode both forget+retain)
- Projecting forget gradient orthogonal to retain removes most of signal
- **Never use gradient_projection for MUSE News**

### Finding 5: post_inner_steps hurts when outer forgetting is insufficient
- C2 (post_inner=50, logit_margin): fk went from 0.523→0.540 (WORSE)
- B5 (post_inner=50, logit_margin): fk=0.581 (WORSE)
- Root cause: logit_margin barely forgot → post_inner undoes it
- **Hypothesis: post_inner should work with NPO (real forgetting). Testing in E2.**

### Finding 6: Core challenge is forget-retain tradeoff
- Old DS-BiAL: fk=0.286 (beats gold!) but rk=0.234 (way below 0.560 target)
- GradDiff: fk=0.330 (≈gold) but rk=0.247 (bad)
- **Nobody has achieved both fk≤0.328 AND rk≥0.560 simultaneously**
- Goal: NPO achieves good forget, then recover retain via masked updates/post-processing

---

## Neuron Trace Summary
- 17% of neurons are forget-dominant (ratio > 1.0)
- 14.9% are retain-dominant
- ~68% neutral (ratio ≈ 1.0)
- Layer 31 is causally retain-critical (from causal tracing)
- `forget_neuron_bitmap.pt` available at: `trace_analysis/figures/traces/analysis/`

---

## All Experiment Results

| Exp | Config | fk↓ | rk↑ | fv↓ | ex↓ | Status |
|-----|--------|-----|-----|-----|-----|--------|
| A1a | logit_margin, ε=0.1 | 0.524 | 0.500 | 0.527 | 0.281 | DONE (ceiling) |
| B5 | logit_margin + post_inner=50, ε=0.1 | 0.581 | 0.503 | 0.543 | 0.285 | DONE (worse) |
| C1 | logit_margin, ε=0.8, K=5, η=2e-4 | 0.523 | 0.461 | 0.457 | 0.213 | DONE (ceiling confirmed) |
| C2 | C1 + post_inner=50 | 0.540 | 0.467 | 0.481 | 0.211 | DONE (post_inner hurts) |
| C4 | logit_margin, ε=0.5, K=3, η=3e-4, post_inner=100 | ⏳ | ⏳ | ⏳ | ⏳ | EVAL RUNNING |
| D1 | NPO β=2.0, ε=0.8, K=5 | ⏳ | ⏳ | ⏳ | ⏳ | TRAINING |
| D2 | NPO β=2.0 + steering[5,6,7] retain_match, ε=0.8 | ⏳ | ⏳ | ⏳ | ⏳ | QUEUED |
| D3 | logit_margin + steering[5,6,7] retain_match, ε=0.8 | ⏳ | ⏳ | ⏳ | ⏳ | QUEUED |
| E1 | NPO + steering + neuron mask (forget-dominant only) | ⏳ | ⏳ | ⏳ | ⏳ | QUEUED |
| E2 | NPO + steering + post_inner=200 (retain recovery) | ⏳ | ⏳ | ⏳ | ⏳ | QUEUED |
| E3 | NPO + steering + ε=0.5, K=10 (tighter+more inner) | ⏳ | ⏳ | ⏳ | ⏳ | QUEUED |
| E4 | NPO + steering + retain_protection_layers=[31] | ⏳ | ⏳ | ⏳ | ⏳ | QUEUED |

---

## Series D: NPO-Based Forgetting (RUNNING 2026-04-08)

**Motivation:** logit_margin has hard ceiling. Old Exp8r (NPO+steering) got fk=0.286 on broken data.
D series reproduces/extends this with correct data pipeline.

| Exp | Key Change | Hypothesis |
|-----|-----------|-----------|
| D1 | NPO β=2.0, ε=0.8, K=5, no steering | Baseline: does NPO forget well without steering? |
| D2 | D1 + steering[5,6,7] retain_match | Exp8r repro: dual-space forget (weight+activation) |
| D3 | logit_margin + steering[5,6,7], ε=0.8 | Control: does steering alone (no NPO) help? |

**Expected:** D2 achieves fk≈0.286-0.371. Retain may be bad (0.23-0.42). D→E series for retain fix.

---

## Series E: Retain Recovery Strategies (QUEUED 2026-04-08)

**Motivation:** NPO gets good forget but bad retain. Need surgical approach.

| Exp | Key Change | Hypothesis |
|-----|-----------|-----------|
| E1 | D2 + neuron mask (forget-dominant only, bitmap.pt) | Only update 17% of neurons → retain-neutral neurons untouched |
| E2 | D2 + post_inner=200, retain_only=true | NPO achieves real forgetting; CE retain steps should recover rk without reversing fk |
| E3 | D2 + ε=0.5, K=10 | Tighter constraint with more inner steps → better retain enforcement per outer step |
| E4 | D2 + retain_protection_layers=[31] | Layer 31 is causally retain-critical; protecting it should improve rk |

**Key insight on E2:** Post_inner failed with logit_margin because forgetting was insufficient (fk still 0.52).
With NPO achieving fk≈0.286 before post_inner, CE retain recovery should not reverse the forgetting
(forget data never re-seen; generalization across domains is limited).

---

## If E Series Succeeds
If best E achieves fk<0.328 AND rk>0.45 (partial), design F series:
- F1: Best E config + stronger steering (steering_coeff=40 or more layers [3,4,5,6,7])
- F2: Best E config + both mask AND layer31 protection
- F3: Staged training: NPO 1 epoch → tighten ε=0.3 for 2 epochs (ε schedule)

---

## Dropped Approaches
- **Gradient projection** (B1/B2): kills forget for same-domain data
- **PDU loss** (B6): unstable, L_ret explodes to 8.6
- **logit_margin variants** (A1, B5, C1, C2, C4): hard ceiling at fk≈0.52

---

## Runner Script
```bash
# All D+E experiments run via:
bash scripts/run_D_E_series.sh
# Progress: /tmp/ablation_DE_progress.log
```
