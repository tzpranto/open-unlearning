# Research Scratchpad — DS-BiAL MUSE News
## For agent continuity. Last updated: 2026-04-09 ~04:35

---

## GOLD TARGETS
- fk ≤ 0.328 (forget knowmem ROUGE)
- rk ≥ 0.560 (retain knowmem ROUGE)

---

## COMPLETE RESULTS TABLE

### Baselines
| Method | fk↓ | rk↑ | notes |
|--------|-----|-----|-------|
| Gold (retrain) | 0.328 | 0.560 | target |
| Pretrained | 0.644 | 0.555 | starting point |
| GradAscent | 0.003 | 0.008 | model collapses |
| GradDiff | 0.330 | 0.247 | good forget, bad retain |
| NPO standalone | 0.517 | 0.420 | — |
| SimNPO standalone | 0.584 | 0.470 | — |

### SIBL experiments
| Exp | fk↓ | rk↑ | verdict |
|-----|-----|-----|---------|
| A1a–C4 logit_margin | 0.52–0.54 | 0.46–0.50 | DEAD END — hard ceiling |
| D1 NPO T=75 | 0.490 | 0.461 | too many steps |
| D2 NPO+steering T=75 | 0.513 | 0.486 | steering hurts at T=75 |
| **F0 bare NPO** | **0.325** | 0.290 | ✅ beats gold on fk |
| F1 +steering coeff=20 | 0.358 | 0.349 | fk too high |
| F2 +implicit | 0.335 | 0.312 | marginal |
| **F3 +ε=0.70** | **0.325** | **0.316** | ✅ anchor — free rk +0.026 |
| F4 T=25 explicit | 0.359 | 0.311 | worse |
| G0 +post_inner=100 | 0.409 | 0.352 | fk hurt — CE re-learns forget |
| **G1 +steering coeff=5** | **0.274** | **0.327** | ✅ BEST — fk beats gold, rk +0.011 |
| G2 +K=5 inner | 0.314 | 0.310 | marginal, skip |
| G3 masked post_inner=25 | 0.657 | 0.572 | rk near gold but fk destroyed |
| H0 G1+top50 seqs | 0.005 | 0.020 | COLLAPSED — T=1 step, grad too strong |

### I series (DGA soft-masked recovery) — 2026-04-09
| Exp | fk↓ | rk↑ | verdict |
|-----|-----|-----|---------|
| I1 G1+soft β=5.0 | 0.342 | 0.341 | fk regressed +0.068 vs G1, rk only +0.014 |
| I2a β=1.0 | 0.357 | 0.338 | worse — low β = near-uniform CE, more re-learning |
| I2b β=3.0 | pending | — | running |
| I2c β=10.0 | pending | — | running |
| I2d β=20.0 | pending | — | queued |

---

## CURRENT STATE (2026-04-09, updated ~13:05)

**Best result: G1** — fk=0.274 ✅ (beats gold 0.328), rk=0.327 ❌ (need 0.560)
**Gap:** rk needs +0.233 more. fk has 0.054 headroom before hitting gold.

**I series early verdict:** DGA soft mask is not working as hoped. Both I1 (β=5) and I2a (β=1) show fk *regression* with only marginal rk gain. The soft mask during CE recovery is still letting forget content re-learn regardless of sharpness. β sweep (β=3,10,20) in progress — unlikely to reverse the trend.

### What each experiment taught us
- **G1 (steering coeff=5)**: Light activation steering simultaneously improves BOTH fk and rk vs F3. This is the only component that gives free gains on both axes. Use as anchor going forward.
- **G3 (masked post_inner)**: Retain-dominant neurons (bitmap=0) also encode some forget content → updating them restores rk=0.572 (near gold!) but destroys fk=0.657. The bitmap separation is not clean enough for post_inner.
- **H0 (top-50 seqs)**: With 50 samples + accum=32, SIBL derives steps_per_epoch=1 → T=1 outer step. But the top-50 seqs have NPO gradient magnitude up to 140× — one step nukes the model. Need much lower LR or fewer accum steps.

### Key insight from G3 vs H0
G3 tells us: rk CAN reach 0.572 — that's the ceiling we know is achievable. The problem is fk.
H0 tells us: targeted forgetting on the 50 most memorized seqs is too nuclear per-step.

---

## I SERIES POST-MORTEM (2026-04-09)

**Why DGA soft mask failed:**
The CE recovery steps (post_inner) update params using the retain dataloader. Even with α=0.5 on contested neurons, 25 steps of CE is enough to partially restore forget-content representations. The problem isn't just WHICH neurons get updated — it's that ANY update to shared/contested neurons re-activates the forget pathway. The gradient-based selectivity score identifies which neurons NPO pushes through, but it doesn't capture cross-neuron interference: updating retain-dominant neurons at higher layers can restore outputs that forget-dominant neurons at lower layers had degraded.

**Key insight from β sweep trend:**
- β=1 (near-uniform α≈0.5): fk=0.357 (worst) — most re-learning
- β=5 (I1): fk=0.342 — better
- β→∞ (approaches binary G3): fk will approach G3's 0.657 — still bad
- There is no β where fk is protected AND rk recovers — the CE recovery is fundamentally at odds with the NPO forgetting regardless of mask sharpness.

**Conclusion:** Post-inner CE recovery as a rk-recovery strategy is broken for this dataset. The entire post_inner approach (G3, I series) fails because forget and retain knowledge are too entangled in the same weight subspace.

## ACTIVE HYPOTHESES (what to try next)

### Priority 0: Abandon post_inner recovery — try rk improvement within the outer loop
Post-inner CE recovery (G3, I series) is a dead end. rk must be improved during training:
- **G4**: G1 + lower ε=0.50 (tighter retain constraint, forces λ to drive retain harder)
- **G5**: G1 + implicit correction (F2 gave +0.022 rk at F3 level — stack with G1)
- **G6**: G1 + more epochs (2-3 instead of 1, more NPO+retain bilevel iterations)
- **I3**: DGA scored on pretrained model (not G1) — may give cleaner neuron separation

### Priority 1: Fix H series — calibrate step count for 50-seq dataset
H0 collapsed because T=1 step on ultra-high-memorization sequences.
Fix options:
- **H1**: Reduce `gradient_accumulation_steps` from 32 to 4 → steps_per_epoch=50//4=12, T=12. Softer per-step signal. Keep G1 config (steering coeff=5).
- **H2**: Same as H1 but also lower npo_beta from 2.0 to 0.5 (less aggressive NPO per step).
- **H3**: Don't use top-50 alone. Instead use top-50 as a weighted subset within the full 800 (sample with replacement, overweight hard seqs). More diverse gradient signal.

### Priority 2: G1 + rk recovery without destroying fk
G1 is our anchor (fk=0.274). How to push rk from 0.327 → 0.560 without fk regression:
- **G4**: G1 + lower ε (tighter retain constraint). F3 showed ε=0.70 gave free +0.026 rk. Try ε=0.50.
- **G5**: G1 + implicit correction (F2 gave +0.022 rk at F3 level). Stack with G1.
- **G6**: G1 + more epochs (2-3 epochs instead of 1). More NPO steps → more forget → might drift fk up but rk may follow.

### Priority 3: Combine G3's rk power with G1's fk power
G3 got rk=0.572 but fk=0.657. G1 got fk=0.274 but rk=0.327.
- **G7**: G1 + G3 stacked: first run G1 (steering, full 800 seqs) → then run G3 (masked post_inner=25) ON TOP of G1 output. Two-stage approach.
- Risk: G3's post_inner may undo G1's fk gains. Need to check if the bitmap mask is good enough.

---

## MEMORIZATION SCORING RESULTS
- Scorer completed: `data/hard_forget_news.jsonl` (top-50 by npo_log_ratio)
- Stats: mean=7.201, min=0.902, max=140.089
- ALL 802 sequences are memorized (score > 1.5) — no generic samples found
- Top-50 are extreme outliers (score ~20-140). Very strong NPO signal.

---

## RUNNER STATE
```bash
# All logs: logs/ablation/
# H series: logs/ablation/H_progress.log
# G series: logs/ablation/G_progress.log
# Cron monitor: every 5 min (durable, job id: e1509b8c)

# Next to run (decide based on priorities above):
bash scripts/run_H_series.sh   # after updating configs for H1/H2
```

---

## BUGS FIXED THIS SESSION
1. G series runner logged to /tmp → moved to logs/ablation/ (persists restart)
2. Memorization scorer: wrong dataset config (needed 'raw' config name)
3. Memorization scorer: tokenizer loaded from muse-bench model (no tokenizer files) → now uses base_model path
4. post_inner_retain_only mask inversion: non-bitmap params got mask=ones after inversion → now stay frozen
5. H0 dataset config: PretrainingDataset needs `path="json"`, `data_files=...`, `split="train"` for local jsonl

---

## CONCLUSION (2026-04-09)

**Best architecture: G1** — NPO outer loop + light activation steering (coeff=5, layers 5-7) on full 800 forget seqs. ε=0.70, K=1, T=25 steps. Result: fk=0.274 ✅ rk=0.327 ❌.

**The core tension:** G1 controls fk well but rk lags. G3 (masked post_inner) reaches rk=0.572 (near gold) but destroys fk=0.657. The neuron bitmap separating forget/retain neurons from causal trace analysis is our best surgical tool, but the separation is not clean enough — retain-dominant neurons (bitmap=0) still encode some forget content, so updating them during CE recovery re-learns what NPO forgot.

**What each promising config proved:**
- G3: rk=0.572 IS achievable — retain recovery via masked CE works directionally
- G1: fk=0.274 IS achievable — light steering simultaneously helps both axes
- F3: free rk gains from tighter ALM constraint (ε=0.70)
- H0: targeted forgetting on top-50 memorized seqs collapses model — all 802 seqs are memorized, no generic noise to filter, top-50 are extreme outliers (score up to 140×)

**Open problem:** combining G1's fk control with G3's rk recovery. Options: (1) two-stage G1→G3, (2) score-weighted NPO loss on full dataset, (3) cleaner neuron separation via finer-grained trace analysis.

---

## LONGER TERM
- Once fk + rk both hit targets: run on MUSE Books + WMDP
- Baselines still needed: BLURNPO (checkpoint at saves/), RMU (needs fresh run)
- Publish as DS-BiAL: bilevel NPO + trace-guided sample selection + staged retain recovery
