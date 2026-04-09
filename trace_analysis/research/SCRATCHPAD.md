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

---

## CURRENT STATE (2026-04-09)

**Best result: G1** — fk=0.274 ✅ (beats gold 0.328), rk=0.327 ❌ (need 0.560)
**Gap:** rk needs +0.233 more. fk has 0.054 headroom before hitting gold.

### What each experiment taught us
- **G1 (steering coeff=5)**: Light activation steering simultaneously improves BOTH fk and rk vs F3. This is the only component that gives free gains on both axes. Use as anchor going forward.
- **G3 (masked post_inner)**: Retain-dominant neurons (bitmap=0) also encode some forget content → updating them restores rk=0.572 (near gold!) but destroys fk=0.657. The bitmap separation is not clean enough for post_inner.
- **H0 (top-50 seqs)**: With 50 samples + accum=32, SIBL derives steps_per_epoch=1 → T=1 outer step. But the top-50 seqs have NPO gradient magnitude up to 140× — one step nukes the model. Need much lower LR or fewer accum steps.

### Key insight from G3 vs H0
G3 tells us: rk CAN reach 0.572 — that's the ceiling we know is achievable. The problem is fk.
H0 tells us: targeted forgetting on the 50 most memorized seqs is too nuclear per-step.

---

## ACTIVE HYPOTHESES (what to try next)

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

## LONGER TERM
- Once fk + rk both hit targets: run on MUSE Books + WMDP
- Baselines still needed: BLURNPO (checkpoint at saves/), RMU (needs fresh run)
- Publish as DS-BiAL: bilevel NPO + trace-guided sample selection + staged retain recovery
