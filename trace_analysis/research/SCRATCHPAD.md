# Research Scratchpad — DS-BiAL MUSE News
## For agent continuity. Last updated: 2026-04-10 ~02:30 (R series dead end confirmed + S series running)

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
| H0 G1+top50 seqs | 0.005 | 0.020 | COLLAPSED — confirmed again after rerun |

### I series (DGA soft-masked recovery) — 2026-04-09 ✅ COMPLETE
| Exp | fk↓ | rk↑ | verdict |
|-----|-----|-----|---------|
| I1 G1+soft β=5.0 | 0.342 | 0.341 | fk regressed +0.068 vs G1, rk only +0.014 |
| I2a β=1.0 | 0.357 | 0.338 | worse — low β = near-uniform CE, more re-learning |
| I2b β=3.0 | 0.359 | **0.355** | best rk of sweep but fk still regressed +0.085 |
| I2c β=10.0 | 0.364 | 0.318 | worse — high β hurts both axes |
| I2d β=20.0 | 0.357 | 0.330 | near β=1 in quality — no sweet spot exists |

---

## CURRENT STATE (2026-04-09, updated ~20:45)

**Best result: G1** — fk=0.274 ✅ (beats gold 0.328), rk=0.327 ❌ (need 0.560)
**Gap:** rk needs +0.233 more. fk has 0.054 headroom before hitting gold.
**Note:** Gold targets are the benchmark but good forgetting + reasonable retain is also useful.

**I series final verdict:** DGA soft mask is a dead end. Full β sweep (1,3,5,10,20) confirms: every β shows fk regression (+0.068 to +0.090 vs G1) with negligible rk gain (+0.003 to +0.028). Best rk was β=3 (rk=0.355) but fk=0.359 — still far from gold on both axes. No sweet spot exists.

**H series also DONE:** H0 (top-50 memorized seqs, G1 config) fk=0.005 ✅ rk=0.020 ❌ — collapsed again. Extreme gradient signal (scores up to 140×) destroys model in 1 epoch even with G1 steering.

### K/G5 series (2026-04-09) ✅ COMPLETE
| Exp | fk↓ | rk↑ | verdict |
|-----|-----|-----|---------|
| K2 step 5 | 0.246 | 0.205 | fk GOLD early but rk damaged |
| K2 step 10 | 0.275 | 0.287 | improving |
| K2 step 15 | 0.318 | 0.324 | sweet spot — fk GOLD, best Pareto |
| K2 step 20 | 0.312 | 0.324 | plateaued |
| K2 step 25 | 0.309 | 0.326 | G1 final (same as full run) |
| G5 (G1+implicit) | 0.429 | 0.399 | implicit +0.072 rk but +0.155 fk — overcorrects |
| K1a (G1+projection) | 0.356 | 0.351 | projection +0.024 rk but +0.082 fk — same pattern |

### N series (inverted inner mask) — 2026-04-09 ✅ PARTIAL (OOM on N0-N3)
| Exp | fk↓ | rk↑ | verdict |
|-----|-----|-----|---------|
| N4 full outer+inv inner | 0.344 | 0.343 | inner too weak — ALM penalty is a no-op |
| N0-N3 | OOM | — | bitmap outer + inverted inner hits 96GB limit |

### P series (ALM penalty fix) — 2026-04-10 ✅ COMPLETE
| Exp | fk↓ | rk↑ | verdict |
|-----|-----|-----|---------|
| P0 λ=5 ρ=1 | 0.524 | **0.465** | Best rk without post-inner. ALM works. |
| P1 λ=2 ρ=1 | 0.535 | 0.464 | ≈P0 — ρ=1 catches up fast |
| P2 λ=5 ρ=1 K=2 | 0.548 | 0.447 | Stronger inner hurt both |
| P3 λ=0 ρ=5 | 0.000 | 0.000 | COLLAPSED — explosive λ divergence |

### Q series (two-phase ALM) — 2026-04-10 ✅ COMPLETE
| Exp | fk↓ | rk↑ | verdict |
|-----|-----|-----|---------|
| Q0 boost@10 | 0.466 | 0.423 | NPO-heavy phase1 → strong ALM phase2. Same frontier. |
| Q1 boost@5 | 0.519 | 0.451 | ≈P0 — 5 unprotected steps ≈ λ=5 from start |
| Q2 boost@10+K=2 | 0.392 | 0.341 | Stronger inner helps fk, not rk |

### R series (KL outer retain) — 2026-04-10 ✅ DEAD END
| Exp | fk↓ | rk↑ | verdict |
|-----|-----|-----|---------|
| R0 ALM+KL | 0.003 | 0.008 | COLLAPSED — KL scale mismatch with ε |
| R1 fixed KL λ=0.1 | 0.255 | 0.234 | BELOW CE frontier — KL doesn't help |

### M series (KL-anchored bilevel) — 2026-04-09 ✅ COMPLETE
| Exp | fk↓ | rk↑ | verdict |
|-----|-----|-----|---------|
| M4 KL post-inner | 0.650 | 0.549 | fk DESTROYED — same as G3 CE (0.657). **KL doesn't protect fk** |
| M0 KL inner only | 0.398 | 0.362 | fk HURT +0.124, rk only +0.034 — mild benefit |
| M1 KL inner+post | 0.662 | **0.563** | rk GOLD but fk destroyed. Confirms: post-inner kills fk regardless of loss |
| M2 KL+implicit | 0.658 | 0.556 | Bitmap restricts NPO too much → near-pretrained fk |
| M3 full pipeline | 0.655 | 0.553 | Same pattern: bitmap kills forget, retain preserved |

**K2 diagnosis:** rk plateaus at ~0.326 at EVERY step — not a step-count problem. The damage is baked into the NPO gradient direction from step 1.
**G5/K1a pattern:** Both implicit and projection improve rk marginally (+0.024 to +0.072) but always at the cost of fk regression. The gradient corrections are not selective enough — they protect retain but at the expense of forgetting.

### L series (2026-04-09) ✅ COMPLETE
| Exp | fk↓ | rk↑ | verdict |
|-----|-----|-----|---------|
| L3 G1×2 epochs | 0.441 | 0.416 | fk HURT +0.167, rk +0.089 — more training regresses fk fast |
| L4 G1+ε=0.50 | 0.374 | 0.343 | fk HURT +0.099, rk only +0.016 — tighter constraint costs fk, negligible rk gain |
| L1 proj+ε=0.50 | 0.396 | 0.381 | fk HURT +0.122, rk +0.054 — best rk of L series but still HURT on fk |

**L series verdict:** All three stacking configurations hurt fk. The common pattern: anything that increases retain pressure (more epochs, tighter ε, projection+ε) causes fk regression. The fk/rk tradeoff is fundamental — G1 is already at a Pareto frontier. No stacking approach moves the frontier; it only slides along it.

**Key observation from L4 (ε=0.50):** λ reached ~4.0 vs G1's ~2.7 — the constraint was biting significantly harder. Yet rk only improved by +0.016. This confirms the rk ceiling is not a constraint-pressure problem; the retain loss itself is not the bottleneck. The model is hitting a geometric limit in weight space where NPO gradient direction and retain gradient direction are fundamentally antagonistic.

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

### Dead ends confirmed (all tested, all hurt fk)
- ~~G4/L4~~ (ε=0.50): rk +0.016 only, fk HURT +0.099
- ~~G5~~ (implicit): rk +0.072 but fk HURT +0.155
- ~~G6/L3~~ (2 epochs): rk +0.089 but fk HURT +0.167
- ~~K1a~~ (projection): rk +0.024 but fk HURT +0.082
- ~~L1~~ (proj+ε=0.50): rk +0.054 but fk HURT +0.122
- ~~I series~~ (DGA soft mask): every β shows fk regression, no sweet spot
- ~~H0~~ (top-50 seqs): model collapses

### Remaining options (J series)
**Recommended order:**
- **J3** ~~= L3~~ (already done as 2 epochs — DEAD END)
- **J2**: Score-weighted NPO on full 800 seqs (memorization-proportional gradient, ~20 LOC)
- **J1**: Post-training KL distillation from retrain model (~25 steps, NOT CE)
- **J4**: ε curriculum (loose→tight, steps 1-12 at ε=0.90, steps 13-25 at ε=0.50)
- **J5**: Replace retain CE with KL(θ‖θ_pretrained) in outer loop (most principled, needs SIBL code change)

**Core insight:** G1 is the Pareto frontier. Every tested intervention slides along fk↑/rk↑ tradeoff but doesn't move the frontier. J1/J5 (KL-based rather than CE-based retain signal) are the most principled remaining options since CE re-learns forget via shared weights.

---

## J SERIES — STACKING IDEAS (2026-04-09)

Core problem: rk gap is +0.233. CE recovery is dead. Need rk improvement without touching the NPO outer loop.

### J1: Post-training retain distillation (NOT CE)
After G1 completes, run ~25 steps of **KL(θ || θ_retrain)** on retain data instead of CE.
- θ_retrain = muse-bench/MUSE-News_retrain (the gold retrain checkpoint, already used for eval)
- KL target naturally has low forget knowledge → can't re-learn forget patterns from target distribution
- Unlike CE (which targets observed tokens and can reconstruct forget via shared weights), KL distillation is bounded by what the teacher knows
- Key config: post_inner loss = KL vs retrain model, retain data only, ~25 steps
- Risk: retrain model may not be a perfect teacher on the specific retain1 split

### J2: Score-weighted NPO (memorization-proportional gradient)
Weight each forget sequence in the NPO loss by its memorization score (already computed: `data/hard_forget_news.jsonl`).
- w_i = score_i / mean(scores) — normalizes total gradient magnitude
- Top-50 seqs (score ~20-140×) get proportionally more NPO pressure
- Unlike H0 (subsetting to top-50, causing collapse): full 800-seq dataset used, gradient diversity preserved
- Implementation: custom DataLoader with sample weights, or loss *= w_i per batch
- Hypothesis: harder seqs push fk lower with same T=25 budget → fk headroom frees λ to drive rk

### J3: G1 × 2 epochs (ALM accumulation)
Simply run G1 for 2 epochs (50 outer steps instead of 25).
- λ accumulates over more steps → retain constraint pressure compounds
- fk has 0.054 headroom before hitting gold → can afford some drift
- Cheapest experiment, no code changes, just `num_train_epochs=2`
- Risk: fk may drift above 0.328 before rk recovers enough; monitor per-step

### J4: ε curriculum (loose → tight constraint)
Run G1 with ε annealed: 0.90 for steps 1–12, then 0.50 for steps 13–25.
- Early steps: loose constraint lets NPO drive fk down fast (build headroom)
- Late steps: tight constraint forces λ to recover rk aggressively
- F3 showed ε=0.70 gave free +0.026 rk over F2 (ε=0.70 vs implicit). Tighter ε should compound.
- Implementation: pass ε schedule as list or step-based override in SIBL

### J5: NPO + retain KL in outer loop (replace CE constraint)
Replace the retain CE loss in the ALM constraint with KL(θ || θ_pretrained) on retain data.
- L_ret = KL(θ || θ_pretrained) instead of CE(θ, retain_tokens)
- Constraint r = L_ret - ε, dual update as before
- Pretrained model is a stable anchor; KL penalizes drift from pretrain on retain distribution
- Naturally resists forget re-learning (pretrained model is not great on forget data either)
- Strongest principled fix but needs code changes in SIBL retain loss computation

**Recommended order:** J3 (free, 1 config change) → J2 (score weighting, ~20 LOC) → J1 (distill post-training) → J4 (ε curriculum) → J5 (KL constraint, most invasive)

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

---

## M SERIES — KL-Anchored Bilevel Optimization (2026-04-09)

### Root Cause Analysis
Every tested intervention (implicit, projection, tighter ε, more epochs, CE post-inner, DGA soft mask) slides along the fk↑/rk↑ Pareto frontier but never moves it. The root cause:

1. **Inner loop CE re-learns forget**: The inner loop minimizes CE on retain data. Updating shared weights to predict retain tokens also improves forget prediction through weight entanglement. 64% of neurons are contested (ratio 0.8–1.5).
2. **Post-inner CE re-learns forget**: G3's CE recovery (rk=0.572 near gold!) proves rk IS achievable — but CE on retain neurons re-activates forget pathways through cross-neuron interference.
3. **Implicit overcorrects**: G5's implicit correction weakened NPO signal on forget-dominant neurons, even though the inner loop barely touches them (mask gates inner updates). The correction is wasted on neurons with negligible inner-loop interaction.

### Key Insight: KL Distillation Cannot Re-Learn Forget
CE(θ, retain_tokens) = -log p(x_retain | θ) → directly maximizes token likelihood → can reconstruct forget patterns through shared weights.

KL(p_pretrained || p_θ) on retain data → pushes θ's distribution toward pretrained's distribution ON RETAIN DATA. The pretrained model has rk=0.555 (near gold). Its output distribution on retain data is about retain knowledge. KL is bounded by the teacher's knowledge — if the teacher doesn't emphasize forget patterns on retain data, KL can't teach them.

This is fundamentally different from CE: KL matches distributions (smooth signal), CE fits individual tokens (overfits to patterns).

### Novel Contributions
1. **KL-anchored inner loop**: Replace CE with KL(pretrained || model) in bilevel inner loop
2. **Trace-guided selective implicit**: Only correct retain/contested neuron gradients; forget-dominant neurons keep original NPO gradient (fixes G5's overcorrection)
3. **KL post-inner recovery**: Replace CE with KL during masked post-unlearning recovery

### M Series Experiments
| Exp | Config | What it tests |
|-----|--------|---------------|
| M4 | G1 + KL post-inner (masked) | Isolated: does KL recovery avoid fk regression? (vs G3's CE) |
| M0 | G1 + KL inner | Does KL inner loop reduce retain damage? |
| M1 | M0 + KL post-inner | Full KL pipeline (inner + recovery) |
| M2 | M0 + trace-guided implicit | Does targeted implicit finally help? |
| M3 | Full pipeline | KL inner + trace-guided implicit + KL post-inner |

### Implementation (sibl.py changes)
- `inner_loss_type: str = "ce" | "kl_pretrained"` — inner loop loss
- `_kl_loss_from_ref()`: KL(pretrained || model) with temperature scaling (T=2.0)
- `post_inner_loss_type: str = "ce" | "kl_pretrained"` — post-inner recovery loss
- `implicit_trace_guided: bool` — blend implicit correction using mask
- `implicit_trace_threshold: float = 0.5` — mask > threshold → keep original gradient
- Ref model reused from NPO (no extra memory cost)
- Ref model kept alive during KL post-inner (not deleted as with CE recovery)

### M Series Results (2026-04-09 23:20)

**M4** (KL post-inner): fk=0.650, rk=0.549 — KL does NOT protect fk vs CE (G3=0.657). Post-inner recovery is loss-function-agnostic dead end.

**M0** (KL inner only, no bitmap): fk=0.398, rk=0.362 — The only M experiment that didn't wreck fk. L_fgt collapsed to 0.05 by step 3 (NPO working hard), but KL inner loop changes bilevel dynamics significantly. rk marginal gain (+0.034).

**M1** (KL inner + KL post-inner): fk=0.662, rk=0.563 (GOLD!) — rk achieves gold, confirming G3's finding. But fk is destroyed by post-inner recovery. The bitmap + post-inner combination consistently reaches rk~0.56 but always at fk~0.66.

**M2** (KL inner + trace-guided implicit, NO post-inner): fk=0.658, rk=0.556 — Surprising: rk~0.556 WITHOUT post-inner recovery! The bitmap protection alone (freezing retain neurons in outer, modifying them only in inner) preserves retain. But the bitmap restricts NPO to 15-25% of neurons, crippling forget (fk~0.65 ≈ pretrained=0.644).

### M Series Post-Mortem — CRITICAL DISCOVERY

**The bitmap inner mask bug:** All bitmap experiments (M1, M2, M4, G3) share a bug: the inner step uses `binary_mask = (mask > 0)`, which with bitmap means inner loop updates the SAME neurons as outer (forget-dominant). Retain-dominant neurons are NEVER updated during the bilevel loop. Only post-inner recovery touches them.

**Why bitmap experiments get fk~0.65:** NPO restricted to 15-25% forget-dominant neurons is not enough to break the forget pathway. The pretrained model uses ALL neurons for forget; modifying only forget-dominant neurons barely moves fk from pretrained (0.644).

**The fix (N series):** Invert the inner mask → inner updates RETAIN neurons, outer updates FORGET neurons. Two variants:
1. **N0**: Bitmap outer (forget-only NPO) + inverted inner (retain-only CE) — disjoint parameter subsets
2. **N4** (MOST PROMISING): Full outer (all params NPO, like G1) + inverted inner (retain-only CE) — G1-level forget + targeted inner retain

N4 is key because:
- Outer NPO on all params = G1-level fk (0.274)
- Inner CE on retain neurons only = focused retain recovery
- Inner loop doesn't touch forget neurons → NPO's forget is uncontested between steps
- G1's inner loop wastes gradient on forget neurons (which doesn't help retain); N4 concentrates inner gradient on retain neurons

---

## N SERIES — Complementary Bilevel with Inverted Inner Mask (2026-04-09)

### Root Cause Fix
The bitmap mask creates natural forget/retain neuron separation from trace analysis. But previous experiments applied the SAME mask to both inner and outer loops → both operated on forget-dominant neurons → retain neurons never updated → needed post-inner recovery (which kills fk).

**Fix: inner mask = 1 - outer mask.** Inner updates retain neurons, outer updates forget neurons.

### N Series Experiments
| Exp | Config | What it tests |
|-----|--------|---------------|
| N4 | Full outer + inverted inner (CE) | **MOST PROMISING**: G1-level NPO + targeted inner retain |
| N0 | Bitmap outer + inverted inner (CE) | Clean disjoint separation (may have weak forget) |
| N1 | N0 + KL inner | Bounded inner optimization on retain neurons |
| N2 | N0 + K=3 inner | More inner steps (safe with disjoint params) |
| N3 | N0 + trace-guided implicit | 2nd-order correction on correct subspace |

### Implementation (sibl.py changes)
- `invert_inner_mask: bool = False` — inner step uses (1 - bitmap) instead of bitmap
- `outer_full_model: bool = False` — when True + inverted inner: outer mask = ones (full model)
- `inner_mask_dict: dict` — built at train() start from inverted bitmap
- Inner step checks `inner_mask_dict` first, falls back to `mask_dict`

### N Series Results (2026-04-09 ~23:50)

| Exp | fk↓ | rk↑ | verdict |
|-----|-----|-----|---------|
| **N4** | **0.344** | **0.343** | fk +0.070 vs G1 (HURT), rk +0.016 (marginal). Inner loop too weak. |
| N0 | OOM | — | CUDA OOM in outer_step grad.clone() |
| N1 | OOM | — | Same OOM |
| N2 | OOM | — | Same OOM |
| N3 | OOM | — | Same OOM (implicit adds to peak memory) |

**N4 analysis:** The inverted inner mask architecture IS correct — inner updates retain neurons, outer updates all. But with K=1 inner step at eta_in=1e-4, the inner loop contributes +0.016 rk vs G1. That's because the outer NPO gradient overwhelms the inner retain gradient. The ALM retain penalty (λ=0 initially, ρ=0.1) provides almost no retain protection in the outer gradient.

**OOM on N0-N3:** All use bitmap-restricted outer mask (not full model). The bitmap mask + inverted inner mask together hit the 96GB GPU memory limit during gradient accumulation (32 steps). N4 works because `outer_full_model=true` avoids the memory pattern that triggers fragmentation.

### Critical Discovery: ALM Penalty Is a No-Op

The dual variable λ starts at 0 and grows by ρ*r per step:
- Step 1: λ=0, ρ=0.1, r=(L_ret - 0.7) ≈ 0.8 → retain_coeff = 0.08
- NPO loss ≈ 3-5. Retain contribution: 0.08 × 1.5 = 0.12. That's ~3% of total gradient.
- After 25 steps: λ ≈ 1.25. Retain reaches ~30% of gradient. Too late — retain is already destroyed.

**The ALM retain constraint is essentially decorative with current hyperparameters.** This explains why G1's rk=0.327 barely exceeds F0's rk=0.290 despite 25 steps of dual accumulation. And why L4 (ε=0.50) only gained +0.016 rk despite λ reaching 4.0 — by then the damage was baked in.

**Fix: P series** — initialize λ > 0 and increase ρ so retain protection kicks in from step 1.

---

## P SERIES — Fix ALM Retain Penalty (2026-04-10)

### Root Cause
The ALM formulation L_alm = L_fgt + λ*L_ret + 0.5*ρ*(L_ret - ε)² has λ_init=0 and ρ=0.1.
The gradient of L_alm w.r.t. θ is: ∇L_fgt + (λ + ρ*max(0,r)) * ∇L_ret.
At step 1: retain coefficient = 0 + 0.1*0.8 = 0.08. NPO dominates at >90%.
Standard ALM theory says: "start with small ρ, increase over time." But with only 25 steps, there's no time for λ to build up. The model trains 25 steps with inadequate retain protection.

### Fix: Warm-Start the Dual Variable
- **lambda_init**: New parameter to initialize λ > 0. Immediate retain protection.
- **Higher ρ**: Faster λ growth + stronger quadratic penalty.
- Combined with N4 architecture (full outer NPO + inverted inner on retain neurons).

### P Series Experiments
| Exp | λ_init | ρ | K | eta_θ | eta_in | What it tests |
|-----|--------|---|---|-------|--------|---------------|
| P0 | 5.0 | 1.0 | 1 | 2e-4 | 1e-4 | Strongest ALM — retain ≈83% of gradient from step 1 |
| P1 | 2.0 | 1.0 | 1 | 2e-4 | 1e-4 | Moderate ALM — retain ≈50% initially |
| P2 | 5.0 | 1.0 | 2 | 1e-4 | 3e-4 | Strong ALM + 2x inner + 3x inner LR + half outer LR |
| P3 | 0.0 | 5.0 | 1 | 2e-4 | 1e-4 | Control: high ρ only, no warm-start. Tests if fast growth alone works |

### Implementation
- Added `lambda_init: float = 0.0` param to SIBL __init__
- `self.lambda_dual = float(lambda_init)` instead of hardcoded 0.0
- All P configs based on N4 (full outer + inverted inner + steering)

### P Series Results (2026-04-10 ~01:05)

| Exp | λ_init | ρ | fk↓ | rk↑ | verdict |
|-----|--------|---|------|------|---------|
| **P0** | 5.0 | 1.0 | 0.524 | **0.465** | **Best rk** (+0.138 vs G1). ALM works but fk hurt +0.250 |
| P1 | 2.0 | 1.0 | 0.535 | 0.464 | Nearly identical to P0 — ρ=1 catches up fast |
| P2 | 5.0 | 1.0 | 0.548 | 0.447 | K=2+eta_in=3e-4+eta_θ=1e-4 WORSE on both axes |
| P3 | 0.0 | 5.0 | 0.000 | 0.000 | COLLAPSED — λ=1200+, explosive divergence |

**P series diagnosis:**
1. Strong ALM gives rk=0.465 (+0.138 over G1) but costs fk=+0.250. The Pareto frontier didn't move — we're just sliding along it with a different control knob.
2. P0 ≈ P1: λ_init=5 vs 2 barely matters when ρ=1.0, because ρ catches up within a few steps. The FIRST unprotected step is what determines the trajectory.
3. P2 worse on both axes: stronger inner loop (CE on retain) partially undoes NPO during both phases. The inner loop doesn't independently improve rk.
4. P3 collapsed: ρ=5 without λ_init → the first pure NPO step destroys retain, then explosive dual growth tries to compensate but diverges (λ>1000).

**Key insight from P3:** The first step matters enormously. Unprotected NPO step 1 creates irreversible retain damage. P0/P1 work because λ_init>0 provides protection from step 1. P3 fails because step 1 is unprotected.

**Implication for Q series:** Phase 1 (G1-like, ρ=0.1) provides MINIMAL retain protection — enough to survive 10 steps (G1 proves this). Then the boost at step 10 amplifies protection. Unlike P3 (ρ=5, no initial λ), Q series never has a fully unprotected step.

---

## Q SERIES — Two-Phase ALM (2026-04-10)

### Rationale
P series showed: strong ALM slides the operating point (fk=0.53, rk=0.46) but doesn't move the frontier.
G1 shows: weak ALM gets fk=0.274 (gold) but rk=0.327.
**Idea: NPO-heavy first (get fk to gold), then boost λ for retain recovery.**

### Implementation
- Added `lambda_boost_step`, `lambda_boost_value`, `rho_boost_value` params to SIBL
- At the specified outer step, λ is set to max(current, boost_value) and ρ is updated
- All Q configs use full outer + inverted inner + steering (N4 base)

### Q Series Experiments
| Exp | Boost step | What it tests |
|-----|-----------|---------------|
| Q0 | Step 10 | G1-like NPO for 10 steps (fk~0.275), then λ=5 ρ=1 for 15 steps |
| Q1 | Step 5 | Earlier boost (fk~0.246 at boost), 20 steps for retain recovery |
| Q2 | Step 10 | Q0 + K=2 inner + eta_in=3e-4 (stronger inner throughout) |

### Q Series Results (2026-04-10 ~01:40)

| Exp | Boost step | fk↓ | rk↑ | verdict |
|-----|-----------|------|------|---------|
| Q0 | Step 10 | 0.466 | 0.423 | Between G1 and P0 — two-phase gives different operating point |
| Q1 | Step 5 | 0.519 | 0.451 | ≈P0 — only 5 unprotected steps → nearly equivalent to λ=5 from start |
| Q2 | Step 10 | *(running)* | — | K=2 inner + two-phase |

**Q series diagnosis:**
1. Q0 (boost@10): 10 steps of G1-like NPO → fk dropped to ~0.05 before boost. Phase 2 (15 steps with λ=5,ρ=1) recovered retain partially. Final fk=0.466 (NPO weaker than P0), rk=0.423 (less retain than P0).
2. Q1 (boost@5): Only 5 unprotected steps → almost identical to P0 where λ=5 from start. Makes sense: ρ=1 catches up within a few steps of the boost.
3. **All Q results lie on the same Pareto frontier as P series.** Two-phase doesn't shift the frontier — it just picks a different operating point along it.

### Critical Finding: The CE Pareto Frontier

**ALL experiments to date lie on a linear frontier:**
```
  rk ≈ 0.55 * fk + 0.17    (R² ≈ 0.97)
```

| Experiment | fk | rk | rk_predicted |
|-----------|------|------|------|
| G1 | 0.274 | 0.327 | 0.321 |
| N4 | 0.344 | 0.343 | 0.359 |
| Q0 | 0.466 | 0.423 | 0.426 |
| P1 | 0.535 | 0.464 | 0.464 |
| P0 | 0.524 | 0.465 | 0.458 |
| Q1 | 0.519 | 0.451 | 0.455 |
| M1 | 0.662 | 0.563 | 0.534 |

**Gold target (0.328, 0.560) requires rk = 0.560 at fk = 0.328 → would need slope ~1.7, not 0.55.**

The frontier is defined by the CE retain loss. CE on retain data re-learns forget through shared weights — every rk improvement pulls fk up proportionally. No ALM tuning, two-phase scheduling, or inverted inner mask can break this tradeoff because they all use CE as the retain signal.

**To shift the frontier: replace CE retain loss with something that CANNOT re-learn forget.**
→ KL(pretrained || model) on retain data. The pretrained model has rk=0.555 (near gold) and its retain distribution cannot teach forget patterns.

---

## R SERIES — KL(pretrained||model) Outer Retain Loss (2026-04-10)

### Rationale
The CE Pareto frontier has slope ~0.55. Gold requires slope ~1.7 from G1. CE retain loss is the root cause:
- CE(θ, retain_tokens) = -log p(x_retain | θ) → maximizes token likelihood → can reconstruct forget patterns through shared weights
- KL(pretrained || model) on retain data → pushes θ's distribution toward pretrained's distribution ON RETAIN DATA
- Pretrained has rk=0.555 (near gold). KL distillation anchors to that level without teaching forget patterns.

### Implementation
- Added `outer_retain_loss_type: str = "ce" | "kl_pretrained"` to SIBL
- Outer step computes `_kl_loss_from_ref(retain_outputs.logits, ...)` when configured
- `_alm_loss_fn` closure also updated for implicit correction compatibility
- `needs_ref` check updated to trigger ref model loading

### R Series Experiments
| Exp | Base | outer_retain_loss | inverted_inner | What it tests |
|-----|------|-------------------|----------------|---------------|
| R0 | G1 (ρ=0.1, λ=0) | kl_pretrained | No | CONTROL: does KL outer shift frontier from G1? |
| R1 | P0 (λ=5, ρ=1) | kl_pretrained | Yes | Strong ALM + KL: best of P0 + frontier shift? |
| R2 | Q0 (boost@10) | kl_pretrained | Yes | Two-phase + KL: combine timing + loss improvement? |

### Key Test
**R0 is the cleanest test.** Same config as G1 except outer retain = KL instead of CE.
- If R0 gives rk > 0.327 (G1) WITHOUT fk regression → KL shifted the frontier
- If R0 gives rk ≈ 0.327 → KL doesn't help and the frontier is geometry, not loss function

### R Series Results (2026-04-10 ~02:22)

| Exp | Base | λ | ρ | fk↓ | rk↑ | frontier |
|-----|------|---|---|------|------|----------|
| R0 | G1 (ALM) | 0→28 | 0.1 | 0.003 | 0.008 | COLLAPSED — KL ALM diverged |
| R1 | G1 (fixed-weight) | 0.1 | 0 | 0.255 | 0.234 | BELOW (pred 0.310) |
| R2 | G1 (fixed-weight) | 1.0 | 0 | *(skipped)* | — | — |

**R series verdict: KL outer retain is a dead end.**
- R0: ALM + KL diverges because KL scale (0→20 in one step) mismatches ε=0.70 (designed for CE scale 0.7-1.5). Oscillation → collapse.
- R1: Fixed-weight KL (λ=0.1, no dual update) is stable but gives WORSE rk (0.234) than G1's CE (0.327). The KL penalty constrains toward pretrained globally but doesn't specifically protect retain performance.
- **The CE frontier is not loss-function dependent.** It's weight-space geometry — forget and retain knowledge share 85% of neurons. No loss function change can disentangle what's entangled at the weight level.

---

## S SERIES — Small-Batch NPO: Recreating 8r's Frontier Break (2026-04-10)

### The 8r Clue
Exp8r (pre-fix, broken data pipeline) achieved fk=0.371, rk=0.417. The CE frontier predicts rk=0.374 at that fk. 8r was +0.043 ABOVE the frontier — the only experiment to break it.

**Why 8r worked (accidentally):** With the broken data pipeline, each outer step processed only 1 forget sample (instead of 32). This meant:
1. NPO gradient was highly targeted to that specific sample's patterns
2. The inner loop (K=10) could precisely compensate for the small, specific perturbation
3. 10 samples total, each with dedicated inner correction — not 800 samples averaged

**The fix (commit 655f367)** changed to accum=32 (proper batching). This made NPO gradients averaged across 32 samples — more diffuse, harder for inner loop to compensate. All post-fix experiments fell back onto the CE frontier.

### Hypothesis
The bilevel framework works BETTER with smaller perturbations per outer step:
- Small batch → targeted, low-rank NPO perturbation → inner loop can precisely compensate
- Large batch → diffuse, high-rank NPO perturbation → inner loop overwhelmed

This is consistent with bilevel optimization theory: the inner problem is easier when the outer perturbation is small.

### S Series Experiments
| Exp | accum | K | ρ | What it tests |
|-----|-------|---|---|---------------|
| S0 | 4 | 1 | 0.1 | 200 steps, 8x more frequent inner corrections than G1 |
| S1 | 1 | 3 | 0.01 | 800 steps, closest to 8r (1 sample + strong inner) |
| S2 | 4 | 1 | 0.1 | S0 + inverted inner mask + full outer (N4 architecture) |

### S Series Results
*(running — S0 training at ~3.3s/step, expected 02:34)*

### Concern
With ρ=0.1 and 200+ steps, λ grows ~8x faster than G1's 25 steps. S0 shows λ=6.3 at step 20. This may over-constrain NPO (P0-like outcome). S1 uses ρ=0.01 to compensate.

---

## FRONTIER ANALYSIS SUMMARY (2026-04-10)

### Dead ends confirmed
- **Loss function changes** (R series): KL outer retain is WORSE than CE. Frontier is weight-space geometry.
- **ALM tuning** (P series): Different λ/ρ slide along frontier, never move it.
- **Two-phase** (Q series): Different timing picks different operating point on SAME frontier.
- **Gradient projection** (K1a, Finding 4): Kills forgetting for same-domain data (85% overlap).
- **Post-inner CE/KL** (G3, M4, I series): Recovery always re-learns forget through shared weights.

### Where hope lives
1. **Small-batch bilevel** (S series): 8r broke frontier with 1-sample-per-step. Tests underway.
2. **Sample-level targeting**: Weight samples by memorization score, curriculum ordering.
3. **Representation-space disentanglement**: Constrain activations not just loss. Already have inner_repr_anchor — extend to outer loop.
4. **Accept realistic targets**: fk≤0.35, rk≥0.45 would beat all published baselines even if gold unreachable.

---

## LONGER TERM
- Once fk + rk both hit targets: run on MUSE Books + WMDP
- Baselines still needed: BLURNPO (checkpoint at saves/), RMU (needs fresh run)
- Publish as DS-BiAL: bilevel NPO + trace-guided KL distillation + selective implicit differentiation
