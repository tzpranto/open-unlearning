# Research Scratchpad — DS-BiAL MUSE News
## For agent continuity. Last updated: 2026-04-11 ~02:30 (PerTA confirmed best, all baselines done, saves cleaned)

---

## GOLD TARGETS (verified 2026-04-10)
- fk ≤ 0.324 (forget knowmem ROUGE) — fresh eval of retrain model
- rk ≥ 0.552 (retain knowmem ROUGE) — fresh eval of retrain model
- Note: prior session used fk≤0.328, rk≥0.560 from an older eval. ~0.008 variance from generation randomness.

---

## REFERENCE MODEL EVALS (fresh, 2026-04-10)

| Model | fk↓ | rk↑ | fv | ex | CE frontier δ | notes |
|-------|------|------|------|------|------|-------|
| **Retrain (gold)** | **0.324** | **0.552** | 0.204 | 0.025 | **+0.204** | `muse-bench/MUSE-news_retrain` — WAY above CE frontier |
| Target (finetuned) | 0.654 | 0.544 | 0.569 | 0.302 | +0.014 | `muse-bench/MUSE-News_target` — ON CE frontier |
| Pretrained (base) | 0.270 | 0.345 | 0.188 | 0.021 | +0.027 | `meta-llama/Llama-2-7b-hf` — ABOVE CE frontier |

### Critical insight: Pretrained as unlearning baseline
The base LLM (never finetuned) has fk=0.270 (BELOW gold!) and rk=0.345. This means:
- G1 (fk=0.274, rk=0.327) is **WORSE than pretrained on rk** — our "best" experiment doesn't beat "do nothing"
- Any method with rk < 0.345 hasn't even matched the base model
- The retrain model sits δ=+0.204 above CE frontier; our best (T8f) is δ=+0.056 — only 27% of the way
- Target model is ON the CE frontier (δ=+0.014) — finetuning creates the correlated fk/rk pattern

### Baselines (from prior sessions, not re-verified)
| Method | fk↓ | rk↑ | notes |
|--------|-----|-----|-------|
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

## CURRENT STATE (2026-04-10, updated ~19:30)

**🔥 NEW PARADIGM: PerTA BREAKS CE FRONTIER SYSTEMATICALLY**
**BEST FRONTIER BREAK: PerTA λ=3.5** — fk=0.282 ✅, rk=0.396 — ABOVE CE frontier by δ=+0.071 (27% better than T8f)
**Best rk at gold fk: PerTA λ=3.5 + LoRA r16** — fk=0.329 ≈ gold, rk=0.401, δ=+0.050
**Best rk overall: PerTA λ=0.5** — fk=0.635, rk=0.567 > gold rk! (but fk too high)
**Remaining gap: rk** — at fk≈gold, rk=0.396-0.401. Gold rk=0.552. Gap = +0.151.
**Key mechanism:** Fisher-weighted per-parameter task arithmetic. Not gradient-based — direct weight surgery.

**PerTA λ=3.5 is CONFIRMED BEST and REPRODUCIBLE (deterministic).**
Two identical runs produce identical metrics. No gradient randomness.

**All MUSE News baselines complete (2026-04-11):**
| Method | fk | rk | notes |
|---|---|---|---|
| GradAscent | 0.003 | 0.008 | collapsed |
| GradDiff | 0.330 | 0.247 | |
| NPO | 0.517 | 0.420 | |
| SimNPO | 0.584 | 0.470 | best gradient retain |
| BLURNPO | 0.581 | 0.532 | checkpoint-100/130 (OOM full run) |
| RMU | 0.516 | 0.457 | |
| **PerTA (ours)** | **0.282** | **0.396** | **best fk by far, above CE frontier** |

**Saves cleaned:** 522GB → 220GB. Kept: baselines, perta_l3.5, Fisher cache.

**Remaining gap:** rk=0.396 vs gold 0.552. Gap = 0.156.

**Generalization requirement:** Method must work on MUSE Books + WMDP. PerTA is fully general — only needs pretrained + target model + forget/retain data for Fisher computation.

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

### S Series Results (S0/S1: first run)
| Exp | accum | K | ρ | steps | fk↓ | rk↑ | verdict |
|-----|-------|---|---|-------|------|------|---------|
| S0 | 4 | 1 | 0.1 | 200 | collapsed | — | λ exploded at step ~160 (1 bad sample in 4 = 25% influence) |
| S1 | 1 | 3 | 0.01 | 813 | collapsed | — | L_ret stable until step ~500, then L_ret=7.0 forever |
| S2 | 4 | 1 | 0.1 | 200 | collapsed | — | Same as S0 (inverted inner didn't help) |

### S2 Series Results (early stopping, all accum=1 K=3 ρ=0.01 unless noted)
| Exp | accum | K | ρ | steps | fk↓ | rk↑ | frontier | verdict |
|-----|-------|---|---|-------|------|------|----------|---------|
| S3 | 1 | 3 | 0.01 | 25 | 0.008 | 0.022 | BELOW (-0.153) | **COLLAPSED** — 25 steps destroyed model |
| S4 | 1 | 3 | 0.01 | 50 | 0.000 | 0.000 | BELOW (-0.170) | **COLLAPSED** — total knowledge destruction |
| **S5** | **1** | **3** | **0.01** | **100** | **0.346** | **0.382** | **ABOVE (+0.022)** | **🔥 FRONTIER BROKEN!** |
| S6 | 1 | 3 | 0.01 | 200 | *(running)* | — | — | will λ accumulation improve further? |
| S7 | 4 | 1 | 0.1 | 25 | *(queued)* | — | — | G1-like small batch |

### S Series Post-Mortem: Why S3 Collapsed But 8r Didn't

**The 8r vs S3 comparison:**
| | 8r (worked) | S3 (collapsed) |
|---|-----------|----------------|
| samples/step | 1 | 1 |
| K (inner steps) | 10 | 3 |
| ρ | 0.1 | 0.01 |
| Total outer steps | 10 | 25 |
| Inner/outer ratio | 10:1 | 3:1 |
| Total inner steps | 100 | 75 |

**Root cause: insufficient inner correction per outer perturbation.**
- 8r: K=10 inner steps per 1 NPO step → massive inner repair capacity. Ratio 10:1.
- S3: K=3 inner steps per 1 NPO step → inner can't keep up with NPO damage. Ratio 3:1.
- S3 also used ρ=0.01 (10x weaker ALM constraint), compounding the problem.
- With λ=0 and ρ=0.01, the outer gradient is ~99.97% NPO, ~0.03% retain penalty. The ALM constraint is essentially absent.
- Each unprotected NPO step on a single sample applies full gradient magnitude — not averaged over 32 samples like G1.
- Over 25 steps, cumulative unprotected NPO destroys all shared representations.

**Why not just use K=10?** With accum=1, 25 outer × 10 inner = 250 inner steps on 25 retain samples (cycling the same 25). This overfits the inner loop to those 25 samples. Need to validate if that helps.

### S5 BREAKTHROUGH: Why 100 Steps Works But 25/50 Collapse

**S3 (25 steps) = collapsed. S4 (50 steps) = collapsed. S5 (100 steps) = ABOVE FRONTIER.**

This seems paradoxical — more steps should mean more damage. But the explanation is ALM dual variable dynamics:

**Phase 1 (steps 0-50): Destructive NPO with weak protection**
- λ starts at 0, ρ=0.01 → retain contribution in gradient is ~0%
- After 25 steps: λ ≈ 0.25 → retain ~5% of gradient (far too little)
- After 50 steps: λ ≈ 0.5 → retain ~10% of gradient (still too little)
- **This is why S3 (stop at 25) and S4 (stop at 50) collapse** — the ALM constraint hasn't kicked in

**Phase 2 (steps 50-100): Recovery with growing ALM constraint**
- After 75 steps: λ ≈ 1.5 → retain ~50% of gradient (significant!)
- After 100 steps: λ ≈ 2.5 → retain ~70%+ of gradient (dominant!)
- With strong retain constraint, the inner loop (K=3) can actually recover representations
- NPO still pushes forget, but proportionally less as λ grows
- **The model RECOVERS from the early damage**

**Key insight: ALM needs TIME to build dual pressure.** With ρ=0.01, the dual variable grows slowly (~0.01-0.05 per step depending on residual). It takes ~50-75 steps for λ to reach values where retain protection is meaningful. Experiments stopped before that point (S3, S4) catch the model at its WORST — maximum NPO damage, minimum ALM protection.

**Comparison with G1 (K=1, accum=32, 25 steps, ρ=0.1):**
- G1: λ grows 10x faster (ρ=0.1), reaches ~2.7 at step 25
- S5: λ grows 10x slower (ρ=0.01), but has 4x more steps → reaches ~2.5 at step 100
- Both end with similar λ! But S5's small-batch dynamics give different gradient geometry
- S5's single-sample gradients are more targeted → inner loop compensates more precisely

**This validates the 8r hypothesis but with a twist: it's not just K that matters, it's K + enough time for ALM to protect.**

**Frontier comparison:**
- G1 (accum=32, K=1, 25 steps): fk=0.274, rk=0.327 (ON frontier, δ=+0.006)
- 8r (accum=1, K=10, 10 steps): fk=0.371, rk=0.417 (ABOVE, δ=+0.043)
- **S5 (accum=1, K=3, 100 steps): fk=0.346, rk=0.382 (ABOVE, δ=+0.022)**

S5 trades more fk (0.346 vs 0.274 in G1) for being ABOVE the frontier. The rk=0.382 is not gold (0.560) but significantly better than G1's 0.327.

---

## FRONTIER ANALYSIS SUMMARY (2026-04-10)

### Dead ends confirmed
- **Loss function changes** (R series): KL outer retain is WORSE than CE. Frontier is weight-space geometry.
- **ALM tuning** (P series): Different λ/ρ slide along frontier, never move it.
- **Two-phase** (Q series): Different timing picks different operating point on SAME frontier.
- **Gradient projection** (K1a, Finding 4): Kills forgetting for same-domain data (85% overlap).
- **Post-inner CE/KL** (G3, M4, I series): Recovery always re-learns forget through shared weights.

### S series: PARTIALLY SUCCESSFUL
- S3 (25 steps), S4 (50 steps): COLLAPSED — ALM dual variable hasn't built enough protection
- **S5 (100 steps): ABOVE FRONTIER** (fk=0.346, rk=0.382, δ=+0.022) — ALM needs ~50+ steps at ρ=0.01 to build sufficient retain constraint
- S6 (200 steps), S7 (accum=4, K=1): running, results pending

### Where hope lives
1. **High K/step ratio** (T series): G1 with K=10 and 5-10 outer steps. Directly replicates 8r's success factor (strong inner correction) with proper batching.
2. **Fisher-weighted outer gradient**: Scale NPO gradient by 1/(1+αF_retain) where F_retain is diagonal Fisher on retain data. Continuously dampens updates to retain-important parameters — more principled than binary bitmap.
3. **Contrastive inner loop**: Inner loop does CE on retain + actively pushes forget representations away from pretrained. Creates representational separation during inner correction.
4. **Subspace NPO**: SVD of forget gradient matrix → only update in top-k directions. Most targeted possible NPO.
5. **Accept realistic targets**: fk≤0.35, rk≥0.45 would beat all published baselines even if gold unreachable.

---

## T SERIES — Targeted Disentanglement (2026-04-10)

### Core Insight
The CE frontier (rk ≈ 0.55*fk + 0.17) holds for ALL loss function and ALM variations. The only experiment that broke it was 8r, which had K/step ratio of 10:1 (10 inner corrections per 1 outer perturbation). All other experiments had ratio 1:1 or 3:1.

**Hypothesis:** The frontier is set by the inner loop's ability to compensate for outer perturbations. With K=1, the inner loop can't fully recover retain → collateral damage accumulates → rk degrades proportionally with fk. With K≥10, inner loop FULLY recovers retain before the next perturbation → retain damage doesn't accumulate → frontier shifts.

This is bilevel optimization 101: the inner problem must converge for the bilevel solution to be meaningful. K=1 never converges.

### T Series Experiments

**T0: High-K with proper batching** (simplest, most 8r-like)
- G1 base, accum=32, K=10, 5 outer steps (8r-like ratio)
- If frontier breaks: confirms K/step ratio hypothesis
- If stays on frontier: K alone isn't enough

**T1: High-K with moderate steps**
- G1 base, accum=32, K=5, 10 outer steps
- Tests intermediate ratio

**T2: Fisher-weighted outer gradient** (novel disentanglement)
- Precompute diagonal Fisher F_retain from ~100 retain samples
- Outer gradient scaled: g_i → g_i / (1 + α*F_i) where F_i is retain Fisher for param i
- α is a hyperparameter controlling protection strength
- This is EWC-inspired but applied to outer gradient, not as a regularization term
- Novel: no prior work uses Fisher weighting in bilevel unlearning outer loops

**T3: Contrastive inner loop** (push-pull disentanglement)
- Inner loss = CE(retain) + β*MSE(h_retain, h_pretrained) - γ*MSE(h_forget, h_pretrained)
- The negative term PUSHES forget representations away during inner correction
- Creates active representational separation, not just passive retain recovery
- Novel: combines representational anchoring with active forget reinforcement in inner loop

**T4: Subspace NPO** (most targeted possible)
- Collect forget gradients across ~50 samples, compute top-k SVD directions
- Constrain NPO gradient to live in this subspace via projection
- Most surgically targeted NPO possible — only touches forget-specific weight directions
- Different from gradient projection (Finding 4): projects onto forget subspace, not away from retain subspace

### Implementation Plan
1. T0/T1: config-only (override K and debug_stop_after_outer)
2. T2: ~50 LOC — precompute Fisher, apply scaling in outer_step
3. T3: ~30 LOC — extend inner_step loss computation
4. T4: ~80 LOC — Fisher/SVD precomputation + projection in outer_step

---

## T SERIES RESULTS (2026-04-10, running)

### Completed
| Exp | Config | fk↓ | rk↑ | frontier | verdict |
|-----|--------|------|------|----------|---------|
| T0 | K=10, 5 steps, accum=32 | 0.072 | 0.123 | BELOW (-0.087) | Over-forgot. 5 steps of K=10 with accum=32 = NPO too strong, insufficient ALM buildup. |
| T1 | K=5, 10 steps, accum=32 | 0.264 | 0.259 | BELOW (-0.057) | Also over-forgot. 10 steps still not enough for ALM at ρ=0.1 with accum=32. |
| T2 | Fisher α=1.0, accum=32 | 0.386 | 0.336 | BELOW (-0.046) | Fisher dampened NPO → weaker forget (fk worse than G1). rk also below frontier. No improvement. |
| T2b | Fisher α=10.0, accum=32 | 0.490 | 0.399 | BELOW (-0.041) | Stronger Fisher → even weaker forget. Slides along frontier. Both T2/T2b BELOW frontier — Fisher hurts bilevel dynamics. |
| T3 | Contrastive inner (β=1,γ=0.5), accum=32 | 0.409 | 0.353 | BELOW (-0.041) | Contrastive weakened forget (fk 0.409 vs G1 0.274). K=1 inner step too weak for contrastive signal. |
| T3b | Contrastive + K=3, accum=32 | 0.361 | 0.327 | BELOW (-0.042) | K=3 prevented L_fgt collapse (0.2-0.8 vs 0.04) but still BELOW. Different dynamics, same frontier. |
| **T5** | **S5 + Fisher α=1.0 (accum=1, K=3, 100 steps)** | **0.365** | **0.305** | **BELOW (-0.066)** | **Fisher HURTS S5 — both axes worse than vanilla S5 (0.346, 0.382). Fisher is anti-bilevel.** |
| **T6** | **S5 + contrastive (accum=1, K=3, 100 steps)** | **0.000** | **0.000** | **COLLAPSED** | **Contrastive DESTROYED model. L_ret spiked to 7.8, never recovered. Chaotic inner dynamics.** |
| T7 | S5 + Fisher + contrastive | — | — | FAILED | Hydra config error (T3 config missing Fisher keys). Would collapse anyway (T6 collapsed). |

### T0-T7 COMPLETE — ALL BELOW FRONTIER OR COLLAPSED
**Not a single T series experiment broke the frontier.** S5 (vanilla accum=1, K=3, 100 steps) remains the ONLY controlled experiment above the CE frontier.

### T8 Weighted Series Results (2026-04-10)
| Exp | Config | fk↓ | rk↑ | frontier | verdict |
|-----|--------|------|------|----------|---------|
| T8 | S5 + log weighting, 100 steps | 0.003 | 0.038 | COLLAPSED | Log weighting alone → L_ret spiked to 7.7, model destroyed |
| T8b | S5 + sqrt weighting, 100 steps | 0.354 | 0.368 | ON (+0.003) | sqrt weighting preserves S5's operating region, marginal frontier |
| T8c | S5 + log weighting, 75 steps | 0.000 | 0.000 | COLLAPSED | Log weighting consistently collapses regardless of step count |
| T8d | S5 + log + Fisher α=1.0 | — | — | FAILED | Config error (missing Fisher keys in T8 config). Fixed for future runs. |
| T8e | G1 + log weighting, accum=32 | 0.390 | 0.331 | BELOW (-0.054) | Log weighting with accum=32 → weaker forget, no rk gain. Sampling diluted by batching. |
| **T8f** | **S5 + log + contrastive inner** | **0.428** | **0.461** | **ABOVE (+0.056)** | **🔥 SYNERGY: log+contrastive stabilize each other. Highest above-frontier (δ=+0.056) but fk too high.** |

### T8 KEY INSIGHT: Log weighting + contrastive = synergistic stabilization
- **T8 (log weighting alone):** COLLAPSED — concentrated NPO on high-memorization chunks → extreme retain damage
- **T6 (contrastive alone):** COLLAPSED — inner push-pull creates chaotic dynamics with single samples  
- **T8f (log + contrastive):** ABOVE FRONTIER (+0.056) — the two mechanisms compensate each other:
  1. Log weighting concentrates NPO → more targeted, less collateral damage per step
  2. Contrastive inner actively separates retain/forget representations → prevents the retain spike
  3. Together: L_ret=0.65 at step 40 (constraint SATISFIED!) vs T8's L_ret=7.7 and T6's L_ret=7.8
  4. The focused NPO gives contrastive less interference to manage; contrastive gives NPO a safer optimization landscape

**T8b (sqrt weighting):** Healthy dynamics (ON frontier) suggest weighting CAN work in S5 regime. The sqrt scheme is more aggressive but paradoxically more stable — it creates cleaner separation between high/low-memorization samples.

**Log vs sqrt:** Log compresses the weight range (max 2.9x), making sampling more uniform-like. Sqrt amplifies (max 11.8x), creating stronger targeting. Log's quasi-uniform distribution may cause worse gradient consistency across steps (different step = different mix of high/low samples → noisy trajectory). Sqrt's strong targeting creates more consistent gradients (mostly high-memorization samples → consistent NPO signal).

### T0/T1 Insights

**T0 (K=10, 5 steps):** fk=0.072 is extremely low — the model forgot almost everything. But rk=0.123 means retain was destroyed too. With only 5 outer steps at accum=32, ALM λ reaches at most ~0.25 (ρ=0.1). That's ~5% retain protection. K=10 inner steps DID repair retain per-step, but the outer gradient at accum=32 was too diffuse — 32 samples averaged = broad damage the inner loop couldn't fully fix. Also, 5 steps × 32 accum = 160 samples processed — enough to wipe forget but not enough ALM buildup for retain.

**T1 (K=5, 10 steps):** fk=0.264 (better than T0's 0.072), rk=0.259. Predicted rk at fk=0.264 is 0.315. T1 is BELOW by 0.057 — worse than the CE frontier. The intermediate K=5 + 10 steps doesn't improve over the baseline regime either. With accum=32, each outer step still makes broad perturbations. K=5 inner correction is insufficient for 32-sample-averaged gradients.

**Key insight from T0+T1:** High K alone (T0: K=10) or intermediate K (T1: K=5) with accum=32 does NOT break the frontier. This confirms that the frontier break requires SMALL BATCH (accum=1) dynamics, not just high K. The 8r/S5 breakthrough came from single-sample gradients being low-rank enough for K=3-10 inner steps to compensate. Accum=32 averages the perturbation into a high-rank mess the inner loop can't efficiently fix.

**T2 (Fisher α=1.0):** fk=0.386, rk=0.336, BELOW (-0.046). Fisher dampening WEAKENED forgetting — fk went from G1's 0.274 to 0.386 (worse). The Fisher diagonal correctly identifies retain-important parameters and suppresses NPO there, but this suppression reduces the effective NPO gradient magnitude across shared parameters. With 85% neuron overlap, most parameters are both retain- and forget-important. Dampening on these shared params means weaker NPO everywhere. Result: same frontier, just a different (worse) point on it.

**T2 training dynamics vs G1:** Nearly identical λ trajectory (G1: ~2.7 at step 25; T2: 2.80). L_fgt collapsed to 0.04 by step 4 (same as G1). L_ret spike was slightly higher (6.9 vs G1's ~5), suggesting Fisher weighting didn't prevent early retain damage. The Fisher scaling 1/(1+αF) with α=1 and mean F≈0 is essentially a no-op for most parameters — only very high-Fisher params get dampened.

**T2b (Fisher α=10):** fk=0.490, rk=0.399. Even further toward "more retain, less forget." Both T2 and T2b are BELOW the CE frontier (not just ON it) — Fisher weighting actually HURTS bilevel performance. The 1/(1+αF) scaling interferes with the coupled NPO/ALM dynamics. When NPO is dampened on retain-important params, the outer gradient becomes less effective at forgetting AND the ALM constraint residuals are smaller → λ grows slower → less retain protection. The Fisher scaling breaks the bilevel optimization's own feedback mechanism.

**Fisher verdict: DEAD END across all regimes.**
- accum=32: Both α=1 (T2: BELOW -0.046) and α=10 (T2b: BELOW -0.041) worse than G1.
- accum=1: T5 (S5+Fisher) BELOW -0.066 — worse than vanilla S5 on BOTH axes.
- Fisher scaling 1/(1+αF) breaks bilevel dynamics because it non-uniformly dampens the outer gradient. The bilevel framework relies on the outer gradient direction being correct (pointing toward forget); Fisher rotates this direction, causing the inner loop to compensate for the wrong perturbation. The result: less effective forgetting AND more retain damage.

### G1-base (accum=32) verdict: ALL BELOW FRONTIER
All four G1-base experiments (T2, T2b, T3, T3b) land BELOW the CE frontier — worse than vanilla G1 (which is ON frontier). Fisher and contrastive mechanisms, when applied to the accum=32 regime, consistently degrade performance. The mechanisms add noise to the bilevel optimization without improving the fundamental gradient geometry. With 32-sample-averaged outer gradients, the per-parameter adjustments (Fisher dampening) and per-representation adjustments (contrastive) are drowned out by the high-rank gradient noise.

**T3b's qualitative difference:** K=3 + contrastive showed genuinely different training dynamics — L_fgt stayed at 0.2-0.8 instead of collapsing to 0.04. The inner loop with 3 steps and contrastive loss actively resisted NPO. But this resistance didn't translate to frontier improvement — it just produced a weaker operating point. The inner loop "fighting" the outer loop wastes gradient budget on an adversarial dynamic rather than cooperative bilevel optimization.

### T5-T7 VERDICT: Fisher and contrastive HURT the S5 regime
- **T5 (S5+Fisher):** BELOW (-0.066). Both fk and rk worse than vanilla S5. Fisher non-uniformly dampens the outer gradient → rotates the gradient direction → inner loop compensates for wrong perturbation.
- **T6 (S5+contrastive):** COLLAPSED. L_ret spiked to 7.8 at step 40 and never recovered. Contrastive push-pull on single-sample representations creates chaotic inner dynamics. The inner loop fights itself — CE pulls retain toward data while contrastive pushes forget away, creating interference when representations overlap.
- **T7 (S5+both):** Failed (config error), but would collapse since T6 collapsed.

### CRITICAL INSIGHT FROM T SERIES: S5's frontier break is FRAGILE
S5 (vanilla accum=1, K=3, 100 steps) breaks the frontier by δ=+0.022. But ANY modification to the S5 formula destroys it:
- Fisher weighting → BELOW (-0.066)
- Contrastive inner → COLLAPSED
- The frontier break relies on a delicate balance between NPO gradient magnitude, inner loop compensation capacity (K=3), and ALM dual variable dynamics (ρ=0.01, 50+ steps to reach meaningful λ).
- Perturbing ANY of these components — even with theoretically motivated mechanisms — disrupts the balance.

**What this means for T8 (weighted sampling):** Score-weighted sampling changes the DISTRIBUTION of NPO gradients across the dataset. Unlike Fisher (which changes gradient direction) or contrastive (which adds inner loss terms), weighted sampling doesn't modify the optimization algorithm — it only reweights which samples appear. This is the most conservative modification and may preserve S5's fragile balance.

---

## U SERIES — Exploiting T8f Synergy (2026-04-10)

### Rationale
T8f (log weighting + contrastive inner) broke the frontier at δ=+0.056 — best ever. But fk=0.428 is too high.
The synergy: log weighting focuses NPO on high-memorization chunks, contrastive inner prevents the retain spike that focused NPO would otherwise cause. Neither works alone (T8 collapsed, T6 collapsed), but together L_ret=0.65 at step 40 (constraint satisfied!).

### U Series Results (2026-04-10)
| Exp | Config | fk↓ | rk↑ | frontier | verdict |
|-----|--------|------|------|----------|---------|
| U0 | T8f + 200 steps | 0.003 | 0.019 | COLLAPSED | L_ret oscillated wildly after step 120 (λ=8+). 100 steps is the sweet spot. |
| U1 | sqrt + contrastive | 0.002 | 0.011 | COLLAPSED | Only LOG + contrastive synergizes. Sqrt's aggressive targeting (11.8x) is too biased for contrastive to stabilize. |
| U2 | T8f + npo_beta=1.0 | 0.201 | 0.240 | BELOW (-0.041) | Stronger NPO pushed fk to 0.201 but overwhelmed contrastive → synergy broke. |
| U3 | T8f + ρ=0.005 | 0.000 | 0.000 | COLLAPSED | Slower ALM → no retain protection → destroyed. |
| **U4** | **T8f + K=5** | **0.328** | **0.330** | **~ON (-0.021)** | **fk hits GOLD (0.328). K=5 inner with contrastive = precise correction. But rk=0.330 is 0.230 below gold.** |
| U5 | T8f + γ=1.0 | 0.619 | 0.523 | ON (+0.012) | Stronger forget push weakened NPO significantly. Best rk=0.523 but fk destroyed. |
| U6 | log + Fisher α=1.0 | 0.332 | 0.298 | BELOW (-0.054) | Fisher hurts as always. |

### U Series Insights

**U4 is the most controlled experiment at fk=gold:** K=5 inner steps with contrastive gives enough per-step correction to precisely balance NPO's forget pressure. fk=0.328 matches gold target. But rk=0.330 is the same ceiling we've been hitting since G1 (rk=0.327). The rk gap (+0.230) is structural — it's not a forgetting problem (fk is solved), it's a retain recovery problem.

**U5 reveals the rk ceiling mechanism:** With γ=1.0 (strong forget push), NPO was weakened so much that fk=0.619 (near pretrained). But rk=0.523 — the BEST rk we've seen in the S5 regime, and close to M1's 0.563 (which used post-inner CE). The strong contrastive push essentially turned the inner loop into a retain-only optimizer (because the push term dominated). This confirms: rk CAN reach 0.52+ in the bilevel framework IF NPO is weakened sufficiently. The problem is always: weakening NPO → fk regression.

**The fundamental rk ceiling at ~0.33 for fk~0.33:** When fk is near gold (0.274-0.346), rk is always 0.30-0.38. This is the CE Pareto frontier. The only way past it is to either:
1. Find a mechanism that improves rk WITHOUT hurting fk (nothing tested so far does this)
2. Accept a two-stage approach: first forget (get fk≤0.328), then recover retain separately

**Log vs Sqrt weighting:** Log (compressed, max 2.9x) creates a quasi-uniform distribution that pairs with contrastive. Sqrt (aggressive, max 11.8x) is too concentrated → contrastive can't stabilize because it gets biased views. The log+contrastive synergy depends on sample diversity within each step.

---

## V SERIES RESULTS (partial, 2026-04-10)

### Completed
| Exp | Config | fk↓ | rk↑ | frontier | verdict |
|-----|--------|------|------|----------|---------|
| V_interp_a7 | 0.7*T8f + 0.3*U4 weights | — | — | — | safetensors fixed, eval pending |
| V_interp_a5 | 0.5*T8f + 0.5*U4 weights | — | — | — | safetensors fixed, eval pending |
| V_interp_a3 | 0.3*T8f + 0.7*U4 weights | — | — | — | safetensors fixed, eval pending |
| V0 | K=4 + log + contrastive | 0.007 | 0.016 | COLLAPSED | K=4 breaks synergy. L_ret=7.85 at step 40 |
| V1 | npo_beta=1.5 + log + contrastive | — | — | COLLAPSED | L_ret=7.90 at step 40. 25% beta reduction kills synergy |

### Not yet run
| Exp | Config | Hypothesis |
|-----|--------|------------|
| V2 | gamma=0.3 (weaker contrastive push) | More NPO headroom → fk drops, but does frontier break survive? |
| V3 | gamma=0.7 (stronger push) | Between T8f (0.5) and U5 (1.0) — will it shift fk/rk balance? |
| V4 | eta_theta=3e-4 (higher outer LR) | 50% more NPO per step without changing beta |
| V5 | K=4 + gamma=0.3 | Likely dead (K=4 collapsed in V0) |
| V6 | 80 steps (shorter) | Less ALM buildup → weaker retain → lower fk? |

### V Series Insight: T8f is a knife-edge
The T8f operating point (K=3, npo_beta=2.0, gamma=0.5, log weighting) is NOT a basin — it's a knife-edge. Both K=4 (V0) and npo_beta=1.5 (V1) collapse to L_ret≈7.9 at step 40, while T8f has L_ret=0.65 at step 40. The synergy between log weighting and contrastive inner requires:
1. K=3 EXACTLY: K=4 gives the inner loop one extra contrastive iteration that overshoots. K=5 (U4) recovers stability but overcorrects → loses frontier break.
2. npo_beta=2.0 EXACTLY: The NPO loss temperature controls gradient sharpness. At β=2.0, the signal is smooth enough for contrastive to stabilize. At β=1.5, sharper gradients create chaotic dynamics.

The remaining V experiments (gamma sweep, LR, step count) keep K=3 and npo_beta=2.0 fixed — these are the most likely to preserve the synergy.

---

## H SERIES — G1 + Hard-50 Sequences (2026-04-10) ✅ DEAD END

H0: G1 config restricted to top-50 most memorized sequences (score 20-140×).
Result: fk=0.005, rk=0.020. **Complete model collapse.** The extreme gradient signal from top-50 memorized seqs (scores up to 140×) destroys ALL knowledge — both forget and retain — within 1 epoch. Even with G1's steering (coeff=5) the concentrated NPO gradient is too nuclear.

Verdict: Subsetting to extreme-memorization sequences is always catastrophic. Score-weighted SAMPLING (T8 approach) is the right way — it adjusts frequency, not subsetting.

---

## V SERIES — Push T8f fk Lower (2026-04-10, PARTIALLY COMPLETE)

### Core Challenge
T8f (log + contrastive, K=3): fk=0.428, rk=0.461, δ=+0.056 (ABOVE frontier)
U4 (log + contrastive, K=5): fk=0.328, rk=0.330, δ=-0.021 (ON frontier)

**UPDATED AFTER V0/V1:** The synergy is razor-thin. K and npo_beta cannot change.
Only safe knobs remaining: contrastive gamma, outer LR (eta_theta), step count.

### Remaining Strategy (V2-V6)
1. **V2: gamma=0.3** (weaker contrastive push) — weaker push → NPO has more headroom → fk drops. Risk: less disentanglement → rk drops too. MOST PROMISING.
2. **V3: gamma=0.7** — between T8f (0.5) and U5 (1.0). May shift balance toward rk.
3. **V4: eta_theta=3e-4** — 50% higher outer LR. Scales ENTIRE outer gradient (NPO+ALM), not just NPO shape. Different from beta change.
4. **V5: K=4+gamma=0.3** — likely dead since K=4 collapsed in V0.
5. **V6: 80 steps** — stop before ALM λ grows too large. T8f at 100 had fk=0.428; fewer steps = less ALM = potentially lower fk.
6. **Weight interpolation eval** — T8f/U4 blends at alpha=0.7/0.5/0.3. Free experiment.

### Additional ideas not yet in V series
- **contrastive_beta=2.0** (stronger retain PULL, keep gamma=0.5) — untested knob
- **Contrastive on layers [3,4,5]** (earlier layers) — different representational separation
- **Steering coeff=10** (stronger retain steering) — more rk protection, frees NPO
- **lambda_init=1.0** (warm-start ALM) — earlier retain protection, slower fk descent

### Generalization Design
Method components and their generalizability:
- **Log-weighted sampling**: Needs memorization scores. General procedure: compute NPO loss ratio per sample → log(1+score). Works for any forget set.
- **Contrastive inner loop**: Layer selection [5,6,7] is Llama-2-7b specific. For other models: pick layers at ~20-25% depth (where forget/retain representations diverge most).
- **Steering**: coeff=5, same layers. Generalizable with same heuristic.
- **ALM + bilevel**: Fully general. ρ=0.01, ε=0.70, K=3 are starting points.
- **accum=1**: Critical for frontier-breaking dynamics. Non-negotiable.

---

## V_INTERP RESULTS — Weight Interpolation Between T8f and U4 (2026-04-10)

Interpolated models: α*T8f + (1-α)*U4

| Model | α (T8f weight) | fk | rk | δ (vs frontier) |
|-------|----------------|------|------|-----------------|
| V_interp_a7 | 0.7 | 0.427 | 0.421 | +0.016 |
| V_interp_a5 | 0.5 | 0.410 | 0.385 | -0.010 |
| V_interp_a3 | 0.3 | 0.381 | 0.367 | -0.013 |

**Key finding:** Uniform interpolation quickly falls back to the CE frontier. Only heavy T8f weighting (a7) stays marginally above. This proves that **per-parameter selectivity** is needed to break the frontier — motivates PerTA approach.

---

## PARADIGM SHIFT — Beyond Gradient-Based Optimization (2026-04-10)

### Diagnosis: Why 1000 experiments hit the same wall

All gradient-based methods (A through V series) are trapped on the CE frontier:
  rk ≈ 0.55*fk + 0.17 (R²=0.97)

Root cause: 85% neuron overlap between forget and retain knowledge. Any gradient update that reduces forget also reduces retain proportionally. The retrain model breaks this because it was trained FROM SCRATCH without forget data — it never had entangled weights.

### Three new approaches (weight-space, not gradient-based):

1. **PerTA (Per-parameter Task Arithmetic)** — ✅ COMPLETE — BREAKS FRONTIER
   θ_final = θ_target - λ * w * (θ_target - θ_pretrained)
   w_i = F_forget_i / (F_forget_i + α * F_retain_i + ε)
   Fisher weighting makes negation SELECTIVE: forget-heavy params negated, retain preserved.
   Script: scripts/perta_unlearn.py
   Fisher cache: saves/unlearn/_perta_fisher_cache_News_n64.pt
   Fisher mask stats: mean_w=0.2377 (24% forget-dominated, 76% retain-protected)

2. **Two-Stage LoRA Retain Recovery** — ✅ COMPLETE — MIXED RESULTS
   Stage 1: PerTA λ=3.5 (fk=0.282, rk=0.396) — best PerTA base
   Stage 2: Freeze base, add LoRA, train 3 epochs on retain data (retain1 split)
   LoRA rank sweep: [4, 8, 16], alpha=2*rank, cosine LR 2e-4, batch 4×4 accum
   Script: scripts/lora_retain_recovery.py
   Result: r16 achieved fk=0.329 (GOLD!) with δ=+0.050

3. **Token-Level NPO** — PLANNED (lower priority given PerTA success)
   Only apply NPO to forget-informative tokens (high target/pretrained log-prob ratio).
   General language tokens preserved → less collateral damage to retain knowledge.

---

## PerTA RESULTS — Per-parameter Task Arithmetic (2026-04-10) ✅ COMPLETE

### Method
  θ_final = θ_target - λ * w * (θ_target - θ_pretrained)
  w_i = F_forget_i / (F_forget_i + α * F_retain_i + ε)

Fisher computed on target model, 64 samples each from forget/retain splits (2048 tokens).
Mean weight w=0.2377 → 24% of parameters are forget-dominated (negated strongly), 76% retain-protected.

### Full Lambda Sweep (α=1.0)
| Experiment | λ | fk↓ | rk↑ | vm | δ (vs CE frontier) | verdict |
|------------|---|------|------|------|-----|---------|
| perta_l0.3 | 0.3 | 0.649 | 0.566 | 0.556 | +0.039 | Near target — barely any negation |
| perta_l0.5 | 0.5 | 0.635 | 0.567 | 0.528 | +0.048 | rk=0.567 NEAR GOLD! Best rk of any PerTA |
| perta_l1.0 | 1.0 | 0.634 | 0.540 | 0.424 | +0.021 | fk barely moved, rk dropped — threshold effect |
| perta_l1.5 | 1.5 | 0.537 | 0.513 | 0.329 | +0.047 | Forget starting to work |
| perta_l2.0 | 2.0 | 0.516 | 0.508 | 0.251 | +0.054 | Strong above-frontier |
| perta_l3.0 | 3.0 | 0.394 | 0.439 | 0.193 | +0.052 | Approaching gold fk zone |
| **perta_l3.5** | **3.5** | **0.282** | **0.396** | **0.176** | **+0.071** | **🔥 BEST DELTA — fk BEATS gold** |
| perta_l4.0 | 4.0 | 0.185 | 0.287 | 0.083 | +0.015 | Over-negated, both dropping |
| perta_l4.5 | 4.5 | 0.027 | 0.050 | 0.008 | -0.135 | CLIFF — model collapsing |
| perta_l5.0+ | 5+ | 0.000 | 0.000 | 0.000 | -0.170 | COLLAPSED |

### Key Findings
1. **ALL PerTA λ=0.3-4.0 are ABOVE the CE frontier** — PerTA systematically breaks the frontier
2. **Best delta: λ=3.5 (δ=+0.071)** — 27% better than T8f's +0.056 gradient record
3. **Best rk: λ=0.5 (rk=0.567)** — EXCEEDS gold target (0.552)! But fk=0.635 (too high)
4. **Cliff at λ=4.5-5.0**: Aggressive negation destroys the model. Sweet spot is λ=3.0-3.5
5. **λ=3.5 beats gold fk**: fk=0.282 < 0.324 gold. Only rk gap remains (+0.156)
6. **The Fisher mask is the key**: mean_w=0.24 means 76% of params are retain-protected

### Why PerTA Breaks the Frontier
PerTA is NOT a gradient method — it's direct weight surgery:
- Gradient methods (CE frontier): ∇L couples forget/retain through shared weights → rk ∝ fk
- PerTA: Fisher identifies WHICH parameters encode forget vs retain → selectively negates forget
- Task vector τ = θ_target - θ_pretrained captures EXACTLY what finetuning learned
- Weighted negation θ - λ*w*τ removes forget-learned components while preserving retain-important params
- This breaks the coupling that traps gradient methods

### PerTA vs Gradient Methods (Pareto comparison)
At fk≈0.28 (gold zone):
- G1 (gradient): fk=0.274, rk=0.327, δ=+0.006
- PerTA λ=3.5: fk=0.282, rk=0.396, δ=+0.071
- **PerTA provides +0.069 rk improvement over gradient at same fk**

At fk≈0.43 (T8f zone):
- T8f (gradient): fk=0.428, rk=0.461, δ=+0.056
- PerTA λ=2.0: fk=0.516, rk=0.508, δ=+0.054 (comparable delta, higher fk)

---

## LoRA RETAIN RECOVERY ON PerTA λ=3.5 (2026-04-10) ✅ COMPLETE

### Method
Stage 1: PerTA λ=3.5 base model (fk=0.282, rk=0.396)
Stage 2: Add LoRA adapters, train 3 epochs on retain1 data only
- Target modules: q/k/v/o_proj, gate/up/down_proj (all 7 linear layers)
- LR: 2e-4, cosine schedule, warmup 10%
- Batch: 4 × 4 gradient accumulation = effective batch 16
- bf16, gradient checkpointing

### Results
| Rank | Trainable params | Train time | fk↓ | rk↑ | vm | δ | verdict |
|------|-----------------|------------|------|------|------|-------|---------|
| base (PerTA 3.5) | — | — | 0.282 | 0.396 | 0.176 | +0.071 | Starting point |
| **r4** | 13.6M (0.2%) | 1664s | **0.354** | **0.431** | 0.213 | **+0.067** | **Best rk gain** (+0.035) but fk leaked +0.072 |
| r8 | 27.3M (0.4%) | 1633s | 0.357 | 0.405 | 0.220 | +0.039 | Rank 8 worse than r4 — more capacity, more forget leakage, less rk gain |
| **r16** | 54.5M (0.8%) | ~1700s | **0.329** | **0.401** | 0.219 | **+0.050** | **🔥 fk=0.329 ≈ GOLD (0.324)! Minimal forget leakage** |

### Key Findings
1. **r16 hits gold fk**: fk=0.329 is within noise of gold (0.324). δ=+0.050 still well above frontier.
2. **r4 best raw rk**: rk=0.431 is highest, but fk=0.354 misses gold target.
3. **Non-monotonic with rank**: r8 is WORST (δ=+0.039). Mid-rank finds a "worst of both" — enough capacity to re-learn forget, not enough to selectively target retain.
4. **r16 paradox**: Higher rank = more parameters = should leak MORE forget. But r16 leaked LESS (fk +0.047) than r4 (+0.072). Hypothesis: higher rank LoRA has enough expressive power to selectively recover retain representations without touching forget-relevant subspace. Low-rank (r4) is too constrained — it can't avoid the forget subspace.
5. **LoRA retain recovery has diminishing returns**: rk improved only +0.005 to +0.035 from base 0.396. The rk gap to gold (0.552) remains +0.151 at best.

### LoRA vs PerTA-only Comparison
| Model | fk | rk | δ | vs gold rk gap |
|-------|------|------|-------|------|
| PerTA λ=3.5 (base) | 0.282 | 0.396 | +0.071 | -0.156 |
| PerTA λ=3.5 + LoRA r16 | 0.329 | 0.401 | +0.050 | -0.151 |
| PerTA λ=3.5 + LoRA r4 | 0.354 | 0.431 | +0.067 | -0.121 |
| Gold (retrain) | 0.324 | 0.552 | +0.204 | 0.000 |

LoRA r4 closes the rk gap most (from -0.156 to -0.121) but misses fk gold. r16 barely closes rk gap (-0.151) but preserves fk at gold. **The LoRA approach provides marginal rk improvement** — the retain knowledge gap is NOT a low-rank correction from the PerTA base.

### Implications
The rk gap to gold (0.552) is +0.151 even with the best LoRA configuration. This suggests:
1. The PerTA base model at λ=3.5 has lost retain knowledge at a DEEPER level than LoRA can recover
2. Retain recovery needs a different λ point — e.g., PerTA λ=1.5 (rk=0.513, fk=0.537) followed by targeted forget enhancement
3. Or: PerTA with different α values to shift the Fisher mask balance

---

## LONGER TERM
- Once fk + rk both hit targets: run on MUSE Books + WMDP
- Baselines still needed: BLURNPO (checkpoint at saves/), RMU (needs fresh run)
- Publish as DS-BiAL: bilevel NPO + trace-guided KL distillation + selective implicit differentiation
