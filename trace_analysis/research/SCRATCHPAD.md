# Research Scratchpad — DS-BiAL MUSE News
## For agent continuity. Last updated: 2026-04-08 ~19:45

---

## BREAKTHROUGH (2026-04-08 18:30)
F0 (bare NPO, T=25, ε=0.95, K=1, no implicit, no steering): **fk=0.325 — BEATS GOLD (0.328)**
Retain is 0.290 (bad). All remaining work = recover retain without hurting fk.

## F SERIES COMPLETE (2026-04-08 ~19:30)
**New anchor: F3** (fk=0.325, rk=0.316, ε=0.70) — strictly better than F0 (same fk, rk +0.026).
Tighter epsilon gives free retain improvement. G series now builds on F3, not F0.

---

## ACTIVE STRATEGY: Strip-and-build on F3 (updated anchor)

1. F3 = anchor (fk=0.325, rk=0.316, ε=0.70). Never cross fk > 0.340.
2. Add one retain-recovery component at a time.
3. Keep it if rk↑ without fk crossing 0.340.
4. Combine keepers into G3/G4, then H series with memorization-scored data.

### Full stacking roadmap
```
Layer 0: F3 — ε=0.70, bare NPO, fk=0.325 rk=0.316
Layer 1: G series — one component at a time on F3
  G0: +post_inner=100   (CE retain after NPO)
  G1: +steering coeff=5 (gentle activation anchor)
  G2: +K=5              (more inner retain enforcement)
Layer 2: Stack keepers → G3 = F3 + best1 + best2
Layer 3: H series — best_G config + top-50 memorized sequences only
  H0 = best_G + memorization-filtered data
Layer 4 (if needed): Neuron masking during post_inner (bitmap from traces)
         Freeze forget-dominant layers during retain recovery
         Implicit correction stacked with keepers
```

---

## ALL RESULTS

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
| Exp | fk↓ | rk↑ | config | conclusion |
|-----|-----|-----|--------|------------|
| A1a–C4 logit_margin | 0.52–0.54 | 0.46–0.50 | various | DEAD END — hard ceiling at 0.52 |
| D1 NPO T=75 | 0.490 | 0.461 | ε=0.8, full dataset | too many steps, dilutes signal |
| D2 NPO+steering T=75 | 0.513 | 0.486 | ε=0.8, full dataset | steering hurts forget at T=75 |
| **F0 bare NPO T=25** | **0.325** | 0.290 | ε=0.95, K=1, 1 epoch | ✅ BEATS GOLD on forget |
| F1 +steering coeff=20 | 0.358 | 0.349 | F0+steering | fk too high (+0.033), rk +0.059 |
| F2 +implicit | 0.335 | 0.312 | F0+Neumann | marginal gains only |
| **F3 +ε=0.70** | **0.325** | **0.316** | tighter constraint | ✅ NEW ANCHOR — free rk +0.026, fk unchanged |
| F4 T=25 explicit | 0.359 | 0.311 | epoch override already = 25 steps | worse, T config doesn't override epoch |
| G0 +post_inner=100 | 0.409 HURT | 0.352 | F3+CE retain after NPO | rk+0.036 but fk blows up — CE re-learns forget |
| **G1 +steering coeff=5** | **0.274** | **0.327** | F3+gentle steering | ✅ KEEPER — fk improved below F3! |
| G2 +K=5 inner | 0.314 | 0.310 | F3+more inner steps | marginal, skip |
| G3 +masked post_inner=100 | ⏳ | ⏳ | F3+post_inner+bitmap mask | running |
| G3b +masked post_inner=25 | ⏳ | ⏳ | F3+25 steps+bitmap mask | queued |

---

## TRACE-GUIDED DATA SAMPLING (NEXT MAJOR IDEA)

### The "Hogwarts vs boarding school" insight
Not all forget sequences are equal:
- **High-memorization** ("Hogwarts"): specific facts, names, dates memorized during fine-tuning.
  Fine-tuned model assigns much lower perplexity than base model on these.
  → GOOD forget targets: forgetting these reduces fk without hurting generic language quality.
- **Low-memorization** ("British boarding school"): generic writing patterns shared with retain.
  Fine-tuned model ≈ base model perplexity.
  → BAD forget targets: training on these damages retain without helping fk.

### Memorization score (proxy mechanism)
```
memorization_score(seq) = loss_base(seq) / loss_finetuned(seq)
npo_log_ratio(seq)      = loss_base(seq) - loss_finetuned(seq)
```
- `npo_log_ratio > 0` → fine-tuned model is MORE confident than base = memorized
- `npo_log_ratio ≈ 0` → generic content, shared with base model knowledge
- High npo_log_ratio = exactly what NPO targets during training

**This is essentially pre-computing the per-sample NPO importance weight before training.**

### Why this should help retain
Current F0 trains on ALL 25 batches = 800 sequences, many generic.
Generic sequences = shared representations with retain → training on them corrupts retain.
If we restrict to top-50 most memorized sequences:
- Forget signal is concentrated on truly memorized content → fk stays low or better
- Generic sequences are skipped → less retain collateral damage → rk improves

### Implementation (ready at scripts/score_forget_memorization.py)
```bash
# Run AFTER GPU is free (needs both models loaded):
python scripts/score_forget_memorization.py \
    --finetuned_model muse-bench/MUSE-News_target \
    --base_model /datadrive/... (local Llama-2-7b-hf) \
    --top_k 50 \
    --hard_forget_path data/hard_forget_news.jsonl
```
Outputs:
- `trace_analysis/figures/traces/analysis/forget_memorization_scores.json` — full ranking
- `data/hard_forget_news.jsonl` — top-50 sequences for targeted training

### Next experiment after scoring
- H0: F0 config but forget dataset = top-50 memorized sequences only
  - T=10 steps over 50 sequences (2 full passes) vs T=25 over 800
  - Expect: fk similar or better, rk better (less generic damage)
- H1: H0 + best G retain component

### Literature backing
- Carlini et al. 2022 "Quantifying Memorization Across Neural LMs" — same metric
- NPO loss is implicitly per-sample importance weighting via log ratio
- ROME/MEMIT: target interventions to most causally relevant facts
- MUSE paper: forget set contains varying memorization levels by design

---

## KEY INSIGHTS (confirmed)

1. **logit_margin = dead end** — hard ceiling at fk≈0.52, never beats gold
2. **T=75 full dataset = too diluted** — NPO with T=25 (1 epoch) gets fk=0.325, T=75 gets 0.490
3. **Steering coeff=20 hurts forget** — too strong, dominates NPO signal. Try coeff=5.
4. **post_inner FAILED before because forgetting was insufficient** — now NPO forgets (fk=0.325), CE retain recovery should work (G0 tests this)
5. **ε=0.95 ≈ unconstrained** — ALM constraint almost never fires, NPO runs free → good forget
6. **K=1** — almost no retain enforcement during training. Retain gap comes from this.

---

## RUNNER STATE
```bash
# F series DONE. G series launching now (2026-04-08 ~19:45).
# G configs updated to use F3 anchor (ε=0.70 instead of 0.95).
# Progress: /tmp/ablation_G_progress.log
bash scripts/run_G_series.sh

# After G series, score memorization and run H series:
python scripts/score_forget_memorization.py --top_k 50 --hard_forget_path data/hard_forget_news.jsonl
bash scripts/run_H_series.sh  # (to be created after G results)
```

---

## G0 FAILED (2026-04-08 ~23:00) — post_inner re-learns forget
G0: fk=0.409 (HURT), rk=0.352 (+0.036). CE steps update forget-dominant neurons too.
→ G3 created: same as G0 but post_inner_retain_only=True + neuron bitmap mask
  Only retain-dominant neurons (bitmap=0) get updated during CE recovery.
  Forget-dominant neurons (bitmap=1) stay frozen.
  bitmap: trace_analysis/figures/traces/analysis/forget_neuron_bitmap.pt

## LONGER TERM
- Once fk + rk both hit targets: run on MUSE Books + WMDP
- Publish as DS-BiAL: bilevel NPO with trace-guided sample selection + staged retain recovery
