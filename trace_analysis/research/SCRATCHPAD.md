# Research Scratchpad — LoRA-BiAL (MUSE News)
## Paper writing guide. Last updated: 2026-04-14.

---

## GOLD TARGETS (verified 2026-04-10)
- fk <= 0.324 (forget knowmem ROUGE) — retrain model
- rk >= 0.552 (retain knowmem ROUGE) — retrain model

---

## REFERENCE MODEL EVALS (2026-04-10)

| Model | fk | rk | fv | ex | CE frontier delta | notes |
|-------|------|------|------|------|------|-------|
| **Retrain (gold)** | **0.324** | **0.552** | 0.204 | 0.025 | **+0.204** | WAY above CE frontier |
| Target (finetuned) | 0.654 | 0.544 | 0.569 | 0.302 | +0.014 | ON CE frontier |
| Pretrained (base) | 0.270 | 0.345 | 0.188 | 0.021 | +0.027 | ABOVE CE frontier |

**Critical insight:** The base LLM (never finetuned) has fk=0.270 (below gold) and rk=0.345. Any method with rk < 0.345 hasn't even matched "do nothing." The retrain model sits delta=+0.204 above CE frontier; our best (LoRA-BiAL Ze0) is delta=+0.120 -- 59% of the way.

---

## CE PARETO FRONTIER (key discovery, 2026-04-10)

```
rk ~ 0.55 * fk + 0.17    (R^2 ~ 0.97)
```

ALL gradient-based bilevel methods (series A through R, 1000+ experiments) are trapped on this line. Root cause: 85% neuron overlap between forget and retain knowledge. Any gradient update that reduces forget also reduces retain proportionally. The retrain model breaks this because it was trained from scratch without forget data.

---

## BASELINES

| Method | fk | rk | notes |
|--------|-----|-----|-------|
| GradAscent | 0.003 | 0.008 | model collapses |
| GradDiff | 0.330 | 0.247 | good forget, bad retain |
| NPO standalone | 0.517 | 0.420 | -- |
| SimNPO standalone | 0.584 | 0.470 | -- |
| BLURNPO | 0.581 | 0.532 | checkpoint-100/130 (OOM full run) |
| RMU | 0.516 | 0.457 | -- |
| **PerTA l=3.5** | **0.282** | **0.396** | delta=+0.071, best non-bilevel |

---

## BEST RESULT: LoRA-BiAL Ze0 (2026-04-14)

| Method | fk | rk | vm | delta |
|---|---|---|---|---|
| PerTA l=3.5 (init) | 0.282 | 0.396 | 0.176 | +0.071 |
| **LoRA-BiAL Ze0** | **0.289** | **0.449** | **0.191** | **+0.120** |

### Champion Config
```
Stage 1: PerTA weight surgery
  lambda=3.5, alpha=1.0
  Fisher cache: saves/unlearn/_perta_fisher_cache_News_n64.pt

Stage 2: LoRA bilevel ALM
  LoRA: r=16, alpha=32, 7 modules (q/k/v/o/gate/up/down)
  NPO: beta=4.0
  Bilevel: K=3 inner, T=25 outer
  Adam: outer_lr=3e-5, inner_lr=2e-4
  ALM: epsilon=0.70, rho=0.1, lambda_init=1.0

Saves: saves/unlearn/Ze0_olr3e5
```

### Compared to PerTA alone
- retain: 0.396 -> 0.449 (+0.053, +13%)
- forget: 0.282 -> 0.289 (+0.007, negligible regression)
- delta: +0.071 -> +0.120 (+69% improvement)

---

## HOW WE GOT HERE: The A-to-Z Journey

### Phase 1: Gradient Bilevel (Series A-R, ~1000 experiments)
All trapped on CE Pareto frontier rk ~ 0.55*fk + 0.17. Every trick tried:
- NPO + steering, post-inner recovery, inverted masks, warm-start ALM, two-phase ALM, KL outer, Fisher-weighted gradients, contrastive inner, small-batch (accum=1).
- Best: T8f (log+contrastive) delta=+0.056. Knife-edge, unreproducible.
- Key discovery: accum=1 breaks frontier (S5, 8r) but fragile.

### Phase 2: PerTA Discovery (2026-04-10)
Weight surgery breaks the frontier entirely. Not gradient-based — directly identifies and negates forget-dominant parameters via Fisher information. lambda=3.5 gives fk=0.282 (below gold!) but rk=0.396 (far from gold 0.552). Zero novelty — it's just Task Arithmetic + Fisher mask.

### Phase 3: Failed Combinations (X-Y Series, 2026-04-13)
- X-series: Full-param bilevel from PerTA init. X1f (accum=1, K=10) gave delta=+0.064 but OOM with any enhancement. Required 40-60GB GPU for retain_graph.
- Y-series: PerTA-masked inits + bilevel. All failed — NPO saturates because masked model is near-identical to reference.

### Phase 4: LoRA-BiAL Innovation (Z-Series, 2026-04-13-14)
**Key insight:** Use LoRA adapters so ref model = base model with LoRA disabled. Zero memory overhead — solves OOM permanently (13.6GB vs 40-60GB).

**Z-series (SGD, 8 experiments):** Manual SGD too slow for LoRA. Best Z3 (K=10) delta=+0.075, barely above PerTA.

**Za-series (Adam, 8 experiments):** Adam unlocked real LoRA updates. Za0 (K=5) delta=+0.077. But K=10 collapsed — Adam too aggressive with many inner steps.

**Zb-series (KL + beta tuning, 8 experiments):**
- Tested KL divergence as forget loss (never saturates). Result: WORSE than NPO.
- **Critical discovery: NPO saturation is a FEATURE.** Fast saturation -> outer loop stops -> inner loop gets free retain recovery steps. Slower saturation (low beta, or KL) keeps outer loop fighting inner loop -> hurts retain.
- Zb5 (NPO K=3 beta=2) delta=+0.089. K=3 is sharply optimal.

**Zc-series (beta > 2, 8 experiments):**
- beta=4.0 peak: Zc2 (beta=4, K=3) delta=+0.099.
- beta=8 overshot (+0.078). K<3 won't saturate NPO (+0.071). K=4 dramatically worse (+0.040).
- T=50 destructive (-0.011) — stale outer Adam momentum after saturation.

**Zd-series (fine-tuning, 8 experiments):**
- Found outer_lr=5e-5 improves: Zd6 delta=+0.107. Gentler outer = less retain damage during initial NPO correction.
- epsilon insensitive (0.50-0.90 all similar). PerTA lambda=3.0 too weak.

**Ze-series (outer LR, 8 experiments):**
- Ze0 (olr=3e-5): delta=+0.120. Even gentler outer keeps winning.
- Ze3 (beta=5, olr=5e-5): delta=+0.118. Close second.
- olr=2e-5 (Ze7): pending, may plateau.

### The Story Arc for the Paper
1. Gradient bilevel is fundamentally limited (CE Pareto frontier, 1000 experiments)
2. Weight surgery (PerTA) breaks the frontier but sacrifices retain quality
3. **LoRA-BiAL bridges the gap**: PerTA init (above frontier) + LoRA bilevel (retain recovery)
4. Key mechanism: NPO saturation acts as implicit scheduling — correction first, then free recovery
5. Zero-overhead ref model via LoRA disable trick solves OOM

---

## KEY FINDINGS FROM Z-SERIES (~45 EXPERIMENTS)

### 1. NPO Saturation is a Feature, Not a Bug
With beta=4.0 and K=3 inner steps, NPO saturates in ~5-8 outer steps (L_fgt -> 0). After saturation, the outer loop effectively becomes a no-op. The remaining ~17 steps are pure inner-loop retain recovery — free from interference.

Evidence:
- beta=0.5 (slow saturation): delta=+0.004
- beta=1.0: delta=+0.023
- beta=2.0: delta=+0.089
- **beta=4.0: delta=+0.099** (peak)
- beta=8.0: delta=+0.078 (overshot)
- KL (never saturates): delta=+0.037

### 2. K=3 is Sharply Optimal
- K=1-2: NPO never saturates (too few inner steps to diverge from ref) -> constant tug-of-war
- **K=3: Just enough inner steps to saturate NPO -> optimal balance**
- K=4: delta drops by 0.06 (!)
- K=5+: Too much forget leakage per outer step with Adam

### 3. Gentle Outer LR is Critical
- olr=1e-4: delta=+0.099
- olr=5e-5: delta=+0.107
- **olr=3e-5: delta=+0.120**
- Less retain damage during the initial correction steps.

### 4. T=25 is Optimal, T=50 Hurts
After NPO saturates, the outer Adam optimizer's momentum becomes stale. With T=50, stale momentum creates erratic updates that undo inner loop's work. T=25 provides enough steps without degradation.

### 5. Adam >> SGD for LoRA
Manual SGD barely moved LoRA weights (40M params at lr=2e-4). Adam's adaptive per-parameter LR provides ~10x larger effective updates. Improvement: +0.05 delta.

### 6. Epsilon is Insensitive
epsilon in {0.50, 0.70, 0.90} all give delta within +/- 0.003. The method is robust to constraint tightness.

---

## PerTA — Per-parameter Task Arithmetic

### Method
```
theta_final = theta_target - lambda * w * (theta_target - theta_pretrained)
w_i = F_forget_i / (F_forget_i + alpha * F_retain_i + epsilon)
```

Fisher computed on target model, 64 samples each from forget/retain splits (2048 tokens).
Mean weight w=0.2377 -- 24% of parameters are forget-dominated (negated strongly), 76% retain-protected.

### Lambda Sweep (alpha=1.0)
| lambda | fk | rk | vm | delta | verdict |
|---|---|---|---|---|---|
| 0.3 | 0.649 | 0.566 | 0.556 | +0.039 | Near target |
| 0.5 | 0.635 | 0.567 | 0.528 | +0.048 | rk=0.567 near gold |
| 1.0 | 0.634 | 0.540 | 0.424 | +0.021 | threshold effect |
| 1.5 | 0.537 | 0.513 | 0.329 | +0.047 | forget starting |
| 2.0 | 0.516 | 0.508 | 0.251 | +0.054 | strong above-frontier |
| 3.0 | 0.394 | 0.439 | 0.193 | +0.052 | approaching gold fk |
| **3.5** | **0.282** | **0.396** | **0.176** | **+0.071** | **BEST delta, fk beats gold** |
| 4.0 | 0.185 | 0.287 | 0.083 | +0.015 | over-negated |
| 4.5+ | ~0.000 | ~0.000 | ~0.000 | collapsed | cliff edge |

---

## PAPER FRAMING: LoRA-BiAL

### Method
- **Stage 1:** PerTA weight surgery — Fisher-weighted task vector negation. Places model ABOVE CE frontier (fk=0.282, rk=0.396). No training required.
- **Stage 2:** LoRA bilevel ALM — LoRA adapters (r=16) with bilevel optimization. Inner loop: retain CE recovery. Outer loop: NPO forget correction. Ref model = base model with LoRA disabled (zero memory overhead).

### Novel Contributions
1. LoRA-based bilevel with zero-overhead reference model (LoRA disable trick)
2. NPO saturation as implicit scheduling mechanism (first to characterize and exploit)
3. PerTA + bilevel combination (weight surgery init for gradient optimization)
4. CE Pareto frontier analysis (1000+ experiments showing fundamental gradient limitation)

### Key Result
Ze0: fk=0.289, rk=0.449, delta=+0.120. Beats all gradient baselines. 69% improvement over PerTA alone.

### Generalization
- PerTA: General — needs pretrained + target + forget/retain Fisher. Works for any forget set.
- LoRA-BiAL: General — beta=4, K=3, T=25, olr=3e-5 as starting config. Memory: ~14GB (fits any single GPU).
- Must validate on MUSE Books + WMDP.

---

## PLAN: MUSE Books Benchmark

### Goal
Validate LoRA-BiAL generalizes from News to Books domain. Same model (Llama-2-7b-hf), different forget/retain splits.

### Step 1: Compute Fisher Cache for Books
```bash
python scripts/perta_masked.py \
    --data_split Books \
    --n_samples 64 \
    --output saves/unlearn/_perta_fisher_cache_Books_n64.pt
```
Uses `muse-bench/MUSE-Books` with forget/retain1 splits. ~30 min on single GPU.

### Step 2: Run PerTA Baselines (Books)
PerTA lambda sweep: {1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0}. Use same eval pipeline:
```bash
python src/eval.py experiment=eval/muse/default.yaml data_split=Books ...
```
Identify the best PerTA lambda for Books (may differ from News's 3.5 due to different forget/retain overlap).

### Step 3: Run Gradient Baselines (Books)
Same baselines as News: GradAscent, GradDiff, NPO, SimNPO, RMU.
Configs already exist — just change `data_split=Books`.
```bash
python src/train.py --config-name=unlearn.yaml experiment=unlearn/muse/<method> data_split=Books ...
```

### Step 4: Run LoRA-BiAL (Books)
Start with Ze0's champion config, only changing data_split and fisher_cache_path:
```bash
python src/train.py --config-name=unlearn.yaml \
    experiment=unlearn/muse/lora_bial \
    data_split=Books \
    task_name=lora_bial_books_v0 \
    trainer.method_args.fisher_cache_path=saves/unlearn/_perta_fisher_cache_Books_n64.pt \
    trainer.method_args.perta_lambda=3.5 \
    trainer.method_args.npo_beta=4.0 \
    trainer.method_args.K=3 \
    trainer.method_args.T=25 \
    trainer.method_args.eta_theta=3e-5 \
    trainer.method_args.eta_in=2e-4 \
    ...
```

### Step 5: Tune if Needed
If Books has different characteristics (e.g., higher forget/retain overlap), may need:
- Different PerTA lambda (try 2.5-4.0 range)
- Slightly different beta (3.0-5.0 range)
- The core recipe (K=3, gentle outer LR, NPO saturation) should transfer

### Expected Timeline
- Fisher cache: ~30 min
- PerTA sweep (7 configs): ~30 min (no training)
- Baselines (5 methods): ~3-4 hours
- LoRA-BiAL (1-3 configs): ~30 min
- Total: ~5 hours

### Success Criteria
- LoRA-BiAL beats all gradient baselines on Books
- LoRA-BiAL beats PerTA-alone on Books (delta improvement)
- Confirms method generalizes across domains
