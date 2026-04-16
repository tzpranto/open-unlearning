# Surgical LLM Unlearning: From Bilevel Optimization to Data-Aware Methods

**Progress Report | April 15, 2026**

---

# Part I: MUSE-News (Llama-2-7b-hf)

## 1. Problem Setup

**Task:** Remove knowledge of specific data (e.g., news articles) from a finetuned LLM while preserving general and retain-set knowledge.

**Primary benchmark:** MUSE-News (Llama-2-7b-hf) -- 888 forget / 1777 retain news articles.

| Reference Model | fk (forget ROUGE, lower=better) | rk (retain ROUGE, higher=better) |
|---|---|---|
| Target (finetuned, pre-unlearn) | 0.654 | 0.544 |
| **Retrain (gold standard)** | **0.324** | **0.552** |
| Pretrained (base, never finetuned) | 0.270 | 0.345 |

**CE Pareto frontier delta** = distance above the line `rk ~ 0.55*fk + 0.17`. Gold: delta=+0.204.

---

## 2. Existing Baselines

| Method | fk | rk | verbmem | extract | Notes |
|---|---|---|---|---|---|
| GradAscent | 0.003 | 0.008 | 0.049 | 0.008 | Complete model collapse |
| GradDiff | 0.330 | 0.247 | 0.005 | 0.008 | Good forget, terrible retain |
| NPO | 0.517 | 0.420 | 0.357 | 0.096 | -- |
| SimNPO | 0.584 | 0.470 | 0.259 | 0.052 | -- |
| BLURNPO | 0.581 | 0.532 | 0.356 | 0.124 | Best existing baseline |
| RMU | 0.516 | 0.457 | 0.285 | 0.055 | -- |

---

## 3. Phase 1: SIBL Bilevel Framework (Exp1-9d, ~40 experiments)

Full-parameter bilevel optimization with augmented Lagrangian (ALM): inner loop = retain CE, outer loop = forget loss + ALM penalty.

### Key discoveries by experiment group:

**Exp1-5 (masking/freezing):**
- Raw bilevel forgets well (fk=0.511, rk=0.469) but damages retain
- Freezing layers 0-7 kills forgetting (fk=0.636) even though trace analysis shows 0% forget neurons there
- Neuron-level masks **hurt** forgetting (fk=0.665 with 15% forget mask vs 0.511 without)
- **Takeaway:** Forgetting requires modifying **shared** representations, not just forget-specific neurons

**Exp6 (gradient projection, 12 experiments):**
- Project outer gradient orthogonal to retain gradient direction
- Full projection: strongest forget ever (fk=0.409, extract=0.062) but retain collapses (rk=0.311)
- **Rescaled projection + decay schedule** (Exp6j): fk=0.484, rk=0.489 -- first to Pareto-dominate all baselines
- **Root cause discovered:** forget/retain gradients are 70-80% correlated in weight space

**Exp7 (activation steering + NPO, 12 experiments):**
- Steering at layers [5,6,7] pushes forget representations toward retain targets
- Combined with NPO: near-gold forgetting. Best results:

| Exp | Config | fk | rk | verbmem | extract |
|---|---|---|---|---|---|
| 7i | NPO beta=3, steer, T=10 | 0.406 | 0.410 | **0.197** | **0.051** |
| 7j | NPO beta=3, K=15, rho=1 | **0.372** | 0.416 | 0.248 | 0.063 |

- **7i beats gold on verbatim memorization** (0.197 vs 0.201)
- Retain plateaus at ~0.41-0.42 regardless of AL strength -- structural limit

**Exp8 (FD-HVP implicit correction, 8 experiments):**
- Solved flash_attention_2 incompatibility (no `create_graph` needed)
- FD-HVP is very sensitive to NPO beta: beta=2.0 optimal, beta=3.0 overcorrects (implicit correction suppresses forgetting)
- **Best: Exp8r** (beta=2.0, FD-HVP): fk=0.371, rk=0.417, vm=0.211, ex=0.053

**Exp9 (post-inner recovery, 4 experiments):**
- Masked post-inner (Exp9a): retain perfectly restored (0.561 = gold) but **forget fully restored too** (0.641)
- **Critical finding:** Same-domain knowledge is non-localizable. Training 85% of params on news restores ALL news knowledge
- Stronger AL (K=15, rho=1) does NOT close the retain gap. It's structural.

### Phase 1 Pareto frontier:
```
Best forget:    Exp8r  (fk=0.371, rk=0.417, vm=0.211, ex=0.053)
Best extract:   Exp7i  (fk=0.406, rk=0.410, vm=0.197, ex=0.051)
Best retain:    Exp6i  (fk=0.534, rk=0.508)
```

**Structural retain ceiling at ~0.42.** Single-pass bilevel cannot close the gap to gold (0.552).

---

## 4. Key Discovery: CE Pareto Frontier (Series A-R, ~1000 experiments)

Rerunning with properly controlled baselines confirmed **all gradient bilevel methods are trapped on a linear frontier:**

```
rk ~ 0.55 * fk + 0.17    (R^2 ~ 0.97)
```

**Root cause:** 85% neuron overlap between forget and retain knowledge. Every gradient step that reduces forget proportionally reduces retain. Tested: NPO+steering, inverted masks, warm-start ALM, two-phase ALM, KL outer, Fisher-weighted gradients, contrastive inner, small-batch. Best gradient-only delta=+0.056 (fragile).

---

## 5. PerTA: Breaking the Frontier via Weight Surgery

**Method:** Fisher-weighted Task Arithmetic (no training, <1 min).
```
theta_final = theta_target - lambda * w * (theta_target - theta_pretrained)
w_i = F_forget_i / (F_forget_i + F_retain_i + eps)
```
~24% of parameters are forget-dominated and selectively negated.

| lambda | fk | rk | delta |
|---|---|---|---|
| 0.5 | 0.635 | 0.567 | +0.048 |
| 2.0 | 0.516 | 0.508 | +0.054 |
| **3.5** | **0.282** | **0.396** | **+0.071** |
| 4.0 | 0.185 | 0.287 | +0.015 |

fk=0.282 beats gold (0.324), but rk=0.396 far from gold 0.552. Weight surgery overshoots on retain.

---

## 6. LoRA-BiAL: Two-Stage Method (Z-series, 45+ experiments)

### Architecture
- **Stage 1:** PerTA weight surgery (lambda=3.5) -- places model above CE frontier
- **Stage 2:** LoRA bilevel ALM (r=16) -- recovers retain quality
  - Inner loop: retain CE minimization
  - Outer loop: NPO forget constraint (beta=4.0)
  - Ref model = base model with LoRA disabled (zero memory overhead, 13.6GB)

### Champion: Ze0

| Method | fk | rk | verbmem | extract | delta |
|---|---|---|---|---|---|
| PerTA alone | 0.282 | 0.396 | 0.176 | 0.020 | +0.071 |
| **LoRA-BiAL Ze0** | **0.289** | **0.449** | **0.191** | **0.028** | **+0.120** |

retain +13% (0.396->0.449), forget essentially unchanged. Delta improves 69%.

### Systematic ablation (Z through Ze):

| Series | What we tested | Key finding | Best delta |
|---|---|---|---|
| Z (8) | SGD vs Adam | SGD too slow for LoRA; Adam essential | +0.075 |
| Za (8) | Inner steps K | K=10 collapses with Adam | +0.077 |
| Zb (8) | KL vs NPO + beta | **NPO saturation is a feature**; KL worse | +0.089 |
| Zc (8) | Beta > 2, K tuning | **beta=4, K=3 sharply optimal** | +0.099 |
| Zd (8) | Outer LR, epsilon | Gentler LR helps; epsilon insensitive | +0.107 |
| Ze (8) | Final LR sweep | **olr=3e-5 optimal** | **+0.120** |

### NPO saturation as implicit scheduling
With beta=4, K=3: NPO saturates in ~5-8 outer steps (loss->0). Remaining ~17 steps = pure inner-loop retain recovery with no interference. This is the key mechanism.

| beta | delta | Mechanism |
|---|---|---|
| 0.5 | +0.004 | Never saturates, constant tug-of-war |
| 2.0 | +0.089 | -- |
| **4.0** | **+0.099** | Fast saturation -> free retain recovery |
| 8.0 | +0.078 | Overshoots |
| KL | +0.037 | Never saturates |

### Constraint sweet spot
Ze0's success = **constrained optimization**: r=16 (40M params), T=25 steps. More capacity is harmful:
- LoRA r=64, 101 steps (Phase 6 v2): fk regresses from 0.376 to 0.423 -- LoRA overwrites PerTA surgery
- Full-epoch training: non-monotonic, best at step 300, degrades after

---

## 7. Seed Stability (Critical Weakness)

Ze0 used deterministic first-64 samples for Fisher estimation. Testing 3 random seeds:

| Fisher seed | fk | rk | delta |
|---|---|---|---|
| **original (first 64)** | **0.289** | **0.449** | **+0.120** |
| seed 42 | 0.295 | 0.405 | +0.073 |
| seed 123 | 0.433 | 0.437 | +0.029 |
| seed 456 | 0.267 | 0.370 | +0.052 |

**Delta varies 4x (0.029 to 0.120).** Fisher sample selection is more impactful than all bilevel hyperparameters combined.

---

## 8. GSP: Gradient Subspace Partitioning (Latest Work)

### Motivation
The seed stability problem + Phase 1's same-domain non-localizability finding suggest the core issue is **data entanglement** -- forget and retain samples share gradient subspaces. We need to understand and exploit per-sample gradient structure.

### Method
1. Compute per-sample gradient signatures (L2-normalized grad norms on `down_proj.weight`, last 4 layers)
2. Build interference matrix: `I[i,j] = |cosine(g_forget_i, g_retain_j)|`
3. Per-sample entanglement: `E_f[i] = mean(top-k retain cosine similarities)`
4. Use entanglement to weight sampling: down-weight entangled forget, up-weight entangled retain

### MUSE News GSP Analysis: Mathematically Futile

| Statistic | E_f (forget) | E_r (retain) |
|---|---|---|
| Mean | 0.972 | 0.971 |
| Std | **0.009** | **0.007** |
| Range | 0.825 - 0.996 | 0.897 - 0.993 |

**E_f std = 0.009 with mean 0.972.** The interference is virtually uniform -- every forget sample is equally entangled with retain (cosine ~0.97). The IQR is only 0.008. GSP's own analysis flagged: "LOW SPREAD: interference is uniform, GSP partition may not help much."

**Root cause confirmed:** Both forget and retain are BBC News articles. Same domain, same language patterns, same gradient subspace. There is no separable structure to exploit.

### GSP-SIBL Experiments on MUSE News (ran anyway)

| Config | fk | rk | extract | Verdict |
|---|---|---|---|---|
| gsp_baseline (a=0, b=0) | 0.164 | 0.192 | 0.017 | Collapsed (over-forgot) |
| gsp_a5b5 | 0.240 | 0.266 | 0.030 | Collapsed (over-forgot) |
| gsp_a20b20 | -- | -- | -- | Training incomplete |

Both completed runs **collapsed** -- fk and rk far below even the pretrained base (0.270/0.345). The bilevel with logit_margin forget loss on epoch-based training over-forgets without PerTA initialization. GSP weighting with such small spread (max weight ratio 1.9%) has negligible effect.

**Conclusion:** GSP is not viable on same-domain benchmarks. The entanglement is structural, not sample-specific.

---

## 9. Pivot: WMDP Cyber (Cross-Domain Test)

WMDP Cyber provides what MUSE News lacks: **semantically orthogonal domains**.

| | MUSE News | WMDP Cyber |
|---|---|---|
| Forget | BBC News articles | Exploit code, backdoor stagers |
| Retain | BBC News articles | Benign code (OpenBSD, SoftFloat) |
| Model | Llama-2-7b | Zephyr-7b-beta |
| Domain overlap | **Same domain** | **Orthogonal domains** |
| Expected E_f std | 0.009 (measured) | ~0.05-0.15 (hypothesis) |

**Status:** GSP signatures computed for WMDP Cyber (1000 forget, 4473 retain). Interference analysis and GSP-SIBL experiments queued. Also testing LoRA-Implicit (FD-HVP bilevel) on WMDP.

**Practical challenge:** WMDP's corpus is substantially larger than MUSE News (4473 retain + 1000 forget vs 1777 + 888), and the model is Zephyr-7b (Mistral architecture). Each training run takes ~12 hours on our single GPU, making hyperparameter tuning impractical. We can run at most 2 configs/day, so we must transfer settings from MUSE News rather than sweep.

**Hypothesis:** Cross-domain data should show bimodal entanglement distribution, enabling GSP to meaningfully differentiate samples. If confirmed, this validates the data-aware approach and provides a novelty angle.

---

## 10. Summary: What We Learned (MUSE)

### Confirmed Findings

| # | Finding | Evidence |
|---|---|---|
| 1 | **CE Pareto frontier**: gradient bilevel is fundamentally limited | 1000+ experiments, R^2=0.97 |
| 2 | **Gradient projection** can decouple forget/retain in weight space | Exp6j Pareto-dominates all baselines |
| 3 | **NPO + steering** achieves near-gold verbmem (0.197 < gold 0.201) | Exp7i, 8r |
| 4 | **Retain gap (~0.14) is structural** for single-pass bilevel | Exp9a-b, stronger AL doesn't help |
| 5 | **PerTA breaks the frontier** (delta=+0.071, no training) | Weight surgery, instant |
| 6 | **LoRA-BiAL Ze0** (PerTA + LoRA bilevel) = **best overall** (delta=+0.120) | 45 ablation experiments |
| 7 | **NPO saturation is a feature**: implicit scheduling mechanism | beta sweep, K=3 sharp optimum |
| 8 | **Constraint sweet spot**: r=16, T=25 > r=64, full epoch | Phase 6 v2 confirmed |
| 9 | **Fisher sample sensitivity**: delta varies 4x | Seed stability test |
| 10 | **Same-domain entanglement is uniform**: GSP futile on MUSE News | E_f std=0.009 |

### Best Numbers per Approach (MUSE News)

| Method | fk | rk | vm | ex | delta | Key strength |
|---|---|---|---|---|---|---|
| Gold (retrain) | 0.324 | 0.552 | 0.204 | 0.025 | +0.204 | -- |
| BLURNPO (best baseline) | 0.581 | 0.532 | 0.356 | 0.124 | ~0 | -- |
| Exp8r (NPO+FD-HVP+steer) | 0.371 | 0.417 | 0.211 | 0.053 | -- | Best forget in bilevel |
| Exp7i (NPO+steer) | 0.406 | 0.410 | **0.197** | **0.051** | -- | Beats gold verbmem |
| PerTA lambda=3.5 | **0.282** | 0.396 | 0.176 | 0.020 | +0.071 | No training |
| **LoRA-BiAL Ze0** | 0.289 | **0.449** | 0.191 | 0.028 | **+0.120** | **Best overall** |

---

# Part II: TOFU (Llama-3.2-1B-Instruct)

## 11. Problem Setting

**Benchmark:** TOFU (Task of Fictitious Unlearning). A Llama-3.2-1B-Instruct model fine-tuned on 4000 fictitious QA pairs. Forget set = 1% (40 QA pairs); retain set = 99% (3960 pairs).

**Metrics:**
- FQ (forget quality): KS test p-value comparing truth_ratio distributions of forget vs. retain. Higher=better; 1.0 = indistinguishable from retrained model.
- MU (model utility): aggregate of retain-set and general knowledge scores.
- ES (extraction strength): how much target info can be extracted. ES(Df) should be low; ES(Dr) should remain high.

**Key challenge -- the cold-start problem.** The fine-tuned model is overfit: CE(forget) and CE(retain) are both ~0.001. All gradient-based forget losses (GA, NPO) depend on CE gradients, which are near-zero at initialization. Standard methods cannot produce meaningful updates.

---

## 12. Our Method: LoRA-Implicit (Two-Phase Bilevel)

### Phase 1: Cold-Start Breaker
- Wrap model with LoRA (r=16), bilevel with **logit_margin** as forget loss
- logit_margin = mean over answer tokens of [max logit - mean logit] (~24 at init)
- Provides strong gradient signal regardless of CE saturation
- Inner loop: K steps of retain CE; Outer: logit_margin + ALM

### Phase 2: Bilevel Refinement
- Fresh LoRA on Phase 1 output, switch to **NPO** forget loss
- Ref model = base with LoRA disabled (zero overhead)
- **FD-HVP implicit correction** via conjugate gradient (accounts for inner loop response)

---

## 13. TOFU Results

### Baselines (from PerTA paper)

| Method | FQ | MU | ES(Df) | ES(Dr) |
|---|---|---|---|---|
| Full (no unlearn) | 0.007 | 0.599 | 0.743 | 0.737 |
| GT (retrain) | 1.000 | 0.599 | 0.069 | 0.751 |
| GA | 0.011 | 0.597 | 0.189 | 0.656 |
| NPO | 0.009 | 0.595 | 0.178 | 0.650 |
| TV | 0.405 | 0.556 | 0.081 | 0.358 |
| PerTA-grad | 0.514 | 0.581 | 0.075 | 0.551 |
| PerTA-fisher | 0.266 | 0.586 | 0.085 | 0.600 |

### Our Phase 1 trajectory (v1: K=1, LR=5e-4, logit_margin)

| Checkpoint | FQ | MU | ES(Df) |
|---|---|---|---|
| step-4 (1 epoch) | **0.054** | **0.573** | 0.120 |
| step-8 (2 epochs) | 0.054 | 0.017 | 0.029 |
| step-12 | 0.007 | 0.536 | 0.029 |
| + Phase 2 (NPO) | 0.003 | 0.563 | 0.036 |

**Status:** v2 run in progress with K=3, LR=2e-4, and answer-masked logit_margin (bug fix: logit_margin was previously computed over all tokens including the prompt).

---

## 14. TOFU Key Insights

1. **Cold-start problem is real.** CE-based losses (GA, NPO) produce near-zero gradients on overfit models. This explains FQ < 0.02 for standard methods.
2. **logit_margin solves cold-start.** Margin ~24 at step 0. After 1 epoch, FQ=0.054 -- already 5x better than NPO.
3. **Over-forgetting hurts FQ.** FQ measures distributional similarity to retrained model. Driving ES(Df) below retrained ES(Df)=0.069 is penalized. Goal = match retrain distribution.
4. **Answer-token masking is critical.** Original logit_margin on all tokens wasted gradient on prompt. Masking to answer tokens focuses forgetting.
5. **Target to beat:** PerTA-grad (FQ=0.514, MU=0.581). Our training approach needs to match or exceed this.

---

# Part III: Open Questions & Next Steps

1. **WMDP Cyber GSP**: Does cross-domain data show structured entanglement? (Running, but ~12h/run limits tuning)
2. **Gradient Diversity Sampling**: Can principled Fisher sample selection stabilize Ze0? (Planned)
3. **MUSE-Books**: Validate LoRA-BiAL generalizes across domains (Planned)
4. **TOFU v2**: Answer-masked logit_margin + K=3 (Running)
5. **Combining approaches**: Can LoRA-BiAL benefit from steering or FD-HVP?

### Paper Framing Options

**Option A (LoRA-BiAL focused):** PerTA + LoRA bilevel pipeline. Contributions: CE frontier analysis, NPO saturation mechanism, constraint sweet spot. Weakness: seed stability.

**Option B (Data-aware unlearning):** GSP + bilevel. Requires WMDP results to validate. Contributions: gradient interference analysis, data-aware sampling. Risk: may not outperform simpler methods.

**Option C (Comprehensive):** Full journey from gradient limitations to weight surgery to data-aware methods. Strongest narrative but needs WMDP + Books + TOFU results.

---

*Total experiments: ~1100+ (A-R: ~1000, SIBL Exp1-9d: ~40, Z-series: ~45, GSP: ~6, TOFU: ~5, misc: ~20)*
