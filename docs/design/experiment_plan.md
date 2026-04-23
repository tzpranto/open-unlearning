# Experiment Plan for EMNLP Submission

Status: Draft (2026-04-23)

## Priority Tiers

**P0 (Must-have for submission)** — Reviewers will reject without these.
**P1 (Strong recommendation)** — Very likely to be requested in revision.
**P2 (Nice to have)** — Strengthens the paper but not blocking.

---

## P0: Component Ablations

**Goal**: Show each component contributes. This is the #1 reviewer concern.

| Ablation | What changes | Expected outcome |
| --- | --- | --- |
| No clamp (unclamped entropy) | Remove ReLU clamp, use raw `τ·H_max - H(t)` | More aggressive forgetting, retain degrades |
| No bilevel (single-loop) | Remove inner loop (K=0), train forget+retain in one step | Retain degrades; forget-retain tradeoff worse |
| Symmetric dual | Replace asymmetric update with standard `λ ← max(0, λ + ρ·r)` | λ oscillates; possible retain spikes |
| Fixed λ (no ALM) | Remove dual update, use λ=λ_init fixed | Either over-forgets (λ low) or under-forgets (λ high) |
| No LoRA (full fine-tune) | Train all params, not just LoRA | More forgetting power but likely retain collapse |

**Run on**: TOFU 1B forget01 (fastest setting, ~15 min per run).
**Estimated time**: 5 ablations × 15 min × 3 seeds = ~4 hours.
**Output**: Ablation table with MU, FQ, fgt_Prob, fgt_ROUGE, HM for each variant.

---

## P0: Multiple Seeds with Error Bars

**Goal**: Show results are reproducible, not lucky seeds.

- Run 3 seeds (42, 123, 456) for **all main results**:
  - TOFU 1B T=100 and T=150
  - TOFU 3B T=100
  - MUSE Books
- Report mean ± std for all metrics.
- Also run 3 seeds for top baselines (PDU, SimNPO, RMU) to get baseline error bars.

**Estimated time**:
- TOFU 1B: 4 configs × 3 seeds × 15 min = 3 hours
- TOFU 3B: 1 config × 3 seeds × 45 min = 2.5 hours
- MUSE Books: 1 config × 3 seeds × 100 min = 5 hours
- Baselines: ~6 hours (3 methods × 2 benchmarks × 3 seeds)
- **Total**: ~16 hours

---

## P0: TOFU forget05 and forget10

**Goal**: Show method works beyond the easiest (1%) forget split.

- forget05: 5% of authors (10 authors, 200 QA pairs)
- forget10: 10% of authors (20 authors, 400 QA pairs)

These are harder — more forget data means more retain interference. Should show LoRA-BiAL's ALM adapts to increased difficulty (λ rises higher).

**Run on**: Llama-3.2-1B (fastest). If time permits, also 3B.
**Hyperparameter note**: May need to adjust ε_mul (try 0.85, 0.90) and T (try 150, 200, 300).
**Estimated time**: 2 splits × ~3 HP configs × 15 min = ~1.5 hours for 1B.

---

## P1: MUSE News

**Goal**: Address the suppressed-results concern. Current HM=0.522, below PDU (0.554).

**Diagnosis**: MUSE News has different characteristics than Books:
- forget=889 documents (vs ~1 book), retain=1777 documents
- Content is news articles (factual, short) vs book text (narrative, long)
- Our method's verbmem=0.388 is high — clamped entropy may need lower τ

**Plan**:
1. Grid search over τ ∈ {0.5, 0.6, 0.7} × ε_mul ∈ {0.80, 0.85, 0.90}
2. Try K=5 (more inner steps for the larger retain set)
3. Try higher T (200, 300 steps)

**If we can't beat PDU**: Include MUSE News honestly in the paper with analysis of why the method works better on Books than News (concentrated vs distributed forget sets). This is still publishable — it shows the method's operating regime.

**Estimated time**: ~9 configs × 70 min = ~10 hours.

---

## P1: Hyperparameter Sensitivity

**Goal**: Show the method is robust to HP choices, not fragile.

Sweep on TOFU 1B forget01 (fast iterations):

| Parameter | Values | Others fixed at |
| --- | --- | --- |
| τ (clamp threshold) | 0.5, 0.6, 0.7, 0.8, 0.9 | default |
| K (inner steps) | 1, 2, 3, 5, 7 | default |
| ε_mul (retain budget) | 0.75, 0.80, 0.85, 0.90, 0.95 | default |
| ρ (penalty coefficient) | 0.01, 0.05, 0.1, 0.2, 0.5 | default |

**Output**: 4 line plots (one per parameter) showing HM vs parameter value.
**Estimated time**: 20 configs × 15 min = 5 hours.

---

## P1: LoRA Baselines for Fair Comparison

**Goal**: All current baselines use full fine-tuning. For fair comparison, also run baselines with LoRA.

Priority baselines to LoRA-ify:
1. **PDU** (main competitor) — Add LoRA config, run on TOFU 1B and MUSE Books
2. **GradDiff** — Simple representative baseline
3. **NPO** — DPO-family representative

**Implementation**: Most baselines in the framework already support LoRA via config. Just need to set `use_lora: true, lora_r: 8, lora_alpha: 16` matching our config.

**Estimated time**: 3 methods × 2 benchmarks × 20 min = ~2 hours.

---

## P1: RWKU — Real World Knowledge Unlearning

**Goal**: Differentiate from the TOFU/MUSE/WMDP crowd with a benchmark that directly tests our core advantage — preserving neighbor knowledge during forgetting.

**Why this benchmark**: RWKU (Jin et al., 2024) is designed to expose exactly the failure mode our ALM constraint prevents. It requires forgetting 200 real-world famous people (e.g., Stephen King) from Llama-3-8B-Instruct or Phi-3-mini-4k, then evaluates:
- **Forget probes** (3 levels): cloze completion, QA, adversarial rephrasing (13,131 probes total)
- **Neighbor perturbation**: Does forgetting "Stephen King" degrade knowledge of related entities (e.g., other horror authors, Maine, The Shining actors)? This is where GA/NPO collapse — they nuke the entire neighborhood.
- **Utility dimensions**: Reasoning, truthfulness, factuality, fluency, multilingual (5 separate axes)
- **MIA**: Membership inference attack resistance

**Why LoRA-BiAL wins here**:
1. Clamped entropy: bounded forgetting signal stops once a person is forgotten — won't keep pushing and destroy neighbors
2. ALM constraint: ratchets up retain protection when neighbor knowledge degrades
3. LoRA: low-rank update physically limits how many representations can be perturbed
4. 200-entity scale stresses consistency over many forget targets — exactly where adaptive λ matters most

**Data/Code**: GitHub `jinzhuoran/RWKU`, HuggingFace `jinzhuoran/RWKU` (CC-BY-4.0). 4.35GB, ~960K rows.

**Integration effort**: Need to write a data loader and eval harness. The forget/retain split structure maps naturally to our framework. Eval requires cloze/QA accuracy + neighbor accuracy + utility benchmarks.

**Risk**: Llama-3-8B-Instruct may need LoRA r=16 or higher for the 200-entity scale. HP tuning could take several iterations. The 8B model is larger than our current 7B MUSE setup but fits on A100 with LoRA.

**Estimated time**: Data integration (3-4 hours) + eval harness (2-3 hours) + HP search (~8 hours GPU) + baselines (~6 hours GPU) = ~20 hours total.

---

## P2: KnowUnDo — Over-Forgetting Stress Test

**Goal**: Secondary differentiation benchmark. KnowUnDo (Gao et al., 2024, EMNLP Findings) was *designed* to penalize excessive unlearning — the exact failure mode we claim to solve.

**What it tests**: Two domains (copyrighted content, private user data). Each has an unlearn split and a retention split of neighboring knowledge. The key metric is whether essential related knowledge survives the unlearning process.

**Why LoRA-BiAL wins here**: The benchmark's core finding is that "existing methods suffer excessive unlearning." Our ALM constraint with asymmetric ratchet is purpose-built to prevent this — λ rises on the first retain spike and stays elevated.

**Data/Code**: GitHub `zjunlp/KnowUnDo`, HuggingFace. Targets Llama-2-7b-chat and Qwen1.5-7B-Chat.

**Risk**: Smaller community adoption than RWKU. Two-domain setup means double the eval work. Lower priority than RWKU.

**Estimated time**: Integration (2-3 hours) + runs (~6 hours GPU) = ~9 hours total.

---

## P2: WMDP Benchmark

**Goal**: Demonstrate on safety-critical unlearning (hazardous knowledge removal).

WMDP (Li et al., 2024) tests removal of weapons/bioweapons knowledge from Zephyr-7B.
- Requires different eval setup (MC accuracy on WMDP-Bio, WMDP-Cyber)
- Need to verify our framework supports the WMDP data format

**Risk**: This is a different domain (safety vs privacy/copyright). Method may need different HP regime. Only attempt if P0 and P1 are solid.

**Estimated time**: Setup (2-3 hours) + runs (~4 hours) = ~7 hours.

---

## P2: Compute-Normalized Comparison

**Goal**: Address "bilevel is just more compute" concern.

- Log wall-clock time and FLOPs for each method (already have train_time in results tables).
- Plot HM vs compute (wall-clock or FLOPs) as a Pareto frontier.
- Key comparison: LoRA-BiAL (104 min, HM=0.798) vs SimNPO (83 min, HM=0.755) on MUSE Books — we're better even accounting for ~25% more compute.

**Estimated time**: Analysis only (no new runs), ~1 hour.

---

## Rebuttal Plan (Issues Not Addressable by Experiments)

### "Beats gold" interpretation
**Response**: Already addressed in paper Sec 4.4 — gold model has residual knowledge (fk=0.303) due to pretraining data overlap. Our lower fk (0.080) represents a different Pareto operating point, not strict superiority. The gold model defines correct behavior.

### HM metric gameability
**Response**: HM penalizes imbalance — you can't game it by zeroing one axis. We also report all individual metrics. Can add a Pareto plot of forget vs retain if reviewers want it.

### Clamped entropy as "minor modification"
**Response**: The clamp is simple to state but has three non-trivial consequences: (1) bounded loss prevents the catastrophic gradient blowup seen in GA/NPO, (2) self-stabilizing property means fewer active tokens over time (show this empirically with an "active token fraction" plot), (3) interacts with ALM — bounded forget loss means λ·(L_ret - ε) can actually dominate when needed.

### Proposition 2 circular assumptions
**Response**: Remove or move to appendix. The empirical evidence (loss dynamics plots) is more compelling than the theoretical claim anyway.

---

## Execution Schedule

Assuming single A100, sequential runs:

| Day | Tasks | Hours |
| --- | --- | --- |
| 1 | P0: Ablations (3 seeds each) | ~4h |
| 1 | P0: TOFU 1B multi-seed (T=100, T=150) | ~3h |
| 2 | P0: TOFU 3B multi-seed | ~2.5h |
| 2 | P0: MUSE Books multi-seed | ~5h |
| 3 | P0: TOFU forget05/10 | ~3h |
| 3 | P1: Baseline multi-seed (PDU, SimNPO, RMU) | ~6h |
| 4 | P1: HP sensitivity sweeps | ~5h |
| 4 | P1: LoRA baselines | ~2h |
| 5 | P1: MUSE News tuning | ~10h |
| 6-7 | P1: RWKU integration + data loader + eval harness | ~6h dev |
| 7-8 | P1: RWKU HP search + baselines | ~14h GPU |
| 9 | P2: KnowUnDo (if time) | ~9h |
| 9 | P2: WMDP (if time) | ~7h |
| 9 | P2: Compute analysis | ~1h |

**Total**: ~7-8 days of continuous GPU time for P0+P1. P2 adds 1-2 more days.
