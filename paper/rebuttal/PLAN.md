# Rebuttal Plan

Concerns from each reviewer, with the response strategy and whether new experiments are needed.

Legend:
- **[TXT]** = text-only response using existing paper/appendix content
- **[EXIST]** = existing numbers already in the repo/paper, but need to be surfaced in rebuttal
- **[NEW-SM]** = small new experiment (single seed, <2 GPU-hours)
- **[NEW-MED]** = medium new experiment (5-20 GPU-hours)
- **[NEW-LG]** = large new experiment (20+ GPU-hours)
- **[STRETCH]** = ideal but may not be feasible in rebuttal window

---

## Reviewer 8hx1 (score 2.5 = Borderline **Findings**) — lowest of three, but still an accept (Findings track). Rebuttal goal: push toward main-conference acceptance by resolving the three specific concerns.

### Concern 1: "Bilevel formulation (Eq. 2) not always easy to follow; Repair-first is intuitive, not mathematical"
- **[TXT]** Point to Prop. 1 (Appendix A.1): bounded gradient norm ≤ (2B+log V)·G. This is *precisely* what makes repair-first work — a bounded outer step is guaranteed recoverable by K SGD retain steps. Add a sentence connecting Prop. 1 → repair-first.
- **[TXT]** Ablation A18 (swapped bilevel: HM=0.506 with λ diverging to 367) is empirical evidence. Cite it explicitly in the response.
- **[TXT]** Formalize the argument: if the outer step ‖Δθ_out‖ ≤ η_out · G_max (bounded by clamped entropy) and inner step has convergence rate μ per step with K steps, then K · μ ≥ η_out · L (Lipschitz constant of L_ret) is a sufficient condition for recovery. This can be stated as a short lemma in the response.

### Concern 2: "Abstract introduces τ · log V without defining τ or V"
- **[TXT]** Concede. State that we'll fix in camera-ready: reword abstract to "clamped-entropy forget loss whose gradient vanishes once each token's entropy passes a target fraction of maximum uncertainty." No math symbols in abstract.

### Concern 3: "Novelty appears less substantial; LoRA, ALM, alternating are all existing"
- **[TXT]** Push back firmly. The paper's *core contribution is the interaction*, and the ablation table proves this directly:
  - A17 (logit_margin + full ALM): HM=0.000 — complete destruction
  - A5 (NPO + full ALM): HM=0.009 — trivial-satisfaction failure
  - A15 (no LoRA + full ALM + bilevel): HM=0.459 — retain collapse
  - A16 (bilevel + clamped_entropy − ALM): HM=0.233 — model destroyed
  - A18 (swapped bilevel + clamped_entropy + ALM + LoRA): HM=0.506 — λ diverges
  - Only the *specific* composition works. This is not a straightforward "combination of known techniques."
- **[TXT]** Note that clamped-entropy is genuinely new (unclamped entropy exists in Yuan 2025; ETW weights per-token but doesn't clamp).

---

## Reviewer 2 (score 3.5 = Borderline Conference) — accept-lean. Rebuttal goal: convert to a firmer conference accept by resolving the three "Major" asks.

### Concern 1 (Major): "Analyze why BLADE performs relatively poorly on MUSE News"
- **[TXT]** Explain the mechanism: clamped entropy maximizes per-token distributional entropy. On MUSE Books, memorized text is *unique* (Harry Potter phrasings) — flattening the distribution kills verbatim recall. On MUSE News, factual knowledge is *redundantly encoded* across 889 articles that discuss overlapping people/events. High-entropy predictions on any one article don't prevent the argmax from recovering facts stored in shared representations elsewhere.
- **[TXT]** Concrete evidence in paper: BLADE has best verbatim (vm=0.211) among competitive methods (PDU=0.092 is comparable), but knowledge memorization (fk=0.545) is harder to lower without collapsing retain.
- **[TXT]** BLADE dominates on the two MUSE News *stress tests* that matter for deployment:
  - Scale: 4× → BLADE HM=0.533 vs PDU=0.005 (100× gap)
  - Sequential: 4 steps → BLADE HM=0.525 vs PDU=0.130 (4× gap)
- **[EXIST]** Point to `results/ablation.md` MUSE News ε-multiplier sweep (all 8 values, K∈{0,3,6}) showing BLADE's HM is stable in [0.51, 0.55] — the gap to PDU is small (0.03) but the failure mode is fundamentally different.

### Concern 2 (Major): "Is MUSE News limitation specific or generalizable?"
- **[TXT]** Argue that the "gap" is dataset-specific because BLADE already wins on the other 6/7 settings. The specific characteristic: MUSE News combines (i) many documents, (ii) redundant factual encoding, (iii) knowmem eval on cloze tasks (not verbatim). No other benchmark in the paper has this profile.
- **[NEW-MED]** [Optional, if time] Run BLADE on WMDP (or another factual/redundant benchmark) to demonstrate generalization. WMDP is standard and cheap on 7B. ~5 GPU-hours per method-seed. Realistic scope: 3 seeds × 2 methods = 6 runs ≈ 30 GPU-h. Reasonable within rebuttal window.
- **[TXT]** Alternatively, argue that MUSE News is *itself* an out-of-distribution stress test that other benchmarks don't replicate, and BLADE still hits within 6% of PDU while dominating the scalability/sustainability axes.

### Concern 3 (Major): "Provide computational cost comparison"
- **[EXIST]** Surface the numbers from `results/gpu_budget_breakdown.md` and per-run wall-clock in `results/muse_news.md` / `results/muse_books.md`:
  - MUSE News BLADE ≈ 2h; PDU ≈ 1h; NPO/SimNPO ≈ 1.5h; BLURNPO ≈ 1h (BLADE ~2× PDU)
  - MUSE Books BLADE ≈ 60m; PDU ≈ 6m (BLADE ~10× PDU here, PDU converges very fast on Books)
  - TOFU 1B BLADE ≈ 12m; baselines ≈ 8m (BLADE ~1.5× baseline)
  - TOFU 3B BLADE ≈ 38m; baselines ≈ 25m (BLADE ~1.5× baseline)
- **[TXT]** Peak memory: same as baselines because we use LoRA (only ~0.2% params trainable) + gradient checkpointing. Ref-free clamped entropy means we don't need a frozen copy of the model (unlike NPO/BLURNPO which double memory).
- **[TXT]** Table format ready to paste into rebuttal.

---

## Reviewer Q1JT (score 3.5 = Borderline Conference) — accept-lean, most positive on soundness/reproducibility. Rebuttal goal: resolve the "shallow unlearning / no MIA/jailbreak" concern so the score holds or ticks up.

### Concern 1: "Re-learning attack recovers unlearning easily; unlearning is shallow"
- **[TXT]** Acknowledge — this is already flagged in Section 6 Limitations. Re-learning is a strictly stronger threat model that assumes: (i) white-box weight access, (ii) possession of the original forget data, (iii) willingness to run gradient updates. No LLM unlearning method in the literature survives this (Dorna et al. 2025 shows GA, NPO, SimNPO, RMU, PDU all recover to >70% of original HM after 1 epoch).
- **[EXIST]** Point to concrete numbers already in paper:
  - TOFU 1B: BLADE Δ=−0.135 (avg); PDU Δ=−0.152 — BLADE is *more* robust
  - TOFU 3B: BLADE Δ=−0.104 (avg); PDU Δ=−0.127 — BLADE is *more* robust
  - MUSE News: BLADE R=0.164; PDU R=0.110 — BLADE is *more* robust
  - KnowUnDo Copyright: BLADE R=0.604; PDU R=0.000 — BLADE >> PDU
- **[TXT]** Frame this as "no method is fully robust, but BLADE is at least as robust as the strongest baseline across all splits."

### Concern 2: "Evaluate against jailbreak attacks"
- **[NEW-MED]** Realistic scope: adversarial prompt paraphrasing on TOFU is already reported (Adv_HM in main paper). Extend with an actual jailbreak template (e.g., DAN-style prefixes on the forget question) on TOFU 1B forget10. ~1 GPU-h per method × 3 methods = trivial.
- **[EXIST]** Also point to LLM judge (Claude Opus 4.7) which is *itself* an adversarial evaluation — the judge specifically looks for semantic leakage that surface metrics miss. BLADE has FL≤0.05 across most settings vs PDU 0.01-0.86.

### Concern 3: "Evaluate against membership inference attack (MIA)"
- **[NEW-MED]** MIA is standard: use loss-based MIA (Yeom et al. / Shokri et al.). Compute L(forget)/L(retain) ratio; a well-unlearned model should have similar losses. Fast to run on existing checkpoints — no retraining needed. ~30 min per method × 3 methods = 1.5 GPU-h.
- **[NEW-MED]** MUSE actually provides an "extraction" attack score (`extract↓`) which is a form of MIA. Already in main paper tables. Highlight it:
  - MUSE News extract: BLADE=0.031; PDU=0.024; GradAscent=0.008 (collapsed); Gold=0.025
  - BLADE close to Gold, only slightly above PDU

### Concern 4: "Evaluate against optimization-based attacks"
- **[TXT]** Argue that re-learning IS an optimization-based attack (gradient descent on forget set). The extraction score in MUSE (which optimizes suffix tokens to elicit memorized content) is another.
- **[STRETCH]** GCG-style adversarial suffix search (Zou et al. 2023) on the forget question. Expensive and hard to run in rebuttal window; note this as future work.

---

## Cross-cutting: prioritized experiments to run for rebuttal

**Must-do (~7 GPU-h total)**:
1. **MIA numbers** on existing BLADE + PDU + top baseline checkpoints on TOFU + MUSE News (~2 GPU-h)
2. **Extraction/verbmem clearer breakdown table** already have — just surface it

**Should-do (~30 GPU-h)**:
3. **Jailbreak attacks** on TOFU (DAN-style prefixes) (~3 GPU-h)
4. **WMDP** or another factual benchmark to show MUSE News gap is dataset-specific (~30 GPU-h if we run 3 seeds × 2 methods)

**Nice-to-have (~50+ GPU-h)**:
5. **GCG-style optimization attack** (expensive)
6. **Compute table** — just numbers we already have, no new runs

---

## Rebuttal format (per user direction)

- One markdown file per reviewer in `paper/rebuttal/`
- Filenames: `reviewer_8hx1.md`, `reviewer_2.md`, `reviewer_q1jt.md`
- Start with existing numbers + basic explanation; note where new experiments would strengthen
- Later convert to HTML for OpenReview
