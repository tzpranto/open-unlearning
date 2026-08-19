# Camera-Ready Checklist — EMNLP 2026 Main

Budget: main paper is 8 pages, +1 extra page allowed (target ≤ 9). Appendix unlimited.

Source of items: meta-review (AC VLUQ) + Reviewer 8hx1 + Reviewer Q1JT + Reviewer xpZ9, and the corresponding author rebuttals in `review.html`.

---

## Main paper (fit within the +1 extra page)

- [x] **1. §3.3 — Repair-first, from an optimization POV** (AC + Rev 8hx1 W1)
  - Rewrite opening: bilevel is asymmetric (K inner steps per outer step); forget is inherently destructive → K unopposed destructive inner steps if forget is inner → therefore retain-in-inner, forget-in-outer.
  - Cite ablation: swapping levels drops HM 0.823 → 0.506 (Table 3).

- [x] **2. §3 intro — Three-contribution framing** (AC + Rev 8hx1 W3)
  - (i) repair-first bilevel formulation — novel for unlearning
  - (ii) asymmetric λ ratchet (one-sided quadratic + slow decay) — new ALM adaptation
  - (iii) clamped-entropy forget loss (bounded, self-stabilizing, reference-free) — novel; proofs App. A
  - Explicitly separate from off-the-shelf: LoRA, vanilla ALM, generic bilevel.

- [x] **3. Symbol-at-first-use audit** (Rev 8hx1 W2)
  - Define τ, log V, ε, λ, K etc. at first appearance (Intro L.65 flagged).
  - Inline fix — no page cost.

- [x] **4. Swap §3.4 ↔ §3.5** (promised to Rev 8hx1)
  - So the flow matches the three-contribution framing.

- [x] **5. §4 — MUSE News discussion paragraph** (AC + Rev xpZ9 W1,W2)
  - MUSE News HM is dataset-driven (BBC forget/retain share distribution → highest lexical entanglement).
  - Point to appendix entanglement table + PISTOL sanity check.
  - Note BLADE still leads MUSE News under scalability/sustainability (§4.6).

- [x] **6. §4 — Computational cost paragraph** (AC + Rev xpZ9 W3)
  - BLADE trades ~1.5–2× training time for ~2–5× lower peak GPU memory (LoRA).
  - Point to appendix tables.

- [x] **7. §4.9 — Adversarial robustness summary** (AC + Rev Q1JT)
  - Bottom-line: BLADE leads on jailbreak ASR, only positive PrivLeak on MIA, best/tied on GCG.
  - Point to appendix tables.

- [x] **8. Limitations — extend** (Rev Q1JT W1)
  - Acknowledge re-learning-attack shallowness.
  - Future direction: dual-threshold clamped entropy separating highly- vs weakly-entangled tokens with distinct τ.

## Appendix (unlimited)

- [x] **A. List of Symbols** — glossary at start of appendix (Rev 8hx1 W2 promise)

- [x] **B. Extended adversarial robustness** (Rev Q1JT W2)
  - [x] Table: Jailbreak ASR (avg of two OpenUnlearning prompts) — TOFU fgt01/05/10
  - [x] Table: MIA PrivLeak (composite: LOSS, ZLib, Min-K% Prob, Min-K++, GradNorm, Reference) — TOFU fgt01/05/10
  - [x] Table: GCG (20-token, 200-step) ASR + mean post-attack ROUGE-L — TOFU fgt01/05/10

- [x] **C. MUSE News entanglement analysis** (Rev xpZ9 W1,W2)
  - [x] Table: TF-IDF cosine, Vocab Jaccard, NE Jaccard across every benchmark (placed in main paper as tab:entanglement)
  - [x] Note on KnowUnDo Privacy dynamics mirroring MUSE News (App. Fig. dynamics_blade_full)
  - ~~PISTOL benchmark experiment table~~ (dropped)

- [x] **D. Computational cost** (Rev xpZ9 W3)
  - [x] Table: mean training time (min) — 8 methods × 6 benchmarks
  - [x] Table: peak GPU memory (GB) — 8 methods × 6 benchmarks

## Non-content housekeeping

- [x] Deanonymize the method-name footnote (real repo URL, replacing anon link from commit `177adf4`)
- [ ] Add Acknowledgements section
- [ ] Verify AI-assistant disclosure matches EMNLP 2026 final policy
- [x] Camera-ready style switch: `\usepackage{acl}` (no `[review]`)
- [x] Author block (Pattern B, superscripts)

---

## Reviewer-side quick reference

| Reviewer | Score | Rebuttal status |
|---|---|---|
| 8hx1 | Findings (3) | "Addresses most of my concern. Please include those in the final version." |
| Q1JT | Borderline (3.5) | "Will maintain the positive score." |
| xpZ9 | Borderline (3.5) | No explicit response (pinged near deadline) |

Meta-review (AC VLUQ) summary of required revisions:
1. Optimization motivation behind repair-first — more rigorous
2. Novelty distinction (new contributions vs. existing optimization components)
3. Additional robustness evaluations — include in manuscript
4. Computational-cost analysis — include in manuscript
5. MUSE News clarification & discussion — improve
