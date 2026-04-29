# BLADE: Bilevel Low-rank Adaptive Data Erasure

## Algorithm

```
Algorithm 1: BLADE
─────────────────────────────────────────────────────────────────────────
Input: Model M with LoRA adapters, forget set D_fgt, retain set D_ret,
       outer LR η_θ, inner LR η_in, safety cap T, inner iterations K,
       epsilon multiplier ε_mul, penalty weight ρ

Output: Unlearned model

 1  Initialize: LoRA(M), λ ← 1.0, SGD_in(lr=η_in), Adam_out(lr=η_θ)

    ┌─────────────────────────────────────────────────────────────────┐
    │ AUTO-EPSILON (step 0)                                           │
    │                                                                 │
 2  │  Run K inner SGD steps on D_ret                                 │
 3  │  ε ← ε_mul × mean(inner retain losses)                         │
    └─────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────┐
    │ MAIN LOOP                                                       │
    │                                                                 │
 4  │  for t = 1, ..., T do                                           │
    │                                                                 │
    │    // ── Inner loop: minimize retain loss ──                    │
 5  │    for k = 1, ..., K do                                         │
 6  │      θ ← θ − η_in · ∇_θ L_CE(D_ret; θ)                        │
    │    end                                                          │
    │                                                                 │
    │    // ── Outer step: minimize forget loss subject to constraint ─│
 7  │    Compute L_fgt = L_forget(D_fgt; θ),  L_ret = L_CE(D_ret; θ) │
 8  │    r ← L_ret − ε                                                │
 9  │    L_ALM ← L_fgt + λ·r + (ρ/2)·max(0, r)²                     │
10  │    θ ← θ − η_θ · ∇_θ L_ALM                                     │
    │                                                                 │
    │    // ── Dual variable update (asymmetric) ──                   │
11  │    if r > 0: λ ← λ + ρ·r                                       │
12  │    else:     λ ← λ + 0.1·ρ·r                                   │
13  │    λ ← clamp(λ, λ_min, λ_max)                                  │
    └─────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────┐
    │ ADAPTIVE SIGNALS                                                │
    │                                                                 │
    │  // ── LR Calibration (one-shot at t = 0.1·T) ──               │
14  │  Measure L_fgt decay rate over calibration window               │
15  │  Scale η_θ so L_fgt drops ~50% over T/3 steps                  │
    │                                                                 │
    │  // ── Phase Transition Detection ──                            │
16  │  T_trans ← first t where r > ε for 3 consecutive steps         │
    │                                                                 │
    │  // ── Convergence Stopping ──                                  │
17  │  After t > 1.6·T_trans and L_fgt dropped > 50%:                 │
18  │    Stop when |vel_ema(L_fgt)| < 0.01 × peak_vel                │
    └─────────────────────────────────────────────────────────────────┘

19  Merge LoRA into base model, return M
```

**Auto-Epsilon** (lines 2–3) calibrates the retain constraint threshold from a single pass over the retain set, eliminating manual tuning.

**Inner Loop** (lines 5–6) runs K SGD steps to restore retain performance before each outer update. This bilevel structure decouples forgetting from retention.

**Outer Step** (lines 7–10) minimizes the forget loss under an Augmented Lagrangian constraint on retain loss. The penalty term activates only when the constraint is violated.

**Asymmetric Dual Update** (lines 11–13) increases λ aggressively when the retain constraint is violated but decreases it gently when satisfied, preventing oscillation around ε.

**LR Calibration** (lines 14–15) adjusts the outer learning rate once, early in training, to match a target forgetting pace regardless of model/dataset scale.

**Phase Transition** (line 16) detects when the retain constraint becomes actively binding — the point where forgetting begins trading off against retention.

**Convergence Stopping** (lines 17–18) halts training when the forget loss velocity drops below 1% of its peak, measured purely in relative terms with no dataset-specific thresholds.

---

## Loss Dynamics: BLADE vs PDU

BLADE figures: red = forget loss, blue = retain loss, purple dashed = λ, orange dash-dot = ε, green dotted = T_trans, gray dashed = convergence.
PDU figures: red (left axis) = forget loss, blue (right axis) = retain loss.

### MUSE Books — Llama-2-7B

| BLADE (HM = 0.823) | PDU (HM = 0.602) |
|---|---|
| ![](figures/fig_muse_books_dynamics.png) | ![](figures/fig_muse_books_PDU_dynamics.png) |

### MUSE News — Llama-2-7B

| BLADE (HM = 0.542) | PDU (HM = 0.581) |
|---|---|
| ![](figures/fig_muse_news_dynamics.png) | ![](figures/fig_muse_news_PDU_dynamics.png) |

### TOFU forget01 — Llama-3.2-1B

| BLADE (HM = 0.811) | PDU (HM = 0.690) |
|---|---|
| ![](figures/fig_tofu_1B_forget01_dynamics.png) | ![](figures/fig_tofu_1B_forget01_PDU_dynamics.png) |

### TOFU forget05 — Llama-3.2-1B

| BLADE (HM = 0.804) | PDU (HM = 0.740) |
|---|---|
| ![](figures/fig_tofu_1B_forget05_dynamics.png) | ![](figures/fig_tofu_1B_forget05_PDU_dynamics.png) |

### TOFU forget10 — Llama-3.2-1B

| BLADE (HM = 0.808) | PDU (HM = 0.797) |
|---|---|
| ![](figures/fig_tofu_1B_forget10_dynamics.png) | ![](figures/fig_tofu_1B_forget10_PDU_dynamics.png) |

### TOFU forget01 — Llama-3.2-3B

| BLADE (HM = 0.849) | PDU (HM = 0.742) |
|---|---|
| ![](figures/fig_tofu_3B_forget01_dynamics.png) | ![](figures/fig_tofu_3B_forget01_PDU_dynamics.png) |

### TOFU forget05 — Llama-3.2-3B

| BLADE (HM = 0.849) | PDU (HM = 0.843) |
|---|---|
| ![](figures/fig_tofu_3B_forget05_dynamics.png) | ![](figures/fig_tofu_3B_forget05_PDU_dynamics.png) |

### TOFU forget10 — Llama-3.2-3B

| BLADE (HM = 0.842) | PDU (HM = 0.857) |
|---|---|
| ![](figures/fig_tofu_3B_forget10_dynamics.png) | ![](figures/fig_tofu_3B_forget10_PDU_dynamics.png) |
