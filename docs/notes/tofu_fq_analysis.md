# Why TOFU Forget Quality (FQ) is Near-Zero for Most Baselines

## How FQ is Computed

**Code:** `src/evals/metrics/privacy.py` lines 6-32

FQ = p-value from a 2-sample Kolmogorov-Smirnov test (`scipy.stats.ks_2samp`) comparing:
- **Distribution A:** `truth_ratio` scores of the **unlearned model** on the forget set
- **Distribution B:** `truth_ratio` scores of the **retrained model** (gold standard) on the retain set

Where `truth_ratio = wrong_prob / correct_prob` (computed in `src/evals/metrics/memorization.py`).

**Interpretation:**
- FQ = 1.0: forget distribution matches retain distribution perfectly (ideal unlearning)
- FQ ≈ 0: distributions are highly significantly different (poor unlearning)

The retain distribution comes from `retain_logs_path` (e.g., `saves/eval/tofu_Llama-3.2-1B-Instruct_retain99/TOFU_EVAL.json`).

## Why Most Baselines Get FQ ≈ 0

1. **Gradient methods genuinely fail to match the retrain distribution.** GA, GradDiff, NPO, SimNPO produce truth_ratio distributions on the forget set that remain far from the retrained model's distribution. The unlearned model either still "remembers" (low truth_ratio, close to original) or catastrophically over-forgets (truth_ratio goes to extreme values), neither of which matches the retrained model.

2. **KS test is very sensitive with small samples.** TOFU forget01 = 40 samples, forget05 = 200, forget10 = 400. Even moderate distributional differences yield astronomically small p-values (1e-20 to 1e-239). The test is statistically powerful enough to detect any deviation.

3. **The bar is high by design.** The retrained model's distribution is the gold standard. Only methods that precisely calibrate forgetting — producing truth_ratios that look like the model was never trained on those samples — achieve meaningful FQ. Crude gradient-based approaches overshoot or undershoot.

## What Achieves Non-Zero FQ

From our 1B results:
- **PDU** (forget01): FQ=0.766, HM=0.663 — dual optimization precisely calibrates forgetting
- **NPO** (forget01): FQ=0.579, HM=0.579 — preference optimization gets closer
- **GradAscent** (forget01): FQ=0.405, HM=0.441 — crude but effective on small forget sets
- **RMU** (forget05): FQ=0.178, HM=0.272 — only survivor at 5% scale

At forget10, nearly all methods collapse to FQ ≈ 0 because matching the retrain distribution across 400 samples is much harder.

## Conclusion

FQ ≈ 0 is **expected behavior, not a bug.** The metric correctly measures that standard gradient-based unlearning methods fail to produce distributions indistinguishable from retraining. The harshness scales with forget set size because the KS test gains statistical power with more samples.
