# Mechanistic Interpretability Trace Report

## Goal

We want to **unlearn** specific knowledge from a language model (make it forget certain news articles) without damaging what it should still know. But blindly updating every parameter risks destroying useful knowledge. So we first need to answer: **where inside the model does the "forget" knowledge actually live?**

This report maps which layers and components of a Llama-2-7b model are most responsible for storing "forget" vs "retain" knowledge, using three complementary techniques plus a neuron-level deep dive. The findings directly inform where to apply (and where to protect from) the SIBL unlearning algorithm.

## Setup

| | |
|---|---|
| **Model** | `muse-bench/MUSE-News_target` -- a Llama-2-7b (32 transformer layers, 7B parameters) fine-tuned on the MUSE News corpus |
| **Forget set** | 889 news article samples the model should unlearn |
| **Retain set** | 1777 samples the model must continue to handle well |
| **Sequence length** | 512 tokens per sample |
| **GPU** | NVIDIA H100 NVL (95 GB) |
| **Date** | 2026-04-01 |

> **Note on set sizes:** The retain set is 2x larger than forget (1777 vs 889). This matters for interpreting raw numbers -- retain signals will naturally be larger in absolute terms. We use **ratios** and **differentials** to account for this.

---

## 1. Gradient Differential -- "Which parameters care most about forget data?"

### What we did

We computed the loss separately for the forget set and the retain set, then backpropagated to get gradients for every parameter. The **gradient magnitude** tells us how much each parameter would need to change to reduce that loss.

By computing the ratio `forget_gradient / retain_gradient` for each parameter:

- **Ratio = 1.0** means the parameter responds equally to both -- it's **balanced**.
- **Ratio < 1.0** means the parameter is **retain-biased** -- it responds more to retain data.
- **Ratio > 1.0** would mean the parameter is **forget-biased**. Very few parameters reach this.

> **A note on "parameters":** Llama-2-7b has ~6.7 billion individual weights, organized into **291 named weight tensors** (e.g., `layers.0.self_attn.q_proj.weight` is one tensor of shape 4096x4096 = 16.7M individual weights). The gradient ratio is computed **per tensor** here. Section 5 breaks this down to individual neurons.

### Results

![Gradient Differential](saves/traces/muse_news_full/gradient_differential.png)

**Distribution of all 291 parameter ratios:**
- Min: 0.22, Max: 1.01, Mean: 0.84, Median: 0.96
- Only **5 of 291 params** have ratio >= 1.0 (barely above parity)
- 77 params have ratio < 0.80 (strongly retain-biased)

**Layer-by-layer summary:**

| Layer Range | Avg Ratio | Verdict |
|-------------|-----------|---------|
| 0-7 (early) | 0.38--0.63 | **Strongly retain-dominant.** Retain gradients are 1.6x--2.6x larger. |
| 8 | 0.82 | Transitional. |
| 9-19 (mid) | 0.94--0.97 | **Balanced** (slight retain bias). |
| 20-26 (mid-late) | 0.97--0.98 | **Balanced** (closest to parity, but NOT forget-dominant). |
| 27-31 (late) | 0.95--0.97 | **Balanced** with high absolute gradients. |

**Key takeaway:** No layer is forget-dominant at the parameter level. The model stores forget knowledge **diffusely**, not in a neat compartment. Layers 20-26 are simply the least retain-biased -- they are near parity, not forget-specific.

![Layer x Component Heatmap](saves/traces/analysis/layer_component_heatmap.png)

The heatmap above shows the gradient ratio for each (layer, component) pair. The diverging colormap is centered at 1.0: red = forget-biased, green = retain-biased, yellow = balanced. The retain-dominated early layers (0-7) are clearly visible as the green block in the top rows.

---

## 2. Activation Traces -- "How does data flow differently for forget vs retain?"

### What we did

We hooked into every layer and recorded the hidden state activations as the model processed forget and retain data. We measured L2 norm, mean absolute value, and variance per layer for: full layer output, MLP sublayer, and attention sublayer.

### Results

![Activation Traces](saves/traces/muse_news_full/activation_traces.png)

![Activation Heatmap (L2 Norm)](saves/traces/analysis/activation_heatmap_l2_norm.png)

**Layer-by-layer activation L2 norm differential (forget - retain):**

| Layer Range | Differential | Absolute Norms | What it means |
|-------------|-------------|----------------|---------------|
| 0-9 | -0.02 to +0.02 | 2 -> 23 | **No meaningful difference.** Both data types produce nearly identical representations. |
| 10-14 | +0.02 to -0.02 | 25 -> 31 | **Negligible.** Tiny forget advantage at layers 10-12, but only ~0.02 on norms of ~25 (0.08%). |
| 15-26 | -0.05 to -0.53 | 33 -> 79 | **Retain activations slightly larger.** The gap widens, but is small: -0.53 out of 79 = 0.7%. |
| 27-31 | -0.54 to **-3.65** | 83 -> 111 | **Retain advantage largest here.** Layer 31 has the biggest gap, but it's still only ~3.3% relative. |

**Interpretation:** The model processes forget and retain data very similarly. There is no separate "forget pathway." Retain produces marginally stronger activations in later layers (likely because the retain set is more diverse -- 1777 samples spanning many topics vs 889 news articles). The differences are small in relative terms.

**How this relates to gradients (not contradictory):** Gradients measure *sensitivity to change*; activations measure *signal magnitude*. A layer can have balanced gradient sensitivity while carrying more retain signal. Both are true simultaneously.

---

## 3. Causal Tracing -- "Which layers are causally necessary for predictions?"

### What we did

For each layer, we **injected calibrated noise** (3x the layer's activation standard deviation) to disrupt that layer's contribution, then measured how much the model's predictions degraded. Higher degradation = layer is more causally important. We run this separately for forget and retain data.

### Results (Full Corpus: 889 forget + 1777 retain)

![Causal Traces](saves/traces/muse_news_full_causal/causal_traces.png)

**Full-corpus causal tracing results:**

| Layer Range | Forget Degrad. | Retain Degrad. | Diff (F-R) | Verdict |
|-------------|---------------|----------------|------------|---------|
| 0 (embed) | 5.73 | 5.59 | +0.13 | Forget-critical |
| 1-8 (early) | 10.74--10.77 | 10.60--10.63 | +0.13 to +0.15 | Forget-critical |
| 9-19 (mid) | 10.29--10.73 | 10.15--10.59 | +0.13 to +0.14 | Forget-critical |
| 20-26 (mid-late) | 9.20--10.14 | 9.07--10.00 | +0.12 to +0.14 | Forget-critical |
| 27-29 | 8.40--8.96 | 8.28--8.83 | +0.12 | Forget-critical |
| 30 | 5.82 | 5.74 | +0.08 | Mildly forget-critical |
| **31** | **4.86** | **4.92** | **-0.06** | **Only retain-critical layer** |

**Key findings:**
- **Every layer except 31 is forget-critical** -- disrupting it hurts forget predictions more than retain.
- The differential is remarkably **uniform** across layers 1-29 (all ~+0.13), meaning forget knowledge is **evenly distributed** throughout the network, not concentrated anywhere.
- **Layer 31 is the sole retain-critical layer.** It is the only layer where disruption hurts retain more. This aligns with the activation findings (largest retain activation gap) and gradients (early layers strongly retain-biased).
- The absolute degradation drops sharply at layers 30-31 (from ~8-10 to ~5), meaning these final layers are less important overall for both data types.

---

## 4. Cross-Technique Synthesis

All three techniques converge on a consistent picture:

### Where is forget knowledge?
| Technique | Finding |
|-----------|---------|
| **Gradients** | No layer is forget-dominant. The least retain-biased layers are 20-26 (ratio 0.97-0.98), but even these are not forget-specific. |
| **Activations** | No meaningful forget-specific signal at any layer. Differences between forget and retain activations are < 1% relative through layer 26. |
| **Causal tracing** | Forget knowledge is **uniformly distributed** across layers 1-29 with a nearly constant differential of +0.13. There is no "forget hub." |

**Bottom line: Forget knowledge is diffuse.** It is not concentrated in any layer or component. This means layer-level masking alone cannot surgically remove forget knowledge without affecting retain.

### Where is retain knowledge concentrated?
| Technique | Finding |
|-----------|---------|
| **Gradients** | **Layers 0-7 are strongly retain-dominant** (ratio 0.38-0.63). These early layers respond 1.6x-2.6x more to retain data. |
| **Activations** | **Layer 31 MLP** has the largest retain-over-forget gap (-1.83 MLP, -3.65 full layer). |
| **Causal tracing** | **Layer 31 is the only retain-critical layer** causally. All others are forget-critical. |

**Bottom line: Retain knowledge has clear hotspots** -- early layers (0-7) for gradient sensitivity, and layer 31 for activation magnitude and causal importance. These must be protected.

---

## 5. Neuron-Level Analysis -- "Are there individual forget-specific neurons?"

Since no layer is forget-dominant overall, we asked: **are there individual neurons within layers that are forget-specific, even if the layer average is balanced?**

### What we did

Instead of averaging gradients across an entire weight matrix (as in Section 1), we computed **per-neuron (per-row) gradient magnitudes**. For a weight matrix of shape `(out_features, in_features)`, each row corresponds to one output neuron. We computed `mean(|grad|)` per row separately for forget and retain data, then took the ratio.

Script: `scripts/analyze_traces.py`

### Results

**Global statistics across all 1,359,872 tracked neurons:**

| Metric | Value |
|--------|-------|
| Total neurons | 1,359,872 |
| Forget-dominant (ratio > 1.0) | **232,034 (17.1%)** |
| Retain-dominant (ratio < 0.5) | 202,600 (14.9%) |
| Ratio range | [0.010, 2.084] |
| Ratio mean | 0.848 |
| Ratio median | 0.945 |

**17% of neurons are forget-dominant** -- a substantial minority, even though no layer is forget-dominant overall. These neurons are masked by the layer average because the other ~83% of neurons in the same layer are retain-biased or balanced.

![Neuron Differential Histogram](saves/traces/analysis/neuron_differential_histogram.png)

The histogram shows the distribution is right-skewed with a long tail above 1.0. Most neurons cluster around 0.9-1.0 (balanced), but there is a meaningful population above 1.0.

### Per-layer forget neuron density

| Layer | Forget Neurons | Total | % Forget | Pattern |
|-------|---------------|-------|----------|---------|
| 0-6 | 0-33 | 42,496 | 0.0-0.1% | Almost zero forget neurons |
| 7 | 477 | 42,496 | 1.1% | First appearance |
| 8 | 3,433 | 42,496 | 8.1% | Ramp-up |
| 9-13 | 6,176-9,055 | 42,496 | 14.5-21.3% | Growing |
| 14-20 | 9,265-10,742 | 42,496 | 21.8-25.3% | Moderate density |
| **21** | **12,911** | **42,496** | **30.4%** | **Peak -- highest forget neuron density** |
| 22-26 | 9,123-12,434 | 42,496 | 21.5-29.3% | High density region |
| 27-30 | 8,182-11,558 | 42,496 | 19.3-27.2% | Still substantial |
| 31 | 6,001 | 42,496 | 14.1% | Drops off |

Layers 17-25 are the richest in forget neurons (~25-30%), with layer 21 peaking at 30.4%.

### Top forget-specific neurons

| Parameter | Neuron | Ratio | Interpretation |
|-----------|--------|-------|----------------|
| layers.22.mlp.up_proj | 7062 | **2.084** | Strongest forget neuron -- 2x more sensitive to forget than retain |
| layers.9.self_attn.k_proj | 3413 | 2.003 | Attention key neuron -- controls what layer 9 attends to |
| layers.10.mlp.gate_proj | 4478 | 1.990 | MLP gating neuron |
| layers.23.mlp.up_proj | 4976 | 1.871 | |
| layers.9.self_attn.k_proj | 3349 | 1.862 | Same attention head region as above |
| layers.26.self_attn.k_proj | 2052 | 1.757 | Layer 26 attention -- consistent with layer-level findings |
| layers.30.self_attn.k_proj | 1768 | 1.692 | Late-layer attention key |

**Attention K-projections** and **MLP up/gate projections** are the most common component types among top forget neurons. Notably, layer 9 and layer 26 each have clusters of forget-specific K-projection neurons, suggesting these layers may contain attention heads specifically involved in forget-knowledge retrieval.

### Neuron heatmaps

The following heatmaps show the forget/retain ratio for every neuron in every layer, organized by component type. Red = forget-dominant, green = retain-dominant. The sparsity of red dots confirms that forget neurons are a minority but clearly present.

**MLP projections:**

![MLP gate_proj Neuron Heatmap](saves/traces/analysis/neuron_heatmap_mlp_gate_proj.png)
![MLP up_proj Neuron Heatmap](saves/traces/analysis/neuron_heatmap_mlp_up_proj.png)
![MLP down_proj Neuron Heatmap](saves/traces/analysis/neuron_heatmap_mlp_down_proj.png)

**Attention projections:**

![Attn Q Neuron Heatmap](saves/traces/analysis/neuron_heatmap_attn_q_proj.png)
![Attn K Neuron Heatmap](saves/traces/analysis/neuron_heatmap_attn_k_proj.png)
![Attn V Neuron Heatmap](saves/traces/analysis/neuron_heatmap_attn_v_proj.png)
![Attn O Neuron Heatmap](saves/traces/analysis/neuron_heatmap_attn_o_proj.png)

### Forget neuron bitmap

The bitmap below shows a binary view: red = forget-dominant (ratio > 1.0), gray = not. This is the raw mask that could be used for neuron-level SIBL targeting.

![Forget Neuron Bitmap](saves/traces/analysis/forget_neuron_bitmap.png)

---

## 6. Implications for SIBL

### What the data says

1. **Forget knowledge is diffuse at the layer level.** No layer is forget-dominant. Causal tracing shows a uniform +0.13 forget advantage across layers 1-29.

2. **Retain knowledge has clear hotspots.** Layers 0-7 (gradient-dominant) and layer 31 (activation/causal-dominant) are where retain knowledge concentrates.

3. **Forget knowledge concentrates at the neuron level.** 17% of neurons (232K out of 1.36M tracked) are forget-dominant, peaking in layers 17-25 at ~25-30% density. The strongest forget neurons have ratios up to 2.08x.

### Recommended SIBL configuration

**Sparsity Mask:**
- Layer-level masking (freeze layers 0-7, update 20-26) is a reasonable starting point for damage minimization, but it cannot precisely target forget knowledge since it is diffuse.
- **Neuron-level masking** (using the bitmap from Section 5) would allow targeting the 17% of neurons that are actually forget-specific. The bitmap is saved at `saves/traces/analysis/forget_neuron_bitmap.pt`.

**Implicit Correction Targeting:**
- Current default: `implicit_block_last_n_layers=2` (layers 30-31)
- Data-informed alternative: apply implicit correction to **layers 0-7 AND layer 31** -- the most retain-critical regions across all three techniques.

**Layer-wise Learning Rate (alternative to binary mask):**
- Zero or very low `eta_theta` for layers 0-7 (strongly retain-dominant)
- Higher `eta_theta` for layers 17-25 (highest forget neuron density)
- This is more nuanced than binary layer masking

---

## 7. Reproducibility

All figures and analysis in this report can be reproduced with the following scripts:

```bash
# Run from repo root:

# Step 1: Collect gradient + activation traces (full corpus)
python trace_analysis/scripts/trace_activations.py \
    --n_samples 10000 --batch_size 4 \
    --output_dir trace_analysis/saves/traces/muse_news_full \
    --skip_causal

# Step 2: Collect causal traces (full corpus, slow)
python trace_analysis/scripts/trace_activations.py \
    --n_samples 10000 --batch_size 4 \
    --output_dir trace_analysis/saves/traces/muse_news_full_causal \
    --skip_gradients --skip_activations

# Step 3: Neuron-level analysis + heatmaps
python trace_analysis/scripts/analyze_traces.py \
    --n_samples 10000 --batch_size 4 \
    --trace_file trace_analysis/saves/traces/muse_news_full/trace_results.pt \
    --output_dir trace_analysis/saves/traces/analysis

# Step 3 with cached neurons (skip GPU, reuse existing neuron_traces.pt):
python trace_analysis/scripts/analyze_traces.py \
    --skip_neuron_collection \
    --trace_file trace_analysis/saves/traces/muse_news_full/trace_results.pt \
    --output_dir trace_analysis/saves/traces/analysis
```

---

## 8. Output Files

| File | Description |
|------|-------------|
| `saves/traces/muse_news_full/trace_results.pt` | Full gradient + activation traces (PyTorch tensors) |
| `saves/traces/muse_news_full/summary.json` | JSON summary with top-50 params and per-layer differentials |
| `saves/traces/muse_news_full/gradient_differential.png` | Per-layer gradient analysis (3 panels) |
| `saves/traces/muse_news_full/activation_traces.png` | Per-layer activation norms (4 panels) |
| `saves/traces/muse_news_full_causal/trace_results.pt` | Full-corpus causal traces |
| `saves/traces/muse_news_full_causal/causal_traces.png` | Per-layer causal importance (3 panels) |
| `saves/traces/muse_news_full_causal/summary.json` | Causal tracing JSON summary |
| `saves/traces/analysis/neuron_traces.pt` | Per-neuron gradient data (15 MB) |
| `saves/traces/analysis/neuron_analysis.json` | Neuron-level analysis summary |
| `saves/traces/analysis/layer_component_heatmap.png` | Layer x component gradient ratio heatmap |
| `saves/traces/analysis/neuron_heatmap_*.png` | Per-component neuron-level heatmaps (7 files) |
| `saves/traces/analysis/neuron_differential_histogram.png` | Distribution of neuron-level ratios |
| `saves/traces/analysis/forget_neuron_bitmap.png` | Binary forget-dominant neuron map |
| `saves/traces/analysis/forget_neuron_bitmap.pt` | Bitmap tensor for downstream use in SIBL |
| `saves/traces/analysis/activation_heatmap_*.png` | Activation heatmaps (L2 norm, mean_abs, variance) |
| `saves/traces/muse_news_llama2_7b/` | 100-sample pilot run (all 3 techniques) |

---

## 9. Runtime

| Technique | Samples | Time | Notes |
|-----------|---------|------|-------|
| Gradient traces | 889 + 1777 | 188s | Full corpus |
| Activation traces | 889 + 1777 | 75s | Full corpus |
| Causal tracing | 889 + 1777 | 2072s (34.5 min) | Full corpus |
| Neuron gradient collection | 889 + 1777 | 190s | Per-row gradients, 7 component types |
| Neuron analysis + plots | -- | ~25s | From cached neuron_traces.pt |
| **Total** | | **~42 min** | |
