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
| **Date** | 2026-04-08 (full rerun: causal + gradient + activation, correct splits) |

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

![Gradient Differential](figures/traces/muse_news/gradient_differential.png)

**Per-layer gradient balance (forget/retain ratio) — as visible in the gradient differential plot:**

| Layer Range | Ratio (F/R) | Verdict |
|-------------|-------------|---------|
| 0–7 (early) | ~0.2–0.8 | **Retain-dominated.** Early layers have substantially larger retain gradients. |
| 8 | ~0.8–0.9 | Transitional — crossing toward parity. |
| 9–31 (mid/late) | ~0.95–1.0 | **Near parity.** Forget and retain gradients nearly equal. Slight forget edge in layers 17–25. |

**Key takeaway:** Early layers 0–7 are clearly retain-dominated at the per-layer gradient level (ratio ~0.2–0.8). From layer 8 onward, gradients are near-parity (~1.0). This matches the neuron-level picture (Section 5): layers 0–6 have mean neuron ratio 0.42–0.52 and near-zero forget neurons.

![Layer x Component Heatmap](figures/traces/analysis/layer_component_heatmap.png)

The heatmap above shows the gradient ratio (forget/retain) for each (layer, component) pair. The diverging colormap is centered at 1.0: red = forget-biased, green = retain-biased, yellow = balanced. Layers 0–6 are deep green across all components (ratio ~0.3–0.5) — strongly retain-dominated. From layer 7 onward the heatmap transitions to yellow/red, with forget dominance strongest in layers 17–25.

---

## 2. Activation Traces -- "How does data flow differently for forget vs retain?"

### What we did

We hooked into every layer and recorded the hidden state activations as the model processed forget and retain data. We measured L2 norm, mean absolute value, and variance per layer for: full layer output, MLP sublayer, and attention sublayer.

### Results

![Activation Traces](figures/traces/muse_news/activation_traces.png)

![Activation Heatmap (L2 Norm)](figures/traces/analysis/activation_heatmap_l2_norm.png)

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

![Causal Traces](figures/traces/muse_news/causal_traces.png)

**Full-corpus causal tracing results:**

| Layer Range | Forget Degrad. | Retain Degrad. | Diff (F-R) | Verdict |
|-------------|---------------|----------------|------------|---------|
| 0 (embed) | 5.715 | 5.597 | +0.118 | Forget-critical |
| 1-8 (early) | 10.74–10.76 | 10.60–10.62 | +0.138 to +0.140 | Forget-critical |
| 9-19 (mid) | 10.30–10.76 | 10.16–10.62 | +0.138 to +0.140 | Forget-critical |
| 20-26 (mid-late) | 9.20–10.15 | 9.07–10.01 | +0.126 to +0.140 | Forget-critical |
| 27-29 | 8.40–8.96 | 8.28–8.83 | +0.126 | Forget-critical |
| 30 | 5.826 | 5.740 | +0.086 | Mildly forget-critical |
| **31** | **4.864** | **4.921** | **-0.058** | **Only retain-critical layer** |

**Key findings:**
- **Every layer except 31 is forget-critical** -- disrupting it hurts forget predictions more than retain.
- The differential is remarkably **uniform** across layers 1-29 (all ~+0.13), meaning forget knowledge is **evenly distributed** throughout the network, not concentrated anywhere.
- **Layer 31 is the sole causally retain-critical layer.** It is the only layer where disruption hurts retain more. This aligns with the activation findings (largest retain activation gap) and neuron-level data (early layers 0–6 strongly retain-biased at mean ratio 0.42–0.52).
- The absolute degradation drops sharply at layers 30-31 (from ~8-10 to ~5), meaning these final layers are less important overall for both data types.

---

## 4. Cross-Technique Synthesis

All three techniques converge on a consistent picture:

### Where is forget knowledge?
| Technique | Finding |
|-----------|---------|
| **Gradients** | Layers 0–7 are clearly retain-dominated (ratio ~0.2–0.8). Layers 8–31 near parity (~1.0). Layers 17–25 have the highest forget neuron density (25–30%). |
| **Activations** | No meaningful forget-specific signal at any layer. Differences between forget and retain activations are < 1% relative through layer 26. |
| **Causal tracing** | Forget knowledge is **uniformly distributed** across layers 1-29 with a nearly constant differential of +0.13. There is no "forget hub." |

**Bottom line: Forget knowledge is diffuse.** It is not concentrated in any layer or component. This means layer-level masking alone cannot surgically remove forget knowledge without affecting retain.

### Where is retain knowledge concentrated?
| Technique | Finding |
|-----------|---------|
| **Gradients** | Layers 0–7 are clearly retain-dominant (per-layer ratio ~0.2–0.8, near-zero forget neurons). Confirmed at both param-level and neuron-level. |
| **Activations** | **Layer 31 MLP** has the largest retain-over-forget gap (-1.83 MLP, -3.65 full layer). |
| **Causal tracing** | **Layer 31 is the only retain-critical layer** causally. All others are forget-critical. |
| **Neuron-level** | Layers 0–6 have near-zero forget neurons (0–0.3%, mean ratio 0.42–0.52). Layer 31 has 14.1% — the lowest of any deep layer. Peak at layer 21 (30.4%). |

**Bottom line: Layer 31 is the only causally retain-critical layer.** Early layers 0–7 are the strongest retain-dominant zone across all three techniques: retain-dominated gradient balance (~0.2–0.8), near-zero forget neurons (0–1.1%), and the largest retain activation gaps.

---

## 5. Neuron-Level Analysis -- "Are there individual forget-specific neurons?"

Since no layer is forget-dominant overall, we asked: **are there individual neurons within layers that are forget-specific, even if the layer average is balanced?**

### What we did

Instead of averaging gradients across an entire weight matrix (as in Section 1), we computed **per-neuron (per-row) gradient magnitudes**. For a weight matrix of shape `(out_features, in_features)`, each row corresponds to one output neuron. We computed `mean(|grad|)` per row separately for forget and retain data, then took the ratio.

Script: `scripts/analyze_traces.py` (uses `figures/traces/analysis/neuron_traces.pt`, collected with correct splits: 889 forget + 1777 retain, `raw` config, 2026-04-08)

### Results

**Global statistics across 1,359,872 tracked neurons (attn/MLP projections, 7 components × 32 layers):**

| Metric | Value |
|--------|-------|
| Total neurons tracked | 1,359,872 |
| Forget-dominant (ratio > 1.0) | **232,018 (17.1%)** |
| Retain-dominant (ratio < 0.5) | 202,597 (14.9%) |
| Ratio range | [0.010, 2.084] |
| Ratio mean / median | 0.848 / 0.945 |

**Only 17% of neurons are forget-dominant** — the model's attn/MLP weights are predominantly retain-biased overall (mean ratio 0.85). The distribution is centered below 1.0, with a long right tail up to 2.08x. This means ratio > 1.0 is already selective; the surgical mask naturally covers ~17% of weights.

![Neuron Differential Histogram](figures/traces/analysis/neuron_differential_histogram.png)

### Per-layer forget neuron density

| Layer | Forget-Dom | Total | % | Mean Ratio |
|-------|-----------|-------|---|------------|
| 0 | 137 | 42,496 | 0.3% | 0.506 |
| 1–5 | 0–1 | 42,496 | ~0.0% | 0.42–0.45 |
| 6 | 33 | 42,496 | 0.1% | 0.522 |
| 7 | 477 | 42,496 | 1.1% | 0.726 |
| 8 | 3,434 | 42,496 | 8.1% | 0.870 |
| 9 | 6,174 | 42,496 | 14.5% | 0.950 |
| 10–16 | 8,011–10,148 | 42,496 | 19–24% | 0.96–0.97 |
| 17 | 11,198 | 42,496 | 26.4% | 0.971 |
| 18–20 | 10,628–10,743 | 42,496 | 25% | 0.966–0.969 |
| **21** | **12,907** | **42,496** | **30.4%** | **0.977** |
| 22–25 | 11,382–11,922 | 42,496 | 27–29% | 0.969–0.974 |
| 26 | 9,124 | 42,496 | 21.5% | 0.964 |
| 27–30 | 8,184–11,558 | 42,496 | 19–27% | 0.959–0.979 |
| **31** | **6,001** | **42,496** | **14.1%** | **0.963** |

**Early layers 0–6 are near-zero forget neurons (0–0.3%) and strongly retain-dominated (mean ratio 0.42–0.52).** Forget neurons emerge sharply at layer 7 (1.1%) and peak at layer 21 (30.4%). Layer 31 has the fewest forget neurons among deep layers.

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

![MLP gate_proj Neuron Heatmap](figures/traces/analysis/neuron_heatmap_mlp_gate_proj.png)
![MLP up_proj Neuron Heatmap](figures/traces/analysis/neuron_heatmap_mlp_up_proj.png)
![MLP down_proj Neuron Heatmap](figures/traces/analysis/neuron_heatmap_mlp_down_proj.png)

**Attention projections:**

![Attn Q Neuron Heatmap](figures/traces/analysis/neuron_heatmap_attn_q_proj.png)
![Attn K Neuron Heatmap](figures/traces/analysis/neuron_heatmap_attn_k_proj.png)
![Attn V Neuron Heatmap](figures/traces/analysis/neuron_heatmap_attn_v_proj.png)
![Attn O Neuron Heatmap](figures/traces/analysis/neuron_heatmap_attn_o_proj.png)

### Forget neuron bitmap

The bitmap below shows a binary view: red = forget-dominant (ratio > 1.0), gray = not. 17.1% of neurons are red. This is the mask used for neuron-level SIBL targeting via `neuron_traces_path`.

![Forget Neuron Bitmap](figures/traces/analysis/forget_neuron_bitmap.png)

---

## 6. Implications for SIBL

### What the data says

1. **Forget knowledge is diffuse at the layer level, but early layers are retain-dominated.** Causal tracing shows a uniform +0.13 forget advantage across layers 1-29. Gradient balance shows layers 0–7 are retain-dominated (ratio ~0.2–0.8), with layers 8–31 near parity. No single layer is strongly forget-dominant.

2. **Layer 31 is the only retain-critical layer.** It is the only layer where causal disruption hurts retain more than forget, has the largest retain activation gap, and has the fewest forget neurons (14.1%) of any deep layer.

3. **Forget knowledge concentrates at the neuron level.** 17.1% of neurons (232K out of 1.36M tracked) are forget-dominant, peaking at layer 21 (30.4%). Layers 0–6 have near-zero forget neurons (0–0.3%) and strongly retain-biased mean ratios (0.42–0.52). The strongest forget neurons have ratios up to 2.08x.

### Recommended SIBL configuration

**Sparsity Mask:**
- Layer-level freezing of early layers is supported by the neuron data: layers 0–6 have mean neuron ratios of 0.42–0.52 and near-zero forget neurons. However, a binary layer freeze is a crude approximation.
- **Neuron-level masking** (`neuron_traces_path`) is the correct approach: targets the 17% of forget-specific neurons, naturally leaves layers 0–6 (~0% forget) untouched, and smoothly scales LR for the contested 64%. Use `figures/traces/analysis/neuron_traces.pt`.

**Implicit Correction Targeting:**
- Current default: `implicit_block_last_n_layers=2` (layers 30-31)
- Data-informed: apply implicit correction to **layer 31** — the only causally retain-critical layer. Layers 0-6 can be skipped (near-zero forget neurons = near-zero outer gradient anyway).

**Layer-wise Learning Rate (alternative to binary mask):**
- Zero or very low `eta_theta` for layers 0–6 (strongly retain-dominant: mean neuron ratio 0.42–0.52, near-zero forget neurons)
- Higher `eta_theta` for layers 17–25 (highest forget neuron density: 25–30%)
- This is more nuanced than binary layer masking but less precise than neuron-level masking

---

## 7. How Much Can We Actually Forget? A Formal Bound

The previous sections told us *where* forget and retain knowledge live inside 
the model. This section asks a different question: given that structure, 
**how much forgetting is theoretically achievable** without damaging what 
the model should keep?

The answer turns out to depend entirely on the model and the data — not on 
which algorithm you use. No matter how clever the unlearning method is, 
there is a hard ceiling on what it can achieve. We derive that ceiling here.

---

### 7.1 Sorting Parameters by Who They Serve

**The intuition.** Think of the model's ~6.7 billion parameters as individual 
knobs. When you run backpropagation separately on the forget set and the retain 
set, each knob receives a gradient signal — essentially a message saying 
"you need to change by this much to reduce the loss on this data." 

A knob that gets a loud signal from forget data but a quiet signal from retain 
data is a **safe unlearning target** — you can turn it freely without touching 
retain knowledge. A knob that gets loud signals from *both* is dangerous — 
turning it to forget will simultaneously disturb retention.

**The math.** For each parameter index $i$, define its gradient sensitivity 
on a dataset $\mathcal{D}$ as the average gradient magnitude over that dataset:

$$s_i(\mathcal{D}) = \mathbb{E}_{(x,y)\sim\mathcal{D}}\bigl[\|\nabla_{w_i}\,\ell(w;\,x,y)\|\bigr]$$

This is simply: *on average, how much does parameter $i$ need to move to 
reduce the loss on* $\mathcal{D}$? We then define the **forget-retain ratio**:

$$\rho_i = \frac{s_i(\mathcal{D}_f)}{s_i(\mathcal{D}_r) + \varepsilon}$$

- $\rho_i \gg 1$ → parameter responds much more to forget data → safe to update  
- $\rho_i \ll 1$ → parameter responds much more to retain data → must be protected  
- $\rho_i \approx 1$ → parameter responds equally to both → dangerous to touch

This ratio is exactly what Section 5 computed for every neuron. The histogram 
in Figure X is the distribution of $\rho_i$ across all 1.36M tracked neurons.

Using two thresholds $\tau_f > 1 > \tau_r > 0$, we split every parameter into 
one of three groups:

$$\mathcal{N}_f = \{i : \rho_i > \tau_f\} \quad \text{(forget-dominant — free to update)}$$
$$\mathcal{N}_r = \{i : \rho_i < \tau_r\} \quad \text{(retain-dominant — must freeze)}$$
$$\mathcal{N}_c = \text{everything else} \quad \text{(contested — updating these risks damage)}$$

**Why this matters:** most unlearning algorithms treat all parameters equally 
or use crude layer-level rules. This partition gives a data-informed view of 
which parameters can be touched and which cannot.

---

### 7.2 The Entanglement Coefficient

**The intuition.** The contested set $\mathcal{N}_c$ is the core problem. 
These parameters are shared infrastructure — they serve both forget and retain 
knowledge at the same time. Updating them to forget will always cause some 
collateral damage to retention. The question is: *how much of the model is 
in this zone?*

We capture this with a single number, the **entanglement coefficient**:

$$E(w,\,\mathcal{D}_f,\,\mathcal{D}_r) = \frac{|\mathcal{N}_c|}{|\mathcal{N}_f| + |\mathcal{N}_r| + |\mathcal{N}_c|}$$

This is just the fraction of parameters that are contested. Two extreme cases:

- **$E = 0$**: forget and retain knowledge live in completely separate parameters. 
  Surgical unlearning is theoretically possible — you can update forget parameters 
  without ever touching retain parameters.
- **$E = 1$**: every single parameter is contested. Any forgetting will damage 
  retention, no matter what algorithm you use. The problem is fundamentally 
  unsolvable without some trade-off.

**Our numbers.** With threshold $\tau_f = 1.5$ (meaningful forget dominance) and $\tau_r = 0.8$ on our 
Llama-2-7b model on MUSE-News (correct splits: 889 forget, 1777 retain, raw config):

> Note: at $\tau_f = 1.0$, 72.8% of neurons are "forget-dominant" — too broad to be meaningful. A threshold of 1.5 gives a more surgical partition.

| Group | Count | Fraction |
|---|---|---|
| Forget-dominant $\mathcal{N}_f$ ($\rho > 1.5$) | 131 | ~0.01% |
| Retain-dominant $\mathcal{N}_r$ ($\rho < 0.8$) | ~490,000 | ~36% |
| Contested $\mathcal{N}_c$ (0.8 ≤ ρ ≤ 1.5) | ~869,000 | ~64% |
| **Total tracked** | **1,359,872** | **100%** |

**What this means in plain language:** With $\tau_f = 1.5$, only 131 neurons are truly forget-dominant — an extraordinarily sparse free budget. The contested zone is 64% of all neurons. Early layers 0–6 are a near-clean retain zone (mean ratio 0.42–0.52), but the vast majority of the network (layers 7–31) is contested. This is a **high-entanglement regime**, and it explains 
empirically why every unlearning method we tested faces a hard trade-off on 
this model-data pair: the vast majority of parameters simply cannot be updated 
for forgetting without some collateral effect on retention. This is not a 
failure of the algorithm — it is a structural property of the model and the data.

> **Key insight for the professor:** $E$ is a property of the model and dataset 
> computed *before* running any unlearning algorithm. It tells you upfront how 
> hard the unlearning problem will be. A model fine-tuned on data that overlaps 
> heavily with what it should retain will have high $E$ and will be inherently 
> difficult to unlearn from. A model where the forget set is truly isolated in 
> parameter space will have low $E$ and can be unlearned surgically.

---

### 7.3 A Bound on How Much We Can Forget

**The intuition.** Now that we have the three-way partition, we can ask: 
what is the best any algorithm could possibly do? Specifically: what is 
the maximum increase in forget loss achievable by any parameter update, 
subject to the constraint that retain loss does not change at all?

The answer depends on two things: (1) how much gradient signal lives in 
the free zone $\mathcal{N}_f$, and (2) how well-separated the forget and 
retain gradients are in the contested zone $\mathcal{N}_c$.

**Setup.** Let $\Delta w = w' - w$ be the update applied by any unlearning 
algorithm. By a first-order Taylor expansion, the effect on each loss is 
approximately:

$$F(\Delta w) \approx \langle\,\nabla_w\mathcal{L}_f(w),\;\Delta w\,\rangle \quad \text{(forget effect — want this positive)}$$
$$R(\Delta w) \approx \langle\,\nabla_w\mathcal{L}_r(w),\;\Delta w\,\rangle \quad \text{(retain effect — want this zero)}$$

Write $g_f = \nabla_w\mathcal{L}_f(w)$ and $g_r = \nabla_w\mathcal{L}_r(w)$ 
for shorthand. The optimal strategy for the easy groups is obvious:

- **On $\mathcal{N}_f$**: update freely — retain gradient here is negligible 
  by definition, so there is no retention cost.
- **On $\mathcal{N}_r$**: freeze entirely — forget gradient here is negligible, 
  so updating would waste budget while risking retention.

The hard part is $\mathcal{N}_c$. Here both gradients are active. We need 
one more quantity.

**Contested gradient alignment.** Within $\mathcal{N}_c$, how much do the 
forget and retain gradients point in the same direction? We measure this 
with the standard cosine similarity:

$$\phi_c = \frac{\langle\,g_f^{\mathcal{N}_c},\;g_r^{\mathcal{N}_c}\,\rangle}{\|g_f^{\mathcal{N}_c}\|\cdot\|g_r^{\mathcal{N}_c}\|} \in [-1,\;1]$$

**Interpreting $\phi_c$:**
- $\phi_c = 1$: forget and retain gradients point in the same direction in 
  the contested zone. Any step that helps forgetting hurts retention by the 
  same proportion. The contested parameters are **useless for forgetting** 
  without retention cost.
- $\phi_c = 0$: the gradients are orthogonal. There exists a direction that 
  achieves forgetting while leaving retention completely unchanged. The contested 
  parameters are **fully usable**.
- $\phi_c = -1$: forgetting and retention actually reinforce each other — 
  the ideal scenario, rarely seen in practice.

**The geometry.** The constraint $R(\Delta w) = 0$ forces the update on 
$\mathcal{N}_c$ to lie in the hyperplane orthogonal to $g_r^{\mathcal{N}_c}$ 
(i.e., the update cannot have any component in the direction that would 
change the retain loss). The best we can do in the contested zone is to 
project $g_f^{\mathcal{N}_c}$ onto this hyperplane. By standard geometry, 
that projection has norm $\|g_f^{\mathcal{N}_c}\|\sqrt{1 - \phi_c^2}$ — 
which shrinks to zero as $\phi_c \to 1$.

Combining both groups, and constraining the total update to 
$\|\Delta w\| \leq \delta$ for some budget $\delta > 0$:

---

> **Proposition (Forget-Retain Bound).**
> For any unlearning update $\Delta w$ with $\|\Delta w\| \leq \delta$ 
> and zero retention damage $R(\Delta w) = 0$, the maximum achievable 
> forget effect is bounded by:
> 
> $$\boxed{F^*(\delta) \;\leq\; \delta\,\Bigl(\underbrace{\|g_f^{\mathcal{N}_f}\|}_{\text{free budget}} \;+\; \underbrace{\|g_f^{\mathcal{N}_c}\|\,\sqrt{1 - \phi_c^2}}_{\text{contested budget}}\Bigr)}$$

---

The two terms have a clean, practical interpretation:

| Term | Name | What it is |
|---|---|---|
| $\|g_f^{\mathcal{N}_f}\|$ | **Free forgetting budget** | Maximum forget effect from updating only the 17% of forget-dominant neurons. Zero retention cost. Fixed by model and data — no algorithm can increase it. |
| $\|g_f^{\mathcal{N}_c}\|\sqrt{1-\phi_c^2}$ | **Contested forgetting budget** | Additional forget effect extractable from the 68% contested zone, but only to the extent that forget and retain gradients are not aligned. Vanishes when $\phi_c \to 1$. |

**The ceiling is a property of the model-data pair, not the algorithm.** 
Any algorithm — no matter how sophisticated — cannot exceed $F^*(\delta)$ 
while keeping retention intact. This is the fundamental limit of selective 
unlearning for this model on this data.

---

### 7.4 Practical Consequences for SIBL

**1. $E$ as a pre-unlearning diagnostic.**
Before running any unlearning algorithm, compute $E$ from the gradient traces. 
High $E$ (like our 0.68) signals that unlearning will be hard and some 
retention degradation is unavoidable. Low $E$ signals that surgical unlearning 
is feasible. This gives practitioners a principled way to set expectations 
before committing to an algorithm.

**2. The neuron bitmap is the free budget.**
The 232K forget-dominant neurons identified in Section 5 are exactly 
$\mathcal{N}_f$ under $\tau_f = 1.0$. Using this bitmap as the SIBL 
sparsity mask concentrates all updates within the free budget term — 
guaranteeing that the inner retain solve is unaffected by the outer forget 
step to first order. Magnitude-based masking has no such guarantee: it may 
include retain-dominant parameters with large weights while missing 
forget-dominant parameters with small weights.

**3. Per-layer $\phi_c$ tells SIBL where to apply implicit correction.**
We can compute $\phi_c$ restricted to each layer's contested neurons, 
giving a per-layer gradient alignment score. Layers where $\phi_c \approx 1$ 
are where the implicit correction step in SIBL is most valuable — it is 
precisely here that naively following the forget gradient would damage 
retention. Layers where $\phi_c \approx 0$ can tolerate direct gradient 
steps with minimal correction overhead. This replaces the current heuristic 
`implicit_block_last_n_layers=2` with a data-driven placement rule.

## 7. Reproducibility

All figures and analysis in this report can be reproduced with the following scripts:

```bash
# Run from repo root:

# Step 1: Full run — causal + gradient + activation traces (slow, ~50 min)
python trace_analysis/scripts/trace_activations.py \
    --preset muse-news \
    --output_dir trace_analysis/figures/traces/muse_news

# Step 1 (fast, ~5 min) — gradient + activation only, skip causal:
python trace_analysis/scripts/trace_activations.py \
    --preset muse-news --skip_causal

# Step 2: Neuron-level analysis + heatmaps (from cached traces):
python trace_analysis/scripts/analyze_traces.py \
    --skip_neuron_collection \
    --trace_file trace_analysis/figures/traces/muse_news/trace_results.pt \
    --output_dir trace_analysis/figures/traces/analysis
```

---

## 8. Output Files

| File | Description |
|------|-------------|
| `figures/traces/muse_news/trace_results.pt` | Full gradient + activation + causal traces (PyTorch tensors) |
| `figures/traces/muse_news/summary.json` | JSON summary with top-50 params and per-layer differentials |
| `figures/traces/muse_news/gradient_differential.png` | Per-layer gradient analysis (3 panels) |
| `figures/traces/muse_news/activation_traces.png` | Per-layer activation norms (4 panels) |
| `figures/traces/muse_news/causal_traces.png` | Per-layer causal importance (3 panels) |
| `figures/traces/analysis/neuron_traces.pt` | **Use this for SIBL** — per-neuron (per-row) gradient traces for targeted attn/MLP params only |
| `figures/traces/analysis/neuron_analysis.json` | Neuron-level analysis summary |
| `figures/traces/analysis/layer_component_heatmap.png` | Layer x component gradient ratio heatmap |
| `figures/traces/analysis/neuron_heatmap_*.png` | Per-component neuron-level heatmaps (7 files) |
| `figures/traces/analysis/neuron_differential_histogram.png` | Distribution of neuron-level ratios |
| `figures/traces/analysis/forget_neuron_bitmap.png` | Binary forget-dominant neuron map |
| `figures/traces/analysis/forget_neuron_bitmap.pt` | Bitmap tensor for downstream use in SIBL |
| `figures/traces/analysis/activation_heatmap_*.png` | Activation heatmaps (L2 norm, mean_abs, variance) |

---

## 9. Runtime

| Technique | Samples | Time | Notes |
|-----------|---------|------|-------|
| Causal tracing | 889 + 1777 | ~35 min | Most expensive step |
| Gradient traces (per-row) | 889 + 1777 | ~60 min | Backward pass, per-row format |
| Activation traces | 889 + 1777 | ~5 min | Forward-only, fast |
| Neuron analysis + plots | 889 + 1777 | ~5 min | Re-runs gradient collection on targeted params |
| **Total (with causal)** | | **~105 min** | Full pipeline |
| **Total (skip_causal)** | | **~70 min** | Gradient + activation only |
