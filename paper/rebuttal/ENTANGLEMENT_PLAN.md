# MUSE News Argument via Forget–Retain Entanglement

## The claim we want to make

Two-part rebuttal argument for Reviewer 2's "why does BLADE trail on MUSE News?" concern:

1. **MUSE News has structurally high forget–retain entanglement** — forget articles and retain articles cover overlapping current-events entities (MPs, IRA, football clubs, etc.). MUSE Books does not (forget = HP novels, retain = HP fan-wiki — different register and vocabulary). This entanglement partly explains why *any* method's forget-side score is bounded on News.
2. **Despite this, BLADE does *not* fail on News overall** — it lands within 0.033 of PDU (0.544 vs 0.577) *and* completely dominates PDU on the stress-tests (4× scale, 4 sequential steps) where PDU collapses to HM≈0.005/0.130. Robustness > raw HM.

We need a concrete **entanglement score** to back up point (1). This doc surveys standard metrics from the literature and picks the ones we can compute cheaply.

---

## Standard entanglement metrics found in the literature

### 1. Cheng et al. 2026 (ICLR) — "Machine Unlearning under Retain–Forget Entanglement"
This is the paper our current draft cites. They operationalize entanglement two ways:

**(a) kNN adjacent-retain fraction (§B.6 of the paper).**
- Extract pretrained-model embeddings for every forget sample.
- For each forget sample, find its top-k=20 nearest neighbours in the retain set.
- Assign each retain sample an "adjacency score" = number of times it appears in these kNN lists.
- Top 10% of retain by score = "adjacent retain set" `D_r^adj`.
- **Entanglement score = |D_r^adj| / |D_r|** relative to the fraction expected under a random baseline; or simpler, the *density* of the top-k neighbourhood.

Reference embedding: they use a pretrained ResNet-18 on CIFAR. For text, standard analog = sentence embedding from a frozen encoder (e.g., MiniLM, Sentence-BERT, or the base LLM itself).

**(b) Wasserstein-2 distance between per-sample loss distributions (§4.2.2).**
- Compute cross-entropy loss `ℓ(f_θ(x_i), y_i)` for every sample in `D_forget` and `D_retain` under the target model.
- Sort the two arrays; the closed-form W₂ distance is `(1/N Σ (ā_i − b̄_i)²)^{1/2}`.
- **Smaller W₂ = more entangled** (forget and retain sit in the same loss regime under the same model).

### 2. EGUP (Wang et al. 2025) — "Entanglement-Guidance with Proxy Constraint"
Also on TOFU. Concrete metric:

**Cosine similarity between mean forget-sentence embedding and mean retain-sentence embedding**:
```
E_D_r = (1/|D_r|) Σ AvgTokenEmbedding(θ_t, x_j)   for x_j ∈ D_r
sim_i = cos(E(x_i), E_D_r)   for x_i ∈ D_f
```
Then aggregate `sim_i` over the forget set. Higher `sim` = more entangled.

This is the simplest, cheapest metric — one forward pass per sample, no training, no attack.

### 3. SKEB (Sun et al. 2025) — "Stimulus-Knowledge Entanglement-Behavior Framework"
Graph-based (concept graphs with connection-strength / eigenvector metrics). Nine metrics total. More complex, less applicable here — designed for individual QA prompts, not corpus-level entanglement.

### 4. Domain-native proxies (no ML)
Simple text-statistic baselines that don't need a model:

- **Vocabulary Jaccard**: `|V_f ∩ V_r| / |V_f ∪ V_r|` on token vocab or top-k content-word set.
- **TF-IDF cosine**: bag-of-words vectors on both corpora, cosine of their L2-normalized centroids.
- **Named-entity overlap**: run an NER tool, compute `|E_f ∩ E_r| / |E_f ∪ E_r|` — targets the "MP-Stuart-McDonald is in both News forget and retain" case directly.

---

## What we should compute

Recommendation: report **three metrics side-by-side** for MUSE News vs MUSE Books vs TOFU vs KnowUnDo. This gives a robust picture without overselling any single metric.

| Metric | Where it comes from | What it captures | Cost |
|---|---|---|---|
| **Cosine of mean embeddings** (EGUP-style) | Wang et al. 2025 | Semantic entanglement at the sentence level | 1 fwd pass per sample × 2 corpora |
| **TF-IDF cosine of corpus centroids** | Standard IR | Lexical / topic overlap without a model | ~seconds |
| **Named-entity Jaccard** | Domain-specific | People/places/orgs that appear in BOTH | ~minutes with spaCy or transformers NER |

Ideally all three tell the same story: **News > Books entanglement**, which frames why News is intrinsically harder than Books.

### Sanity prediction (before running)

- **MUSE News**: high — same journalistic domain, same time-window, overlapping public figures
- **MUSE Books**: low — HP novels vs HP fan-wiki have completely different style, only entities overlap
- **TOFU (fgt10 vs ret90)**: low — synthetic biographies, forget/retain authors are structurally disjoint by design
- **KnowUnDo Copyright/Privacy**: moderate — same domain injections, but structured Q-A pairs are less entangled than free-form articles

If the numbers come out this way, the argument for the rebuttal is clean:

> "**MUSE News exhibits substantially higher forget–retain entanglement than every other benchmark in this paper** (Table X). Under this structural constraint, the forget-side ceiling on any unlearning method's HM is lower on News than on the other benchmarks — no method can push MUSE News HM as high as MUSE Books HM. BLADE's HM=0.544 on News is within 0.033 of the strongest baseline (PDU=0.577), while dominating PDU under the scale/sustainability stress-tests §5.6 that matter for repeated deployment."

---

## Implementation plan (when you approve)

1. **Compute all three metrics on 4 benchmarks** — TOFU fgt10, MUSE News, MUSE Books, KnowUnDo (copyright + privacy).
2. **Embedding metric**: use `sentence-transformers/all-MiniLM-L6-v2` (small, standard, ~90 MB, no gated repo). Alternative: mean-pool token embeddings from `Llama-3.2-1B-Instruct` on both corpora — closer to what our unlearning operates over.
3. **TF-IDF**: sklearn one-liner over the raw text. Report cosine of L2-normalized centroids.
4. **NER Jaccard**: spaCy `en_core_web_sm` — cheap. Report Jaccard over the set of entities that appear at least twice.
5. **Report the numbers as an appendix table** ("Benchmark Entanglement Profile") and cite it from a new paragraph in §4.5 (MUSE) as the framing for the News gap.

Expected compute: **~30 min total** on one CPU box, no GPU needed for TF-IDF / NER. Embedding pass is small enough to run on 1 GPU in a few minutes.

---

## What NOT to include

- **Full Wasserstein loss-distribution metric** (Cheng 2026) — requires running the target model on both corpora, computing per-sample CE, sorting. Doable but more complex; and the interpretation ("smaller W₂ = more entangled") is less immediately intuitive than cosine or Jaccard for a rebuttal.
- **SKEB graph metrics** — too heavy, poor fit for corpus-level statement.
- **Custom entanglement definitions** — reviewers will ask "why not use the existing published metric?" Better to use two literature-standard ones (Cheng / EGUP) + one classical IR baseline.
