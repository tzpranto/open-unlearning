# BLADE: Bilevel Low-rank Adaptive Data Erasure

A fork of [locuslab/open-unlearning](https://github.com/locuslab/open-unlearning), extended with **BLADE** — our method for LLM unlearning via bilevel constrained optimization with LoRA. Evaluated on TOFU and MUSE benchmarks.

---

## Setup

```bash
git clone <this-repo> && cd open-unlearning
conda create -n unlearning python=3.11
conda activate unlearning
pip install -e .[lm_eval]
pip install flash-attn --no-build-isolation

# Download evaluation reference logs (retain set evals for TOFU/MUSE)
python setup_data.py --eval
```

---

## Reproducing Results

### Baselines

**TOFU** (GradAscent, GradDiff, NPO, SimNPO, RMU, BLURNPO, PDU; 1B + 3B, 5 seeds):

```bash
# Edit SEEDS, QUEUE, TRAINERS in the script as needed
nohup bash scripts/tofu_baselines.sh > saves/unlearn/tofu_baselines.log 2>&1 &
```

**MUSE** (GA, GradDiff, NPO, SimNPO, RMU, BLURNPO, PDU; Llama-2-7b-hf, per-split params):

```bash
# MUSE News
nohup DATA_SPLIT=News bash scripts/muse_baselines.sh > saves/unlearn/muse_news_baselines.log 2>&1 &

# MUSE Books
nohup DATA_SPLIT=Books bash scripts/muse_baselines.sh > saves/unlearn/muse_books_baselines.log 2>&1 &

# Multiple seeds
nohup DATA_SPLIT=News SEEDS="42 123 456 789 1337" bash scripts/muse_baselines.sh > saves/unlearn/muse_news_baselines.log 2>&1 &
```

**WMDP Bio+Cyber** (GA, GradDiff, NPO, SimNPO, RMU, BLURNPO; Zephyr-7b-beta):

```bash
nohup bash scripts/wmdp_baselines.sh > saves/unlearn/wmdp_baselines.log 2>&1 &
```

RMU uses paper-correct Zephyr notebook params (centerforaisafety/wmdp): steering_coeff=6.5, alpha=1200, lr=5e-5, max_steps=150, trainable=layers.5-7.mlp.down_proj. Other methods use epoch-based training with upstream defaults.

All split-specific params (BLUR beta/lr, PDU eps) are set automatically by the script based on `DATA_SPLIT`:

| Method | Config | News params | Books params | Source |
|--------|--------|-------------|--------------|--------|
| RMU | `unlearn/muse/rmu.yaml` | lr=5e-5, layers.5-7.mlp.down_proj | same | open-unlearning repro |
| BLUR-NPO | `unlearn/muse/blurnpo_muse.yaml` | beta=0.05, lr=2.5e-5 | beta=0.4, lr=1e-5 | arXiv:2506.08164, Table 6 |
| PDU | `unlearn/muse/pdu.yaml` | alpha=50, eps=1.5 | alpha=50, eps=0.1 | arXiv:2506.05314 |

BLUR-NPO on MUSE uses raw-logits NPO (full-vocab `logsigmoid(beta * (ref_logits - model_logits)).mean()`), matching the authors' MUSE implementation. TOFU and WMDP use standard per-token NLL NPO.

PDU TOFU overrides (`alpha=100, eps=0.3, dual_step_size=5`) are from `community/methods/PDU/run.sh` (arXiv:2506.05314).

### BLADE (ours)

**TOFU** (1B + 3B, 5 seeds × 3 splits):

```bash
# BLADE with auto-epsilon (eps_multiplier=0.85)
# T=250 for forget01/05, T=500 for forget10
nohup bash scripts/tofu_blade.sh > saves/unlearn/tofu_blade.log 2>&1 &
```

**MUSE** (Llama-2-7b-hf, Books or News):

```bash
# MUSE Books (eps_multiplier=3.2, T=250, conv_patience=20)
nohup DATA_SPLIT=Books bash scripts/muse_blade.sh > saves/unlearn/muse_blade_books.log 2>&1 &

# MUSE News (eps_multiplier=1.3, T=150)
nohup DATA_SPLIT=News bash scripts/muse_blade.sh > saves/unlearn/muse_blade_news.log 2>&1 &
```

All scripts auto-skip completed runs and clean model weights after eval. Results are saved to `results/` as CSV and markdown.

### Running a single method manually

```bash
# Train
CUDA_VISIBLE_DEVICES=0 python src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/tofu/default.yaml \
  trainer=GradAscent \
  model=Llama-3.2-3B-Instruct \
  model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_Llama-3.2-3B-Instruct_full \
  forget_split=forget10 retain_split=retain90 \
  trainer.args.seed=42 \
  task_name=my_run

# Eval
CUDA_VISIBLE_DEVICES=0 python src/eval.py \
  experiment=eval/tofu/default.yaml \
  model=Llama-3.2-3B-Instruct \
  model.model_args.pretrained_model_name_or_path=saves/unlearn/my_run \
  paths.output_dir=saves/unlearn/my_run/evals \
  forget_split=forget10 holdout_split=holdout10 \
  retain_logs_path=saves/eval/tofu_Llama-3.2-3B-Instruct_retain90/TOFU_EVAL.json
```

---

## LLM Judge Evaluation

The LLM judge uses an API-based model (CPU-only, no GPU needed) to score generations on forget leakage, retain accuracy, and response quality.

### Run LLM judge on TOFU baselines

```bash
# All methods, all seeds, specific model + splits
bash scripts/llm_judge_run.sh --model 1B --splits "forget01 forget05 forget10"
bash scripts/llm_judge_run.sh --model 3B --splits "forget10"

# Specific methods and seeds
bash scripts/llm_judge_run.sh --model 1B --splits "forget01" --methods "NPO SimNPO" --seeds "42 123"

# Our method
bash scripts/llm_judge_run.sh --model 3B --splits "forget10" --ours
```

### Run LLM judge on MUSE baselines

```bash
# By method name (auto-discovers eval dirs)
bash scripts/llm_judge_run.sh --benchmark muse --model Llama-2-7b-hf \
  --splits "Books" --methods "GradAscent GradDiff NPO SimNPO RMU PDU" --seeds "42"

# Or point to specific eval directories
bash scripts/llm_judge_run.sh --benchmark muse \
  --eval-dirs "saves/unlearn/muse_Llama-2-7b-hf_Books_NPO_s42/evals"
```

### Run in background (recommended for large jobs)

```bash
nohup bash scripts/llm_judge_run.sh --model 1B --splits "forget01 forget05 forget10" \
  > saves/unlearn/llm_judge.log 2>&1 &
```

Results are appended to `results/tofu_llm_judge.csv` or `results/muse_llm_judge.csv`. Already-judged runs are auto-skipped.

---

## Results

### Where to find eval results

Auto-eval results (JSON) are saved per run under `saves/unlearn/<task_name>/evals/`:

| Benchmark | Files |
|-----------|-------|
| TOFU | `TOFU_EVAL.json` (per-example), `TOFU_SUMMARY.json` (aggregated) |
| MUSE | `MUSE_EVAL.json` (per-example), `MUSE_SUMMARY.json` (aggregated) |

The baselines scripts also aggregate results into CSV files:

| File | Description |
|------|-------------|
| `results/tofu_baselines.csv` | TOFU auto-eval metrics (MU, FQ, ES, fgt_Prob, fgt_ROUGE, HM) per method/seed |
| `results/tofu_llm_judge.csv` | TOFU LLM judge scores per method/seed |
| `results/muse_llm_judge.csv` | MUSE LLM judge scores per method/seed |
| `results/tofu.md` | TOFU results summary tables |
| `results/muse_books.md` | MUSE Books results + LLM judge |
| `results/muse_news.md` | MUSE News results + LLM judge |

### Harmonic Mean (HM) calculation

We use the harmonic mean as a single aggregate score that penalizes imbalance — any weak axis tanks the score.

**TOFU:**
```
HM = harmonic_mean(MU, 1 - fgt_Prob, 1 - fgt_ROUGE)
```
- `MU` = model utility (retain knowledge + real-world QA, higher = better)
- `fgt_Prob` = forget set probability (lower = better forgetting)
- `fgt_ROUGE` = forget set ROUGE (lower = better forgetting)
- If any component is zero, HM = 0.

**MUSE (auto-eval):**
```
HM = harmonic_mean(1 - forget_knowmem, 1 - forget_verbmem, retain_knowmem)
```
- `forget_knowmem` = forget set knowledge memorization ROUGE (lower = better)
- `forget_verbmem` = forget set verbatim memorization ROUGE (lower = better)
- `retain_knowmem` = retain set knowledge memorization ROUGE (higher = better)

**MUSE (LLM judge):**

LLM judge scores (FL, RA, RQ) are on a 0-2 scale. The HM in the LLM judge table is computed separately when building the markdown results tables. The raw judge CSV (`results/muse_llm_judge.csv`) contains per-run scores without HM.

---

## Project Structure

```
configs/
  experiment/unlearn/   # Experiment configs (tofu/, muse/, wmdp/)
  experiment/eval/      # Eval configs
  trainer/              # Method configs (GradAscent, NPO, RMU, BLADE, etc.)
  model/                # Model configs
scripts/                # Ready-to-run experiment scripts
src/
  train.py              # Training entry point
  eval.py               # Evaluation entry point
  trainer/unlearn/      # Method implementations
results/                # CSV and markdown result tables
community/              # Community-contributed method configs (PDU, etc.)
```

---

## License

MIT. See [LICENSE](LICENSE).

## Acknowledgements

Built on [open-unlearning](https://github.com/locuslab/open-unlearning) by Vineeth Dorna and Anmol Mekala. See the upstream repo for the full technical report and documentation.
