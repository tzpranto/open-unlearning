# BLADE: Bilevel Low-rank Adaptive Data Erasure

A fork of [locuslab/open-unlearning](https://github.com/locuslab/open-unlearning), extended with **BLADE** — our method for LLM unlearning via bilevel constrained optimization with LoRA. Evaluated on TOFU, MUSE, and KnowUnDo benchmarks.

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
nohup bash scripts/tofu_baselines.sh > saves/unlearn/tofu_baselines.log 2>&1 &
```

**MUSE** (GA, GradDiff, NPO, SimNPO, RMU, BLURNPO, PDU; Llama-2-7b-hf, per-split params):

```bash
# MUSE News
nohup DATA_SPLIT=News bash scripts/muse_baselines.sh > saves/unlearn/muse_news_baselines.log 2>&1 &

# MUSE Books
nohup DATA_SPLIT=Books bash scripts/muse_baselines.sh > saves/unlearn/muse_books_baselines.log 2>&1 &

# Multiple seeds
nohup DATA_SPLIT=News SEEDS="42 123 456 789 1024" bash scripts/muse_baselines.sh > saves/unlearn/muse_news_baselines.log 2>&1 &
```

**KnowUnDo** (GA, GradDiff, NPO, SimNPO, RMU, BLURNPO, PDU, MemFlex; Llama-2-7b-chat):

```bash
# Finetune target model first (use the original KnowUnDo repo: https://github.com/zjunlp/KnowUnDo)
# See KnowUnDo/pretrain/pretrain.py with config KnowUnDo/pretrain/config/finetune_lora.yaml
# LoRA r=8, alpha=16, dropout=0.1, all-linear; lr=1e-4, 10 epochs, BS=2×16, seed=100

# Run baselines
nohup bash scripts/knowundo_baselines.sh > saves/unlearn/knowundo_baselines.log 2>&1 &
```

Split-specific params (BLUR beta/lr, PDU eps) are set automatically by each script:

| Method | News params | Books params | Source |
|--------|-------------|--------------|--------|
| BLUR-NPO | beta=0.05, lr=2.5e-5 | beta=0.4, lr=1e-5 | arXiv:2506.08164, Table 6 |
| PDU | alpha=50, eps=1.5 | alpha=50, eps=0.1 | arXiv:2506.05314 |

### BLADE (ours)

**TOFU** (1B + 3B, 5 seeds × 3 splits):

```bash
nohup bash scripts/tofu_blade.sh > saves/unlearn/tofu_blade.log 2>&1 &
```

**MUSE** (Llama-2-7b-hf, Books or News):

```bash
# MUSE Books (T=250, converges ~step 78)
nohup DATA_SPLIT=Books bash scripts/muse_blade.sh > saves/unlearn/muse_blade_books.log 2>&1 &

# MUSE News (T=300, runs full duration)
nohup DATA_SPLIT=News bash scripts/muse_blade.sh > saves/unlearn/muse_blade_news.log 2>&1 &
```

**KnowUnDo** (copyright + privacy):

```bash
# Uses configs/experiment/unlearn/knowundo/blade.yaml
nohup bash scripts/knowundo_baselines.sh > saves/unlearn/knowundo_blade.log 2>&1 &
```

BLADE hyperparams: eps_mul=3.2, eta_theta=3e-5, K=3, tau=0.7, rho=0.1, lora_r=16, conv_patience=20. Same across all MUSE/KnowUnDo benchmarks; only T differs (250 for Books, 300 for News).

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

The LLM judge uses Claude Opus 4.7 via Bedrock (CPU-only, no GPU needed) to score generations on forget leakage, retain accuracy, and response quality.

```bash
# TOFU
python scripts/llm_judge.py --eval-dir saves/unlearn/<task>/evals --benchmark tofu

# MUSE
python scripts/llm_judge.py --eval-dir saves/unlearn/<task>/checkpoint-0/evals --benchmark muse

# KnowUnDo
python scripts/llm_judge.py --eval-dir saves/unlearn/<task>/evals --benchmark knowundo

# Batch mode (all entries in a CSV)
python scripts/llm_judge.py --csv results/tofu_baselines.csv --all --benchmark tofu
```

Results are appended to `results/{tofu,muse,knowundo}_llm_judge.csv`. Already-judged runs are auto-skipped.

---

## Results

### Where to find eval results

Auto-eval results (JSON) are saved per run under `saves/unlearn/<task_name>/evals/`:

| Benchmark | Files |
|-----------|-------|
| TOFU | `TOFU_EVAL.json`, `TOFU_SUMMARY.json` |
| MUSE | `MUSE_EVAL.json`, `MUSE_SUMMARY.json` |
| KnowUnDo | `KnowUnDo_EVAL.json` |

Summary tables:

| File | Description |
|------|-------------|
| `results/tofu.md` | TOFU results (1B + 3B, 5 seeds) |
| `results/muse_books.md` | MUSE Books results + LLM judge |
| `results/muse_news.md` | MUSE News results + LLM judge |
| `results/muse_sust_scal.md` | MUSE sustainability & scalability |
| `results/knowundo_copyright.md` | KnowUnDo copyright results |
| `results/knowundo_privacy.md` | KnowUnDo privacy results |
| `results/ablation.md` | BLADE ablation study |

### Harmonic Mean (HM) calculation

**TOFU:**
```
HM = harmonic_mean(MU, 1 - fgt_Prob, 1 - fgt_ROUGE)
```

**MUSE:**
```
HM = harmonic_mean(1 - forget_knowmem, 1 - forget_verbmem, retain_knowmem)
```

**KnowUnDo:**
```
HM = harmonic_mean(forget_efficacy, retain_ROUGE, MMLU)
```

---

## Project Structure

```
configs/
  experiment/unlearn/   # Experiment configs (tofu/, muse/, knowundo/)
  experiment/eval/      # Eval configs
  trainer/              # Method configs (GradAscent, NPO, RMU, BLADE, etc.)
  model/                # Model configs
scripts/                # Experiment and evaluation scripts
src/
  train.py              # Training entry point
  eval.py               # Evaluation entry point
  trainer/unlearn/      # Method implementations
    lora_bial.py            # BLADE base: bilevel loop, LoRA, ALM dual update
    lora_bial_adaptive.py   # BLADE adaptive: auto LR calibration + convergence detection
    lora_bial_losses.py     # Forget losses: clamped_entropy, GA, NPO, logit_margin
results/                # CSV and markdown result tables
```

---

## License

MIT. See [LICENSE](LICENSE).

## Acknowledgements

Built on [open-unlearning](https://github.com/locuslab/open-unlearning) by Vineeth Dorna and Anmol Mekala. See the upstream repo for the full technical report and documentation.
