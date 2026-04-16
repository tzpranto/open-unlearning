# Developer Guide

Quick-reference for contributors and AI agents working in this repo.

---

## Folder Structure

```
open-unlearning/
  configs/              # Hydra config files
    experiment/         #   Full experiment configs (dataset + trainer + eval)
    trainer/            #   Trainer-specific configs (SIBL.yaml, LoRABiAL.yaml, ...)
    model/              #   Model configs
    data/               #   Dataset configs
    eval/               #   Evaluator configs
  src/                  # Source code
    trainer/            #   Trainer implementations
      unlearn/          #     Unlearning methods (sibl.py, lora_bial.py, npo.py, ...)
      base.py           #     FinetuneTrainer base class
      sparsity.py       #     SparsityManager for mask creation
    data/               #   Dataset loading and collation
    evals/              #   Evaluation metrics
    model/              #   Model loading utilities
  docs/
    design/             #   Paper PDFs, method descriptions, algorithm specs
    results/            #   Benchmark result reports (MUSE, TOFU, WMDP)
    notes/              #   Agent handover notes, briefings, session logs
    *.md                #   Upstream framework docs (components, evaluation, hydra, ...)
  scripts/              # Runner scripts, utilities, report generation
  saves/                # Model checkpoints and experiment outputs (gitignored heavy files)
  data/                 # Downloaded datasets (gitignored)
  paper/                # LaTeX source for the paper
```

---

## Guidelines

### 1. Python Environment

Always use the `unlearning` conda environment. Without it, flash attention and other dependencies will fail.

```bash
# Option A: use the binary directly
/datadrive/conda/envs/unlearning/bin/python src/train.py ...

# Option B: activate first
source /datadrive/conda/etc/profile.d/conda.sh && conda activate unlearning
```

Flash Attention is not installed in this env. Always pass:
```
model.model_args.attn_implementation=sdpa
```

### 2. Clean Code and Folder Structure

- Keep `src/` for source code only. No scripts, reports, or notebooks.
- Keep `scripts/` for runner shell scripts and standalone utilities.
- Keep `docs/` organized by subfolder: `design/`, `results/`, `notes/`.
- Do not leave stale artifacts (temp files, debug outputs, cache dirs) in the repo root.

### 3. Keep This Guide Updated

When you add new folders, methods, or change conventions, update this file. It is the single source of truth for repo structure and workflow.

### 4. Git Discipline

- Commit with meaningful messages that describe *why*, not just *what*.
- Push to remote after meaningful commits (especially before ending a session).
- Maintain the changelog below.

### 5. Adding a New Unlearning Method

If a method is significantly different from existing ones, create a new trainer:

1. **Implement**: Create `src/trainer/unlearn/your_method.py` with a class that extends `UnlearnTrainer` (from `trainer.unlearn.base`).
2. **Register**: In `src/trainer/__init__.py`, add:
   ```python
   from trainer.unlearn.your_method import YourMethod
   _register_trainer(YourMethod)
   ```
3. **Config**: Create `configs/trainer/YourMethod.yaml` with:
   ```yaml
   defaults:
     - finetune
   handler: YourMethod
   method_args:
     your_param: value
   args:
     per_device_train_batch_size: 2
     ...
   ```
4. **Experiment config** (optional): Create `configs/experiment/unlearn/<benchmark>/your_method.yaml` to wire up dataset + trainer + eval.

The `handler` field in the YAML must match the class name exactly (it is looked up in `TRAINER_REGISTRY`).

### 6. Changelog

Maintain a running log of meaningful changes alongside git pushes.

| Date | Commit | Summary |
|------|--------|---------|
| 2026-04-16 | `fb70338` | Clean SIBL to match Algorithm 1; fix 9 reviewer bugs; delete Fisher/GSP variants |
| 2026-04-16 | *(this)* | Reorganize docs/reports/notes into `docs/` subfolders; add DEVELOPER_GUIDE.md |

### 7. Code Review Protocol

Every code change must be reviewed by a separate subagent before finalizing:

1. **Write** the code changes.
2. **Launch a reviewer subagent** to independently verify correctness (logic, formulation match, edge cases).
3. **Fix** any issues found by the reviewer.
4. **Final review** pass to confirm fixes are correct.
5. Only then commit and push.
