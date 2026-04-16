# MUSE News Handover

## What works (TOFU)

On TOFU forget 1% (Llama-3.2-1B-Instruct), our best result:
- **v2 Phase 1 step-8: FQ=0.405, MU=0.581** (approaches PerTA-grad FQ=0.514, MU=0.581)
- Config: logit_margin answer-masked, K=3 inner retain, LR=2e-4, no implicit
- Key fix: `_compute_logit_margin_loss` now masks to answer tokens only (labels != -100)
- Code: `src/trainer/unlearn/lora_implicit.py:201-212`

## MUSE News task

- Model: Llama-2-7b-hf, target: `muse-bench/MUSE-News_target`
- Data: `muse-bench/MUSE-News`, subset `raw`, split `forget` (889 samples, 813 chunks@1024), `retain1` (1777 samples, 1603 chunks)
- Experiment config: `configs/experiment/unlearn/muse/lora_implicit.yaml`
- LoRA: r=16, 7 modules, ~40M params
- Dataloader: anchor=forget, 203 steps/epoch at bs=4
- Gold targets: forget_knowmem ≤ 0.324, retain_knowmem ≥ 0.552
- Retain logs: `saves/eval/muse_Llama-2-7b-hf_News_retrain/MUSE_EVAL.json`
- Eval metrics: forget_knowmem_ROUGE, forget_verbmem_ROUGE, retain_knowmem_ROUGE, extraction_strength, privleak

## What failed on MUSE News

1. **Phase 1 pure forget LR=1e-4**: Destroyed everything. retain=0.000, forget=0.003. Way too aggressive.
2. **Phase 1 bilevel LR=5e-5 K=1**: ALM lambda hit cap (5.0), L_fgt barely moved (23→20). Too conservative — single inner step can't protect retain on 7B, so ALM dominates and blocks forgetting. Result: forget=0.500 (target ≤0.324), retain=0.478 (target ≥0.552).
3. **Phase 1 pure forget LR=1e-5**: Running when killed. L_fgt went 23.4→22.3 in 20/204 steps, L_ret already at 0.99. Likely still too destructive over full epoch.

## What needs to happen

Phase 1 should NOT be pure forget. It needs gentle forgetting WITH retain protection. The TOFU insight was K=3 inner retain steps — this keeps MU stable while outer pushes forget. But on MUSE 7B:
- K=1 was too weak (ALM dominated)
- K=0 (pure forget) destroyed retain

**Suggested approach**: K=2 or K=3, logit_margin answer-masked, LR=3e-5 (matching our proven LoRA-BiAL outer_lr), 1 epoch. Then Phase 2 continues with same setup for 2-3 more epochs, eval after each.

## Prior art (LoRA-BiAL Ze0 — our best on MUSE News)

Best config from 40+ ablations: PerTA λ=3.5 init → LoRA r=16 bilevel, NPO β=4.0, K=3, T=25 steps, outer_lr=3e-5, inner_lr=2e-4.
- Result: forget_knowmem=0.289, retain=0.449, extraction=0.028
- But NPO saturates instantly on this data (same cold-start as TOFU). Logit_margin should do better.

## Key files

- `src/trainer/unlearn/lora_implicit.py` — trainer (logit_margin fix at line 201-212)
- `configs/experiment/unlearn/muse/lora_implicit.yaml` — experiment config
- `configs/trainer/LoRAImplicit.yaml` — default trainer params
- `scripts/run_muse_news_implicit.sh` — current runner (needs redesign)
- `docs/results/muse_news.md` — all baseline results
- `docs/results/tofu.md` — TOFU results showing the method works

## Run command template

```bash
python src/train.py --config-name=unlearn.yaml \
    experiment=unlearn/muse/lora_implicit \
    data_split=News \
    task_name=muse_news_XXX \
    retain_logs_path=saves/eval/muse_Llama-2-7b-hf_News_retrain/MUSE_EVAL.json \
    trainer.args.per_device_train_batch_size=4 \
    trainer.args.gradient_accumulation_steps=1 \
    trainer.args.gradient_checkpointing=true \
    trainer.method_args.forget_loss_type=logit_margin \
    trainer.method_args.K=3 \
    trainer.method_args.eta_theta=3e-5 \
    trainer.method_args.eta_in=2e-4 \
    trainer.method_args.epsilon=0.50 \
    trainer.method_args.lambda_max=5.0 \
    trainer.method_args.use_implicit=false \
    trainer.method_args.adaptive_lr=false \
    trainer.method_args.checkpoint_every_epoch=true \
    trainer.args.num_train_epochs=4
```
