#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#   TOFU v2 Phase 2: NPO bilevel from step-8 checkpoint
#   Fresh LoRA on step-8 merged model, implicit ON
#   K=3, lower LR, 5 epochs = 20 steps
# ═══════════════════════════════════════════════════════════════

set -euo pipefail
export PATH="/datadrive/conda/envs/unlearning/bin:$PATH"

SAVES="saves/unlearn"
P1_CKPT="$SAVES/tofu_1b_v2_p1/step-8"
RETAIN_LOGS="saves/eval/tofu_Llama-3.2-1B-Instruct_retain99/TOFU_EVAL.json"

echo "[$(date)] Phase 2: NPO bilevel from step-8, implicit ON"

OUTDIR="$SAVES/tofu_1b_v2_p2"
if [[ -d "$OUTDIR/evals" ]]; then
    echo "[SKIP] already done"
    exit 0
fi
rm -rf "$OUTDIR" 2>/dev/null

python src/train.py --config-name=unlearn.yaml \
    experiment=unlearn/tofu/lora_implicit_1b \
    model.model_args.pretrained_model_name_or_path="$P1_CKPT" \
    forget_split=forget01 \
    retain_split=retain99 \
    holdout_split=holdout01 \
    task_name="tofu_1b_v2_p2" \
    retain_logs_path="$RETAIN_LOGS" \
    trainer.args.per_device_train_batch_size=12 \
    trainer.args.gradient_accumulation_steps=1 \
    trainer.args.gradient_checkpointing=true \
    trainer.method_args.checkpoint_every_epoch=true \
    trainer.method_args.forget_loss_type=logit_margin \
    trainer.method_args.K=3 \
    trainer.method_args.eta_theta=1e-4 \
    trainer.method_args.eta_in=2e-4 \
    trainer.method_args.epsilon=0.05 \
    trainer.method_args.lambda_max=5.0 \
    trainer.method_args.use_implicit=true \
    trainer.method_args.implicit_warmup_steps=4 \
    trainer.method_args.adaptive_lr=false \
    trainer.method_args.eval_at_steps=[4,8,12,16,20] \
    trainer.args.num_train_epochs=5 \
    2>&1 | tail -80

echo "[$(date)] Phase 2 done"
if [[ -f "$OUTDIR/checkpoint-0/evals/TOFU_SUMMARY.json" ]]; then
    cat "$OUTDIR/checkpoint-0/evals/TOFU_SUMMARY.json"
fi
