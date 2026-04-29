#!/bin/bash
# Clean up PDU epoch sweep checkpoints after evals are done
# Keeps only the evals/ subfolder in each checkpoint dir
set -euo pipefail
cd /datadrive/forked/open-unlearning

OUTDIR="saves/unlearn/muse_Llama-2-7b-hf_Books_PDU_epoch_sweep_s42"

echo "PDU checkpoint cleanup"
echo "======================"

for ckpt_dir in "$OUTDIR"/checkpoint-*; do
    [ -d "$ckpt_dir" ] || continue
    ckpt_name=$(basename "$ckpt_dir")

    if [[ ! -f "${ckpt_dir}/evals/MUSE_SUMMARY.json" ]]; then
        echo "[SKIP] ${ckpt_name} — not yet evaluated, keeping"
        continue
    fi

    # Remove model weights, keep evals/
    before=$(du -sh "$ckpt_dir" | cut -f1)
    find "$ckpt_dir" -maxdepth 1 -type f -delete
    after=$(du -sh "$ckpt_dir" | cut -f1)
    echo "[CLEANED] ${ckpt_name}: ${before} → ${after}"
done

# Also clean the final model weights in the main output dir
for f in "$OUTDIR"/model*.safetensors "$OUTDIR"/model.safetensors.index.json \
         "$OUTDIR"/pytorch_model* "$OUTDIR"/config.json "$OUTDIR"/generation_config* \
         "$OUTDIR"/tokenizer* "$OUTDIR"/special_tokens* "$OUTDIR"/added_tokens* \
         "$OUTDIR"/optimizer* "$OUTDIR"/scheduler* "$OUTDIR"/training_args*; do
    [ -f "$f" ] && rm -f "$f" && echo "[RM] $(basename $f)"
done
find "$OUTDIR" -maxdepth 1 -name "*.safetensors" -delete 2>/dev/null
find "$OUTDIR" -maxdepth 1 -name "*.bin" -delete 2>/dev/null

echo ""
echo "Done. Remaining:"
du -sh "$OUTDIR"
