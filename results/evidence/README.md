# MUSE Evaluation Evidence

Raw evaluation outputs backing the tables in `results/muse_news.md` and `results/muse_books.md`.

## Structure

```
evidence/
├── muse_news/
│   ├── eval_outputs/
│   │   ├── eval_results.json        # All 30 MUSE_SUMMARY results (6 methods × 5 seeds)
│   │   └── retrain_summary.json     # Gold (retrain) baseline eval
│   └── llm_judge/
│       ├── muse_llm_judge_news_5fold.csv  # 6 methods × 5 seeds (30 rows)
│       └── judge_run_log.txt         # Full run log with per-sample scores
├── muse_books/
│   ├── eval_outputs/
│   │   ├── eval_results.json        # All 30 MUSE_SUMMARY results (6 methods × 5 seeds)
│   │   └── retrain_summary.json     # Gold (retrain) baseline eval
│   └── llm_judge/
│       ├── muse_llm_judge_books_5fold.csv  # 6 methods × 5 seeds (30 rows)
│       └── judge_run_log.txt         # Full run log
├── tofu/                            # TOFU benchmark evidence (see tofu/README.md)
├── llm_judge_seed42.csv             # Original seed=42 judge results (both splits)
├── reproduce_eval.sh                # Script to reproduce any eval from checkpoint
└── regenerate_tables.py             # Regenerate markdown tables from evidence JSONs/CSVs
```

## Eval Metrics (MUSE_SUMMARY.json)

Each entry contains:
- `forget_knowmem_ROUGE`: Knowledge memorization on forget set (lower = better forgetting)
- `forget_verbmem_ROUGE`: Verbatim memorization on forget set (lower = better)
- `retain_knowmem_ROUGE`: Knowledge retention (higher = better)
- `extraction_strength`: Membership inference attack success (lower = better)
- `privleak`: Privacy leakage score

## LLM Judge Metrics (CSV)

Scores on 0-2 scale per sample (100 samples per split):
- `forget_leakage`: Mean of knowmem + verbmem leakage (lower = better)
- `forget_leakage_knowmem`: Knowledge leakage on forget set
- `forget_leakage_verbmem`: Verbatim leakage on forget set
- `retain_accuracy`: Factual accuracy on retain set (higher = better)
- `response_quality`: Overall response quality
- `forget_rq` / `retain_rq`: Response quality split by forget/retain

## Reproducing

```bash
# Run eval for a specific checkpoint
bash results/evidence/reproduce_eval.sh Books RMU 42

# Requires: model checkpoint + retrain eval in saves/
```

## Regenerating Tables

```bash
# Regenerate all 5-fold tables from evidence files
python results/evidence/regenerate_tables.py

# Output matches the tables in results/muse_news.md and results/muse_books.md
# This verifies the markdown tables are correctly derived from raw data
```

## Configuration

- Model: Llama-2-7b-hf
- Seeds: 42, 123, 456, 789, 1024
- LLM Judge: Claude 3.5 Haiku via AWS Bedrock (us.anthropic.claude-3-5-haiku-20241022-v1:0)
- Eval config: configs/eval/muse/default.yaml
