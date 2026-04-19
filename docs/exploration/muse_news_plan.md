# MUSE Exploration Plan (LoRA-BiAL)

Started: 2026-04-19

## Mission

Act as an autonomous researcher targeting EMNLP publication. Explore, innovate, and produce results that demonstrate genuine novelty — not incremental tweaks or copies from existing unlearning papers. Borrow ideas from other ML domains (optimization, continual learning, meta-learning, privacy, etc.) that haven't been applied to unlearning yet.

## Targets

- **MUSE Books first**, then News. Books has clearer signal (SimNPO already beats gold).
- Books: beat SimNPO HM=0.755, gold=0.739
- News: beat PDU HM=0.554, gold=0.660

## Our Core Method

Bilevel optimization with ALM constraint + implicit differentiation, operating on LoRA adapters. Inner loop protects retain. Outer loop forgets. ALM adapts the balance. Implicit corrects for inner loop response.

## What Must Work for the Paper

1. **Implicit differentiation** — differentiates us from naive bilevel. Debug and fix, don't drop.
2. **ALM constraint** — principled adaptive retain protection. Already functional. Tune per dataset.
3. **A principled forget loss** — current options may not be ideal. Open to redesign. Must be theoretically motivated. Consider ideas from other domains (information geometry, optimal transport, continual learning, etc.) that haven't appeared in unlearning.

## Operating Principles

- No shortcuts. No sloppy mistakes. Verify configs, batch sizes, and outputs before trusting results.
- Every experiment has a hypothesis. Log reasoning before running, results after.
- Use subagents for code review and result analysis — catch bugs before they waste GPU hours.
- These phases are guidelines, not gospel. Adapt based on what the results tell us.
- Don't copy methods from unlearning papers. Borrow from other domains if needed.

## Phases (Flexible)

### Phase 1: Validate on Books (~1h)
Run current LoRA-BiAL on Books. Establish baseline. Confirm pipeline.

### Phase 2: Fix Implicit Differentiation (~3h)
This is our novelty. Understand why FD-HVP hasn't helped. Instrument the code — log gradient norms, HVP norms, Neumann convergence. Try CG solver. Don't give up until we understand the root cause.

### Phase 3: Forget Loss Innovation (~3h)
Explore alternatives to NPO. Think outside unlearning literature. What loss captures "behave as if you never saw this data" without model collapse? Consider ideas from information theory, optimal transport, continual learning, or elsewhere.

### Phase 4: Transfer to News (~2h)
Best Books config on News. Adjust if needed — different domain, different balance.

### Phase 5: Final Sweep + Seeds (~3h)
Best config, 3 seeds, both datasets.

**Phases may be reordered, merged, or new phases added based on findings.**

---

## Results Log

| Date | ID | Dataset | Config | fk | vm | rk | extract | HM | Hypothesis & Notes |
|------|----|---------|--------|----|----|----|---------|----|-------------------|
| | | | | | | | | | |

## Decision Log

| Date | Decision | Reasoning |
|------|----------|-----------|
| | | |
