# Internal fixes list (camera-ready)

Working list of presentation issues we surfaced while drafting the response to Reviewer 8hx1's W2 but that we chose not to enumerate in the rebuttal itself. All of these should be applied at camera-ready time (in addition to the items already promised in `reviewer_8hx1.md` §W2). Locations are cited as (page, PDF line number) against the current submission PDF.

---

## Symbol collisions

- **α collision (LoRA scaling vs. asymmetric-decay coefficient).** α in Eq. 1 [p. 3, l. 196] denotes the LoRA scaling factor, but the same symbol is reused for the asymmetric dual-update decay coefficient in Eq. 6 / Algorithm 1 [p. 4, l. 311]. Fix: rename the dual-update coefficient to **α_dual** everywhere (algorithm, equation, hyperparameter table, appendix).
- **B collision (logit-magnitude bound vs. LoRA adapter matrix).** Assumption A2 in Proposition 1 [p. 11, ll. 808–809] uses B for the ‖z_t‖_∞ bound, colliding with the LoRA adapter matrix B ∈ ℝ^{d×r} in Eq. 1 [p. 3, l. 196]. Fix: rename the logit bound to **B_z** in Proposition 1 and its proof.
- **η_θ vs. η_out.** Appendix Table 4 [p. 12, l. 928] uses η_θ for the outer learning rate, but §3 (Algorithm 1 and p. 4, l. 237) calls the same quantity η_out. Fix: standardise on **η_out** everywhere.
- **T (safety cap) vs. T (sequence length in Appendix A.1).** Appendix A.1 [p. 11, around Eq. 8: L_fgt(θ) = (1/T)∑_{t=1}^{T} ℓ_t(θ)] uses T for the sequence length, while the main paper uses T for the outer-loop safety cap (Algorithm 1 REQUIRE list, Table on p. 12). Fix: rename the appendix's sequence length to **N** (matching the main text) or vice versa, and use one letter consistently.

## Metric / column naming inconsistencies

- **"ret" vs. "rk" in the ablation table.** The MUSE-Books ablation table [p. 7] uses column header "ret" for the retain-knowledge quantity that MUSE Table 2 [p. 6] and §B.2 [p. 12, l. 906] call "rk". Fix: unify to **rk**.
- **Subscript J on judge columns.** FL_J, RA_J, rRQ_J columns in the ablation table [p. 7] use a subscript J that is not explained in the caption; the unsubscripted FL, RA, rRQ are defined only in §B.2 [p. 12, ll. 916–919] and HM_J only at p. 13, l. 944. Fix: add a caption footnote "subscript J denotes LLM-judge scores" and cross-reference §B.2.
- **fgt_Acc, ret_Acc in KnowUnDo table.** Table 10 [p. 16] uses accuracy-based column headers fgt_Acc / ret_Acc, but §B.2 [p. 12, ll. 907–909] only defines fgt_R, ret_R, and MMLU. Fix: add definitions of the accuracy metrics to §B.2 (or drop the accuracy columns if redundant with ROUGE).
- **ES (Extraction Score) in the adversarial-robustness section.** ES [p. 14, ll. 1052–1053] is expanded in prose but the underlying metric is never defined (unlike PP = ParaProb and PtP = PertProb, which point at a probe procedure). Fix: add a one-line definition of ES with the source citation.

## Notes for the "List of Symbols" appendix

While assembling the symbol table, cross-check each of the following against its point-of-use, and make sure the table records both the meaning and the first-use location:

- θ, θ*, θ₀
- L_fgt, L_CE, L_ret, L_outer, L_forget (Eq. 5), L_ent, L_GA
- D_forget, D_retain, B_ret, B'_ret
- p_t, z_t, H_t, H_max
- τ, V, N, T (after disambiguation above)
- ε, ε_mul, ρ, λ, λ_0, α_dual
- η_in, η_out (after disambiguation above)
- K, r, d (LoRA rank / hidden dim — flag these as two separate d's if applicable)
- G (Lipschitz constant), B_z (logit magnitude bound)
- FL, RA, rRQ, HM, HM_J, MU, Prob, RG, fk, vm, rk, fgt_R, ret_R, MMLU, ES, PP, PtP

Anything not yet in this list should be swept out of the manuscript in a final pass.
