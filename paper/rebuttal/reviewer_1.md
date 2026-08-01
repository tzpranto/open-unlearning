We thank the reviewer for the time taken to review our manuscript and for the strengths and weaknesses identified.

---

## Response to Weakness 1 (W1)

Below, we first justify repair-first from an optimization point of view and explain how each of the major components complements this principle.

Forget and retain are entangled, and forget is destructive, and prior work has documented both properties, which we cite in §3.3. A bilevel program is inherently asymmetric: for every one outer step, the inner subproblem runs $K > 1$ steps, and in our formulation, the outer objective is additionally constrained on the inner objective's value (retain budget). Given that forgetting is destructive, placing it in the unconstrained inner loop would give $K$ unopposed destructive steps between every outer step, a trajectory that the constrained outer step cannot recover reliably. The foundation is therefore repair-first: retain runs unconstrained in the inner loop, so the model first stabilizes on the shared representation before any forget update is applied. In the ablation study (Table 3), swapping the two levels significantly degrades performance compared to BLADE (HM $0.823 \to 0.506$).

The outer forget objective is wrapped in an Augmented Lagrangian (Eq. 4) with asymmetric dual update of $\lambda$ so that forgetting does not undo the inner retain step, and the ablation confirms this ("ALM off": HM $= 0.233$, retain $= 0.092$). To enforce this cleanly, the outer forget loss itself must be bounded, self-stabilizing, and reference-free (§3.3, L. 264–267); clamped entropy is designed to satisfy exactly these properties. The ablation shows that other forget losses fail to achieve this: HM $= 0.160$ for GA, HM $= 0.009$ for NPO, HM $= 0.000$ for logit margin.

In the revised version, we will introduce the repair-first principle from the optimization point of view as well, where it is first stated (§3.3).

---

## Response to Weakness 2 (W2)

The submitted abstract itself does not use $\tau$ or $V$ symbols. The reviewer's underlying concern is nevertheless well-founded: $\tau \cdot \log V$ is introduced in the introduction at L. 65 with a forward-reference to §3.4. Prompted by the reviewer's observation, we re-audited the manuscript and identified several such instances. To resolve these issues, we will (i) introduce each definition at its point of first use, and (ii) add a "List of Symbols" at the start of the appendix showing symbols and their meanings.

---

## Response to Weakness 3 (W3)

Our technical contribution is threefold: (i) the repair-first bilevel formulation, (ii) the augmented Lagrangian with asymmetric $\lambda$ ratchet, and (iii) the clamped-entropy forget loss. These three mechanisms are highlighted in the manuscript's Introduction (P. 1, L. 61–77), each targeting a specific failure mode of prior unlearning methods. Our response to W1 above already discusses these three components; in addition to that, we would like to add the following.

**Repair-first bilevel formulation.** While bilevel and constrained formulations are not novel individually, the repair-first-guided formulation and the supporting mechanisms around it that we designed are novel for unlearning.

**Augmented Lagrangian with asymmetric $\lambda$ ratchet (§3.5, Eq. 6).** While standard ALM uses symmetric updates on an equality constraint, we enforce the constraint as an inequality on retain ($\mathcal{L}_\text{ret} \leq \varepsilon$) via a one-sided quadratic penalty that activates only when retain drifts past $\varepsilon$. Besides, once a retain violation elevates $\lambda$, it stays elevated and decays only slowly on the recovery side, so retain protection is not immediately released while forgetting resumes gradually. This produces the smooth three-phase training dynamics BLADE exhibits (warm-up, spike-and-ratchet, smooth convergence; §4.8). These enhancements are not incidental design choices but adaptations to complement the repair-first principle.

**Clamped-entropy forget loss.** As discussed in W1's response, we designed clamped entropy to satisfy bounded, self-stabilizing, and reference-free properties, and provided formal proofs in Appendix A.

We will swap §3.4 and §3.5 in the revised version so that the flow of information is more consistent with our responses for W1 and W3.

---

## Reproducibility

The anonymous repository (https://anonymous.4open.science/r/blade-5981/) contains all files and configurations needed to reproduce both BLADE and every baseline in the paper. We have updated the README so that the configuration files are easier to navigate and the environment setup and per-benchmark experiments can be run by following the README end-to-end.
