# SIBL Memory Utilization Plan (1× A100)

**Goal:** Run SIBL on a single A100 without OOM, while keeping the run valid for testing.  
**Context:** NPO runs; SIBL previously OOM’d. This plan is based on the current codebase (no code changes applied yet).

---

## 1. Why NPO Fits and SIBL OOMs

| Aspect | NPO | SIBL |
|--------|-----|------|
| **Training loop** | Standard Trainer loop (`_inner_training_loop`): one forward/backward per batch, gradient accumulation. | **Custom** `train()`: inner loop (K steps on retain) + outer step (forget + retain, AL + optional implicit). |
| **Gradient checkpointing** | **Enabled** in `_inner_training_loop` before training (script passes `gradient_checkpointing=true`). | **Never enabled**: SIBL overrides `train()` and never calls `_inner_training_loop`, so `model.gradient_checkpointing_enable()` is never run. |
| **Extra model in memory** | One **reference model** (frozen copy) for DPO loss. | No ref model for default `logit_margin`; optional ref only if `forget_loss_type=npo`. |
| **Peak memory pattern** | One forward + backward, ref_model forward (no grad). | **Inner loop:** K forwards/backwards on retain (grads cleared each step). **Outer step:** forget forward + retain forward + AL backward; if `use_implicit=True`: **second-order graph** (L_inner with `create_graph=True`) + **HVP** (Hessian–vector product, another `create_graph=True`), plus full gradient copy `g_alm_dict`. |
| **Batch size** | Script: e.g. 8 per device (tofu uses 12). Muse script overrides to 8. | SIBL config defaults to **2**; script **overrides to 8** for muse_baselines.sh. So SIBL runs with **8** per device. |
| **Sequence length** | Same data. | Same: MUSE **max_length 2048** (MUSE_forget / MUSE_retain). |

**Main memory drivers for SIBL:**

1. **Gradient checkpointing not applied** → full activation storage in inner/outer steps.  
2. **Large effective batch** (8) with long sequences (2048) → big activation and gradient footprint.  
3. **Implicit differentiation** (when `use_implicit=True`): retain graph + L_inner with `create_graph=True` + multiple HVP with `create_graph=True` → very large computation graph.  
4. **Full gradient clone** in `outer_step`: `g_alm_dict` holds a copy of gradients for all parameters.  
5. **mask_dict**: one mask per parameter (same count as params; usually acceptable).

---

## 2. Levers to Reduce Memory (in order of impact)

### 2.1 Enable gradient checkpointing for SIBL (code)

- **Where:** SIBL uses a custom `train()` and never enters `_inner_training_loop`, so the Trainer’s `gradient_checkpointing_enable()` is never called.
- **Action:** Before the custom loop in `SIBL.train()`, if `self.args.gradient_checkpointing` is True, call `self.model.gradient_checkpointing_enable(...)` (same pattern as in `base.py`).
- **Effect:** Large reduction in activation memory in both inner and outer steps (recompute in backward instead of storing).

### 2.2 Lower per-device batch size for SIBL (config / script)

- **Current:** Script overrides to `per_device_train_batch_size=8` (muse_baselines.sh) and SIBL config has 2.
- **Action:** For 1 GPU + 7B + 2048 seq, use **2** (or at most 4) for SIBL. Either:
  - In **muse_baselines.sh**: pass a SIBL-specific batch size (e.g. 2) when trainer is SIBL, or  
  - In **configs/experiment/unlearn/muse/sibl.yaml**: set `trainer.args.per_device_train_batch_size: 2` and do **not** override it in the script for SIBL (or override to 2).
- **Effect:** Linear reduction in activation and gradient memory per step.

### 2.3 Keep implicit differentiation off (config)

- **Current:** `configs/experiment/unlearn/muse/sibl.yaml` already has `use_implicit: false`.
- **Action:** Ensure it stays **false** for 1-GPU runs (no code change; just don’t turn it on).
- **Effect:** Avoids second-order graph and HVP; large memory save.

### 2.4 Reduce inner steps K (config)

- **Current:** K=10 in muse/sibl.yaml; SIBL default T=20, K=10.
- **Action:** Try **K=4 or 5** for testing to reduce time spent in inner loop (each step still does one forward/backward on retain).
- **Effect:** Slightly lower peak memory in inner loop and faster iteration; may slightly affect convergence (tune if needed).

### 2.5 Reduce outer iterations T (config)

- **Current:** T=10 in muse/sibl.yaml.
- **Action:** Only reduce T if still OOM after the above (e.g. T=5 for a quick test).
- **Effect:** Fewer outer steps; mainly for fitting, not the first lever.

### 2.6 Reduce sequence length for training (config / data)

- **Current:** MUSE_forget / MUSE_retain use **max_length: 2048** (data config).
- **Action:** Add or override a smaller **max_length** for the **training** datasets only (e.g. 1024 or 512) if the benchmark allows. Leave eval datasets at 2048 if required by the benchmark.
- **Effect:** Large reduction in activation memory (roughly linear in seq length for attention).

### 2.7 DeepSpeed / ZeRO (already in use)

- **Current:** `single_gpu_config.yaml` uses DeepSpeed ZeRO-3, no CPU offload.
- **Optional:** If still OOM, consider enabling **optimizer** or **param** CPU offload in `configs/accelerate/zero_stage3_offload_config.json` (trade memory for speed).
- **Effect:** Frees GPU memory at the cost of slower steps.

### 2.8 Avoid ref model for SIBL (config)

- **Current:** Default SIBL forget loss is `logit_margin` (no ref model). If you switch to `forget_loss_type: npo`, a second model is created.
- **Action:** For 1 GPU, keep `forget_loss_type: logit_margin` (or `simnpo` / `grad_ascent`) so no ref model is ever allocated.
- **Effect:** Saves one full model copy if you were to use NPO-style loss inside SIBL.

---

## 3. Recommended order of operations (no code yet)

1. **Enable gradient checkpointing in SIBL** (code: call `gradient_checkpointing_enable()` at the start of `SIBL.train()` when args say so).  
2. **Use batch size 2 for SIBL on 1 GPU** (script or experiment config: do not override to 8 for SIBL; keep 2 or set 2 explicitly).  
3. **Confirm `use_implicit: false`** in the experiment config.  
4. **Optionally reduce K** (e.g. to 4–5) in `configs/experiment/unlearn/muse/sibl.yaml` for testing.  
5. If still OOM: **lower training max_length** (e.g. 1024) for MUSE train splits only, then re-run.  
6. If still OOM: **enable DeepSpeed CPU offload** (optimizer or params) for the single-GPU config.

---

## 4. Files to touch (when implementing)

| Change | File(s) |
|--------|--------|
| Enable gradient checkpointing for SIBL | `src/trainer/unlearn/sibl.py` (start of `train()`) |
| SIBL batch size 2 on 1 GPU | `scripts/muse_baselines.sh` and/or `configs/experiment/unlearn/muse/sibl.yaml` |
| use_implicit / K / T | `configs/experiment/unlearn/muse/sibl.yaml` |
| Training max_length | Data config used by unlearn (e.g. MUSE_forget / MUSE_retain or experiment overrides) |
| DeepSpeed offload | `configs/accelerate/zero_stage3_offload_config.json` |

---

## 5. Quick reference: SIBL vs NPO code paths

- **NPO:** `NPO.compute_loss()` used inside Trainer’s training step → `_inner_training_loop` runs → gradient checkpointing enabled → standard backward.
- **SIBL:** `SIBL.train()` overrides and runs its own loop → `inner_loop()` (K× `inner_step`) and `outer_step()` → **no** call to `_inner_training_loop` → gradient checkpointing never enabled unless we add it in `SIBL.train()`.

This plan is intended to be implemented in the order above; no code has been changed yet.
