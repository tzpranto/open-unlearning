"""
LoRA-Implicit: LoRA Bilevel with FD-HVP Implicit Correction
============================================================

Clean bilevel unlearning with proper implicit differentiation.
LoRA makes FD-HVP tractable (~40M params, no create_graph needed).

Two bilevel modes:
  forget_outer: inner=retain CE, outer=forget+ALM  (standard)
  forget_inner: inner=forget,    outer=retain+ALM  (inverted)

Implicit correction accounts for inner loop's response to outer
update via Truncated Neumann series with finite-difference HVP.

Adaptive LR: reduces outer LR when retain quality degrades.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import logging
import os
import json
from typing import Optional

from trainer.unlearn.base import UnlearnTrainer

logger = logging.getLogger(__name__)


class LoRAImplicit(UnlearnTrainer):

    def __init__(
        self,
        *args,
        # LoRA
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.0,
        lora_target_modules: Optional[list] = None,
        # Bilevel structure
        bilevel_mode: str = "forget_outer",  # "forget_outer" or "forget_inner"
        T: int = -1,  # T<=0 = epoch-based, T>0 = fixed steps
        K: int = 3,   # inner steps per outer step
        eta_theta: float = 3e-5,   # outer LR
        eta_in: float = 2e-4,      # inner LR
        # ALM constraint (always on L_ret <= epsilon)
        epsilon: float = 0.70,
        rho: float = 0.1,
        lambda_init: float = 1.0,
        lambda_max: float = 0.0,
        # Forget loss
        forget_loss_type: str = "npo",
        npo_beta: float = 4.0,
        ga_clip: float = 1.0,
        # Implicit correction (FD-HVP + Truncated Neumann)
        use_implicit: bool = True,
        neumann_steps: int = 5,
        neumann_mu: float = 0.01,
        neumann_alpha_default: float = 0.1,
        neumann_alpha_min: float = 1e-6,
        neumann_alpha_max: float = 1.0,
        neumann_use_probe_alpha: bool = True,
        neumann_max_growth_ratio: float = 10.0,
        fd_hvp_eps: float = 0.01,
        implicit_warmup_steps: int = 20,  # skip implicit for first N steps (LoRA=0 at init)
        # Adaptive LR: reduce outer LR when retain degrades
        adaptive_lr: bool = True,
        adapt_ema_decay: float = 0.95,
        adapt_check_every: int = 20,
        adapt_degrad_threshold: float = 0.05,
        adapt_lr_factor: float = 0.5,
        adapt_lr_min: float = 1e-6,
        # LR schedule (applied on top of adaptive)
        lr_schedule: str = "constant",  # "constant" or "cosine"
        warmup_fraction: float = 0.0,
        # Layer control: freeze specific layers during inner/outer
        freeze_inner_layers: Optional[list] = None,  # e.g. [0,1,2,3]
        freeze_outer_layers: Optional[list] = None,
        # Post-unlearning retain recovery
        # After bilevel, run pure retain CE steps (no forget loss) to recover rk
        recovery_epochs: int = 0,      # 0 = disabled; >0 = extra retain-only epochs
        recovery_lr: float = 1e-5,     # LR for recovery (typically lower than eta_in)
        # Checkpointing
        checkpoint_every_epoch: bool = False,
        eval_at_steps: Optional[list] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # LoRA config
        self.lora_r = lora_r
        self.lora_alpha_val = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules or [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
        # Bilevel config
        self.bilevel_mode = bilevel_mode
        assert bilevel_mode in ("forget_outer", "forget_inner"), \
            f"bilevel_mode must be 'forget_outer' or 'forget_inner', got {bilevel_mode}"
        self.T = T
        self.K = K
        self.eta_theta = eta_theta
        self.eta_in = eta_in
        # ALM
        self.epsilon = epsilon
        self.rho = rho
        self.lambda_init = lambda_init
        self.lambda_max = lambda_max
        self.lambda_dual = float(lambda_init)
        # Forget loss
        self.forget_loss_type = forget_loss_type
        self.npo_beta = npo_beta
        self.ga_clip = ga_clip
        # Implicit
        self.use_implicit = use_implicit
        self.neumann_steps = neumann_steps
        self.neumann_mu = neumann_mu
        self.neumann_alpha_default = neumann_alpha_default
        self.neumann_alpha_min = neumann_alpha_min
        self.neumann_alpha_max = neumann_alpha_max
        self.neumann_use_probe_alpha = neumann_use_probe_alpha
        self.neumann_max_growth_ratio = neumann_max_growth_ratio
        self.fd_hvp_eps = fd_hvp_eps
        self.implicit_warmup_steps = implicit_warmup_steps
        # Adaptive LR
        self.adaptive_lr = adaptive_lr
        self.adapt_ema_decay = adapt_ema_decay
        self.adapt_check_every = adapt_check_every
        self.adapt_degrad_threshold = adapt_degrad_threshold
        self.adapt_lr_factor = adapt_lr_factor
        self.adapt_lr_min = adapt_lr_min
        # LR schedule
        self.lr_schedule = lr_schedule
        self.warmup_fraction = warmup_fraction
        # Layer control
        self.freeze_inner_layers = set(freeze_inner_layers) if freeze_inner_layers else set()
        self.freeze_outer_layers = set(freeze_outer_layers) if freeze_outer_layers else set()
        # Recovery
        self.recovery_epochs = recovery_epochs
        self.recovery_lr = recovery_lr
        # Checkpointing
        self.checkpoint_every_epoch = checkpoint_every_epoch
        self.eval_at_steps = set(eval_at_steps) if eval_at_steps else set()

    # ------------------------------------------------------------------
    # LoRA setup
    # ------------------------------------------------------------------
    def _wrap_with_lora(self):
        from peft import get_peft_model, LoraConfig, TaskType
        config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha_val,
            target_modules=self.lora_target_modules,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.model = get_peft_model(self.model, config)
        self.model.print_trainable_parameters()
        logger.info(f"LoRA: r={self.lora_r}, alpha={self.lora_alpha_val}, "
                     f"modules={self.lora_target_modules}")

    # ------------------------------------------------------------------
    # Loss functions
    # ------------------------------------------------------------------
    def _compute_ce_loss(self, batch, device):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch.get("labels", input_ids).to(device)
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        return outputs.loss

    def _compute_npo_loss(self, batch, device):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch.get("labels", input_ids).to(device)
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        shifted_labels = labels[..., 1:].contiguous()
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")

        # Ref = base model (LoRA disabled)
        self.model.disable_adapter_layers()
        with torch.no_grad():
            ref_logits = self.model(**inputs).logits[..., :-1, :].contiguous()
            ref_nll = loss_fn(ref_logits.transpose(-1, -2), shifted_labels).sum(dim=-1)

        self.model.enable_adapter_layers()
        model_logits = self.model(**inputs).logits[..., :-1, :].contiguous()
        model_nll = loss_fn(model_logits.transpose(-1, -2), shifted_labels).sum(dim=-1)

        log_ratio = -(model_nll - ref_nll)
        loss = -2 / self.npo_beta * F.logsigmoid(self.npo_beta * (-log_ratio)).mean()
        return loss

    def _compute_logit_margin_loss(self, batch, device):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        margins = logits.max(dim=-1)[0] - logits.mean(dim=-1)
        return margins.mean()

    def _compute_forget_loss(self, batch, device):
        if self.forget_loss_type == "npo":
            return self._compute_npo_loss(batch, device)
        elif self.forget_loss_type == "ga":
            return -self._compute_ce_loss(batch, device)
        elif self.forget_loss_type == "logit_margin":
            return self._compute_logit_margin_loss(batch, device)
        else:
            raise ValueError(f"Unknown forget_loss_type: {self.forget_loss_type}")

    # ------------------------------------------------------------------
    # FD-HVP + Truncated Neumann
    # ------------------------------------------------------------------
    def _unflatten_params(self, flat_vec, params_list):
        out, offset = [], 0
        for p in params_list:
            n = p.numel()
            out.append(flat_vec[offset:offset + n].reshape(p.shape))
            offset += n
        return out

    def _compute_hvp_fd(self, loss_fn, params, v):
        """H*v via central finite differences. Works with flash attention."""
        eps = self.fd_hvp_eps
        v_norm = v.norm().clamp(min=1e-12)
        v_unit = v / v_norm
        v_list = self._unflatten_params(v_unit, params)

        # +eps perturbation
        with torch.no_grad():
            for p, dv in zip(params, v_list):
                p.data.add_(eps * dv.to(p.dtype))
        loss_p = loss_fn()
        grads_p = torch.autograd.grad(loss_p, params, allow_unused=True)
        grads_p = [g.detach().clone() if g is not None else torch.zeros_like(p)
                   for g, p in zip(grads_p, params)]
        del loss_p

        # -eps perturbation (from +eps, subtract 2*eps)
        with torch.no_grad():
            for p, dv in zip(params, v_list):
                p.data.sub_(2.0 * eps * dv.to(p.dtype))
        loss_m = loss_fn()
        grads_m = torch.autograd.grad(loss_m, params, allow_unused=True)
        grads_m = [g.detach().clone() if g is not None else torch.zeros_like(p)
                   for g, p in zip(grads_m, params)]
        del loss_m

        # Restore
        with torch.no_grad():
            for p, dv in zip(params, v_list):
                p.data.add_(eps * dv.to(p.dtype))

        hvp_list = [(gp - gm) / (2.0 * eps) for gp, gm in zip(grads_p, grads_m)]
        return torch.cat([h.reshape(-1) for h in hvp_list]) * v_norm.item()

    def _implicit_correction(self, params, v, inner_loss_fn, outer_loss_fn):
        """Implicit gradient correction via CG on (H_inner + μI)h = v.

        Conjugate gradient is naturally stable — no step-size tuning,
        no divergence risk. Minimizes ||Ah - v|| in Krylov subspace.
        Each CG iteration costs 1 HVP (2 fwd+bwd).

        After solving h ≈ (H_inner + μI)^{-1} v:
          g_corr = v - H_outer(h)
        """
        if not torch.isfinite(v).all():
            return v, "fallback_nonfinite_v"

        v_norm = v.norm().clamp(min=1e-12).item()

        def A(x):
            """(H_inner + μI) x"""
            return self._compute_hvp_fd(inner_loss_fn, params, x) + self.neumann_mu * x

        # CG solve: Ah = v
        h = torch.zeros_like(v)
        r = v.clone()         # residual = v - A(h) = v (since h=0)
        p = r.clone()         # search direction
        rs_old = r.dot(r)

        for i in range(self.neumann_steps):
            Ap = A(p)
            if not torch.isfinite(Ap).all():
                logger.warning(f"CG iter {i}: non-finite HVP, stopping early")
                break

            pAp = p.dot(Ap)
            if pAp <= 0:
                logger.debug(f"CG iter {i}: non-positive pAp={pAp.item():.4e}, Hessian indefinite — stopping")
                break

            alpha = rs_old / pAp
            h = h + alpha * p
            r = r - alpha * Ap

            rs_new = r.dot(r)
            if rs_new.sqrt().item() < 1e-6 * v_norm:
                logger.debug(f"CG converged at iter {i}, residual={rs_new.sqrt().item():.6f}")
                break

            p = r + (rs_new / rs_old) * p
            rs_old = rs_new

            # Safety: if h grows too large, something is wrong
            h_norm = h.norm().item()
            if not np.isfinite(h_norm) or h_norm > 1e6:
                logger.warning(f"CG iter {i}: h exploded ({h_norm:.1f}), fallback")
                return v, "fallback_cg_exploded"

        # Outer HVP: g_corr = v - H_outer(h)
        c = self._compute_hvp_fd(outer_loss_fn, params, h)
        if not torch.isfinite(c).all():
            return v, "fallback_nonfinite_c"

        g_corr = v - c
        g_norm = g_corr.norm().item()

        # Clip if correction is too large
        if g_norm > self.neumann_max_growth_ratio * v_norm:
            scale = self.neumann_max_growth_ratio * v_norm / max(g_norm, 1e-12)
            g_corr = scale * g_corr
            return g_corr, f"clipped({scale:.3f})"

        return g_corr, "ok"

    # ------------------------------------------------------------------
    # Layer freeze helper
    # ------------------------------------------------------------------
    def _zero_layer_grads(self, layer_indices):
        """Zero gradients for specified transformer layer indices."""
        if not layer_indices:
            return
        for name, param in self.model.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue
            for idx in layer_indices:
                if f".layers.{idx}." in name:
                    param.grad.zero_()
                    break

    # ------------------------------------------------------------------
    # Inner and outer steps (dispatch by bilevel_mode)
    # ------------------------------------------------------------------
    def inner_step(self, forget_batch, retain_batch, device):
        """Inner loop step. Returns loss value."""
        self.model.enable_adapter_layers()
        self.model.train()
        self._inner_opt.zero_grad()

        if self.bilevel_mode == "forget_outer":
            loss = self._compute_ce_loss(retain_batch, device)
        else:  # forget_inner
            loss = self._compute_forget_loss(forget_batch, device)

        loss.backward()
        self._zero_layer_grads(self.freeze_inner_layers)
        self._inner_opt.step()
        return loss.item()

    def outer_step(self, forget_batch, retain_batch, device, global_step=0):
        """Outer step with ALM and optional implicit correction."""
        self.model.enable_adapter_layers()
        self.model.train()
        self._outer_opt.zero_grad()

        # Compute both losses
        L_fgt = self._compute_forget_loss(forget_batch, device)
        L_ret = self._compute_ce_loss(retain_batch, device)

        # ALM on retain: r = max(0, L_ret - epsilon)
        r_val = max(0.0, L_ret.item() - self.epsilon)
        alm_coeff = self.lambda_dual + self.rho * r_val

        # Compose outer loss based on mode
        if self.bilevel_mode == "forget_outer":
            # Outer = forget + ALM * retain
            L_outer = L_fgt + alm_coeff * L_ret
        else:  # forget_inner
            # Outer = retain + ALM * retain (retain-focused, ALM amplifies when degraded)
            L_outer = L_ret + alm_coeff * L_ret

        L_outer.backward()
        self._zero_layer_grads(self.freeze_outer_layers)

        # Clip for GA
        if self.forget_loss_type == "ga" and self.ga_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.ga_clip,
            )

        # Implicit correction (skip during warmup — LoRA=0 at init makes Hessian degenerate)
        implicit_status = "off"
        if self.use_implicit and global_step >= self.implicit_warmup_steps:
            lora_params = [p for p in self.model.parameters()
                          if p.requires_grad and p.grad is not None]
            v = torch.cat([p.grad.reshape(-1) for p in lora_params])
            torch.cuda.empty_cache()

            # Loss closures for FD-HVP — must match inner/outer assignment
            if self.bilevel_mode == "forget_outer":
                def _inner_loss_fn():
                    return self._compute_ce_loss(retain_batch, device)
                def _outer_loss_fn():
                    lf = self._compute_forget_loss(forget_batch, device)
                    lr = self._compute_ce_loss(retain_batch, device)
                    r = max(0.0, lr.item() - self.epsilon)
                    c = self.lambda_dual + self.rho * r
                    return lf + c * lr
            else:  # forget_inner
                def _inner_loss_fn():
                    return self._compute_forget_loss(forget_batch, device)
                def _outer_loss_fn():
                    lr = self._compute_ce_loss(retain_batch, device)
                    r = max(0.0, lr.item() - self.epsilon)
                    c = self.lambda_dual + self.rho * r
                    return lr + c * lr

            g_corr, implicit_status = self._implicit_correction(
                lora_params, v, _inner_loss_fn, _outer_loss_fn
            )

            # Write corrected gradient back
            offset = 0
            for p in lora_params:
                n = p.numel()
                p.grad = g_corr[offset:offset + n].reshape(p.shape).to(p.dtype)
                offset += n

        self._outer_opt.step()

        # Dual variable update
        self.lambda_dual = max(0.0, self.lambda_dual + self.rho * r_val)
        if self.lambda_max > 0:
            self.lambda_dual = min(self.lambda_dual, self.lambda_max)

        return L_fgt.item(), L_ret.item(), r_val, implicit_status

    # ------------------------------------------------------------------
    # Adaptive LR
    # ------------------------------------------------------------------
    def _adapt_lr_check(self, global_step, L_ret):
        """Update EMA and reduce outer LR if retain is degrading."""
        if not self.adaptive_lr:
            return

        if not hasattr(self, '_adapt_ema'):
            self._adapt_ema = L_ret
            self._adapt_prev_ema = L_ret
            self._adapt_lr_reductions = 0
            return

        d = self.adapt_ema_decay
        self._adapt_ema = d * self._adapt_ema + (1 - d) * L_ret

        if global_step > 0 and global_step % self.adapt_check_every == 0:
            if self._adapt_ema > self._adapt_prev_ema * (1 + self.adapt_degrad_threshold):
                # Retain is degrading — reduce outer LR
                for pg in self._outer_opt.param_groups:
                    old_lr = pg['lr']
                    new_lr = max(old_lr * self.adapt_lr_factor, self.adapt_lr_min)
                    pg['lr'] = new_lr
                self._adapt_lr_reductions += 1
                logger.info(f"  [AdaptLR] L_ret EMA {self._adapt_ema:.4f} > "
                            f"prev {self._adapt_prev_ema:.4f} * {1+self.adapt_degrad_threshold:.2f} "
                            f"-> outer LR {old_lr:.2e} -> {new_lr:.2e} "
                            f"(reduction #{self._adapt_lr_reductions})")
            self._adapt_prev_ema = self._adapt_ema

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------
    def train(self):
        device = self.args.device

        # Stage 1: LoRA
        logger.info("=" * 60)
        logger.info("LoRA-Implicit: Bilevel with FD-HVP Correction")
        logger.info("=" * 60)
        self._wrap_with_lora()

        # Optimizers
        lora_params = [p for p in self.model.parameters() if p.requires_grad]
        self._inner_opt = torch.optim.Adam(lora_params, lr=self.eta_in)
        self._outer_opt = torch.optim.Adam(lora_params, lr=self.eta_theta)
        n_params = sum(p.numel() for p in lora_params)
        logger.info(f"Optimizers: inner lr={self.eta_in}, outer lr={self.eta_theta}, "
                     f"params={n_params:,}")

        # Gradient checkpointing
        if getattr(self.args, "gradient_checkpointing", False):
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        # Data
        train_dataloader = self.get_train_dataloader()
        batch_size = self.args.per_device_train_batch_size
        steps_per_epoch = len(train_dataloader)
        num_epochs = max(1, int(self.args.num_train_epochs))

        if self.T > 0:
            max_steps = self.T
        else:
            max_steps = num_epochs * steps_per_epoch

        # LR schedulers (on top of adaptive)
        if self.lr_schedule == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
            warmup_steps = int(self.warmup_fraction * max_steps)
            if warmup_steps > 0:
                wo = LinearLR(self._outer_opt, start_factor=0.1, total_iters=warmup_steps)
                co = CosineAnnealingLR(self._outer_opt, T_max=max_steps - warmup_steps)
                self._outer_sched = SequentialLR(self._outer_opt, [wo, co], milestones=[warmup_steps])
                wi = LinearLR(self._inner_opt, start_factor=0.1, total_iters=warmup_steps)
                ci = CosineAnnealingLR(self._inner_opt, T_max=max_steps - warmup_steps)
                self._inner_sched = SequentialLR(self._inner_opt, [wi, ci], milestones=[warmup_steps])
            else:
                self._outer_sched = CosineAnnealingLR(self._outer_opt, T_max=max_steps)
                self._inner_sched = CosineAnnealingLR(self._inner_opt, T_max=max_steps)
        else:
            self._outer_sched = None
            self._inner_sched = None

        # Logging
        mode_desc = ("inner=retain, outer=forget+ALM" if self.bilevel_mode == "forget_outer"
                     else "inner=forget, outer=retain+ALM")
        logger.info(f"  Mode: {self.bilevel_mode} ({mode_desc})")
        logger.info(f"  Epochs={num_epochs}, steps/epoch={steps_per_epoch}, "
                     f"max_steps={max_steps}, K={self.K}")
        logger.info(f"  Batch={batch_size}, forget_loss={self.forget_loss_type}"
                     f" (beta={self.npo_beta})")
        logger.info(f"  ALM: eps={self.epsilon}, rho={self.rho}, lambda={self.lambda_init}")
        logger.info(f"  Implicit: {'ON' if self.use_implicit else 'OFF'}"
                     f" (neumann={self.neumann_steps}, mu={self.neumann_mu}, "
                     f"fd_eps={self.fd_hvp_eps})")
        logger.info(f"  Adaptive LR: {'ON' if self.adaptive_lr else 'OFF'}"
                     f" (check_every={self.adapt_check_every}, "
                     f"threshold={self.adapt_degrad_threshold})")
        if self.freeze_inner_layers:
            logger.info(f"  Freeze inner layers: {sorted(self.freeze_inner_layers)}")
        if self.freeze_outer_layers:
            logger.info(f"  Freeze outer layers: {sorted(self.freeze_outer_layers)}")
        logger.info(f"  LR schedule: {self.lr_schedule}, warmup={self.warmup_fraction}")

        if torch.cuda.is_available():
            logger.info(f"  GPU memory: {torch.cuda.memory_allocated()/1e9:.1f} GB")

        # Training
        history = []
        global_step = 0

        for epoch in range(num_epochs):
            epoch_start = time.time()
            epoch_fgt, epoch_ret = [], []

            for batch_idx, combined_batch in enumerate(train_dataloader):
                if global_step >= max_steps:
                    break

                t0 = time.time()
                forget_batch = combined_batch["forget"]
                retain_batch = combined_batch["retain"]

                # Inner loop: K steps
                inner_losses = []
                for _ in range(self.K):
                    l_in = self.inner_step(forget_batch, retain_batch, device)
                    inner_losses.append(l_in)

                # Outer step with implicit correction
                L_fgt, L_ret, r_val, imp_status = self.outer_step(
                    forget_batch, retain_batch, device, global_step=global_step
                )

                # LR scheduling
                if self._outer_sched is not None:
                    self._outer_sched.step()
                    self._inner_sched.step()

                # Adaptive LR
                self._adapt_lr_check(global_step, L_ret)

                dt = time.time() - t0

                step_info = {
                    "step": global_step, "epoch": epoch,
                    "L_fgt": L_fgt, "L_ret": L_ret, "r": r_val,
                    "lambda": self.lambda_dual,
                    "inner_mean": sum(inner_losses) / len(inner_losses) if inner_losses else 0.0,
                    "implicit": imp_status, "dt": dt,
                }
                history.append(step_info)
                epoch_fgt.append(L_fgt)
                epoch_ret.append(L_ret)

                # Periodic logging (~10 per epoch)
                log_every = max(1, steps_per_epoch // 10)
                if global_step % log_every == 0 or global_step == max_steps - 1:
                    olr = self._outer_opt.param_groups[0]['lr']
                    imp_tag = f" imp={imp_status}" if self.use_implicit else ""
                    logger.info(
                        f"  [{global_step:4d}/{max_steps}|e{epoch+1}] "
                        f"L_fgt={L_fgt:.4f} L_ret={L_ret:.4f} "
                        f"r={r_val:+.4f} lam={self.lambda_dual:.3f} "
                        f"inner={step_info['inner_mean']:.4f} "
                        f"olr={olr:.2e}{imp_tag} dt={dt:.1f}s"
                    )

                global_step += 1

                # Intermediate checkpoint
                if global_step in self.eval_at_steps:
                    self._save_checkpoint(global_step, history)

                # Collapse detection
                if L_ret > 10.0:
                    logger.warning(f"  L_ret={L_ret:.1f} > 10 — collapsed, stopping.")
                    break

            # Epoch summary
            if epoch_fgt:
                logger.info(
                    f"  Epoch {epoch+1}/{num_epochs}: {batch_idx+1} steps, "
                    f"mean_fgt={sum(epoch_fgt)/len(epoch_fgt):.4f}, "
                    f"mean_ret={sum(epoch_ret)/len(epoch_ret):.4f}, "
                    f"dt={time.time()-epoch_start:.0f}s"
                )

            if self.checkpoint_every_epoch and epoch < num_epochs - 1 and epoch_fgt:
                ckpt_dir = os.path.join(self.args.output_dir, f"checkpoint-epoch{epoch+1}")
                os.makedirs(ckpt_dir, exist_ok=True)
                self.model.save_pretrained(ckpt_dir)
                logger.info(f"  Checkpoint: {ckpt_dir}")

            if global_step >= max_steps or L_ret > 10.0:
                break

        # ============================================================
        # Post-unlearning retain recovery
        # Pure retain CE — no forget loss, no outer loop.
        # LoRA has learned a "forget direction"; retain CE recovers rk
        # without fully restoring fk because the forget signal is gone.
        # ============================================================
        if self.recovery_epochs > 0:
            logger.info("=" * 60)
            logger.info(f"Retain recovery: {self.recovery_epochs} epochs, "
                         f"lr={self.recovery_lr}")
            logger.info("=" * 60)

            # Fresh optimizer at recovery LR
            recovery_opt = torch.optim.Adam(
                [p for p in self.model.parameters() if p.requires_grad],
                lr=self.recovery_lr,
            )
            self.model.enable_adapter_layers()
            self.model.train()

            for rec_epoch in range(self.recovery_epochs):
                rec_start = time.time()
                rec_losses = []
                for rec_idx, combined_batch in enumerate(train_dataloader):
                    retain_batch = combined_batch["retain"]
                    recovery_opt.zero_grad()
                    loss = self._compute_ce_loss(retain_batch, device)
                    loss.backward()
                    recovery_opt.step()
                    rec_losses.append(loss.item())

                mean_loss = sum(rec_losses) / len(rec_losses)
                logger.info(
                    f"  Recovery epoch {rec_epoch+1}/{self.recovery_epochs}: "
                    f"mean_ret={mean_loss:.4f}, dt={time.time()-rec_start:.0f}s"
                )
                history.append({
                    "step": global_step, "epoch": num_epochs + rec_epoch + 1,
                    "L_fgt": 0.0, "L_ret": mean_loss, "r": 0.0,
                    "lambda": self.lambda_dual, "inner_mean": mean_loss,
                    "implicit": "recovery", "dt": time.time() - rec_start,
                })

        # Merge LoRA and save
        logger.info("Merging LoRA into base model...")
        self.model = self.model.merge_and_unload()
        os.makedirs(self.args.output_dir, exist_ok=True)
        self.model.save_pretrained(self.args.output_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(self.args.output_dir)

        with open(os.path.join(self.args.output_dir, "training_history.json"), "w") as f:
            json.dump(history, f, indent=2)

        logger.info(f"Saved to {self.args.output_dir}")
        if history:
            logger.info(f"Final: lam={self.lambda_dual:.3f}, "
                         f"L_fgt={history[-1]['L_fgt']:.4f}, "
                         f"L_ret={history[-1]['L_ret']:.4f}")

        self.evaluate()

    def _save_checkpoint(self, step, history):
        """Save intermediate merged checkpoint."""
        ckpt_dir = os.path.join(self.args.output_dir, f"step-{step}")
        os.makedirs(ckpt_dir, exist_ok=True)
        logger.info(f"  Saving checkpoint at step {step}...")
        self.model.eval()
        self.model.merge_adapter()
        peft_sd = self.model.base_model.model.state_dict()
        clean_sd = {k.replace(".base_layer", ""): v
                    for k, v in peft_sd.items() if "lora_" not in k}
        self.model.base_model.model.save_pretrained(ckpt_dir, state_dict=clean_sd)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(ckpt_dir)
        self.model.unmerge_adapter()
        self.model.train()
        with open(os.path.join(ckpt_dir, "training_history.json"), "w") as f:
            json.dump(history, f, indent=2)
