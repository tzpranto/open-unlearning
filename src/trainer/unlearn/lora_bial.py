"""
LoRA-BiAL: LoRA-based Bilevel Augmented Lagrangian for Unlearning
=================================================================

Approach:
1. (Optional) PerTA weight surgery on base model
2. LoRA bilevel optimization for unlearning

Key innovations vs standard bilevel (SIBL):
- Zero-overhead ref model: ref = base model with LoRA adapters disabled
- Naturally constrained update space: LoRA limits collateral damage
- Solves OOM: ~55MB LoRA gradients vs ~14GB full model gradients
- FD-HVP implicit correction: flash-attention-compatible bilevel differentiation

The bilevel dynamics:
- Inner loop: LoRA learns retain CE → some forget knowledge spillover
- Outer loop: NPO/GA detects spillover, corrects LoRA to reduce forget probs
- ALM constraint: prevents over-correction
- Implicit correction: accounts for inner loop's response to outer update
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


class LoRABiAL(UnlearnTrainer):
    """LoRA Bilevel Augmented Lagrangian for Unlearning.

    Stage 1: Apply PerTA weight surgery (if perta_lambda > 0)
    Stage 2: Wrap model with LoRA adapters
    Stage 3: Bilevel optimization — inner CE on retain, outer NPO/GA on forget
    """

    def __init__(
        self,
        *args,
        # PerTA parameters (stage 1)
        perta_lambda: float = 3.5,
        perta_alpha: float = 1.0,
        fisher_cache_path: str = "saves/unlearn/_perta_fisher_cache_News_n64.pt",
        pretrained_model_name: str = "meta-llama/Llama-2-7b-hf",
        # LoRA parameters (stage 2)
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.0,
        lora_target_modules: Optional[list] = None,
        # Bilevel parameters (stage 3)
        T: int = 25,
        K: int = 5,
        eta_theta: float = 1e-4,
        eta_in: float = 2e-4,
        # ALM parameters
        epsilon: float = 0.70,
        rho: float = 0.1,
        lambda_init: float = 1.0,
        lambda_max: float = 0.0,  # 0 = no cap; >0 = cap dual variable
        # Forget loss
        forget_loss_type: str = "npo",  # "npo" or "ga"
        npo_beta: float = 2.0,
        ga_clip: float = 1.0,
        # Epoch-aware training (new)
        lr_schedule: str = "constant",  # "constant" or "cosine"
        warmup_fraction: float = 0.0,  # fraction of total steps for warmup
        npo_saturation_threshold: float = 0.01,
        saturation_patience: int = 5,
        retain_only_after_saturation: bool = False,
        checkpoint_every_epoch: bool = False,
        # Intermediate checkpoints for dynamics analysis
        # Saves merged model at these steps; eval them after training
        eval_at_steps: Optional[list] = None,  # e.g. [25, 50, 100, 200]
        # Implicit differentiation (FD-HVP)
        use_implicit: bool = False,
        neumann_steps: int = 5,       # J: truncated Neumann series terms
        neumann_mu: float = 0.01,     # damping for (H + μI)
        neumann_alpha_default: float = 0.1,   # fallback step size
        neumann_alpha_min: float = 1e-6,
        neumann_alpha_max: float = 1.0,
        neumann_use_probe_alpha: bool = True,  # adapt α from curvature probe
        neumann_max_growth_ratio: float = 10.0,
        fd_hvp_eps: float = 0.01,     # finite-difference perturbation size
        implicit_offload_cpu: bool = False,  # offload Neumann vectors to CPU
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.perta_lambda = perta_lambda
        self.perta_alpha = perta_alpha
        self.fisher_cache_path = fisher_cache_path
        self.pretrained_model_name = pretrained_model_name
        self.lora_r = lora_r
        self.lora_alpha_val = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules or [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
        self.T = T
        self.K = K
        self.eta_theta = eta_theta
        self.eta_in = eta_in
        self.epsilon = epsilon
        self.rho = rho
        self.lambda_init = lambda_init
        self.lambda_max = lambda_max
        self.forget_loss_type = forget_loss_type
        self.npo_beta = npo_beta
        self.ga_clip = ga_clip
        self.lr_schedule = lr_schedule
        self.warmup_fraction = warmup_fraction
        self.npo_saturation_threshold = npo_saturation_threshold
        self.saturation_patience = saturation_patience
        self.retain_only_after_saturation = retain_only_after_saturation
        self.checkpoint_every_epoch = checkpoint_every_epoch
        self.eval_at_steps = set(eval_at_steps) if eval_at_steps else set()
        self.use_implicit = use_implicit
        self.neumann_steps = neumann_steps
        self.neumann_mu = neumann_mu
        self.neumann_alpha_default = neumann_alpha_default
        self.neumann_alpha_min = neumann_alpha_min
        self.neumann_alpha_max = neumann_alpha_max
        self.neumann_use_probe_alpha = neumann_use_probe_alpha
        self.neumann_max_growth_ratio = neumann_max_growth_ratio
        self.fd_hvp_eps = fd_hvp_eps
        self.implicit_offload_cpu = implicit_offload_cpu

        # Runtime state
        self.lambda_dual = float(lambda_init)

    # ------------------------------------------------------------------
    # Stage 1: PerTA weight surgery
    # ------------------------------------------------------------------
    def _apply_perta(self):
        """Apply PerTA weight surgery in-place: θ = θ_target - λ*w*τ."""
        from transformers import AutoModelForCausalLM

        device = next(self.model.parameters()).device

        # Load Fisher cache
        logger.info(f"Loading Fisher cache: {self.fisher_cache_path}")
        cached = torch.load(
            self.fisher_cache_path, map_location="cpu", weights_only=True
        )
        fisher_forget = cached["forget"]
        fisher_retain = cached["retain"]

        # Load pretrained state dict (CPU only — we just need the weights)
        logger.info(f"Loading pretrained model: {self.pretrained_model_name}")
        pretrained = AutoModelForCausalLM.from_pretrained(
            self.pretrained_model_name,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
        )
        pretrained_sd = pretrained.state_dict()
        del pretrained

        # Apply PerTA: θ = θ_target - λ*w*τ
        target_sd = {k: v.clone().cpu() for k, v in self.model.state_dict().items()}
        new_sd = {}
        total_w = 0.0
        n_w = 0

        for name in target_sd:
            theta_t = target_sd[name].float()
            theta_p = pretrained_sd[name].float()
            tau = theta_t - theta_p

            if name in fisher_forget and name in fisher_retain:
                ff = fisher_forget[name].float()
                fr = fisher_retain[name].float()
                w = ff / (ff + self.perta_alpha * fr + 1e-8)
                total_w += w.mean().item()
                n_w += 1
            else:
                w = torch.full_like(theta_t, 0.3)

            new_sd[name] = (theta_t - self.perta_lambda * w * tau).to(
                target_sd[name].dtype
            )

        self.model.load_state_dict(new_sd)
        self.model.to(device)

        del target_sd, new_sd, pretrained_sd, fisher_forget, fisher_retain
        torch.cuda.empty_cache()

        logger.info(
            f"PerTA applied: λ={self.perta_lambda:.2f}, α={self.perta_alpha:.1f}, "
            f"mean_w={total_w / max(n_w, 1):.4f}"
        )

    # ------------------------------------------------------------------
    # Stage 2: LoRA wrapping
    # ------------------------------------------------------------------
    def _wrap_with_lora(self):
        """Add LoRA adapters to the model."""
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
        logger.info(
            f"LoRA applied: r={self.lora_r}, alpha={self.lora_alpha_val}, "
            f"modules={self.lora_target_modules}"
        )

    # ------------------------------------------------------------------
    # Loss computation helpers
    # ------------------------------------------------------------------
    def _compute_ce_loss(self, batch, device):
        """Standard CE loss on a batch."""
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch.get("labels", input_ids).to(device)

        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        return outputs.loss

    def _compute_npo_loss(self, batch, device):
        """NPO loss: model (LoRA enabled) vs ref (LoRA disabled)."""
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch.get("labels", input_ids).to(device)

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        shifted_labels = labels[..., 1:].contiguous()
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")

        # Ref forward: disable LoRA adapters (base model = PerTA init)
        self.model.disable_adapter_layers()
        with torch.no_grad():
            ref_outputs = self.model(**inputs)
            ref_logits = ref_outputs.logits[..., :-1, :].contiguous()
            ref_nll = loss_fn(
                ref_logits.transpose(-1, -2), shifted_labels
            ).sum(dim=-1)

        # Model forward: enable LoRA adapters
        self.model.enable_adapter_layers()
        model_outputs = self.model(**inputs)
        model_logits = model_outputs.logits[..., :-1, :].contiguous()
        model_nll = loss_fn(
            model_logits.transpose(-1, -2), shifted_labels
        ).sum(dim=-1)

        # NPO: -2/β * log σ(β * (model_nll - ref_nll))
        log_ratio = -(model_nll - ref_nll)
        loss = -2 / self.npo_beta * F.logsigmoid(
            self.npo_beta * (-log_ratio)
        ).mean()

        return loss

    def _compute_kl_loss(self, batch, device):
        """KL(model || ref) on forget data — never saturates unlike NPO."""
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch.get("labels", input_ids).to(device)

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        shifted_labels = labels[..., 1:].contiguous()
        mask = (shifted_labels != -100).float()

        # Ref forward: disable LoRA (base = PerTA init)
        self.model.disable_adapter_layers()
        with torch.no_grad():
            ref_logits = self.model(**inputs).logits[..., :-1, :].contiguous()
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)

        # Model forward: enable LoRA
        self.model.enable_adapter_layers()
        model_logits = self.model(**inputs).logits[..., :-1, :].contiguous()
        model_log_probs = F.log_softmax(model_logits, dim=-1)

        # KL(model || ref) = sum model_prob * (log model_prob - log ref_prob)
        # We want model to DIVERGE from ref on forget → minimize -KL
        # Or equivalently: maximize KL → use -KL as loss to minimize
        kl = F.kl_div(ref_log_probs, model_log_probs, log_target=True, reduction="none")
        kl = kl.sum(dim=-1)  # sum over vocab
        kl = (kl * mask).sum() / mask.sum().clamp(min=1)

        # Negative KL: we want model to diverge FROM ref on forget data
        return -kl

    def _compute_logit_margin_loss(self, batch, device):
        """Logit margin flattening: minimize (max_logit - mean_logit).

        Pushes model output toward uniform distribution on forget data.
        Never saturates — there's always room to flatten further.
        """
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        max_logits = logits.max(dim=-1)[0]
        mean_logits = logits.mean(dim=-1)
        margins = max_logits - mean_logits
        return margins.mean()

    def _compute_forget_loss(self, batch, device):
        """Dispatch to configured forget loss."""
        if self.forget_loss_type == "npo":
            return self._compute_npo_loss(batch, device)
        elif self.forget_loss_type == "ga":
            # Gradient ascent: -CE on forget data
            return -self._compute_ce_loss(batch, device)
        elif self.forget_loss_type == "kl":
            return self._compute_kl_loss(batch, device)
        elif self.forget_loss_type == "logit_margin":
            return self._compute_logit_margin_loss(batch, device)
        else:
            raise ValueError(f"Unknown forget_loss_type: {self.forget_loss_type}")

    # ------------------------------------------------------------------
    # Implicit differentiation (FD-HVP + Truncated Neumann)
    # ------------------------------------------------------------------
    def _unflatten_params(self, flat_vec, params_list):
        """Unflatten vector back to parameter shapes."""
        out = []
        offset = 0
        for p in params_list:
            n = p.numel()
            out.append(flat_vec[offset:offset + n].reshape(p.shape))
            offset += n
        return out

    def _compute_hvp_fd(self, loss_fn, params, v):
        """Finite-difference HVP: H*v ≈ (∇L(θ+εv̂) - ∇L(θ-εv̂)) / (2ε) * ||v||

        Works with any attention backend (no create_graph needed).
        Only perturbs LoRA params (~55M), so the perturbation is cheap.
        """
        eps = self.fd_hvp_eps
        v_norm = v.norm().clamp(min=1e-12)
        v_unit = v / v_norm
        v_list = self._unflatten_params(v_unit, params)

        # +ε perturbation
        with torch.no_grad():
            for p, dv in zip(params, v_list):
                p.data.add_(eps * dv.to(p.dtype))
        loss_p = loss_fn()
        grads_p = torch.autograd.grad(loss_p, params, allow_unused=True)
        grads_p = [
            g.detach().clone() if g is not None else torch.zeros_like(p)
            for g, p in zip(grads_p, params)
        ]
        del loss_p

        # -ε perturbation (from +ε → -ε = subtract 2ε)
        with torch.no_grad():
            for p, dv in zip(params, v_list):
                p.data.sub_(2.0 * eps * dv.to(p.dtype))
        loss_m = loss_fn()
        grads_m = torch.autograd.grad(loss_m, params, allow_unused=True)
        grads_m = [
            g.detach().clone() if g is not None else torch.zeros_like(p)
            for g, p in zip(grads_m, params)
        ]
        del loss_m

        # Restore params
        with torch.no_grad():
            for p, dv in zip(params, v_list):
                p.data.add_(eps * dv.to(p.dtype))

        hvp_list = [(gp - gm) / (2.0 * eps) for gp, gm in zip(grads_p, grads_m)]
        hvp_flat = torch.cat([h.reshape(-1) for h in hvp_list])
        return hvp_flat * v_norm.item()

    def _truncated_neumann(self, params_list, v, inner_loss_fn, alm_loss_fn):
        """Truncated Neumann implicit correction.

        Approximates h ≈ (H_inner + μI)^{-1} v via Richardson iteration,
        then computes corrected gradient: g_corr = v - H_outer(h).

        v: flat outer gradient (on same device as params)
        Returns (g_corr_flat, status_string).
        """
        device = v.device
        dtype = v.dtype
        offload = self.implicit_offload_cpu

        if not torch.isfinite(v).all():
            logger.warning("Neumann: non-finite v, skipping correction")
            return v, "fallback_nonfinite_v"

        # Damped inner Hessian: H_tilde(x) = H_inner(x) + μx
        def H_in(x):
            # If offloaded, move to GPU for HVP then back
            if offload:
                x_gpu = x.to(device)
            else:
                x_gpu = x
            Hv = self._compute_hvp_fd(inner_loss_fn, params_list, x_gpu)
            result = Hv + self.neumann_mu * x_gpu
            if offload:
                return result.cpu()
            return result

        # Choose α via curvature probe
        alpha = self.neumann_alpha_default
        if self.neumann_use_probe_alpha:
            u = torch.randn_like(v)
            u = u / u.norm().clamp(min=1e-12)
            Hu = H_in(u)
            L_est = Hu.norm().clamp(min=1e-12).item()
            alpha = 0.5 / (L_est + 1e-12)
            if not np.isfinite(alpha) or alpha <= 0:
                alpha = self.neumann_alpha_default
        alpha = float(np.clip(alpha, self.neumann_alpha_min, self.neumann_alpha_max))

        # Richardson iteration: h_{k+1} = h_k + α(v - H_tilde h_k)
        work_v = v.cpu() if offload else v
        h = torch.zeros_like(work_v)
        for j in range(self.neumann_steps + 1):
            residual = work_v - H_in(h)
            if not torch.isfinite(residual).all():
                logger.warning(f"Neumann step {j}: non-finite residual, fallback")
                return v, "fallback_nonfinite"
            h = h + alpha * residual
            if not torch.isfinite(h).all():
                logger.warning(f"Neumann step {j}: non-finite h, fallback")
                return v, "fallback_nonfinite"

        # Outer HVP: c = H_alm(h)
        h_gpu = h.to(device) if offload else h
        c = self._compute_hvp_fd(alm_loss_fn, params_list, h_gpu)
        g_corr = v - c

        # Safety: fallback if correction explodes
        v_norm = v.norm().clamp(min=1e-12).item()
        g_corr_norm = g_corr.norm().item()
        h_norm = h_gpu.norm().item()
        if g_corr_norm > self.neumann_max_growth_ratio * v_norm:
            logger.warning(
                f"Neumann: correction exploded ||g_corr||={g_corr_norm:.4f} "
                f"> {self.neumann_max_growth_ratio}*||v||={v_norm:.4f}, fallback"
            )
            return v, "fallback_exploded"

        logger.debug(
            f"Neumann: α={alpha:.6f} ||v||={v_norm:.4f} "
            f"||h||={h_norm:.4f} ||g_corr||={g_corr_norm:.4f}"
        )
        return g_corr, "ok"

    # ------------------------------------------------------------------
    # Bilevel inner and outer steps
    # ------------------------------------------------------------------
    def inner_step(self, retain_batch, device):
        """Inner loop: CE on retain data, updating LoRA params only."""
        self.model.enable_adapter_layers()
        self.model.train()
        self._inner_opt.zero_grad()

        loss = self._compute_ce_loss(retain_batch, device)
        loss.backward()
        self._inner_opt.step()

        return loss.item()

    def outer_step(self, forget_batch, retain_batch, device):
        """Outer step: ALM loss = L_fgt + retain_coeff * L_ret.

        When use_implicit=True, applies FD-HVP Neumann correction to the
        outer gradient before the optimizer step. This accounts for the
        inner loop's response to the outer update (proper bilevel).
        """
        self.model.enable_adapter_layers()
        self.model.train()
        self._outer_opt.zero_grad()

        # Forget loss (NPO or GA)
        L_fgt = self._compute_forget_loss(forget_batch, device)

        # Retain loss for ALM constraint
        L_ret = self._compute_ce_loss(retain_batch, device)

        # ALM: L_alm = L_fgt + (λ + ρ*max(0, r)) * L_ret
        r_val = max(0.0, L_ret.item() - self.epsilon)
        retain_coeff = self.lambda_dual + self.rho * r_val
        L_alm = L_fgt + retain_coeff * L_ret

        L_alm.backward()

        # Clip gradients for GA (can be aggressive)
        if self.forget_loss_type == "ga" and self.ga_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.ga_clip,
            )

        # Implicit correction: replace .grad with Neumann-corrected gradient
        if self.use_implicit:
            lora_params = [p for p in self.model.parameters() if p.requires_grad]
            # Capture outer gradient as v
            v = torch.cat([p.grad.reshape(-1) for p in lora_params])

            # Free autograd graph before FD-HVP forward passes
            torch.cuda.empty_cache()

            # FD-HVP closures — recompute losses from current model state
            def _inner_loss_fn():
                return self._compute_ce_loss(retain_batch, device)

            def _alm_loss_fn():
                l_f = self._compute_forget_loss(forget_batch, device)
                l_r = self._compute_ce_loss(retain_batch, device)
                r_t = max(0.0, l_r.item() - self.epsilon)
                coeff = self.lambda_dual + self.rho * r_t
                return l_f + coeff * l_r

            g_corr, status = self._truncated_neumann(
                lora_params, v, _inner_loss_fn, _alm_loss_fn
            )

            # Write corrected gradient back to .grad
            offset = 0
            for p in lora_params:
                n = p.numel()
                p.grad = g_corr[offset:offset + n].reshape(p.shape).to(p.dtype)
                offset += n

            if status != "ok":
                logger.info(f"  Implicit: {status} (using uncorrected gradient)")

        self._outer_opt.step()

        # Dual variable update
        self.lambda_dual = max(0.0, self.lambda_dual + self.rho * r_val)
        if self.lambda_max > 0:
            self.lambda_dual = min(self.lambda_dual, self.lambda_max)

        return L_fgt.item(), L_ret.item(), L_ret.item() - self.epsilon

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------
    def train(self):
        """Main loop: PerTA → LoRA → Bilevel ALM."""
        device = self.args.device

        # Stage 1: PerTA weight surgery
        if self.perta_lambda > 0:
            logger.info("=" * 60)
            logger.info("Stage 1: PerTA weight surgery")
            logger.info("=" * 60)
            self._apply_perta()
        else:
            logger.info("Skipping PerTA (lambda=0)")

        # Stage 2: LoRA adapters
        logger.info("=" * 60)
        logger.info("Stage 2: LoRA adapters")
        logger.info("=" * 60)
        self._wrap_with_lora()

        # Create Adam optimizers for LoRA parameters
        lora_params = [p for p in self.model.parameters() if p.requires_grad]
        self._inner_opt = torch.optim.Adam(lora_params, lr=self.eta_in)
        self._outer_opt = torch.optim.Adam(lora_params, lr=self.eta_theta)
        logger.info(
            f"Adam optimizers: inner lr={self.eta_in}, outer lr={self.eta_theta}, "
            f"params={sum(p.numel() for p in lora_params)}"
        )

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

        # Determine total training steps
        # T > 0: fixed step count (backward compat with Ze0: T=25)
        # T <= 0: run full epochs (new mode for data coverage)
        if self.T > 0:
            max_outer_steps = self.T
        else:
            max_outer_steps = num_epochs * steps_per_epoch

        # LR schedulers
        if self.lr_schedule == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
            warmup_steps = int(self.warmup_fraction * max_outer_steps)
            if warmup_steps > 0:
                warmup_outer = LinearLR(self._outer_opt, start_factor=0.1, total_iters=warmup_steps)
                cosine_outer = CosineAnnealingLR(self._outer_opt, T_max=max_outer_steps - warmup_steps)
                self._outer_scheduler = SequentialLR(self._outer_opt, [warmup_outer, cosine_outer], milestones=[warmup_steps])
                warmup_inner = LinearLR(self._inner_opt, start_factor=0.1, total_iters=warmup_steps)
                cosine_inner = CosineAnnealingLR(self._inner_opt, T_max=max_outer_steps - warmup_steps)
                self._inner_scheduler = SequentialLR(self._inner_opt, [warmup_inner, cosine_inner], milestones=[warmup_steps])
            else:
                self._outer_scheduler = CosineAnnealingLR(self._outer_opt, T_max=max_outer_steps)
                self._inner_scheduler = CosineAnnealingLR(self._inner_opt, T_max=max_outer_steps)
        else:
            self._outer_scheduler = None
            self._inner_scheduler = None

        logger.info("=" * 60)
        logger.info("Stage 3: Bilevel ALM optimization")
        logger.info("=" * 60)
        logger.info(f"  Mode: {'epoch-based' if self.T <= 0 else f'fixed T={self.T}'}")
        logger.info(f"  Epochs={num_epochs}, steps/epoch={steps_per_epoch}, "
                     f"max_steps={max_outer_steps}, K={self.K}")
        logger.info(f"  Batch size={batch_size}, "
                     f"samples/epoch={steps_per_epoch * batch_size}")
        logger.info(f"  Forget loss: {self.forget_loss_type} (beta={self.npo_beta})")
        logger.info(f"  ALM: ε={self.epsilon}, ρ={self.rho}, λ_init={self.lambda_init}"
                     f"{f', λ_max={self.lambda_max}' if self.lambda_max > 0 else ''}")
        logger.info(f"  LR: outer={self.eta_theta}, inner={self.eta_in}, "
                     f"schedule={self.lr_schedule}")
        logger.info(f"  LoRA: r={self.lora_r}, alpha={self.lora_alpha_val}")
        if self.perta_lambda > 0:
            logger.info(f"  PerTA: λ={self.perta_lambda}, α={self.perta_alpha}")
        if self.use_implicit:
            logger.info(f"  Implicit: FD-HVP Neumann, steps={self.neumann_steps}, "
                         f"μ={self.neumann_mu}, eps={self.fd_hvp_eps}, "
                         f"probe_α={self.neumann_use_probe_alpha}, "
                         f"offload_cpu={self.implicit_offload_cpu}")
        if self.retain_only_after_saturation:
            logger.info(f"  Saturation: threshold={self.npo_saturation_threshold}, "
                         f"patience={self.saturation_patience}")

        # GPU memory check
        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated() / 1e9
            logger.info(f"  GPU memory before bilevel: {mem:.1f} GB")

        history = []
        global_step = 0
        saturated_count = 0
        total_saturated_steps = 0  # steps where we skipped outer

        for epoch in range(num_epochs):
            epoch_start = time.time()
            epoch_inner_losses = []
            epoch_fgt_losses = []
            epoch_ret_losses = []

            for batch_idx, combined_batch in enumerate(train_dataloader):
                # Check max steps
                if global_step >= max_outer_steps:
                    break

                t_start = time.time()
                forget_batch = combined_batch["forget"]
                retain_batch = combined_batch["retain"]

                # Inner loop: K steps of retain CE (K=0 → no bilevel, outer only)
                inner_losses = []
                for _k in range(self.K):
                    l_in = self.inner_step(retain_batch, device)
                    inner_losses.append(l_in)

                # Saturation check: skip outer step if NPO has saturated
                if (self.retain_only_after_saturation
                        and saturated_count >= self.saturation_patience):
                    L_fgt, L_ret, r = 0.0, inner_losses[-1], 0.0
                    total_saturated_steps += 1
                else:
                    # Outer step: ALM(forget + retain constraint)
                    L_fgt, L_ret, r = self.outer_step(
                        forget_batch, retain_batch, device
                    )

                    # Track NPO saturation
                    if L_fgt < self.npo_saturation_threshold:
                        saturated_count += 1
                    else:
                        saturated_count = 0

                # Step LR schedulers
                if self._outer_scheduler is not None:
                    self._outer_scheduler.step()
                    self._inner_scheduler.step()

                dt = time.time() - t_start

                step_info = {
                    "step": global_step,
                    "epoch": epoch,
                    "L_fgt": L_fgt,
                    "L_ret": L_ret,
                    "r": r,
                    "lambda": self.lambda_dual,
                    "inner_loss_mean": sum(inner_losses) / len(inner_losses) if inner_losses else 0.0,
                    "saturated": saturated_count >= self.saturation_patience,
                    "dt": dt,
                }
                history.append(step_info)
                epoch_inner_losses.extend(inner_losses)
                epoch_fgt_losses.append(L_fgt)
                epoch_ret_losses.append(L_ret)

                # Periodic logging
                log_every = max(1, steps_per_epoch // 10)  # ~10 logs per epoch
                if global_step % log_every == 0 or global_step == max_outer_steps - 1:
                    lr_info = ""
                    if self._outer_scheduler is not None:
                        lr_info = f" olr={self._outer_opt.param_groups[0]['lr']:.2e}"
                    sat_info = " [SAT]" if saturated_count >= self.saturation_patience else ""
                    logger.info(
                        f"  [{global_step:4d}/{max_outer_steps}|e{epoch+1}] "
                        f"L_fgt={L_fgt:.4f} L_ret={L_ret:.4f} "
                        f"r={r:+.4f} λ={self.lambda_dual:.3f} "
                        f"inner={sum(inner_losses) / len(inner_losses) if inner_losses else 0.0:.4f}"
                        f"{lr_info}{sat_info} dt={dt:.1f}s"
                    )

                global_step += 1

                # Intermediate checkpoint: non-destructive merge → save → unmerge
                if global_step in self.eval_at_steps:
                    ckpt_dir = os.path.join(
                        self.args.output_dir, f"step-{global_step}"
                    )
                    logger.info(f"  Saving intermediate checkpoint at step {global_step}...")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    self.model.eval()
                    # Merge LoRA into base (reversible)
                    self.model.merge_adapter()
                    # Extract state dict with clean key names (strip PEFT wrapper)
                    peft_sd = self.model.base_model.model.state_dict()
                    clean_sd = {}
                    for key, val in peft_sd.items():
                        clean_key = key.replace(".base_layer", "")
                        if "lora_" in clean_key:
                            continue
                        clean_sd[clean_key] = val
                    # Save as standard model (handles sharding automatically)
                    self.model.base_model.model.save_pretrained(
                        ckpt_dir, state_dict=clean_sd
                    )
                    if self.tokenizer is not None:
                        self.tokenizer.save_pretrained(ckpt_dir)
                    # Unmerge to restore LoRA state for continued training
                    self.model.unmerge_adapter()
                    self.model.train()
                    # Save history snapshot
                    with open(os.path.join(ckpt_dir, "lora_bial_history.json"), "w") as f:
                        json.dump(history, f, indent=2)
                    logger.info(f"  Checkpoint step-{global_step} saved (non-destructive).")

                # Early termination: model collapsed
                if L_ret > 10.0:
                    logger.warning(
                        f"  L_ret={L_ret:.1f} > 10.0 — model collapsed. "
                        f"Stopping at step {global_step}."
                    )
                    break

            # End of epoch summary
            if epoch_fgt_losses:
                n_samples_seen = (batch_idx + 1) * batch_size
                logger.info(
                    f"  Epoch {epoch+1}/{num_epochs} done: "
                    f"{batch_idx + 1} steps, ~{n_samples_seen} samples, "
                    f"mean_fgt={sum(epoch_fgt_losses)/len(epoch_fgt_losses):.4f}, "
                    f"mean_ret={sum(epoch_ret_losses)/len(epoch_ret_losses):.4f}, "
                    f"saturated_steps={total_saturated_steps}, "
                    f"dt={time.time()-epoch_start:.0f}s"
                )

            # Per-epoch checkpoint
            if (self.checkpoint_every_epoch
                    and epoch < num_epochs - 1
                    and epoch_fgt_losses):
                ckpt_dir = os.path.join(self.args.output_dir, f"checkpoint-epoch{epoch+1}")
                os.makedirs(ckpt_dir, exist_ok=True)
                # Save LoRA state (not merged) for checkpoint
                self.model.save_pretrained(ckpt_dir)
                logger.info(f"  Checkpoint saved: {ckpt_dir}")

            if global_step >= max_outer_steps:
                break
            if L_ret > 10.0:
                break

        # Merge LoRA into base and save
        logger.info("Merging LoRA adapters into base model...")
        self.model = self.model.merge_and_unload()

        output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)

        # Save tokenizer if available
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Save training history
        history_path = os.path.join(output_dir, "lora_bial_history.json")
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        logger.info(f"Model saved to {output_dir}")
        if history:
            logger.info(
                f"Final: λ={self.lambda_dual:.3f}, "
                f"L_fgt={history[-1]['L_fgt']:.4f}, "
                f"L_ret={history[-1]['L_ret']:.4f}"
            )

        # Run evaluation if configured
        self.evaluate()
