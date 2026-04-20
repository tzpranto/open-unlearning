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

from torch.utils.data import DataLoader
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
        focal_gamma: float = 2.0,
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
        implicit_warmup_steps: int = 0,  # skip implicit for first N steps (zero-init LoRA)
        # Batching
        gradient_accumulation_steps: int = 1,
        inner_accumulation_steps: int = 0,  # 0 = same as gradient_accumulation_steps
        inner_warmup_steps: int = 0,  # outer-only for first N steps (K=0), then K=configured
        max_grad_norm: float = 1.0,
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
        self.focal_gamma = focal_gamma
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
        self.implicit_warmup_steps = implicit_warmup_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.inner_accumulation_steps = inner_accumulation_steps if inner_accumulation_steps > 0 else gradient_accumulation_steps
        self.inner_warmup_steps = inner_warmup_steps
        self.max_grad_norm = max_grad_norm

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
        """Logit margin flattening: minimize (max_logit - mean_logit)."""
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        max_logits = logits.max(dim=-1)[0]
        mean_logits = logits.mean(dim=-1)
        margins = max_logits - mean_logits
        mask = attention_mask.float()
        return (margins * mask).sum() / mask.sum().clamp(min=1e-8)

    def _compute_focal_logit_margin_loss(self, batch, device):
        """Focal logit margin: upweight tokens where model is still confident."""
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        max_logits = logits.max(dim=-1)[0]
        mean_logits = logits.mean(dim=-1)
        margins = max_logits - mean_logits

        mask = attention_mask.float()
        margin_max = (margins * mask).amax(dim=-1, keepdim=True).clamp(min=1e-8)
        norm_margins = (margins / margin_max).clamp(min=1e-6)
        focal_weights = (norm_margins.detach() ** self.focal_gamma) * mask
        weighted_sum = (focal_weights * margins).sum()
        return weighted_sum / focal_weights.sum().clamp(min=1e-8)

    def _compute_entropy_max_loss(self, batch, device):
        """Entropy maximization: push model toward uniform on forget data.
        Reference-free — works from zero-init LoRA."""
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        return -entropy

    def _compute_repr_orthogonal_loss(self, forget_batch, retain_batch, device):
        """Representation orthogonality: minimize cosine similarity between
        forget and retain hidden states. Reference-free, bounded in [-1, 1]."""
        def get_mean_repr(batch):
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            out = self.model(input_ids=input_ids, attention_mask=attn,
                             output_hidden_states=True)
            h = out.hidden_states[-1]
            mask = attn.unsqueeze(-1).float()
            return (h * mask).sum(1) / mask.sum(1).clamp(min=1)

        h_f = get_mean_repr(forget_batch)
        with torch.no_grad():
            h_r = get_mean_repr(retain_batch).detach()

        h_f_n = F.normalize(h_f, dim=-1)
        h_r_n = F.normalize(h_r, dim=-1)
        sim = h_f_n @ h_r_n.T
        return sim.mean()

    def _compute_focal_repr_ortho_loss(self, forget_batch, retain_batch, device):
        """Focal representation orthogonality: per-sample cosine similarity
        weighted by sim^γ. Concentrates gradient on forget samples still
        similar to retain (hard-to-forget), ignores already-orthogonal ones."""
        def get_repr(batch):
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            out = self.model(input_ids=input_ids, attention_mask=attn,
                             output_hidden_states=True)
            h = out.hidden_states[-1]
            mask = attn.unsqueeze(-1).float()
            return (h * mask).sum(1) / mask.sum(1).clamp(min=1)

        h_f = get_repr(forget_batch)
        with torch.no_grad():
            h_r = get_repr(retain_batch).detach()

        h_f_n = F.normalize(h_f, dim=-1)
        h_r_n = F.normalize(h_r, dim=-1)

        # Per-sample: mean similarity to retain samples
        sim = h_f_n @ h_r_n.T  # [B_f, B_r]
        per_sample_sim = sim.mean(dim=-1).clamp(min=0)  # [B_f]

        # Focal weight: sim^γ (detached). High sim = hard = high weight.
        focal_weight = per_sample_sim.detach() ** self.focal_gamma
        return (focal_weight * per_sample_sim).sum() / focal_weight.sum().clamp(min=1e-8)

    def _compute_per_token_repr_ortho_loss(self, forget_batch, retain_batch, device):
        """Per-token representation orthogonality: for each forget token, find
        its most similar retain token and push away. Much stronger gradient
        signal than mean-pooled variant (~seq_len times more supervision)."""
        def get_token_reprs(batch):
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            out = self.model(input_ids=input_ids, attention_mask=attn,
                             output_hidden_states=True)
            h = out.hidden_states[-1]  # [B, T, D]
            return h, attn

        h_f, mask_f = get_token_reprs(forget_batch)
        with torch.no_grad():
            h_r, mask_r = get_token_reprs(retain_batch)

        # Flatten to [N_f, D] and [N_r, D] (valid tokens only)
        h_f_flat = h_f[mask_f.bool()]  # [N_f, D]
        h_r_flat = h_r[mask_r.bool()].detach()  # [N_r, D]

        h_f_n = F.normalize(h_f_flat, dim=-1)
        h_r_n = F.normalize(h_r_flat, dim=-1)

        # Per forget token: max similarity to any retain token
        sim_matrix = h_f_n @ h_r_n.T  # [N_f, N_r]
        max_sim = sim_matrix.max(dim=1)[0]  # [N_f]
        return max_sim.mean()

    def _compute_focal_per_token_repr_ortho_loss(self, forget_batch, retain_batch, device):
        """Focal per-token repr orthogonality: upweight forget tokens still
        similar to retain (hard to orthogonalize)."""
        def get_token_reprs(batch):
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            out = self.model(input_ids=input_ids, attention_mask=attn,
                             output_hidden_states=True)
            h = out.hidden_states[-1]
            return h, attn

        h_f, mask_f = get_token_reprs(forget_batch)
        with torch.no_grad():
            h_r, mask_r = get_token_reprs(retain_batch)

        h_f_flat = h_f[mask_f.bool()]
        h_r_flat = h_r[mask_r.bool()].detach()

        h_f_n = F.normalize(h_f_flat, dim=-1)
        h_r_n = F.normalize(h_r_flat, dim=-1)

        sim_matrix = h_f_n @ h_r_n.T
        max_sim = sim_matrix.max(dim=1)[0].clamp(min=0)

        focal_weights = max_sim.detach() ** self.focal_gamma
        weighted_sum = (focal_weights * max_sim).sum()
        return weighted_sum / focal_weights.sum().clamp(min=1e-8)

    def _compute_forget_loss(self, batch, device, retain_batch=None):
        """Dispatch to configured forget loss."""
        if self.forget_loss_type == "npo":
            return self._compute_npo_loss(batch, device)
        elif self.forget_loss_type == "ga":
            return -self._compute_ce_loss(batch, device)
        elif self.forget_loss_type == "kl":
            return self._compute_kl_loss(batch, device)
        elif self.forget_loss_type == "logit_margin":
            return self._compute_logit_margin_loss(batch, device)
        elif self.forget_loss_type == "focal_logit_margin":
            return self._compute_focal_logit_margin_loss(batch, device)
        elif self.forget_loss_type == "entropy_max":
            return self._compute_entropy_max_loss(batch, device)
        elif self.forget_loss_type == "repr_orthogonal":
            assert retain_batch is not None, "repr_orthogonal requires retain_batch"
            return self._compute_repr_orthogonal_loss(batch, retain_batch, device)
        elif self.forget_loss_type == "focal_repr_ortho":
            assert retain_batch is not None, "focal_repr_ortho requires retain_batch"
            return self._compute_focal_repr_ortho_loss(batch, retain_batch, device)
        elif self.forget_loss_type == "per_token_repr_ortho":
            assert retain_batch is not None, "per_token_repr_ortho requires retain_batch"
            return self._compute_per_token_repr_ortho_loss(batch, retain_batch, device)
        elif self.forget_loss_type == "focal_per_token_repr_ortho":
            assert retain_batch is not None, "focal_per_token_repr_ortho requires retain_batch"
            return self._compute_focal_per_token_repr_ortho_loss(batch, retain_batch, device)
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

    def _truncated_neumann(self, params_list, v, inner_loss_fn, outer_loss_fn):
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
        for j in range(self.neumann_steps):
            residual = work_v - H_in(h)
            if not torch.isfinite(residual).all():
                logger.warning(f"Neumann step {j}: non-finite residual, fallback")
                return v, "fallback_nonfinite"
            h = h + alpha * residual
            if not torch.isfinite(h).all():
                logger.warning(f"Neumann step {j}: non-finite h, fallback")
                return v, "fallback_nonfinite"

        # Outer HVP: c = H_outer(h)
        h_gpu = h.to(device) if offload else h
        c = self._compute_hvp_fd(outer_loss_fn, params_list, h_gpu)
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
    # Batch iterators (separate for inner/outer, fresh each call)
    # ------------------------------------------------------------------
    def _next_retain_batch(self):
        try:
            return next(self._retain_iter)
        except StopIteration:
            self._retain_iter = iter(self._retain_dataloader)
            return next(self._retain_iter)

    def _next_forget_batch(self):
        try:
            return next(self._forget_iter)
        except StopIteration:
            self._forget_iter = iter(self._forget_dataloader)
            return next(self._forget_iter)

    # ------------------------------------------------------------------
    # Bilevel inner and outer steps
    # ------------------------------------------------------------------
    def inner_step(self, retain_batch, device):
        """Single micro-batch forward+backward (no optimizer step).
        Loss is divided by inner accumulation steps for correct averaging."""
        loss = self._compute_ce_loss(retain_batch, device)
        (loss / self.inner_accumulation_steps).backward()
        return loss.item()

    def inner_loop(self, device):
        """K inner optimizer steps, each accumulating over inner_accumulation_steps
        micro-batches with FRESH retain data per micro-batch."""
        self.model.enable_adapter_layers()
        self.model.train()
        inner_losses = []
        for k in range(self.K):
            self._inner_opt.zero_grad()
            accum_loss = 0.0
            for micro_i in range(self.inner_accumulation_steps):
                batch = self._next_retain_batch()
                accum_loss += self.inner_step(batch, device)
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.max_grad_norm,
            )
            self._inner_opt.step()
            inner_losses.append(accum_loss / self.inner_accumulation_steps)
        return inner_losses

    def outer_step(self, device, global_step=0):
        """Outer ALM step: L_alm = L_fgt + λ·(L_ret - ε) + ρ/2·max(0, L_ret - ε)².

        Full ALM gradient when λ > 0 or L_ret > ε; otherwise effectively
        forget-only (ALM terms vanish when λ=0 and constraint is satisfied).
        """
        self.model.enable_adapter_layers()
        self.model.train()
        self._outer_opt.zero_grad()

        total_L_fgt = 0.0
        total_L_ret = 0.0
        last_forget_batch = None
        last_retain_batch = None

        for micro_i in range(self.gradient_accumulation_steps):
            forget_batch = self._next_forget_batch()
            retain_batch = self._next_retain_batch()

            L_fgt = self._compute_forget_loss(forget_batch, device, retain_batch=retain_batch)
            L_ret = self._compute_ce_loss(retain_batch, device)

            r_micro = L_ret - self.epsilon
            r_plus = torch.clamp(r_micro, min=0.0)
            L_alm = L_fgt + self.lambda_dual * r_micro + 0.5 * self.rho * (r_plus ** 2)
            (L_alm / self.gradient_accumulation_steps).backward()

            total_L_fgt += L_fgt.item()
            total_L_ret += L_ret.item()
            last_forget_batch = forget_batch
            last_retain_batch = retain_batch

        avg_L_fgt = total_L_fgt / self.gradient_accumulation_steps
        avg_L_ret = total_L_ret / self.gradient_accumulation_steps
        avg_r = avg_L_ret - self.epsilon

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad],
            self.max_grad_norm,
        )

        # Implicit correction (after warmup period for zero-init LoRA)
        use_implicit_this_step = (
            self.use_implicit and global_step >= self.implicit_warmup_steps
        )
        if use_implicit_this_step:
            lora_params = [p for p in self.model.parameters() if p.requires_grad]
            v = torch.cat([
                p.grad.reshape(-1) if p.grad is not None
                else torch.zeros(p.numel(), device=device)
                for p in lora_params
            ])
            torch.cuda.empty_cache()

            def _inner_loss_fn():
                return self._compute_ce_loss(last_retain_batch, device)

            def _outer_loss_fn():
                l_f = self._compute_forget_loss(last_forget_batch, device, retain_batch=last_retain_batch)
                l_r = self._compute_ce_loss(last_retain_batch, device)
                r_t = l_r - self.epsilon
                r_p = torch.clamp(r_t, min=0.0)
                return l_f + self.lambda_dual * r_t + 0.5 * self.rho * (r_p ** 2)

            g_corr, status = self._truncated_neumann(
                lora_params, v, _inner_loss_fn, _outer_loss_fn
            )

            offset = 0
            for p in lora_params:
                n = p.numel()
                p.grad = g_corr[offset:offset + n].reshape(p.shape).to(p.dtype)
                offset += n

            if status != "ok":
                logger.info(f"  Implicit: {status} (using uncorrected gradient)")

            # Re-clip after implicit correction (corrected gradient may exceed max_grad_norm)
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.max_grad_norm,
            )

        self._outer_opt.step()

        # Dual update on averaged residual (standard ALM for inequality g(x) <= 0)
        self.lambda_dual = max(0.0, self.lambda_dual + self.rho * avg_r)
        if self.lambda_max > 0:
            self.lambda_dual = min(self.lambda_dual, self.lambda_max)

        return avg_L_fgt, avg_L_ret, avg_r

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

        # Create optimizers for LoRA parameters
        # Inner uses SGD: no momentum state to go stale between inner/outer phases.
        # Outer uses Adam: needs adaptive LR for the harder forget+ALM objective.
        lora_params = [p for p in self.model.parameters() if p.requires_grad]
        self._inner_opt = torch.optim.SGD(lora_params, lr=self.eta_in)
        self._outer_opt = torch.optim.Adam(lora_params, lr=self.eta_theta)
        logger.info(
            f"Optimizers: inner=SGD(lr={self.eta_in}), outer=Adam(lr={self.eta_theta}), "
            f"params={sum(p.numel() for p in lora_params)}"
        )

        # Gradient checkpointing
        if getattr(self.args, "gradient_checkpointing", False):
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        # Data — separate dataloaders for forget and retain
        combined_dataset = self.train_dataset
        forget_ds = combined_dataset.forget
        retain_ds = combined_dataset.retain
        collator = self.data_collator
        forget_dl = DataLoader(
            forget_ds, batch_size=self.args.per_device_train_batch_size,
            shuffle=True, collate_fn=collator, drop_last=False,
            pin_memory=True,
        )
        retain_dl = DataLoader(
            retain_ds, batch_size=self.args.per_device_train_batch_size,
            shuffle=True, collate_fn=collator, drop_last=False,
            pin_memory=True,
        )
        self._forget_dataloader = forget_dl
        self._retain_dataloader = retain_dl
        self._forget_iter = iter(forget_dl)
        self._retain_iter = iter(retain_dl)

        batch_size = self.args.per_device_train_batch_size
        effective_bs = batch_size * self.gradient_accumulation_steps
        inner_effective_bs = batch_size * self.inner_accumulation_steps
        # Epoch = one pass through the forget dataset (anchor)
        forget_micro_batches = len(self._forget_dataloader)
        retain_micro_batches = len(self._retain_dataloader)
        steps_per_epoch = max(1, forget_micro_batches // self.gradient_accumulation_steps)
        num_epochs = max(1, int(self.args.num_train_epochs))

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
        logger.info(f"  Epochs={num_epochs}, micro_bs={batch_size}, "
                     f"outer_accum={self.gradient_accumulation_steps} (eff_bs={effective_bs}), "
                     f"inner_accum={self.inner_accumulation_steps} (eff_bs={inner_effective_bs})")
        logger.info(f"  forget_batches/epoch={forget_micro_batches}, "
                     f"retain_batches/epoch={retain_micro_batches}, "
                     f"outer_steps/epoch={steps_per_epoch}, "
                     f"max_steps={max_outer_steps}, K={self.K}")
        logger.info(f"  Forget loss: {self.forget_loss_type} (beta={self.npo_beta})")
        logger.info(f"  ALM: ε={self.epsilon}, ρ={self.rho}, λ_init={self.lambda_init}"
                     f"{f', λ_max={self.lambda_max}' if self.lambda_max > 0 else ''}")
        logger.info(f"  LR: outer={self.eta_theta}, inner={self.eta_in}, "
                     f"schedule={self.lr_schedule}")
        logger.info(f"  LoRA: r={self.lora_r}, alpha={self.lora_alpha_val}")
        if self.inner_warmup_steps > 0:
            logger.info(f"  Inner warmup: outer-only for first {self.inner_warmup_steps} steps")
        if self.perta_lambda > 0:
            logger.info(f"  PerTA: λ={self.perta_lambda}, α={self.perta_alpha}")
        if self.use_implicit:
            logger.info(f"  Implicit: FD-HVP Neumann, steps={self.neumann_steps}, "
                         f"μ={self.neumann_mu}, eps={self.fd_hvp_eps}, "
                         f"warmup={self.implicit_warmup_steps}")
        if self.retain_only_after_saturation:
            logger.info(f"  Saturation: threshold={self.npo_saturation_threshold}, "
                         f"patience={self.saturation_patience}")

        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated() / 1e9
            logger.info(f"  GPU memory before bilevel: {mem:.1f} GB")

        history = []
        global_step = 0
        saturated_count = 0
        total_saturated_steps = 0

        for t in range(max_outer_steps):
            t_start = time.time()
            epoch = t // steps_per_epoch if steps_per_epoch > 0 else 0

            # Inner loop: skip during warmup to let outer establish forgetting direction
            if t < self.inner_warmup_steps:
                inner_losses = []
            else:
                inner_losses = self.inner_loop(device)

            # Saturation check
            if (self.retain_only_after_saturation
                    and saturated_count >= self.saturation_patience):
                L_fgt, L_ret, r = 0.0, inner_losses[-1] if inner_losses else 0.0, 0.0
                total_saturated_steps += 1
            else:
                L_fgt, L_ret, r = self.outer_step(device, global_step=t)

                # Adaptive inner: extra recovery steps when retain spikes
                extra_inner = 0
                while L_ret > 2 * self.epsilon and extra_inner < self.K * 3:
                    recovery = self.inner_loop(device)
                    with torch.no_grad():
                        rb = self._next_retain_batch()
                        L_ret = self._compute_ce_loss(rb, device).item()
                    r = L_ret - self.epsilon
                    extra_inner += self.K
                if extra_inner > 0:
                    logger.info(f"  Adaptive inner: {extra_inner} extra steps, L_ret={L_ret:.4f}")

                if L_fgt >= 0 and L_fgt < self.npo_saturation_threshold:
                    saturated_count += 1
                else:
                    saturated_count = 0

            if self._outer_scheduler is not None:
                self._outer_scheduler.step()
                if t >= self.inner_warmup_steps:
                    self._inner_scheduler.step()

            dt = time.time() - t_start

            step_info = {
                "step": t,
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

            log_every = max(1, max_outer_steps // 20)
            if t % log_every == 0 or t == max_outer_steps - 1:
                lr_info = ""
                if self._outer_scheduler is not None:
                    lr_info = f" olr={self._outer_opt.param_groups[0]['lr']:.2e}"
                sat_info = " [SAT]" if saturated_count >= self.saturation_patience else ""
                logger.info(
                    f"  [{t:4d}/{max_outer_steps}|e{epoch+1}] "
                    f"L_fgt={L_fgt:.4f} L_ret={L_ret:.4f} "
                    f"r={r:+.4f} λ={self.lambda_dual:.3f} "
                    f"inner={sum(inner_losses) / len(inner_losses) if inner_losses else 0.0:.4f}"
                    f"{lr_info}{sat_info} dt={dt:.1f}s"
                )

            global_step = t + 1

            # Intermediate checkpoint
            if global_step in self.eval_at_steps:
                ckpt_dir = os.path.join(self.args.output_dir, f"step-{global_step}")
                logger.info(f"  Saving intermediate checkpoint at step {global_step}...")
                os.makedirs(ckpt_dir, exist_ok=True)
                self.model.eval()
                self.model.merge_adapter()
                peft_sd = self.model.base_model.model.state_dict()
                clean_sd = {}
                for key, val in peft_sd.items():
                    clean_key = key.replace(".base_layer", "")
                    if "lora_" in clean_key:
                        continue
                    clean_sd[clean_key] = val
                self.model.base_model.model.save_pretrained(ckpt_dir, state_dict=clean_sd)
                if self.tokenizer is not None:
                    self.tokenizer.save_pretrained(ckpt_dir)
                self.model.unmerge_adapter()
                self.model.train()
                with open(os.path.join(ckpt_dir, "lora_bial_history.json"), "w") as f:
                    json.dump(history, f, indent=2)
                logger.info(f"  Checkpoint step-{global_step} saved.")

            # Per-epoch checkpoint
            if (self.checkpoint_every_epoch
                    and steps_per_epoch > 0
                    and global_step % steps_per_epoch == 0
                    and global_step < max_outer_steps):
                ep_num = global_step // steps_per_epoch
                ckpt_dir = os.path.join(self.args.output_dir, f"checkpoint-epoch{ep_num}")
                os.makedirs(ckpt_dir, exist_ok=True)
                self.model.eval()
                self.model.merge_adapter()
                peft_sd = self.model.base_model.model.state_dict()
                clean_sd = {}
                for key, val in peft_sd.items():
                    clean_key = key.replace(".base_layer", "")
                    if "lora_" in clean_key:
                        continue
                    clean_sd[clean_key] = val
                self.model.base_model.model.save_pretrained(ckpt_dir, state_dict=clean_sd)
                if self.tokenizer is not None:
                    self.tokenizer.save_pretrained(ckpt_dir)
                self.model.unmerge_adapter()
                self.model.train()
                with open(os.path.join(ckpt_dir, "lora_bial_history.json"), "w") as f:
                    json.dump(history, f, indent=2)
                logger.info(f"  Epoch {ep_num} checkpoint saved: {ckpt_dir}")

            if L_ret > 10.0:
                logger.warning(f"  L_ret={L_ret:.1f} > 10.0 — model collapsed at step {global_step}.")
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
