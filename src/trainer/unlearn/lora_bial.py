"""
LoRA-BiAL: LoRA-based Bilevel Augmented Lagrangian for Unlearning
=================================================================

Two-stage approach:
1. PerTA weight surgery on base model (selective per-param negation)
2. LoRA bilevel optimization for retain recovery

Key innovations vs standard bilevel (SIBL):
- Zero-overhead ref model: ref = base model with LoRA adapters disabled
- Naturally constrained update space: LoRA limits collateral damage
- Solves OOM: ~55MB LoRA gradients vs ~14GB full model gradients
- PerTA init places model ABOVE CE frontier; bilevel recovers retain

The bilevel dynamics:
- Inner loop: LoRA learns retain CE → some forget knowledge spillover
- Outer loop: NPO/GA detects spillover, corrects LoRA to reduce forget probs
- ALM constraint: prevents over-correction (maintains fk quality from PerTA)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        # Forget loss
        forget_loss_type: str = "npo",  # "npo" or "ga"
        npo_beta: float = 2.0,
        ga_clip: float = 1.0,
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
        self.forget_loss_type = forget_loss_type
        self.npo_beta = npo_beta
        self.ga_clip = ga_clip

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

    def _compute_forget_loss(self, batch, device):
        """Dispatch to configured forget loss."""
        if self.forget_loss_type == "npo":
            return self._compute_npo_loss(batch, device)
        elif self.forget_loss_type == "ga":
            # Gradient ascent: -CE on forget data
            return -self._compute_ce_loss(batch, device)
        elif self.forget_loss_type == "kl":
            return self._compute_kl_loss(batch, device)
        else:
            raise ValueError(f"Unknown forget_loss_type: {self.forget_loss_type}")

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
        """Outer step: ALM loss = L_fgt + retain_coeff * L_ret."""
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

        self._outer_opt.step()

        # Dual variable update
        self.lambda_dual = max(0.0, self.lambda_dual + self.rho * r_val)

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
        accum_steps = max(1, self.args.gradient_accumulation_steps)
        steps_per_epoch = max(1, len(train_dataloader) // accum_steps)
        num_epochs = max(1, int(self.args.num_train_epochs))
        total_outer_steps = num_epochs * steps_per_epoch
        effective_T = min(self.T, total_outer_steps)

        data_iter = iter(train_dataloader)

        logger.info("=" * 60)
        logger.info("Stage 3: Bilevel ALM optimization")
        logger.info("=" * 60)
        logger.info(f"  T={effective_T}, K={self.K}, accum={accum_steps}")
        logger.info(f"  Forget loss: {self.forget_loss_type} (beta={self.npo_beta})")
        logger.info(f"  ALM: ε={self.epsilon}, ρ={self.rho}, λ_init={self.lambda_init}")
        logger.info(f"  LR: outer={self.eta_theta}, inner={self.eta_in}")
        logger.info(f"  LoRA: r={self.lora_r}, alpha={self.lora_alpha_val}")
        if self.perta_lambda > 0:
            logger.info(f"  PerTA: λ={self.perta_lambda}, α={self.perta_alpha}")

        # GPU memory check
        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated() / 1e9
            logger.info(f"  GPU memory before bilevel: {mem:.1f} GB")

        history = []

        for t in range(effective_T):
            t_start = time.time()

            # Collect batches
            forget_batches = []
            retain_batches = []
            for _ in range(accum_steps):
                try:
                    combined_batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(train_dataloader)
                    combined_batch = next(data_iter)
                forget_batches.append(combined_batch["forget"])
                retain_batches.append(combined_batch["retain"])

            if not forget_batches:
                break

            # Inner loop: K steps of retain CE
            inner_losses = []
            retain_cycle = (
                retain_batches * ((self.K // len(retain_batches)) + 1)
            )[: self.K]
            for rb in retain_cycle:
                l_in = self.inner_step(rb, device)
                inner_losses.append(l_in)

            # Outer step: ALM(forget + retain constraint)
            L_fgt, L_ret, r = self.outer_step(
                forget_batches[0], retain_batches[0], device
            )

            dt = time.time() - t_start

            step_info = {
                "step": t,
                "L_fgt": L_fgt,
                "L_ret": L_ret,
                "r": r,
                "lambda": self.lambda_dual,
                "inner_loss_mean": sum(inner_losses) / len(inner_losses),
                "dt": dt,
            }
            history.append(step_info)

            if t % 5 == 0 or t == effective_T - 1:
                logger.info(
                    f"  [{t:3d}/{effective_T}] L_fgt={L_fgt:.4f} L_ret={L_ret:.4f} "
                    f"r={r:+.4f} λ={self.lambda_dual:.3f} "
                    f"inner={sum(inner_losses) / len(inner_losses):.4f} "
                    f"dt={dt:.1f}s"
                )

            # Early termination check: model collapsed
            if L_ret > 10.0:
                logger.warning(
                    f"  L_ret={L_ret:.1f} > 10.0 — model likely collapsed. "
                    f"Stopping early at step {t}."
                )
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
