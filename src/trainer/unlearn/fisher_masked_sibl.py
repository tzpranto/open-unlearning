"""
Fisher-Masked SIBL v2: PerTA + LoRA Bilevel with Gradual Unfreezing
=====================================================================

Three-stage architecture:
  Stage 1: PerTA weight surgery — breaks CE frontier (optional but recommended)
  Stage 2: LoRA r=64 adapters — higher capacity than r=16
  Stage 3: Bilevel ALM with gradual unfreezing

Bilevel structure (flipped from LoRA-BiAL):
- Inner loop (K steps): FORGET via NPO on forget data
- Outer loop (1 step): RETAIN via CE on retain data + ALM constraint
- Ref model = LoRA disabled (zero overhead)

Key innovations over LoRA-BiAL (Ze0):
- Higher LoRA rank (r=64 vs r=16): 4x more capacity for retain recovery
- Gradual unfreezing: start with last N layers, progressively open more
- LR decay as more layers open (prevents late-stage damage)
- Gradient accumulation (accum=4) for smoother updates
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


class FisherMaskedSIBL(UnlearnTrainer):
    """LoRA Bilevel with Gradual Unfreezing for Unlearning."""

    def __init__(
        self,
        *args,
        # PerTA parameters (stage 1)
        perta_lambda: float = 3.5,
        perta_alpha: float = 1.0,
        fisher_cache_path: str = "saves/unlearn/_perta_fisher_cache_News_ps_full.pt",
        pretrained_model_name: str = "meta-llama/Llama-2-7b-hf",
        # LoRA parameters
        lora_r: int = 64,
        lora_alpha: int = 128,
        lora_dropout: float = 0.0,
        lora_target_modules: Optional[list] = None,
        # Bilevel parameters
        T: int = -1,   # T <= 0 = epoch-based; T > 0 = fixed step count
        K: int = 3,
        eta_theta: float = 3e-5,   # outer LR (retain)
        eta_in: float = 2e-4,      # inner LR (forget)
        # ALM parameters
        epsilon: float = 0.70,
        rho: float = 0.1,
        lambda_init: float = 1.0,
        lambda_max: float = 5.0,
        # NPO parameters
        forget_loss_type: str = "npo",  # "npo" or "ga"
        npo_beta: float = 4.0,
        # Bilevel mode
        bilevel_mode: str = "forget_inner",  # "forget_inner" or "retain_inner"
        # Gradual unfreezing
        # Schedule: list of [step_fraction, n_layers_from_end]
        # e.g. [[0.0, 4], [0.25, 12], [0.6, 32]]
        # Step fractions are relative to total optimizer steps.
        # When gradual_unfreeze=False, all layers trainable from start.
        gradual_unfreeze: bool = True,
        unfreeze_schedule: Optional[list] = None,
        # LR decay phases: list of [step_fraction, lr_multiplier]
        # e.g. [[0.0, 1.0], [0.6, 0.5], [0.85, 0.1]]
        lr_schedule_phases: Optional[list] = None,
        # Checkpoints
        eval_at_steps: Optional[list] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # PerTA
        self.perta_lambda = perta_lambda
        self.perta_alpha = perta_alpha
        self.fisher_cache_path = fisher_cache_path
        self.pretrained_model_name = pretrained_model_name
        # LoRA
        self.lora_r = lora_r
        self.lora_alpha_val = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules or [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
        # Bilevel
        self.T = T
        self.K = K
        self.eta_theta = eta_theta
        self.eta_in = eta_in
        # ALM
        self.epsilon = epsilon
        self.rho = rho
        self.lambda_init = lambda_init
        self.lambda_max = lambda_max
        # NPO
        self.forget_loss_type = forget_loss_type
        self.npo_beta = npo_beta
        # Bilevel mode
        assert bilevel_mode in ("forget_inner", "retain_inner"), \
            f"bilevel_mode must be 'forget_inner' or 'retain_inner', got {bilevel_mode}"
        self.bilevel_mode = bilevel_mode
        # Unfreezing
        self.gradual_unfreeze = gradual_unfreeze
        self.unfreeze_schedule = unfreeze_schedule or [
            [0.0, 4], [0.25, 12], [0.6, 32]
        ]
        self.lr_schedule_phases = lr_schedule_phases or [
            [0.0, 1.0], [0.6, 0.5], [0.85, 0.1]
        ]
        # Checkpoints
        self.eval_at_steps = set(eval_at_steps) if eval_at_steps else set()
        # Runtime
        self.lambda_dual = float(lambda_init)
        self._current_unfreeze_idx = -1

    # ------------------------------------------------------------------
    # Stage 1: PerTA weight surgery
    # ------------------------------------------------------------------
    def _apply_perta(self):
        """Apply PerTA weight surgery in-place: θ = θ_target - λ*w*τ."""
        from transformers import AutoModelForCausalLM

        device = next(self.model.parameters()).device

        logger.info(f"Loading Fisher cache: {self.fisher_cache_path}")
        cached = torch.load(
            self.fisher_cache_path, map_location="cpu", weights_only=True
        )
        fisher_forget = cached["forget"]
        fisher_retain = cached["retain"]

        logger.info(f"Loading pretrained model: {self.pretrained_model_name}")
        pretrained = AutoModelForCausalLM.from_pretrained(
            self.pretrained_model_name,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
        )
        pretrained_sd = pretrained.state_dict()
        del pretrained

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

        del target_sd, new_sd, pretrained_sd, fisher_forget, fisher_retain, cached
        torch.cuda.empty_cache()

        logger.info(
            f"PerTA applied: λ={self.perta_lambda:.2f}, α={self.perta_alpha:.1f}, "
            f"mean_w={total_w / max(n_w, 1):.4f}"
        )

    # ------------------------------------------------------------------
    # LoRA setup
    # ------------------------------------------------------------------
    def _wrap_with_lora(self):
        """Add LoRA adapters to all target modules."""
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
    # Layer unfreezing
    # ------------------------------------------------------------------
    def _get_layer_id(self, param_name):
        """Extract layer index from parameter name. Returns None if not a layer param."""
        # Matches patterns like 'base_model.model.model.layers.30.self_attn.q_proj.lora_A...'
        parts = param_name.split(".")
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts):
                try:
                    return int(parts[i + 1])
                except ValueError:
                    pass
        return None

    def _count_model_layers(self):
        """Count total transformer layers."""
        max_layer = -1
        for name, _ in self.model.named_parameters():
            lid = self._get_layer_id(name)
            if lid is not None and lid > max_layer:
                max_layer = lid
        return max_layer + 1 if max_layer >= 0 else 32  # fallback

    def _apply_unfreeze(self, n_layers_from_end, total_layers):
        """Freeze/unfreeze LoRA params based on layer position."""
        cutoff = total_layers - n_layers_from_end  # layers >= cutoff are trainable
        n_frozen = 0
        n_trainable = 0
        for name, param in self.model.named_parameters():
            if "lora_" not in name:
                continue  # skip base model params (already frozen by PEFT)
            lid = self._get_layer_id(name)
            if lid is not None and lid < cutoff:
                param.requires_grad = False
                n_frozen += param.numel()
            else:
                param.requires_grad = True
                n_trainable += param.numel()
        logger.info(
            f"  Unfreeze: layers >= {cutoff}/{total_layers} "
            f"({n_layers_from_end} from end), "
            f"trainable={n_trainable:,}, frozen={n_frozen:,}"
        )
        return n_trainable

    def _check_unfreeze(self, step, max_steps, total_layers):
        """Check if we need to unfreeze more layers at this step."""
        if not self.gradual_unfreeze:
            return
        frac = step / max(max_steps, 1)
        # Find which phase we should be in
        target_idx = 0
        for i, (phase_frac, _) in enumerate(self.unfreeze_schedule):
            if frac >= phase_frac:
                target_idx = i
        if target_idx != self._current_unfreeze_idx:
            self._current_unfreeze_idx = target_idx
            n_layers = self.unfreeze_schedule[target_idx][1]
            self._apply_unfreeze(n_layers, total_layers)

    def _get_lr_multiplier(self, step, max_steps):
        """Get current LR multiplier from schedule."""
        frac = step / max(max_steps, 1)
        mult = 1.0
        for phase_frac, phase_mult in self.lr_schedule_phases:
            if frac >= phase_frac:
                mult = phase_mult
        return mult

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------
    def _compute_ce_loss(self, batch, device):
        """Standard cross-entropy loss."""
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch.get("labels", input_ids).to(device)
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        return outputs.loss

    def _compute_npo_loss(self, batch, device):
        """NPO loss. Ref = LoRA disabled (theta_target, zero overhead)."""
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

        # Ref forward: disable LoRA (base = theta_target)
        self.model.disable_adapter_layers()
        with torch.no_grad():
            ref_logits = self.model(**inputs).logits[..., :-1, :].contiguous()
            ref_nll = loss_fn(
                ref_logits.transpose(-1, -2), shifted_labels
            ).sum(dim=-1)
        self.model.enable_adapter_layers()

        # Model forward (with LoRA)
        model_logits = self.model(**inputs).logits[..., :-1, :].contiguous()
        model_nll = loss_fn(
            model_logits.transpose(-1, -2), shifted_labels
        ).sum(dim=-1)

        # NPO: push model_nll >> ref_nll
        log_ratio = -(model_nll - ref_nll)
        loss = -2 / self.npo_beta * F.logsigmoid(
            self.npo_beta * (-log_ratio)
        ).mean()
        return loss

    def _compute_forget_loss(self, batch, device):
        """Dispatch to configured forget loss."""
        if self.forget_loss_type == "npo":
            return self._compute_npo_loss(batch, device)
        elif self.forget_loss_type == "ga":
            return -self._compute_ce_loss(batch, device)
        else:
            raise ValueError(f"Unknown forget_loss_type: {self.forget_loss_type}")

    # ------------------------------------------------------------------
    # Inner step: FORGET (NPO on forget data, K steps)
    # ------------------------------------------------------------------
    def inner_step(self, forget_batch, device):
        """One inner step: NPO on forget data."""
        self.model.enable_adapter_layers()
        self.model.train()
        self._inner_opt.zero_grad()

        loss = self._compute_forget_loss(forget_batch, device)
        loss.backward()
        self._inner_opt.step()
        return loss.item()

    # ------------------------------------------------------------------
    # Outer step: RETAIN (CE on retain data + ALM, 1 step)
    # ------------------------------------------------------------------
    def outer_step(self, forget_batch, retain_batch, device):
        """One outer step: CE on retain + ALM constraint."""
        self.model.enable_adapter_layers()
        self.model.train()
        self._outer_opt.zero_grad()

        # Retain CE
        L_ret = self._compute_ce_loss(retain_batch, device)

        # ALM
        r_val = max(0.0, L_ret.item() - self.epsilon)
        retain_coeff = 1.0 + self.lambda_dual + self.rho * r_val
        L_total = retain_coeff * L_ret
        L_total.backward()
        self._outer_opt.step()

        # Dual update
        self.lambda_dual = max(0.0, self.lambda_dual + self.rho * r_val)
        if self.lambda_max > 0:
            self.lambda_dual = min(self.lambda_dual, self.lambda_max)

        # Check forget quality (no grad)
        with torch.no_grad():
            L_fgt = self._compute_forget_loss(forget_batch, device).item()

        return L_fgt, L_ret.item(), r_val

    # ------------------------------------------------------------------
    # Checkpoint saving (merge LoRA → clean keys → save → unmerge)
    # ------------------------------------------------------------------
    def _save_checkpoint(self, step, history):
        """Save intermediate checkpoint with merged LoRA weights."""
        ckpt_dir = os.path.join(self.args.output_dir, f"step-{step}")
        logger.info(f"  Saving checkpoint at step {step}...")
        os.makedirs(ckpt_dir, exist_ok=True)
        self.model.eval()

        # Merge LoRA into base (reversible)
        self.model.merge_adapter()
        peft_sd = self.model.base_model.model.state_dict()
        clean_sd = {}
        for key, val in peft_sd.items():
            clean_key = key.replace(".base_layer", "")
            if "lora_" in clean_key:
                continue
            clean_sd[clean_key] = val
        self.model.base_model.model.save_pretrained(
            ckpt_dir, state_dict=clean_sd
        )
        self.model.unmerge_adapter()
        self.model.train()

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(ckpt_dir)
        with open(os.path.join(ckpt_dir, "fisher_sibl_history.json"), "w") as f:
            json.dump(history, f, indent=2)
        logger.info(f"  Checkpoint step-{step} saved.")

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------
    def train(self):
        """Main loop: PerTA → LoRA → optional unfreeze → bilevel ALM."""
        device = self.args.device

        # Stage 1: PerTA weight surgery
        if self.perta_lambda > 0:
            logger.info("=" * 60)
            logger.info("Stage 1: PerTA weight surgery")
            logger.info("=" * 60)
            self._apply_perta()
        else:
            logger.info("Skipping PerTA (lambda=0)")

        # Stage 2: LoRA
        logger.info("=" * 60)
        logger.info("Stage 2: LoRA adapters")
        logger.info("=" * 60)
        self._wrap_with_lora()
        total_layers = self._count_model_layers()
        logger.info(f"  Model has {total_layers} transformer layers")

        # Gradient checkpointing
        if getattr(self.args, "gradient_checkpointing", False):
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        # Data
        train_dataloader = self.get_train_dataloader()
        batch_size = self.args.per_device_train_batch_size
        accum_steps = max(1, self.args.gradient_accumulation_steps)
        steps_per_epoch = len(train_dataloader)
        num_epochs = max(1, int(self.args.num_train_epochs))

        # Effective optimizer steps (accounting for accumulation)
        batches_per_epoch = steps_per_epoch
        opt_steps_per_epoch = batches_per_epoch // accum_steps
        if self.T > 0:
            max_opt_steps = self.T
        else:
            max_opt_steps = num_epochs * opt_steps_per_epoch

        # Initial unfreezing
        if self.gradual_unfreeze:
            # Freeze all LoRA params first
            for name, param in self.model.named_parameters():
                if "lora_" in name:
                    param.requires_grad = False
            # Apply first phase
            self._check_unfreeze(0, max_opt_steps, total_layers)
        else:
            n_trainable = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
            logger.info(f"  All LoRA params trainable: {n_trainable:,}")

        # Create Adam optimizers for ALL LoRA params (frozen ones get no grad → skipped)
        all_lora_params = [p for n, p in self.model.named_parameters() if "lora_" in n]
        self._inner_opt = torch.optim.Adam(all_lora_params, lr=self.eta_in)
        self._outer_opt = torch.optim.Adam(all_lora_params, lr=self.eta_theta)
        total_lora = sum(p.numel() for p in all_lora_params)
        logger.info(
            f"  Adam optimizers: inner lr={self.eta_in}, outer lr={self.eta_theta}, "
            f"total LoRA params={total_lora:,}"
        )

        # Logging
        retain_inner = (self.bilevel_mode == "retain_inner")

        logger.info("=" * 60)
        logger.info(f"Stage 3: Bilevel ALM ({self.bilevel_mode})")
        logger.info("=" * 60)
        logger.info(f"  Mode: {'epoch-based' if self.T <= 0 else f'fixed T={self.T}'}")
        logger.info(f"  Epochs={num_epochs}, batches/epoch={batches_per_epoch}, "
                     f"accum={accum_steps}, opt_steps={max_opt_steps}")
        if retain_inner:
            logger.info(f"  K={self.K} inner RETAIN steps per outer FORGET step")
        else:
            logger.info(f"  K={self.K} inner FORGET steps per outer RETAIN step")
        logger.info(f"  Forget: {self.forget_loss_type} (beta={self.npo_beta})")
        logger.info(f"  ALM: eps={self.epsilon}, rho={self.rho}, "
                     f"lam_init={self.lambda_init}, lam_max={self.lambda_max}")
        logger.info(f"  LoRA: r={self.lora_r}, alpha={self.lora_alpha_val}")
        if self.perta_lambda > 0:
            logger.info(f"  PerTA: λ={self.perta_lambda}, α={self.perta_alpha}")
        if self.gradual_unfreeze:
            logger.info(f"  Unfreeze schedule: {self.unfreeze_schedule}")
            logger.info(f"  LR decay phases: {self.lr_schedule_phases}")
        else:
            logger.info(f"  Unfreezing: all layers from start")

        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated() / 1e9
            logger.info(f"  GPU memory: {mem:.1f} GB")

        history = []
        opt_step = 0
        batch_buffer = []

        for epoch in range(num_epochs):
            epoch_start = time.time()
            epoch_fgt = []
            epoch_ret = []
            epoch_inner = []

            for batch_idx, combined_batch in enumerate(train_dataloader):
                if opt_step >= max_opt_steps:
                    break

                batch_buffer.append(combined_batch)

                # Wait until we have enough batches for accumulation
                if len(batch_buffer) < accum_steps:
                    continue

                t_start = time.time()

                # Check unfreezing schedule
                self._check_unfreeze(opt_step, max_opt_steps, total_layers)

                # Apply LR multiplier
                lr_mult = self._get_lr_multiplier(opt_step, max_opt_steps)
                for pg in self._inner_opt.param_groups:
                    pg["lr"] = self.eta_in * lr_mult
                for pg in self._outer_opt.param_groups:
                    pg["lr"] = self.eta_theta * lr_mult

                if retain_inner:
                    # --- Ze0-style: inner=RETAIN(CE+ALM), outer=FORGET(NPO) ---

                    # Inner loop: K steps of RETAIN + ALM
                    inner_losses = []
                    for _k in range(self.K):
                        self._inner_opt.zero_grad()
                        accum_loss = 0.0
                        for buf_batch in batch_buffer:
                            L_ret_i = self._compute_ce_loss(
                                buf_batch["retain"], device
                            )
                            r_val_i = max(0.0, L_ret_i.item() - self.epsilon)
                            coeff = 1.0 + self.lambda_dual + self.rho * r_val_i
                            L_total_i = coeff * L_ret_i
                            (L_total_i / accum_steps).backward()
                            accum_loss += L_ret_i.item() / accum_steps
                        self._inner_opt.step()
                        inner_losses.append(accum_loss)

                    # Outer step: 1 step of FORGET (NPO, pure)
                    self._outer_opt.zero_grad()
                    accum_fgt_loss = 0.0
                    for buf_batch in batch_buffer:
                        L_fgt_i = self._compute_forget_loss(
                            buf_batch["forget"], device
                        )
                        (L_fgt_i / accum_steps).backward()
                        accum_fgt_loss += L_fgt_i.item() / accum_steps
                    self._outer_opt.step()

                    L_fgt = accum_fgt_loss
                    accum_ret_loss = inner_losses[-1]

                else:
                    # --- Flipped: inner=FORGET(NPO), outer=RETAIN(CE+ALM) ---

                    # Inner loop: K steps of FORGET
                    inner_losses = []
                    for _k in range(self.K):
                        self._inner_opt.zero_grad()
                        accum_loss = 0.0
                        for buf_batch in batch_buffer:
                            loss = self._compute_forget_loss(
                                buf_batch["forget"], device
                            )
                            (loss / accum_steps).backward()
                            accum_loss += loss.item() / accum_steps
                        self._inner_opt.step()
                        inner_losses.append(accum_loss)

                    # Outer step: 1 step of RETAIN + ALM
                    self._outer_opt.zero_grad()
                    accum_ret_loss = 0.0
                    for buf_batch in batch_buffer:
                        L_ret_i = self._compute_ce_loss(
                            buf_batch["retain"], device
                        )
                        r_val_i = max(0.0, L_ret_i.item() - self.epsilon)
                        coeff = 1.0 + self.lambda_dual + self.rho * r_val_i
                        L_total_i = coeff * L_ret_i
                        (L_total_i / accum_steps).backward()
                        accum_ret_loss += L_ret_i.item() / accum_steps
                    self._outer_opt.step()

                    # Check forget quality (no grad)
                    with torch.no_grad():
                        L_fgt = self._compute_forget_loss(
                            batch_buffer[-1]["forget"], device
                        ).item()

                # Dual update (always on retain loss)
                r_val = max(0.0, accum_ret_loss - self.epsilon)
                self.lambda_dual = max(0.0, self.lambda_dual + self.rho * r_val)
                if self.lambda_max > 0:
                    self.lambda_dual = min(self.lambda_dual, self.lambda_max)

                dt = time.time() - t_start

                step_info = {
                    "step": opt_step,
                    "epoch": epoch,
                    "L_fgt": L_fgt,
                    "L_ret": accum_ret_loss,
                    "r": r_val,
                    "lambda": self.lambda_dual,
                    "inner_loss_mean": sum(inner_losses) / len(inner_losses),
                    "lr_mult": lr_mult,
                    "dt": dt,
                }
                history.append(step_info)
                epoch_fgt.append(L_fgt)
                epoch_ret.append(accum_ret_loss)
                epoch_inner.extend(inner_losses)

                # Logging
                log_every = max(1, max_opt_steps // 20)
                if opt_step % log_every == 0 or opt_step == max_opt_steps - 1:
                    logger.info(
                        f"  [{opt_step:4d}/{max_opt_steps}|e{epoch+1}] "
                        f"L_fgt={L_fgt:.4f} L_ret={accum_ret_loss:.4f} "
                        f"r={r_val:+.4f} lam={self.lambda_dual:.3f} "
                        f"inner={sum(inner_losses)/len(inner_losses):.4f} "
                        f"lr_m={lr_mult:.2f} dt={dt:.1f}s"
                    )

                opt_step += 1
                batch_buffer = []

                # Intermediate checkpoint
                if opt_step in self.eval_at_steps:
                    self._save_checkpoint(opt_step, history)

                # Early stop
                if accum_ret_loss > 10.0:
                    logger.warning(f"  L_ret={accum_ret_loss:.1f} > 10 — collapsed.")
                    break

            # Epoch summary
            if epoch_fgt:
                logger.info(
                    f"  Epoch {epoch+1}/{num_epochs}: "
                    f"{opt_step} opt steps, "
                    f"mean_fgt={sum(epoch_fgt)/len(epoch_fgt):.4f}, "
                    f"mean_ret={sum(epoch_ret)/len(epoch_ret):.4f}, "
                    f"dt={time.time()-epoch_start:.0f}s"
                )

            if opt_step >= max_opt_steps:
                break
            if accum_ret_loss > 10.0:
                break

        # Merge LoRA and save final model
        logger.info("Merging LoRA adapters into base model...")
        self.model = self.model.merge_and_unload()

        output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        history_path = os.path.join(output_dir, "fisher_sibl_history.json")
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        logger.info(f"Model saved to {output_dir}")
        if history:
            logger.info(
                f"Final: lam={self.lambda_dual:.3f}, "
                f"L_fgt={history[-1]['L_fgt']:.4f}, "
                f"L_ret={history[-1]['L_ret']:.4f}"
            )

        self.evaluate()
