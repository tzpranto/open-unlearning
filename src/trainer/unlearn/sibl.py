"""
S-BiAL: Sparse Bilevel Augmented Lagrangian for Machine Unlearning
===================================================================

Implements Algorithm 1 from the S-BiAL paper.  Bilevel optimization with
mask-based sparsity, Augmented Lagrangian constraint on retain quality,
and optional implicit differentiation via CG or Truncated Neumann series.

Bilevel structure:
  Inner loop (K steps): retain CE + regularization on active set (mask)
  Outer loop (1 step):  ALM = L_forget + λ·L_retain + ρ/2·(L_retain - ε)²
  Dual update:          λ ← max(0, λ + ρ·(L_retain - ε))

Optional implicit correction approximates H⁻¹·v (CG or Neumann) on active
coords, then adjusts the outer gradient for the inner solver's response.
"""

import copy
import torch
import time
import logging
import os
import json

import numpy as np

from trainer.unlearn.base import UnlearnTrainer
from trainer.sparsity import SparsityManager
from trainer.unlearn.loss_functions import (
    get_forget_loss_fn,
    get_regularization_fn,
    AVAILABLE_FORGET_LOSSES,
    AVAILABLE_REGULARIZATIONS,
)

logger = logging.getLogger(__name__)

try:
    from scipy.sparse.linalg import LinearOperator, cg
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available, will use PyTorch CG implementation")


class SIBL(UnlearnTrainer):
    """Sparse Bilevel Augmented Lagrangian for Unlearning."""

    def __init__(
        self,
        *args,
        # Sparsity
        use_sparsity: bool = True,
        sparsity: float = 0.9,
        sparsity_method: str = "layerwise_magnitude",
        # Bilevel
        T: int = 20,
        K: int = 10,
        eta_theta: float = 1e-4,
        eta_in: float = 1e-4,
        # ALM
        epsilon: float = 0.1,
        rho: float = 1.0,
        lambda_init: float = 0.0,
        # Regularization
        gamma: float = 1e-4,
        regularization_type: str = "l1",
        # Implicit correction
        use_implicit: bool = False,
        implicit_solver: str = "neumann",  # "neumann" or "cg"
        cg_iters: int = 10,
        cg_tol: float = 1e-3,
        cg_damping: float = 0.0,
        # Neumann solver parameters
        neumann_steps: int = 4,
        neumann_mu: float = 0.01,
        neumann_alpha: float = 0.1,
        neumann_max_growth: float = 10.0,
        # Forget loss
        forget_loss_type: str = "logit_margin",
        npo_beta: float = 1.0,
        # Checkpointing
        checkpoint_every_steps: int = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # Sparsity
        self.use_sparsity = use_sparsity
        self.sparsity = sparsity
        self.sparsity_method = sparsity_method

        # Bilevel
        self.T = T
        self.K = K
        self.eta_theta = eta_theta
        self.eta_in = eta_in

        # ALM
        self.epsilon = epsilon
        self.rho = rho
        self.lambda_dual = float(lambda_init)

        # Regularization
        self.gamma = gamma
        self.regularization_type = regularization_type
        if regularization_type not in AVAILABLE_REGULARIZATIONS:
            raise ValueError(f"Unknown regularization_type: {regularization_type}")
        self._regularization_fn = get_regularization_fn(regularization_type)

        # Implicit
        self.use_implicit = use_implicit
        self.implicit_solver = implicit_solver.lower()
        assert self.implicit_solver in ("neumann", "cg"), \
            f"implicit_solver must be 'neumann' or 'cg', got {implicit_solver}"
        self.cg_iters = cg_iters
        self.cg_tol = cg_tol
        self.cg_damping = cg_damping
        self.neumann_steps = neumann_steps
        self.neumann_mu = neumann_mu
        self.neumann_alpha = neumann_alpha
        self.neumann_max_growth = neumann_max_growth

        # Forget loss
        self.forget_loss_type = forget_loss_type
        self.npo_beta = npo_beta
        if forget_loss_type not in AVAILABLE_FORGET_LOSSES:
            raise ValueError(f"Unknown forget_loss_type: {forget_loss_type}")
        self._forget_loss_fn = get_forget_loss_fn(forget_loss_type)

        # Checkpointing
        self.checkpoint_every_steps = checkpoint_every_steps

        # Runtime state
        self.mask_dict = None
        self.ref_model = None
        self.history = {
            "iter": [], "L_forget": [], "L_retain": [],
            "residual": [], "lambda": [], "time": [],
        }

        logger.info(
            f"SIBL: forget={forget_loss_type}, reg={regularization_type}, "
            f"sparsity={sparsity if use_sparsity else 'off'}"
        )

    # ------------------------------------------------------------------
    # Mask initialization
    # ------------------------------------------------------------------
    def _initialize_mask(self):
        if self.use_sparsity:
            logger.info(
                f"Creating sparsity mask: {self.sparsity_method} "
                f"at {self.sparsity:.0%} sparsity"
            )
            self.mask_dict = SparsityManager.create_mask(
                self.model,
                sparsity=self.sparsity,
                method=self.sparsity_method,
                device=self.args.device,
            )
        else:
            logger.info("No sparsity — using full model")
            self.mask_dict = {
                name: torch.ones_like(param.data).to(self.args.device)
                for name, param in self.model.named_parameters()
            }

    # ------------------------------------------------------------------
    # Reference model (for NPO)
    # ------------------------------------------------------------------
    def _prepare_ref_model(self):
        if self.ref_model is not None:
            return
        device = next(self.model.parameters()).device
        logger.info("Creating frozen reference model (deepcopy)...")
        self.model.to("cpu")
        torch.cuda.empty_cache()
        self.ref_model = copy.deepcopy(self.model)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False
        self.ref_model.to(device)
        self.model.to(device)

    # ------------------------------------------------------------------
    # Loss functions
    # ------------------------------------------------------------------
    def compute_forget_loss(self, batch):
        if self.forget_loss_type == "npo":
            self._prepare_ref_model()
        return self._forget_loss_fn(
            self.model, batch, self.args.device,
            ref_model=self.ref_model, beta=self.npo_beta,
        )

    def compute_retain_loss(self, batch):
        input_ids = batch["input_ids"].to(self.args.device)
        attention_mask = batch["attention_mask"].to(self.args.device)
        labels = batch.get("labels", input_ids).to(self.args.device)
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        return outputs.loss

    def compute_regularizer(self):
        return self._regularization_fn(
            self.model, self.mask_dict, self.gamma, self.args.device
        )

    # ------------------------------------------------------------------
    # Inner step: retain CE + regularization, masked SGD
    # ------------------------------------------------------------------
    def inner_step(self, batch):
        self.model.train()
        input_ids = batch["input_ids"].to(self.args.device)
        attention_mask = batch["attention_mask"].to(self.args.device)
        labels = batch.get("labels", input_ids).to(self.args.device)

        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        loss = outputs.loss + self.compute_regularizer()
        loss.backward()

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.grad is not None and name in self.mask_dict:
                    binary_mask = (self.mask_dict[name] > 0).float()
                    param.data.sub_(self.eta_in * param.grad * binary_mask)
                param.grad = None

        return loss.item()

    def inner_loop(self, retain_dataloader):
        """K inner steps, each with a fresh retain minibatch."""
        if not hasattr(self, '_retain_iter'):
            self._retain_iter = iter(retain_dataloader)
        for _ in range(self.K):
            try:
                batch = next(self._retain_iter)
            except StopIteration:
                self._retain_iter = iter(retain_dataloader)
                batch = next(self._retain_iter)
            # The collator wraps everything as {"forget": ..., "retain": ...}
            # so extract the retain portion if nested
            if isinstance(batch, dict) and "retain" in batch:
                batch = batch["retain"]
            self.inner_step(batch)

    # ------------------------------------------------------------------
    # HVP and CG for implicit correction
    # ------------------------------------------------------------------
    def flatten_params(self, params_list):
        return torch.cat([p.reshape(-1) for p in params_list])

    def unflatten_params(self, flat_vec, params_list):
        parts, offset = [], 0
        for p in params_list:
            n = p.numel()
            parts.append(flat_vec[offset : offset + n].reshape(p.shape))
            offset += n
        return parts

    def flatten_mask(self):
        masks = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name in self.mask_dict:
                masks.append(self.mask_dict[name].reshape(-1))
            else:
                masks.append(torch.ones(param.numel(), device=self.args.device))
        return torch.cat(masks)

    def compute_hvp_fd(self, loss_fn, params, v, eps=1e-3):
        """Finite-difference HVP: H·v ≈ (∇(θ+εv̂) - ∇(θ-εv̂)) / (2ε) · ||v||."""
        v_norm = v.norm().clamp(min=1e-12)
        v_unit = v / v_norm
        v_parts = self.unflatten_params(v_unit, params)

        # +ε perturbation
        with torch.no_grad():
            for p, dv in zip(params, v_parts):
                p.data.add_(eps * dv.to(p.dtype))
        loss_p = loss_fn()
        grads_p = [
            (g.detach().clone() if g is not None else torch.zeros_like(p))
            for g, p in zip(torch.autograd.grad(loss_p, params, allow_unused=True), params)
        ]
        del loss_p

        # -ε perturbation (from +ε → -ε = subtract 2ε)
        with torch.no_grad():
            for p, dv in zip(params, v_parts):
                p.data.sub_(2.0 * eps * dv.to(p.dtype))
        loss_m = loss_fn()
        grads_m = [
            (g.detach().clone() if g is not None else torch.zeros_like(p))
            for g, p in zip(torch.autograd.grad(loss_m, params, allow_unused=True), params)
        ]
        del loss_m

        # Restore
        with torch.no_grad():
            for p, dv in zip(params, v_parts):
                p.data.add_(eps * dv.to(p.dtype))

        hvp_flat = torch.cat([(gp - gm).reshape(-1) / (2.0 * eps) for gp, gm in zip(grads_p, grads_m)])
        return hvp_flat * v_norm.item()

    def conjugate_gradient(self, hvp_func, b):
        """Solve H·x = b via CG."""
        if SCIPY_AVAILABLE:
            return self._cg_scipy(hvp_func, b)
        return self._cg_torch(hvp_func, b)

    def _cg_scipy(self, hvp_func, b):
        device, dtype = b.device, b.dtype
        n = b.numel()
        b_np = b.detach().cpu().float().numpy()

        def matvec(v):
            vt = torch.from_numpy(v).to(device=device, dtype=dtype)
            return hvp_func(vt).detach().cpu().float().numpy()

        A = LinearOperator((n, n), matvec=matvec, dtype=b_np.dtype)
        x_np, info = cg(A, b_np, maxiter=self.cg_iters, atol=self.cg_tol, rtol=0)
        return torch.from_numpy(x_np).to(device=device, dtype=dtype)

    def _cg_torch(self, hvp_func, b):
        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        rs_old = (r * r).sum()

        for i in range(self.cg_iters):
            Ap = hvp_func(p)
            alpha = rs_old / ((p * Ap).sum() + 1e-10)
            x = x + alpha * p
            r = r - alpha * Ap
            rs_new = (r * r).sum()
            if rs_new.sqrt() < self.cg_tol:
                break
            p = r + (rs_new / (rs_old + 1e-10)) * p
            rs_old = rs_new

        return x

    # ------------------------------------------------------------------
    # Truncated Neumann implicit correction
    # ------------------------------------------------------------------
    def _neumann_correction(self, inner_hvp_fn, outer_hvp_fn, v, mask_flat):
        """Approximate h ≈ (H_inner + μI)⁻¹ v via Neumann series,
        then compute g_corr = v - H_outer(h).

        Richardson iteration: h_{k+1} = h_k + α·(v - H̃·h_k)
        where H̃ = mask ⊙ H_inner(mask ⊙ ·) + μ·(mask ⊙ ·).

        Returns corrected gradient or original v on failure.
        """
        mu = self.neumann_mu
        alpha = self.neumann_alpha

        def H_tilde(x):
            x_act = x * mask_flat
            Hv = inner_hvp_fn(x_act)
            return Hv * mask_flat + mu * x_act

        # Richardson iteration
        h = torch.zeros_like(v)
        for _ in range(self.neumann_steps):
            residual = v - H_tilde(h)
            if not torch.isfinite(residual).all():
                logger.info("Neumann: non-finite residual, falling back to uncorrected gradient")
                return v
            h = h + alpha * residual
            if not torch.isfinite(h).all():
                logger.info("Neumann: non-finite h, falling back to uncorrected gradient")
                return v

        # Outer HVP for final correction
        h_act = h * mask_flat
        c = outer_hvp_fn(h_act) * mask_flat
        g_corr = v - c

        # Safety: reject if correction explodes
        v_norm = v.norm().clamp(min=1e-12).item()
        g_corr_norm = g_corr.norm().item()
        if g_corr_norm > self.neumann_max_growth * v_norm:
            logger.info(
                f"Neumann: correction exploded ({g_corr_norm:.4f} > "
                f"{self.neumann_max_growth}×{v_norm:.4f}), falling back"
            )
            return v

        logger.info(
            f"Neumann: ||v||={v_norm:.4f} ||h||={h.norm().item():.4f} "
            f"||g_corr||={g_corr_norm:.4f} α={alpha:.4f}"
        )
        return g_corr

    # ------------------------------------------------------------------
    # Outer step: ALM gradient + optional implicit correction + masked update
    # ------------------------------------------------------------------
    def outer_step(self, forget_batch, retain_batch, outer_iter=None):
        self.model.train()

        # Compute ALM loss: L_fgt + λ·L_ret + ρ/2·(L_ret - ε)²
        L_fgt = self.compute_forget_loss(forget_batch)
        L_ret = self.compute_retain_loss(retain_batch)

        if not torch.isfinite(L_fgt) or not torch.isfinite(L_ret):
            logger.warning(f"Non-finite loss: L_fgt={L_fgt.item()}, L_ret={L_ret.item()}")
            return float("nan"), float("nan"), float("nan")

        r = L_ret - self.epsilon
        L_alm = L_fgt + self.lambda_dual * L_ret + 0.5 * self.rho * (r ** 2)
        L_alm.backward()

        # Collect gradients
        g_alm_dict = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                g_alm_dict[name] = param.grad.detach()
                param.grad = None
            else:
                g_alm_dict[name] = torch.zeros_like(param.data)

        # Optional implicit correction
        if self.use_implicit:
            def _inner_loss_fn():
                return self.compute_retain_loss(retain_batch) + self.compute_regularizer()

            def _alm_loss_fn():
                lf = self.compute_forget_loss(forget_batch)
                lr = self.compute_retain_loss(retain_batch)
                rt = lr - self.epsilon
                return lf + self.lambda_dual * lr + 0.5 * self.rho * (rt ** 2)

            params_list = [p for p in self.model.parameters() if p.requires_grad]
            mask_flat = self.flatten_mask()
            g_alm_flat = torch.cat([
                g_alm_dict[n].reshape(-1)
                for n, p in self.model.named_parameters() if p.requires_grad
            ])

            v = g_alm_flat * mask_flat

            def inner_hvp(x):
                return self.compute_hvp_fd(_inner_loss_fn, params_list, x)

            def outer_hvp(x):
                return self.compute_hvp_fd(_alm_loss_fn, params_list, x)

            if self.implicit_solver == "neumann":
                g_corrected_flat = self._neumann_correction(inner_hvp, outer_hvp, v, mask_flat)
            else:
                # CG solver: solve H·s = v on supp(m), then g_corr = v - H_ALM·s
                def hvp_func(vec):
                    vec_masked = vec * mask_flat
                    hvp = self.compute_hvp_fd(_inner_loss_fn, params_list, vec_masked)
                    return hvp * mask_flat + self.cg_damping * vec_masked

                h = self.conjugate_gradient(hvp_func, v)
                h_act = h * mask_flat
                correction = self.compute_hvp_fd(_alm_loss_fn, params_list, h_act) * mask_flat
                g_corrected_flat = v - correction

            corrected_parts = self.unflatten_params(g_corrected_flat, params_list)
            for (name, _), g_update in zip(
                ((n, p) for n, p in self.model.named_parameters() if p.requires_grad),
                corrected_parts,
            ):
                g_alm_dict[name] = g_update

        # Primal masked update: θ ← θ - η_θ · g · m
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.mask_dict and name in g_alm_dict:
                    param.data.sub_(self.eta_theta * g_alm_dict[name] * self.mask_dict[name])
                param.grad = None

        # Dual update: λ ← max(0, λ + ρ·r)
        r_val = r.item()
        self.lambda_dual = max(0.0, self.lambda_dual + self.rho * r_val)

        return L_fgt.item(), L_ret.item(), r_val

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------
    def train(self):
        # Initialize mask
        if self.mask_dict is None:
            self._initialize_mask()

        # Gradient checkpointing
        if getattr(self.args, "gradient_checkpointing", False):
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        # Data
        train_dataloader = self.get_train_dataloader()
        steps_per_epoch = len(train_dataloader)
        num_epochs = max(1, int(self.args.num_train_epochs))
        total_outer_steps = num_epochs * steps_per_epoch
        effective_T = min(self.T, total_outer_steps) if self.T < total_outer_steps else total_outer_steps

        logger.info(f"\nS-BiAL: {effective_T} outer steps "
                     f"({num_epochs} epochs × {steps_per_epoch} steps/epoch)")
        logger.info(f"  K={self.K} inner steps, ε={self.epsilon}, ρ={self.rho}")
        logger.info(f"  η_θ={self.eta_theta}, η_in={self.eta_in}, γ={self.gamma}")
        logger.info(f"  Forget: {self.forget_loss_type}, Reg: {self.regularization_type}")
        if self.use_implicit:
            if self.implicit_solver == "neumann":
                logger.info(f"  Implicit: neumann (steps={self.neumann_steps}, "
                             f"α={self.neumann_alpha}, μ={self.neumann_mu})")
            else:
                logger.info(f"  Implicit: CG (iters={self.cg_iters}, tol={self.cg_tol})")
        else:
            logger.info(f"  Implicit: off")

        data_iter = iter(train_dataloader)
        for t in range(effective_T):
            t_start = time.time()

            try:
                combined_batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_dataloader)
                combined_batch = next(data_iter)

            forget_batch = combined_batch["forget"]

            # Inner loop: K steps of retain CE + regularization (fresh batches each step)
            self.inner_loop(train_dataloader)

            # Fresh retain batch for outer step (independent of inner loop batches)
            try:
                outer_combined = next(data_iter)
            except StopIteration:
                data_iter = iter(train_dataloader)
                outer_combined = next(data_iter)
            retain_batch = outer_combined["retain"]

            # Outer step: ALM + optional implicit correction
            L_fgt, L_ret, r = self.outer_step(forget_batch, retain_batch, outer_iter=t)

            if not np.isfinite(L_fgt) or not np.isfinite(L_ret):
                logger.warning(f"Non-finite at step {t}, stopping.")
                break

            dt = time.time() - t_start
            self.history["iter"].append(t)
            self.history["L_forget"].append(L_fgt)
            self.history["L_retain"].append(L_ret)
            self.history["residual"].append(r)
            self.history["lambda"].append(self.lambda_dual)
            self.history["time"].append(dt)

            if t % max(1, effective_T // 20) == 0 or t == effective_T - 1:
                logger.info(
                    f"  [{t:3d}/{effective_T}] L_fgt={L_fgt:.4f} L_ret={L_ret:.4f} "
                    f"r={r:+.4f} λ={self.lambda_dual:.3f} dt={dt:.1f}s"
                )

            self.state.global_step = t + 1

            if self.checkpoint_every_steps > 0 and (t + 1) % self.checkpoint_every_steps == 0:
                ckpt_dir = os.path.join(self.args.output_dir, f"checkpoint-step-{t + 1}")
                os.makedirs(ckpt_dir, exist_ok=True)
                self.model.save_pretrained(ckpt_dir)
                logger.info(f"  Checkpoint: {ckpt_dir}")

        # Save final model + history
        os.makedirs(self.args.output_dir, exist_ok=True)
        self.model.save_pretrained(self.args.output_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(self.args.output_dir)

        history_path = os.path.join(self.args.output_dir, "sibl_history.json")
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)

        logger.info(f"Model saved to {self.args.output_dir}")
        if self.history["iter"]:
            logger.info(
                f"Final: λ={self.lambda_dual:.3f}, "
                f"L_fgt={self.history['L_forget'][-1]:.4f}, "
                f"L_ret={self.history['L_retain'][-1]:.4f}"
            )

        self.evaluate()
