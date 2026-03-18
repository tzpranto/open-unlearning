"""
SIBL (Sparse Bilevel Augmented Lagrangian) Unlearning Method
==============================================================

Implements the S-BiAL algorithm for machine unlearning with sparsity constraints.
Uses bilevel optimization with Augmented Lagrangian method and implicit differentiation.

Supports multiple forget loss functions and regularization methods:
- Forget losses: logit_margin (default), grad_ascent, grad_diff, npo, simnpo, pdu, rmu
- Regularization: l1 (default), l2, elastic_net, none
"""

import torch
import torch.nn as nn
import time
import copy
import logging
import os
import json
import re
from typing import Dict, Optional
import numpy as np
from trainer.unlearn.base import UnlearnTrainer
from trainer.sparsity import SparsityManager
from trainer.unlearn.loss_functions import (
    get_forget_loss_fn,
    get_regularization_fn,
    AVAILABLE_FORGET_LOSSES,
    AVAILABLE_REGULARIZATIONS
)

logger = logging.getLogger(__name__)

try:
    from scipy.sparse.linalg import LinearOperator, cg
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available, will use PyTorch CG implementation")


class SIBL(UnlearnTrainer):
    """Sparse Bilevel Augmented Lagrangian for TOFU Unlearning."""

    def __init__(
        self,
        *args,
        # S-BiAL specific parameters (keyword-only)
        use_sparsity: bool = True,
        sparsity: float = 0.9,
        sparsity_method: str = "layerwise_magnitude",
        epsilon: float = 0.1,  # Retain loss budget
        T: int = 20,  # Number of outer iterations
        K: int = 10,  # Number of inner iterations
        eta_theta: float = 1e-4,  # Outer learning rate
        eta_in: float = 1e-4,  # Inner learning rate
        rho: float = 1.0,  # Penalty parameter for AL
        gamma: float = 1e-4,  # Regularization coefficient
        use_implicit: bool = False,  # Use implicit differentiation
        implicit_solver: str = "neumann",  # "neumann" (Truncated Neumann) or "cg"
        implicit_blockwise: bool = False,  # Apply implicit correction layer-by-layer
        implicit_block_last_n_layers: int = 2,  # Number of last transformer layers for blockwise implicit
        implicit_block_include_attn: bool = True,  # Include self-attention params in blockwise implicit
        implicit_block_include_mlp: bool = True,  # Include MLP params in blockwise implicit
        cg_iters: int = 10,  # Conjugate gradient iterations
        cg_tol: float = 1e-3,  # CG tolerance
        cg_damping: float = 0.0,  # Hessian damping for CG stability
        # Truncated Neumann implicit correction (when implicit_solver == "neumann")
        neumann_steps: int = 4,  # J: number of Neumann steps (3–5 typical)
        neumann_mu: float = 0.01,  # μ: damping for (H + μI), stabilizes curvature
        neumann_alpha_default: float = 0.1,  # fallback α if probe off or fails
        neumann_alpha_min: float = 1e-6,  # clamp lower bound for α (stability)
        neumann_alpha_max: float = 1.0,  # clamp upper bound for α (stability)
        neumann_use_probe_alpha: bool = True,  # adapt α from curvature probe each step
        neumann_clip_norm: Optional[float] = None,  # clip v and g_corr; None = no clip
        neumann_max_growth_ratio: float = 10.0,  # fallback if correction explodes
        neumann_variant: str = "legacy",  # "legacy" or "richardson" (proper H^{-1} Neumann)
        # Debug controls for implicit-correction diagnostics
        debug_implicit: bool = False,
        debug_dir: Optional[str] = None,
        debug_save_arrays: bool = False,
        debug_stop_after_outer: Optional[int] = None,  # stop after this outer iter (inclusive)
        debug_condition_probes: int = 3,  # random Rayleigh probes for conditioning proxy
        # New configurable loss and regularization parameters
        forget_loss_type: str = "logit_margin",  # Type of forget loss function
        regularization_type: str = "l1",  # Type of regularization (l1, l2, elastic_net, none)
        # Loss-specific parameters
        npo_beta: float = 1.0,  # Beta for NPO loss
        simnpo_beta: float = 1.0,  # Beta for SimNPO loss
        simnpo_delta: float = 0.0,  # Delta offset for SimNPO loss
        elastic_net_l1_ratio: float = 0.5,  # L1 ratio for elastic net (0.5 = equal L1 and L2)
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        # Store configuration
        self.use_sparsity = use_sparsity
        self.sparsity = sparsity
        self.sparsity_method = sparsity_method
        self.epsilon = epsilon
        self.T = T
        self.K = K
        self.eta_theta = eta_theta
        self.eta_in = eta_in
        self.rho = rho
        self.gamma = gamma
        self.use_implicit = use_implicit
        self.implicit_solver = implicit_solver.lower()
        self.implicit_blockwise = implicit_blockwise
        self.implicit_block_last_n_layers = max(1, int(implicit_block_last_n_layers))
        self.implicit_block_include_attn = implicit_block_include_attn
        self.implicit_block_include_mlp = implicit_block_include_mlp
        self.cg_iters = cg_iters
        self.cg_tol = cg_tol
        self.cg_damping = cg_damping
        self.neumann_steps = neumann_steps
        self.neumann_mu = neumann_mu
        self.neumann_alpha_default = neumann_alpha_default
        self.neumann_alpha_min = neumann_alpha_min
        self.neumann_alpha_max = neumann_alpha_max
        self.neumann_use_probe_alpha = neumann_use_probe_alpha
        self.neumann_clip_norm = neumann_clip_norm
        self.neumann_max_growth_ratio = neumann_max_growth_ratio
        self.neumann_variant = neumann_variant.lower()
        if self.implicit_solver not in ("neumann", "cg"):
            raise ValueError(
                f"implicit_solver must be 'neumann' or 'cg', got {implicit_solver!r}"
            )
        if not self.implicit_block_include_attn and not self.implicit_block_include_mlp:
            raise ValueError(
                "At least one of implicit_block_include_attn or implicit_block_include_mlp must be True"
            )
        if self.neumann_variant not in ("legacy", "richardson"):
            raise ValueError(
                f"neumann_variant must be 'legacy' or 'richardson', got {neumann_variant!r}"
            )
        if self.neumann_alpha_min <= 0 or self.neumann_alpha_max <= 0:
            raise ValueError("neumann_alpha_min and neumann_alpha_max must be > 0")
        if self.neumann_alpha_min >= self.neumann_alpha_max:
            raise ValueError("neumann_alpha_min must be < neumann_alpha_max")

        self.debug_implicit = debug_implicit
        self.debug_save_arrays = debug_save_arrays
        self.debug_stop_after_outer = debug_stop_after_outer
        self.debug_condition_probes = max(1, int(debug_condition_probes))
        self.debug_dir = debug_dir or os.path.join(self.args.output_dir, "debug")
        self._debug_records = []

        # Store loss and regularization configuration
        self.forget_loss_type = forget_loss_type
        self.regularization_type = regularization_type
        self.npo_beta = npo_beta
        self.simnpo_beta = simnpo_beta
        self.simnpo_delta = simnpo_delta
        self.elastic_net_l1_ratio = elastic_net_l1_ratio

        # Validate loss and regularization types
        if forget_loss_type not in AVAILABLE_FORGET_LOSSES:
            raise ValueError(
                f"Unknown forget_loss_type: {forget_loss_type}. "
                f"Available: {AVAILABLE_FORGET_LOSSES}"
            )
        if regularization_type not in AVAILABLE_REGULARIZATIONS:
            raise ValueError(
                f"Unknown regularization_type: {regularization_type}. "
                f"Available: {AVAILABLE_REGULARIZATIONS}"
            )

        # Get the loss and regularization functions
        self._forget_loss_fn = get_forget_loss_fn(forget_loss_type)
        self._regularization_fn = get_regularization_fn(regularization_type)

        # Initialize reference model for NPO if needed
        self.ref_model = None
        if forget_loss_type == "npo":
            logger.info("NPO loss requires reference model - will be initialized on first use")

        # Initialize dual variable
        self.lambda_dual = 0.0

        # History tracking
        self.history = {
            'iter': [],
            'L_forget': [],
            'L_retain': [],
            'residual': [],
            'lambda': [],
            'time': []
        }

        # Initialize sparsity mask (will be created when training starts)
        self.mask_dict = None

        # Log configuration
        logger.info(f"SIBL configured with forget_loss_type={forget_loss_type}, "
                   f"regularization_type={regularization_type}")
        if self.use_implicit:
            logger.info(f"Neumann variant: {self.neumann_variant}")
        if self.debug_implicit:
            os.makedirs(self.debug_dir, exist_ok=True)
            logger.info(f"Implicit debug enabled. Outputs: {self.debug_dir}")

    def _save_debug_array(self, outer_iter, name, vec):
        """Persist a 1D tensor as numpy for offline diagnostics."""
        if not self.debug_implicit or not self.debug_save_arrays:
            return
        os.makedirs(self.debug_dir, exist_ok=True)
        npy_path = os.path.join(self.debug_dir, f"outer_{outer_iter:04d}_{name}.npy")
        np.save(npy_path, vec.detach().float().cpu().numpy())

    def _log_implicit_debug_record(self, record: Dict):
        """Append structured debug record and keep JSONL evidence."""
        if not self.debug_implicit:
            return
        os.makedirs(self.debug_dir, exist_ok=True)
        self._debug_records.append(record)
        jsonl_path = os.path.join(self.debug_dir, "implicit_debug.jsonl")
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _rayleigh_condition_proxy(self, hvp_func, v, mask_flat):
        """
        Lightweight conditioning proxy from random Rayleigh quotients:
        reports min/max/ratio and number of non-positive curvature probes.
        """
        min_rayleigh = float("inf")
        max_rayleigh = 0.0
        nonpos = 0
        valid = 0
        for _ in range(self.debug_condition_probes):
            q = torch.randn_like(v) * mask_flat
            qn = q.norm().clamp(min=1e-12)
            q = q / qn
            Hq = hvp_func(q)
            rq = (q * Hq).sum().item()
            if not np.isfinite(rq):
                continue
            valid += 1
            min_rayleigh = min(min_rayleigh, rq)
            max_rayleigh = max(max_rayleigh, rq)
            if rq <= 0:
                nonpos += 1
        if valid == 0:
            return {
                "valid_probes": 0,
                "min_rayleigh": None,
                "max_rayleigh": None,
                "rayleigh_ratio": None,
                "nonpos_count": None,
            }
        ratio = float("inf") if min_rayleigh <= 1e-12 else max_rayleigh / min_rayleigh
        return {
            "valid_probes": valid,
            "min_rayleigh": float(min_rayleigh),
            "max_rayleigh": float(max_rayleigh),
            "rayleigh_ratio": float(ratio),
            "nonpos_count": int(nonpos),
        }

    def _initialize_mask(self):
        """Initialize sparsity mask for the model."""
        if self.use_sparsity:
            logger.info(f"Creating sparsity mask with {self.sparsity_method} "
                       f"at {self.sparsity} sparsity...")
            self.mask_dict = SparsityManager.create_mask(
                self.model,
                sparsity=self.sparsity,
                method=self.sparsity_method,
                device=self.args.device
            )
        else:
            # No sparsity: all ones mask
            logger.info("No sparsity constraints - using full model")
            self.mask_dict = {
                name: torch.ones_like(param.data).to(self.args.device)
                for name, param in self.model.named_parameters()
            }

    def _prepare_ref_model(self):
        """Prepare reference model for NPO loss (lazy initialization)."""
        if self.ref_model is None and self.forget_loss_type == "npo":
            logger.info("Creating reference model for NPO loss...")
            self.ref_model = copy.deepcopy(self.model)
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False
            # Move to same device
            self.ref_model = self.ref_model.to(self.args.device)
            logger.info("Reference model created and frozen")

    def compute_forget_loss(self, batch):
        """
        Compute forget loss using the configured loss function.

        Supports: logit_margin (default), grad_ascent, grad_diff, npo, simnpo, pdu, rmu
        """
        # Prepare reference model for NPO if needed
        if self.forget_loss_type == "npo":
            self._prepare_ref_model()

        # Build kwargs for the loss function
        loss_kwargs = {
            'ref_model': self.ref_model,
            'beta': self.npo_beta if self.forget_loss_type == "npo" else self.simnpo_beta,
            'delta': self.simnpo_delta,
        }

        return self._forget_loss_fn(
            self.model,
            batch,
            self.args.device,
            **loss_kwargs
        )

    def compute_retain_loss(self, batch):
        """Compute retain loss (standard cross-entropy)."""
        input_ids = batch['input_ids'].to(self.args.device)
        attention_mask = batch['attention_mask'].to(self.args.device)
        labels = batch.get('labels', input_ids).to(self.args.device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs.loss

    def compute_sparsity_regularizer(self):
        """
        Compute sparsity regularizer using the configured regularization type.

        Supports: l1 (default), l2, elastic_net, none
        """
        if self.regularization_type == "elastic_net":
            return self._regularization_fn(
                self.model,
                self.mask_dict,
                self.gamma,
                self.args.device,
                l1_ratio=self.elastic_net_l1_ratio
            )
        else:
            return self._regularization_fn(
                self.model,
                self.mask_dict,
                self.gamma,
                self.args.device
            )

    def inner_step(self, batch):
        """Single inner optimization step on retain set."""
        self.model.train()

        input_ids = batch['input_ids'].to(self.args.device)
        attention_mask = batch['attention_mask'].to(self.args.device)
        labels = batch.get('labels', input_ids).to(self.args.device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss

        # Add regularization using the configured type
        reg_loss = self.compute_sparsity_regularizer()
        loss_total = loss + reg_loss

        loss_total.backward()

        # Masked gradient update
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.grad is not None and name in self.mask_dict:
                    param.data.sub_(self.eta_in * param.grad * self.mask_dict[name])
                param.grad = None

        return loss.item()

    def inner_loop(self, retain_loader):
        """Inner loop: Optimize on retain set."""
        retain_iter = iter(retain_loader)

        for k in range(self.K):
            try:
                batch = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_loader)
                batch = next(retain_iter)

            self.inner_step(batch)

    def flatten_params(self, params_list):
        """Flatten list of parameters to single vector."""
        return torch.cat([p.reshape(-1) for p in params_list])

    def _flatten_tensors(self, ts):
        """Flatten a tensor list into one vector."""
        if len(ts) == 0:
            return torch.empty(0, device=self.args.device)
        return torch.cat([t.reshape(-1) for t in ts])

    def unflatten_params(self, flat_vec, params_list):
        """Unflatten vector back to parameter shapes."""
        unflattened = []
        offset = 0
        for param in params_list:
            numel = param.numel()
            unflattened.append(flat_vec[offset:offset+numel].reshape(param.shape))
            offset += numel
        return unflattened

    def _layer_id_from_param_name(self, name):
        """Extract transformer layer index from parameter name."""
        match = re.search(r"model\.layers\.(\d+)\.", name)
        if match is None:
            return None
        return int(match.group(1))

    def _build_implicit_blocks(self):
        """
        Build layer-wise parameter blocks for implicit correction.
        Returns: list[(block_name, [(param_name, param), ...])]
        """
        by_layer = {}
        max_layer = -1
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name not in self.mask_dict:
                continue
            layer_id = self._layer_id_from_param_name(name)
            if layer_id is None:
                continue
            is_attn = ".self_attn." in name
            is_mlp = ".mlp." in name
            if (is_attn and not self.implicit_block_include_attn) or (
                is_mlp and not self.implicit_block_include_mlp
            ):
                continue
            by_layer.setdefault(layer_id, []).append((name, param))
            max_layer = max(max_layer, layer_id)

        if max_layer < 0:
            return []

        min_keep = max(0, max_layer - self.implicit_block_last_n_layers + 1)
        blocks = []
        for layer_id in sorted(by_layer.keys()):
            if layer_id < min_keep:
                continue
            entries = by_layer[layer_id]
            if len(entries) == 0:
                continue
            blocks.append((f"layer_{layer_id}", entries))
        return blocks

    def _apply_blockwise_implicit_correction(
        self,
        g_alm_dict,
        L_alm,
        L_inner,
        outer_iter: Optional[int] = None,
    ):
        """
        Apply implicit correction block-by-block to avoid full-model flatten OOM.
        Updates g_alm_dict in-place.
        """
        blocks = self._build_implicit_blocks()
        if len(blocks) == 0:
            logger.warning("Blockwise implicit enabled but no valid blocks found. Skipping implicit correction.")
            return

        for block_name, entries in blocks:
            names = [name for name, _ in entries]
            params = [param for _, param in entries]
            g_block = self._flatten_tensors([g_alm_dict[name] for name in names]).to(self.args.device)
            mask_block = self._flatten_tensors([self.mask_dict[name] for name in names]).to(self.args.device)
            if g_block.numel() == 0:
                continue

            if self.implicit_solver == "neumann":
                g_corr_flat, neumann_status, meta = self._truncated_neumann_correction(
                    params_list=params,
                    g_alm_flat=g_block,
                    mask_flat=mask_block,
                    L_alm=L_alm,
                    L_inner=L_inner,
                )
                corr_parts = self.unflatten_params(g_corr_flat, [g_alm_dict[name] for name in names])
                for name, corr in zip(names, corr_parts):
                    g_alm_dict[name] = corr

                if self.debug_implicit:
                    self._log_implicit_debug_record({
                        "outer_iter": outer_iter,
                        "solver": "neumann",
                        "variant": self.neumann_variant,
                        "mode": "blockwise",
                        "block": block_name,
                        "dim": int(g_block.numel()),
                        "status": neumann_status,
                        "metrics": meta,
                    })
            else:
                v_block = g_block * mask_block

                def hvp_func(vec):
                    vec_masked = vec * mask_block
                    hvp = self.compute_hvp(L_inner, params, vec_masked)
                    return hvp * mask_block + self.cg_damping * vec

                h = self.conjugate_gradient(hvp_func, v_block)
                corr = self.compute_hvp(L_alm, params, h)
                corr_parts = self.unflatten_params(corr, [g_alm_dict[name] for name in names])
                for name, corr_part in zip(names, corr_parts):
                    g_alm_dict[name] = g_alm_dict[name] - corr_part

                lin_res = ((hvp_func(h) - v_block).norm() / (v_block.norm().clamp(min=1e-12))).item()
                cond_proxy = self._rayleigh_condition_proxy(hvp_func, v_block, mask_block)
                if self.debug_implicit:
                    self._log_implicit_debug_record({
                        "outer_iter": outer_iter,
                        "solver": "cg",
                        "variant": "cg",
                        "mode": "blockwise",
                        "block": block_name,
                        "dim": int(v_block.numel()),
                        "status": "ok",
                        "metrics": {
                            "v_norm": float(v_block.norm().item()),
                            "h_norm": float(h.norm().item()),
                            "linear_residual": float(lin_res),
                            "condition_proxy": cond_proxy,
                        },
                    })

    def flatten_mask(self):
        """Flatten all masks to single vector."""
        masks = []
        for name, param in self.model.named_parameters():
            if name in self.mask_dict:
                masks.append(self.mask_dict[name].reshape(-1))
            else:
                masks.append(torch.ones(param.numel(), device=self.args.device))
        return torch.cat(masks)

    def compute_hvp(self, loss, params, v):
        """Compute Hessian-vector product H*v."""
        grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
        flat_grad = torch.cat([g.reshape(-1) for g in grads])

        grad_v = (flat_grad * v).sum()

        hvp_grads = torch.autograd.grad(grad_v, params, retain_graph=True)
        flat_hvp = torch.cat([h.reshape(-1) for h in hvp_grads])

        return flat_hvp

    def conjugate_gradient_scipy(self, hvp_func, b):
        """Solve H*x = b using Conjugate Gradient (scipy implementation)."""
        # Store original device and dtype
        device = b.device
        dtype = b.dtype
        n = b.numel()

        # Convert b to numpy (convert to float32 first as numpy doesn't support bfloat16)
        b_np = b.detach().cpu().float().numpy()

        # Define the matrix-vector product function for LinearOperator
        def matvec(v):
            # Convert numpy array to torch tensor
            v_torch = torch.from_numpy(v).to(device=device, dtype=dtype)
            # Compute Hessian-vector product
            Hv_torch = hvp_func(v_torch)
            # Convert back to numpy (convert to float32 first as numpy doesn't support bfloat16)
            return Hv_torch.detach().cpu().float().numpy()

        # Create LinearOperator
        A = LinearOperator((n, n), matvec=matvec, dtype=b_np.dtype)

        # Solve using scipy's CG
        x_np, info = cg(A, b_np, maxiter=self.cg_iters, atol=self.cg_tol, rtol=0)

        # Convert solution back to torch tensor
        x = torch.from_numpy(x_np).to(device=device, dtype=dtype)

        # Log convergence info
        if info == 0:
            logger.debug("CG converged successfully")
        elif info > 0:
            logger.debug(f"CG did not converge, {info} iterations reached")
        else:
            logger.warning("CG illegal input or breakdown")

        return x

    def conjugate_gradient_torch(self, hvp_func, b):
        """Solve H*x = b using Conjugate Gradient (PyTorch implementation)."""
        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        rs_old = torch.sum(r * r)

        for i in range(self.cg_iters):
            Ap = hvp_func(p)
            alpha = rs_old / (torch.sum(p * Ap) + 1e-10)
            x = x + alpha * p
            r = r - alpha * Ap
            rs_new = torch.sum(r * r)

            if torch.sqrt(rs_new) < self.cg_tol:
                logger.debug(f"CG converged at iteration {i}")
                break

            beta = rs_new / (rs_old + 1e-10)
            p = r + beta * p
            rs_old = rs_new

        return x

    def conjugate_gradient(self, hvp_func, b):
        """Solve H*x = b using Conjugate Gradient."""
        if SCIPY_AVAILABLE:
            return self.conjugate_gradient_scipy(hvp_func, b)
        else:
            return self.conjugate_gradient_torch(hvp_func, b)

    def _clip_by_global_norm(self, flat_tensor, max_norm):
        """Clip flat tensor by global L2 norm; in-place if possible."""
        norm = flat_tensor.norm().clamp(min=1e-12)
        if norm <= max_norm:
            return flat_tensor
        return flat_tensor * (max_norm / norm)

    def _truncated_neumann_correction(
        self,
        params_list,
        g_alm_flat,
        mask_flat,
        L_alm,
        L_inner,
    ):
        """
        Truncated Neumann implicit correction: approximate h ≈ (H_inner + μI)^{-1} v
        with v = masked outer gradient, then g_corr = v - H_outer(h_correct).

        Uses only Hessian-vector products.
        Returns (g_corr_flat, status_string, meta_dict).
        """
        device = g_alm_flat.device
        dtype = g_alm_flat.dtype

        # 1) Masked outer gradient v
        v = g_alm_flat * mask_flat
        if not torch.isfinite(v).all():
            logger.info("Neumann: fallback_no_correction (non-finite v)")
            v = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            return v, "fallback_no_correction_nonfinite_v", {"alpha": None}
        if self.neumann_clip_norm is not None:
            v = self._clip_by_global_norm(v, self.neumann_clip_norm)

        # 2) Damped masked HVP for inner Hessian: H_tilde(x) = mask ⊙ H_inner(mask ⊙ x) + μ (mask ⊙ x)
        def H_in_tilde(x):
            x_act = x * mask_flat
            Hv = self.compute_hvp(L_inner, params_list, x_act)
            Hv_act = Hv * mask_flat
            return Hv_act + self.neumann_mu * x_act

        # 3) Choose α (probe or default)
        alpha = self.neumann_alpha_default
        if self.neumann_use_probe_alpha:
            u = torch.randn_like(v, device=device, dtype=dtype)
            u = u * mask_flat
            u_norm = u.norm().clamp(min=1e-12)
            u = u / u_norm
            Hu = H_in_tilde(u)
            # u is normalized, so ||u||=1 and L_est ≈ ||H u||.
            L_est = Hu.norm().clamp(min=1e-12).item()
            alpha = 0.5 / (L_est + 1e-12)
            if not np.isfinite(alpha) or alpha <= 0:
                alpha = self.neumann_alpha_default

        # Always clamp alpha to a conservative stability interval.
        alpha = float(np.clip(alpha, self.neumann_alpha_min, self.neumann_alpha_max))

        # 4) Approximate H_tilde^{-1} v with either:
        #    - legacy accumulation (kept for reproducibility), or
        #    - Richardson/Neumann on (I - αH), which directly approximates H^{-1}v.
        if self.neumann_variant == "legacy":
            h = torch.zeros_like(v, device=device, dtype=dtype)
            p = v.clone()
            for _ in range(self.neumann_steps + 1):
                h = h + p
                if not torch.isfinite(h).all():
                    logger.info(f"Neumann: fallback_no_correction (non-finite h); ||v||={v.norm().item():.6f}")
                    return v, "fallback_no_correction", {"alpha": alpha}
                p = alpha * H_in_tilde(p)
                if not torch.isfinite(p).all():
                    logger.info(f"Neumann: fallback_no_correction (non-finite p); ||v||={v.norm().item():.6f}")
                    return v, "fallback_no_correction", {"alpha": alpha}
            h_correct = alpha * h
        else:
            # Richardson fixed-point:
            # h_{k+1} = h_k + α (v - H_tilde h_k)
            # => h ≈ α Σ_{i=0}^J (I - αH_tilde)^i v
            h_correct = torch.zeros_like(v, device=device, dtype=dtype)
            for _ in range(self.neumann_steps + 1):
                residual = v - H_in_tilde(h_correct)
                if not torch.isfinite(residual).all() or not torch.isfinite(h_correct).all():
                    logger.info(f"Neumann(Richardson): fallback_no_correction (non-finite residual); ||v||={v.norm().item():.6f}")
                    return v, "fallback_no_correction", {"alpha": alpha}
                h_correct = h_correct + alpha * residual
                if not torch.isfinite(h_correct).all():
                    logger.info(f"Neumann(Richardson): fallback_no_correction (non-finite h); ||v||={v.norm().item():.6f}")
                    return v, "fallback_no_correction", {"alpha": alpha}

        # 5) Outer HVP for correction: c = mask ⊙ H_AL(h_correct)
        h_act = h_correct * mask_flat
        c = self.compute_hvp(L_alm, params_list, h_act)
        c = c * mask_flat
        g_corr = v - c

        # 6) Safety: fallback if correction explodes
        v_norm = v.norm().clamp(min=1e-12).item()
        g_corr_norm = g_corr.norm().item()
        h_norm = h_correct.norm().item()
        if g_corr_norm > self.neumann_max_growth_ratio * v_norm:
            logger.info(
                f"Neumann: fallback_exploding_correction ||g_corr||={g_corr_norm:.6f} "
                f"> {self.neumann_max_growth_ratio}*||v||={v_norm:.6f}; ||h||={h_norm:.6f}"
            )
            return v, "fallback_exploding_correction", {"alpha": alpha}

        if self.neumann_clip_norm is not None:
            g_corr = self._clip_by_global_norm(g_corr, self.neumann_clip_norm)

        # 7) Solve quality and condition proxy for debugging
        Hh = H_in_tilde(h_correct)
        lin_res = ((Hh - v).norm() / (v.norm().clamp(min=1e-12))).item()
        cond_proxy = self._rayleigh_condition_proxy(H_in_tilde, v, mask_flat)

        logger.info(
            f"Neumann: ||v||={v_norm:.6f} ||h||={h_norm:.6f} ||g_corr||={g_corr_norm:.6f} alpha={alpha:.6f}"
        )
        return g_corr, "ok", {
            "alpha": float(alpha),
            "v_norm": float(v_norm),
            "h_norm": float(h_norm),
            "g_corr_norm": float(g_corr_norm),
            "linear_residual": float(lin_res),
            "condition_proxy": cond_proxy,
        }

    def outer_step(self, forget_batch, retain_batch, outer_iter: Optional[int] = None):
        """Outer loop: Update parameters to forget while respecting budget."""
        self.model.train()

        # Compute forget loss using the configured loss function
        L_fgt = self.compute_forget_loss(forget_batch)

        # Compute retain loss
        retain_ids = retain_batch['input_ids'].to(self.args.device)
        retain_mask = retain_batch['attention_mask'].to(self.args.device)
        retain_labels = retain_batch.get('labels', retain_ids).to(self.args.device)

        retain_outputs = self.model(
            input_ids=retain_ids,
            attention_mask=retain_mask,
            labels=retain_labels
        )
        L_ret = retain_outputs.loss

        if not torch.isfinite(L_fgt) or not torch.isfinite(L_ret):
            logger.warning(
                "Outer step received non-finite loss: "
                f"L_fgt={L_fgt.item() if torch.isfinite(L_fgt) else 'nan/inf'} "
                f"L_ret={L_ret.item() if torch.isfinite(L_ret) else 'nan/inf'}. "
                "Skipping outer update."
            )
            return float("nan"), float("nan"), float("nan")

        # Constraint residual (keep tensor form for correct ALM gradient term)
        r_tensor = L_ret - self.epsilon
        r = r_tensor.item()

        # Augmented Lagrangian objective
        L_alm = L_fgt + self.lambda_dual * L_ret + 0.5 * self.rho * (r_tensor ** 2)

        # Compute raw gradient
        L_alm.backward(retain_graph=self.use_implicit)

        # Store raw gradients
        g_alm_dict = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                g_alm_dict[name] = param.grad.clone()
            else:
                g_alm_dict[name] = torch.zeros_like(param.data)

        # Clear gradients
        self.model.zero_grad()

        # Implicit correction (if enabled): use outer gradient as RHS (masked g_alm)
        if self.use_implicit:
            L_ret_v = self.compute_retain_loss(retain_batch)
            R_theta_v = self.compute_sparsity_regularizer()
            L_inner = L_ret_v + R_theta_v
            if self.implicit_blockwise:
                self._apply_blockwise_implicit_correction(
                    g_alm_dict=g_alm_dict,
                    L_alm=L_alm,
                    L_inner=L_inner,
                    outer_iter=outer_iter,
                )
            else:
                params_list = [p for p in self.model.parameters() if p.requires_grad]
                mask_flat = self.flatten_mask()
                g_alm_flat = torch.cat([
                    g_alm_dict[name].reshape(-1)
                    for name, param in self.model.named_parameters()
                    if param.requires_grad
                ])

                if self.implicit_solver == "neumann":
                    g_corr_flat, neumann_status, meta = self._truncated_neumann_correction(
                        params_list=params_list,
                        g_alm_flat=g_alm_flat,
                        mask_flat=mask_flat,
                        L_alm=L_alm,
                        L_inner=L_inner,
                    )
                    if neumann_status != "ok":
                        logger.debug(f"Neumann correction: {neumann_status}")
                    if self.debug_implicit:
                        if outer_iter is not None:
                            self._save_debug_array(outer_iter, "v", g_alm_flat * mask_flat)
                            self._save_debug_array(outer_iter, "g_corr", g_corr_flat)
                        self._log_implicit_debug_record({
                            "outer_iter": outer_iter,
                            "solver": "neumann",
                            "variant": self.neumann_variant,
                            "mode": "full",
                            "status": neumann_status,
                            "metrics": meta,
                        })
                    correction_unflattened = self.unflatten_params(g_corr_flat, params_list)
                    for (name, param), g_update in zip(
                        self.model.named_parameters(), correction_unflattened
                    ):
                        if name in g_alm_dict:
                            g_alm_dict[name] = g_update
                else:
                    # CG path: solve (H_inner + damping)*h = v with v = masked g_alm (outer gradient)
                    v = g_alm_flat * mask_flat

                    def hvp_func(vec):
                        vec_masked = vec * mask_flat
                        hvp = self.compute_hvp(L_inner, params_list, vec_masked)
                        return hvp * mask_flat + self.cg_damping * vec

                    h = self.conjugate_gradient(hvp_func, v)
                    hvp_correction = self.compute_hvp(L_alm, params_list, h)
                    lin_res = ((hvp_func(h) - v).norm() / (v.norm().clamp(min=1e-12))).item()
                    cond_proxy = self._rayleigh_condition_proxy(hvp_func, v, mask_flat)
                    if self.debug_implicit:
                        if outer_iter is not None:
                            self._save_debug_array(outer_iter, "v", v)
                            self._save_debug_array(outer_iter, "h", h)
                            self._save_debug_array(outer_iter, "hvp_correction", hvp_correction)
                        self._log_implicit_debug_record({
                            "outer_iter": outer_iter,
                            "solver": "cg",
                            "variant": "cg",
                            "mode": "full",
                            "status": "ok",
                            "metrics": {
                                "v_norm": float(v.norm().item()),
                                "h_norm": float(h.norm().item()),
                                "hvp_correction_norm": float(hvp_correction.norm().item()),
                                "linear_residual": float(lin_res),
                                "condition_proxy": cond_proxy,
                            },
                        })
                    correction_unflattened = self.unflatten_params(hvp_correction, params_list)
                    for (name, param), correction in zip(
                        self.model.named_parameters(), correction_unflattened
                    ):
                        if name in g_alm_dict:
                            g_alm_dict[name] = g_alm_dict[name] - correction

        # Primal update
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.mask_dict and name in g_alm_dict:
                    mask = self.mask_dict[name]
                    param.data.sub_(self.eta_theta * g_alm_dict[name] * mask)
                param.grad = None

        # Dual update
        self.lambda_dual = max(0.0, self.lambda_dual + self.rho * r)

        return L_fgt.item(), L_ret.item(), r

    def train(self):
        """Override the train method to implement custom S-BiAL training loop."""
        # Initialize mask
        if self.mask_dict is None:
            self._initialize_mask()

        # Enable gradient checkpointing for memory (SIBL bypasses _inner_training_loop).
        # use_reentrant=False is required when use_implicit=True (HVPs use autograd.grad).
        if getattr(self.args, "gradient_checkpointing", False):
            kwargs = getattr(self.args, "gradient_checkpointing_kwargs", None) or {}
            if self.use_implicit:
                kwargs = dict(kwargs)
                kwargs["use_reentrant"] = False
                logger.info("Gradient checkpointing: use_reentrant=False (required for implicit/HVP).")
            else:
                kwargs.setdefault("use_reentrant", False)
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=kwargs
            )

        # Get data loaders
        train_dataloader = self.get_train_dataloader()

        logger.info(f"\nStarting S-BiAL unlearning for {self.T} iterations...")
        logger.info(f"Forget loss type: {self.forget_loss_type}")
        logger.info(f"Regularization type: {self.regularization_type}")
        logger.info(f"Retain budget ε = {self.epsilon:.4f}")
        logger.info(f"Use implicit correction: {self.use_implicit}")
        if self.use_implicit:
            logger.info(f"Implicit solver: {self.implicit_solver}")
            logger.info(
                f"Implicit mode: {'blockwise' if self.implicit_blockwise else 'full'}"
            )
            if self.implicit_blockwise:
                logger.info(
                    f"Blockwise settings: last_n_layers={self.implicit_block_last_n_layers}, "
                    f"include_attn={self.implicit_block_include_attn}, include_mlp={self.implicit_block_include_mlp}"
                )
            if self.implicit_solver == "cg" and self.cg_damping > 0:
                logger.info(f"CG damping (for Hessian conditioning): {self.cg_damping}")
            elif self.implicit_solver == "neumann":
                logger.info(
                    f"Neumann: steps={self.neumann_steps}, mu={self.neumann_mu}, "
                    f"probe_alpha={self.neumann_use_probe_alpha}, variant={self.neumann_variant}"
                )

        for t in range(self.T):
            t_start = time.time()

            # Get data iterators
            data_iter = iter(train_dataloader)

            # Get forget and retain batches
            try:
                combined_batch = next(data_iter)
                forget_batch = combined_batch['forget']
                retain_batch = combined_batch['retain']
            except (StopIteration, KeyError) as e:
                logger.error(f"Error getting batches: {e}")
                break

            # Inner loop
            # Create a simple retain loader from the current batch
            retain_loader = [retain_batch] * self.K

            self.inner_loop(retain_loader)

            # Outer loop
            L_fgt, L_ret, r = self.outer_step(forget_batch, retain_batch, outer_iter=t)

            if not np.isfinite(L_fgt) or not np.isfinite(L_ret) or not np.isfinite(r):
                logger.warning(f"Stopping training early at iter={t} due to non-finite metrics.")
                break

            # Track history
            t_elapsed = time.time() - t_start
            self.history['iter'].append(t)
            self.history['L_forget'].append(L_fgt)
            self.history['L_retain'].append(L_ret)
            self.history['residual'].append(r)
            self.history['lambda'].append(self.lambda_dual)
            self.history['time'].append(t_elapsed)

            # Log progress
            if t % 2 == 0 or t == self.T - 1:
                logger.info(
                    f"[{t:3d}/{self.T}] L_fgt={L_fgt:.4f} | L_ret={L_ret:.4f} | "
                    f"r={r:+.4f} | λ={self.lambda_dual:.3f} | t={t_elapsed:.2f}s"
                )

            # Update training state
            self.state.global_step = t + 1

            if (
                self.debug_implicit
                and self.debug_stop_after_outer is not None
                and t >= self.debug_stop_after_outer
            ):
                logger.info(
                    f"Debug stop requested at outer_iter={t} "
                    f"(debug_stop_after_outer={self.debug_stop_after_outer})"
                )
                break

            # Evaluation and checkpointing
            # if self.args.evaluation_strategy != "no" and (t + 1) % self.args.eval_steps == 0:
            #     self.evaluate()

            # if self.args.save_strategy != "no" and (t + 1) % self.args.save_steps == 0:
            #     self.save_model()

        self.evaluate()
        if self.debug_implicit and len(self._debug_records) > 0:
            summary_path = os.path.join(self.debug_dir, "implicit_debug_summary.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(self._debug_records, f, indent=2)
            logger.info(f"Saved implicit debug summary to {summary_path}")
        logger.info("Unlearning complete!")

        # Return training output
        return type('TrainOutput', (), {'global_step': self.T, 'training_loss': L_ret})()

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute loss for evaluation.
        This is only used during evaluation, not during training.
        """
        # For evaluation, just compute standard loss
        if 'forget' in inputs:
            forget_inputs = inputs['forget']
            forget_inputs = {
                'input_ids': forget_inputs['input_ids'],
                'attention_mask': forget_inputs['attention_mask'],
                'labels': forget_inputs.get('labels', forget_inputs['input_ids']),
            }
            outputs = model(**forget_inputs)
            loss = outputs.loss
        else:
            outputs = model(**inputs)
            loss = outputs.loss

        return (loss, outputs) if return_outputs else loss