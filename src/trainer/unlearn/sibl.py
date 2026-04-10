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
import torch.nn.functional as F
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
        lambda_init: float = 0.0,  # Initial dual variable (0 = standard ALM; >0 = early retain protection)
        # Two-phase ALM: run NPO-heavy first, then boost λ/ρ for retain recovery
        lambda_boost_step: int = 0,  # Outer step at which to boost (0 = disabled)
        lambda_boost_value: float = 5.0,  # λ target after boost
        rho_boost_value: float = 1.0,  # ρ target after boost
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
        neumann_backtrack_factor: float = 0.5,  # backtrack step reduction factor
        neumann_backtrack_max_tries: int = 4,  # max backtrack attempts
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
        # Surgical unlearning: neuron-level bitmap mask from trace analysis
        neuron_bitmap_path: Optional[str] = None,  # Path to forget_neuron_bitmap.pt (legacy)
        neuron_traces_path: Optional[str] = None,  # Path to neuron_traces.pt (raw ratios)
        mask_th_low: float = 0.5,  # ratio < th_low -> mask=0 (frozen)
        mask_th_high: float = 1.0,  # ratio >= th_high -> full outer LR; else smooth * outer LR
        outer_lr_smooth: float = 0.6,  # LR multiplier for neurons with th_low <= ratio < th_high
        mask_freeze_layers: Optional[list] = None,  # Layer indices forced to mask=0 everywhere
        outer_freeze_layers: Optional[list] = None,  # Layer indices frozen in outer step only (inner still updates)
        proportional_outer_lr: bool = False,  # Use ratio as LR scale (vs fixed smooth)
        retain_protection_layers: Optional[list] = None,  # Layer indices for aggressive retention LR
        retain_lr_multiplier: float = 2.0,  # Inner LR multiplier for retain-protection layers
        implicit_block_skip_layers: Optional[list] = None,  # Layer indices to SKIP in implicit correction
        # Gradient disentanglement: project forget gradient orthogonal to retain gradient
        gradient_projection: bool = False,  # Enable orthogonal gradient projection
        gradient_projection_scope: str = "layer",  # "param", "layer", "global", or "aggressive"
        projection_strength: float = 1.0,  # Fraction of retain-aligned component to remove (0=none, 1=full)
        projection_schedule: str = "constant",  # "constant" or "linear_decay" (1→0 over T iters)
        projection_rescale: bool = False,  # Rescale projected gradient to maintain original norm
        projection_layers: Optional[list] = None,  # Layer indices to apply projection to (None = all)
        # Activation steering: representation-space forget intervention
        use_steering: bool = False,  # Enable activation steering as forget loss
        steering_layers: Optional[list] = None,  # Layer indices for steering (default: [5,6,7])
        steering_coeff: float = 20.0,  # Magnitude of random control vectors
        steering_alpha: float = 1.0,  # Weight of steering loss (vs logit loss)
        steering_only: bool = True,  # If True, use ONLY steering loss (no logit loss)
        steering_retain_match: bool = False,  # If True, match retain activations instead of random
        # Surgical neuron-masked steering: steer only forget-dominant hidden dims
        steering_neuron_mask_path: Optional[str] = None,  # Path to forget_neuron_bitmap.pt
        # Inner representation anchor: add activation MSE to inner retain loop
        inner_repr_anchor: bool = False,  # Add representation anchor to inner loop
        inner_repr_alpha: float = 1.0,  # Weight of representation anchor loss
        inner_repr_layers: Optional[list] = None,  # Layers for inner anchor (default: steering_layers)
        # Post-unlearning retention recovery: pure inner steps after T outer iterations
        post_unlearn_inner_steps: int = 0,  # N pure CE inner steps after main loop to recover retention
        post_inner_retain_only: bool = False,  # If True, invert bitmap mask during post-inner (update only retain-dominant neurons)
        post_inner_soft_mask_path: Optional[str] = None,  # DGA selectivity .pt for soft-masked recovery (overrides binary inversion)
        post_inner_soft_mask_beta: float = 5.0,  # Sharpness β: α = σ(-β*s); contested neurons get partial updates
        # KL-anchored inner loop: replace CE with KL(pretrained || model) to prevent forget re-learning
        inner_loss_type: str = "ce",  # "ce" (default) or "kl_pretrained" — inner loop retain loss
        outer_retain_loss_type: str = "ce",  # "ce" (default) or "kl_pretrained" — outer ALM retain constraint
        inner_kl_temperature: float = 2.0,  # Temperature for KL distillation (higher = softer)
        post_inner_loss_type: str = "ce",  # "ce" or "kl_pretrained" — post-inner recovery loss
        # Inverted inner mask: when bitmap loaded, inner loop updates RETAIN-dominant neurons (1-mask)
        # instead of forget-dominant neurons. Creates complementary inner/outer parameter subsets.
        invert_inner_mask: bool = False,
        # When True + bitmap + invert_inner_mask: outer uses full model (mask=ones) while inner uses
        # inverted bitmap. Gets G1-level forget (full NPO) + targeted inner retain recovery.
        outer_full_model: bool = False,
        # Trace-guided implicit: only correct retain-dominant neuron gradients (prevents overcorrection)
        implicit_trace_guided: bool = False,  # Blend implicit correction using mask (retain-only correction)
        implicit_trace_threshold: float = 0.5,  # Mask values above this → skip implicit (keep original forget gradient)
        checkpoint_every_steps: int = 0,  # Save checkpoint every N outer steps (0 = disabled); for trajectory analysis
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
        self.lambda_init = lambda_init
        self.lambda_boost_step = lambda_boost_step
        self.lambda_boost_value = lambda_boost_value
        self.rho_boost_value = rho_boost_value
        self._lambda_boosted = False
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

        # Surgical unlearning configuration
        self.neuron_bitmap_path = neuron_bitmap_path
        self.neuron_traces_path = neuron_traces_path
        self.mask_th_low = mask_th_low
        self.mask_th_high = mask_th_high
        self.outer_lr_smooth = outer_lr_smooth
        self.mask_freeze_layers = set(mask_freeze_layers or [])
        self.outer_freeze_layers = set(outer_freeze_layers or [])
        self.proportional_outer_lr = proportional_outer_lr
        self.retain_protection_layers = set(retain_protection_layers or [])
        self.retain_lr_multiplier = retain_lr_multiplier
        self.implicit_block_skip_layers = set(implicit_block_skip_layers or [])
        self.gradient_projection = gradient_projection
        self.gradient_projection_scope = gradient_projection_scope
        self.projection_strength = projection_strength
        self.projection_schedule = projection_schedule
        self._projection_strength_base = projection_strength  # Store base for scheduling
        self.projection_rescale = projection_rescale
        self.projection_layers = set(projection_layers) if projection_layers is not None else None

        # Activation steering configuration
        self.use_steering = use_steering
        self.steering_layers = steering_layers or [5, 6, 7]
        self.steering_coeff = steering_coeff
        self.steering_alpha = steering_alpha
        self.steering_only = steering_only
        self.steering_retain_match = steering_retain_match
        self._steering_vectors = {}  # Cache: layer_idx → random control vector
        self._ref_retain_activations = {}  # Cache: layer_idx → retain activation (if steering_retain_match)
        # Neuron-masked steering
        self.steering_neuron_mask_path = steering_neuron_mask_path
        self._steering_neuron_masks = {}  # layer_idx → bool tensor [hidden_dim]
        if steering_neuron_mask_path is not None:
            self._build_steering_neuron_masks(steering_neuron_mask_path)
        # Inner representation anchor
        self.inner_repr_anchor = inner_repr_anchor
        self.inner_repr_alpha = inner_repr_alpha
        self.inner_repr_layers = inner_repr_layers if inner_repr_layers is not None else self.steering_layers
        self._repr_anchor_model = None  # Frozen reference model for inner anchor
        self.post_unlearn_inner_steps = post_unlearn_inner_steps
        self.post_inner_retain_only = post_inner_retain_only
        self.post_inner_soft_mask_path = post_inner_soft_mask_path
        self.post_inner_soft_mask_beta = post_inner_soft_mask_beta
        self.inner_loss_type = inner_loss_type
        self.inner_kl_temperature = inner_kl_temperature
        self.post_inner_loss_type = post_inner_loss_type
        self.outer_retain_loss_type = outer_retain_loss_type
        self.invert_inner_mask = invert_inner_mask
        self.outer_full_model = outer_full_model
        self.implicit_trace_guided = implicit_trace_guided
        self.implicit_trace_threshold = implicit_trace_threshold
        self.checkpoint_every_steps = checkpoint_every_steps

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
        self.lambda_dual = float(lambda_init)
        if lambda_init > 0:
            logger.info(f"ALM dual variable initialized to λ={lambda_init:.2f} (early retain protection)")

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
        self.inner_mask_dict = None  # Built after bitmap loading when invert_inner_mask=True

        # Log configuration
        logger.info(f"SIBL configured with forget_loss_type={forget_loss_type}, "
                   f"regularization_type={regularization_type}")
        if self.inner_loss_type != "ce":
            logger.info(f"Inner loop loss: {self.inner_loss_type} (T={self.inner_kl_temperature})")
        if self.post_inner_loss_type != "ce":
            logger.info(f"Post-inner recovery loss: {self.post_inner_loss_type}")
        if self.implicit_trace_guided:
            logger.info(f"Trace-guided implicit: threshold={self.implicit_trace_threshold}")
        if self.use_implicit:
            logger.info(f"Neumann variant: {self.neumann_variant}")
        if self.debug_implicit:
            os.makedirs(self.debug_dir, exist_ok=True)
            logger.info(f"Implicit debug enabled. Outputs: {self.debug_dir}")

    def _build_steering_neuron_masks(self, bitmap_path: str):
        """Build per-layer hidden-dim neuron masks from forget_neuron_bitmap.pt.

        For each steering layer, creates a boolean mask of shape (hidden_dim,) marking
        forget-dominant output neurons from down_proj (MLP output) and o_proj (attn output).
        Steering MSE will be applied ONLY at these dimensions.
        """
        try:
            bitmap = torch.load(bitmap_path, map_location='cpu', weights_only=False)
        except TypeError:
            bitmap = torch.load(bitmap_path, map_location='cpu')

        if not isinstance(bitmap, dict):
            logger.warning(f"Neuron bitmap at {bitmap_path} is not a dict — skipping neuron masking")
            return

        for layer_idx in self.steering_layers:
            o_key = f"model.layers.{layer_idx}.self_attn.o_proj.weight"
            d_key = f"model.layers.{layer_idx}.mlp.down_proj.weight"
            o_bitmap = bitmap.get(o_key, None)
            d_bitmap = bitmap.get(d_key, None)
            if o_bitmap is None and d_bitmap is None:
                continue
            o_arr = o_bitmap if o_bitmap is not None else np.zeros(4096, dtype=np.int32)
            d_arr = d_bitmap if d_bitmap is not None else np.zeros(4096, dtype=np.int32)
            combined = torch.tensor((o_arr.astype(bool) | d_arr.astype(bool)), dtype=torch.bool)
            n_active = combined.sum().item()
            if n_active > 0:
                self._steering_neuron_masks[layer_idx] = combined
                logger.info(f"Neuron mask layer {layer_idx}: {n_active}/{len(combined)} forget-dominant dims")
            else:
                logger.info(f"Neuron mask layer {layer_idx}: no forget-dominant dims — layer will use full steering")

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

    def _load_neuron_bitmap_mask(self):
        """Load neuron-level forget bitmap and convert to weight-level mask.

        The bitmap (from trace analysis) maps param_name -> np.ndarray(int32)
        of shape (out_features,) where 1 = forget-dominant neuron.
        We broadcast each row indicator to a full weight mask:
            mask[i, :] = bitmap[i]  for 2D weight matrices.
        Parameters not in the bitmap (layernorm, embeddings, lm_head) get zeros.
        """
        bitmap_path = self.neuron_bitmap_path
        logger.info(f"Loading neuron bitmap mask from {bitmap_path}")
        bitmap = torch.load(bitmap_path, map_location="cpu", weights_only=False)

        mask_dict = {}
        self._bitmap_params = set()  # track which params are actually in the bitmap
        total_active = 0
        total_params = 0

        for name, param in self.model.named_parameters():
            if name in bitmap:
                # Broadcast per-neuron (row) bitmap to full weight shape
                row_mask = torch.from_numpy(bitmap[name]).float()  # (out_features,)
                if param.dim() == 2:
                    # Weight matrix: expand rows -> (out_features, in_features)
                    weight_mask = row_mask.unsqueeze(1).expand_as(param.data)
                elif param.dim() == 1:
                    weight_mask = row_mask
                else:
                    weight_mask = row_mask.view(-1, *([1] * (param.dim() - 1))).expand_as(param.data)
                mask_dict[name] = weight_mask.to(self.args.device)
                self._bitmap_params.add(name)
            else:
                # Not in bitmap (layernorm, embeddings, etc.) -> frozen in both outer and post_inner
                mask_dict[name] = torch.zeros_like(param.data).to(self.args.device)

            n_active = (mask_dict[name] > 0).sum().item()
            total_active += n_active
            total_params += param.numel()

        pct = 100.0 * total_active / max(total_params, 1)
        logger.info(f"Neuron bitmap mask: {total_active:,}/{total_params:,} active ({pct:.2f}%)")

        # Log per-layer stats
        layer_stats = {}
        for name in mask_dict:
            layer_id = self._layer_id_from_param_name(name)
            if layer_id is not None:
                if layer_id not in layer_stats:
                    layer_stats[layer_id] = {"active": 0, "total": 0}
                layer_stats[layer_id]["active"] += (mask_dict[name] > 0).sum().item()
                layer_stats[layer_id]["total"] += mask_dict[name].numel()
        for lid in sorted(layer_stats):
            s = layer_stats[lid]
            lpct = 100.0 * s["active"] / max(s["total"], 1)
            logger.info(f"  Layer {lid:2d}: {s['active']:>10,}/{s['total']:>12,} active ({lpct:.2f}%)")

        return mask_dict

    def _build_dga_soft_mask(self, selectivity_dict: dict, beta: float) -> dict:
        """Build a continuous post-inner recovery mask from DGA selectivity scores.

        α_n = σ(-β * s_n):
          s_n > 0  (forget-dominant) → α ≈ 0  (frozen during recovery)
          s_n < 0  (retain-dominant) → α ≈ 1  (fully updated)
          s_n ≈ 0  (contested)       → α ≈ 0.5 (partial update, damped)

        selectivity_dict: {param_name: tensor(out_features,)}  per-neuron scores
        beta: sharpness of sigmoid; higher = closer to binary G3 behavior.
        """
        mask_dict = {}
        for name, param in self.model.named_parameters():
            if name in selectivity_dict:
                s = selectivity_dict[name].float().to(self.args.device)
                # Expand per-neuron (out_features,) to full weight shape
                if s.dim() == 1 and param.dim() == 2:
                    s = s.unsqueeze(1).expand_as(param.data)
                elif s.dim() == 1 and param.dim() > 2:
                    s = s.view(-1, *([1] * (param.dim() - 1))).expand_as(param.data)
                alpha = torch.sigmoid(-beta * s)
                mask_dict[name] = alpha.contiguous()
            else:
                # Not in selectivity dict (embeddings, layernorm, lm_head) → frozen
                mask_dict[name] = torch.zeros_like(param.data)

        n_active = sum((m > 0.01).sum().item() for m in mask_dict.values())
        n_total = sum(m.numel() for m in mask_dict.values())
        logger.info(f"DGA soft mask built: β={beta}, effective active (>0.01): "
                    f"{n_active:,}/{n_total:,} ({100*n_active/max(n_total,1):.1f}%)")
        return mask_dict

    def _load_neuron_traces_mask(self):
        """Load raw neuron traces and build tiered mask with embedded LR scale.

        Mask values encode both trainability and outer LR scale:
          - 0.0 : frozen (ratio < th_low, or frozen layer)
          - outer_lr_smooth : mixed neuron (th_low <= ratio < th_high)
          - 1.0 : forget-dominant neuron (ratio >= th_high)

        Inner step uses (mask > 0) for binary gating.
        Outer step uses mask values directly as LR scale.
        This avoids storing a separate LR scale dict (saves ~27GB GPU RAM).
        """
        traces_path = self.neuron_traces_path
        logger.info(f"Loading neuron traces from {traces_path}")
        traces = torch.load(traces_path, map_location="cpu", weights_only=False)
        forget_traces = traces["forget"]
        retain_traces = traces["retain"]

        mask_dict = {}
        total_active = 0
        total_full_lr = 0
        total_smooth_lr = 0
        total_params = 0

        for name, param in self.model.named_parameters():
            layer_id = self._layer_id_from_param_name(name)

            # Freeze layers in mask_freeze_layers, or params not in traces
            if (layer_id is not None and layer_id in self.mask_freeze_layers) or name not in forget_traces:
                mask_dict[name] = torch.zeros(param.data.shape, dtype=torch.float16, device=self.args.device)
                total_params += param.numel()
                continue

            # Compute per-neuron ratio
            f_raw = forget_traces[name]
            r_raw = retain_traces[name]
            f_act = f_raw.float() if isinstance(f_raw, torch.Tensor) else torch.from_numpy(f_raw).float()
            r_act = r_raw.float() if isinstance(r_raw, torch.Tensor) else torch.from_numpy(r_raw).float()
            ratio = f_act / (r_act + 1e-8)

            # Build tiered mask encoding outer LR scale
            if self.proportional_outer_lr:
                # Proportional: use min(ratio, 1.0) as LR scale
                neuron_mask = torch.where(
                    ratio >= self.mask_th_high,
                    torch.ones_like(ratio),
                    torch.where(
                        ratio >= self.mask_th_low,
                        ratio.clamp(max=1.0),  # ratio itself as LR scale
                        torch.zeros_like(ratio),
                    ),
                )
            else:
                # Fixed smooth factor
                neuron_mask = torch.where(
                    ratio >= self.mask_th_high,
                    torch.ones_like(ratio),
                    torch.where(
                        ratio >= self.mask_th_low,
                        torch.full_like(ratio, self.outer_lr_smooth),
                        torch.zeros_like(ratio),
                    ),
                )

            # Expand to weight shape
            if param.dim() == 2:
                weight_mask = neuron_mask.unsqueeze(1).expand_as(param.data)
            elif param.dim() == 1:
                weight_mask = neuron_mask
            else:
                weight_mask = neuron_mask.view(-1, *([1] * (param.dim() - 1))).expand_as(param.data)

            mask_dict[name] = weight_mask.half().to(self.args.device)

            n_active = (weight_mask > 0).sum().item()
            n_full = (weight_mask >= 1.0 - 1e-6).sum().item()
            n_smooth = n_active - n_full
            total_active += n_active
            total_full_lr += n_full
            total_smooth_lr += n_smooth
            total_params += param.numel()

        pct = 100.0 * total_active / max(total_params, 1)
        logger.info(f"Traces mask: {total_active:,}/{total_params:,} active ({pct:.2f}%)")
        logger.info(f"  Full outer LR (ratio>={self.mask_th_high}): {total_full_lr:,} params")
        logger.info(f"  Smooth outer LR ({self.mask_th_low}<=ratio<{self.mask_th_high}): {total_smooth_lr:,} params")

        # Log per-layer stats
        layer_stats = {}
        for name in mask_dict:
            lid = self._layer_id_from_param_name(name)
            if lid is not None:
                if lid not in layer_stats:
                    layer_stats[lid] = {"active": 0, "total": 0}
                layer_stats[lid]["active"] += (mask_dict[name] > 0).sum().item()
                layer_stats[lid]["total"] += mask_dict[name].numel()
        for lid in sorted(layer_stats):
            s = layer_stats[lid]
            lpct = 100.0 * s["active"] / max(s["total"], 1)
            logger.info(f"  Layer {lid:2d}: {s['active']:>10,}/{s['total']:>12,} active ({lpct:.2f}%)")

        return mask_dict

    def _initialize_mask(self):
        """Initialize sparsity mask for the model."""
        if self.neuron_traces_path:
            self.mask_dict = self._load_neuron_traces_mask()
        elif self.neuron_bitmap_path:
            self.mask_dict = self._load_neuron_bitmap_mask()
        elif self.use_sparsity:
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

    def _kl_loss_from_ref(self, student_logits, input_ids, attention_mask, labels):
        """Compute KL(p_ref || p_model) for distillation from frozen reference model.

        Used in inner loop and post-inner recovery when loss_type='kl_pretrained'.
        The reference model (pretrained) serves as teacher — its retain-data distribution
        anchors the student without re-learning forget content through shared weights.
        """
        T = self.inner_kl_temperature
        with torch.no_grad():
            ref_out = self.ref_model(input_ids=input_ids, attention_mask=attention_mask)
            ref_logits = ref_out.logits

        # Shift for autoregressive alignment (predict next token)
        s_logits = student_logits[..., :-1, :].contiguous() / T
        r_logits = ref_logits[..., :-1, :].contiguous() / T
        valid = (labels[..., 1:] != -100).float()

        # KL(teacher || student) per token, summed over vocab
        kl = F.kl_div(
            F.log_softmax(s_logits, dim=-1),
            F.softmax(r_logits, dim=-1),
            reduction='none'
        ).sum(dim=-1)  # (batch, seq-1)

        # Masked average, scaled by T^2 (standard distillation scaling)
        return (kl * valid).sum() / valid.sum().clamp(min=1) * (T ** 2)

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

    def _forward_with_hooks_on_model(self, model, batch, layer_indices):
        """Forward pass with activation hooks on an arbitrary model (e.g. frozen anchor)."""
        caches = {}
        hooks = []
        for layer_idx in layer_indices:
            module = model.model.layers[layer_idx]
            def make_hook(idx):
                def hook_fn(mod, inp, out):
                    caches[idx] = out[0] if isinstance(out, tuple) else out
                return hook_fn
            hooks.append(module.register_forward_hook(make_hook(layer_idx)))
        input_ids = batch['input_ids'].to(self.args.device)
        attention_mask = batch['attention_mask'].to(self.args.device)
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
        for h in hooks:
            h.remove()
        return caches

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

        # KL distillation from pretrained model instead of CE for inner retain loss
        if self.inner_loss_type == "kl_pretrained" and self.ref_model is not None:
            loss = self._kl_loss_from_ref(outputs.logits, input_ids, attention_mask, labels)
        else:
            loss = outputs.loss

        # Representation anchor: penalize drift of retain activations from reference model
        if self.inner_repr_anchor and self._repr_anchor_model is not None:
            # Get reference activations (no grad)
            ref_caches = self._forward_with_hooks_on_model(
                self._repr_anchor_model, batch, self.inner_repr_layers
            )
            # Get current model activations at same layers
            cur_caches, _ = self._forward_with_hooks(
                batch, self.inner_repr_layers
            )
            repr_labels = batch.get('labels', batch['input_ids']).to(self.args.device)
            repr_token_mask = (repr_labels != -100).float()
            repr_loss = torch.tensor(0.0, device=self.args.device)
            for layer_idx in self.inner_repr_layers:
                if layer_idx not in ref_caches or layer_idx not in cur_caches:
                    continue
                ref_act = ref_caches[layer_idx].detach()
                cur_act = cur_caches[layer_idx]
                min_seq = min(cur_act.shape[1], ref_act.shape[1])
                cur_act = cur_act[:, :min_seq, :]
                ref_act = ref_act[:, :min_seq, :]
                lmask = repr_token_mask[:, :min_seq]
                diff = (cur_act - ref_act) ** 2
                lmask_exp = lmask.unsqueeze(-1).expand_as(diff)
                # Apply neuron mask if available (only anchor forget-dominant dims)
                if layer_idx in self._steering_neuron_masks:
                    nm = self._steering_neuron_masks[layer_idx].to(cur_act.device)
                    nm_exp = nm.unsqueeze(0).unsqueeze(0).expand_as(diff).float()
                    n_active = nm.sum().clamp(min=1).float()
                    per_sample = (diff * lmask_exp * nm_exp).sum(dim=(1, 2)) / (
                        lmask.sum(dim=1).clamp(min=1).float() * n_active
                    )
                else:
                    per_sample = (diff * lmask_exp).mean(dim=2).sum(dim=1) / lmask.sum(dim=1).clamp(min=1)
                repr_loss = repr_loss + per_sample.mean()
            repr_loss = repr_loss / max(len(self.inner_repr_layers), 1)
            loss = loss + self.inner_repr_alpha * repr_loss

        # Add regularization using the configured type
        reg_loss = self.compute_sparsity_regularizer()
        loss_total = loss + reg_loss

        loss_total.backward()

        # Masked gradient update with layer-wise LR scaling
        # When inner_mask_dict is set (inverted bitmap), inner updates RETAIN-dominant neurons.
        # Otherwise falls back to default (mask > 0 = forget-dominant neurons).
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.grad is not None and name in self.mask_dict:
                    lr = self.eta_in
                    if self.retain_protection_layers:
                        layer_id = self._layer_id_from_param_name(name)
                        if layer_id is not None and layer_id in self.retain_protection_layers:
                            lr = self.eta_in * self.retain_lr_multiplier
                    if self.inner_mask_dict is not None and name in self.inner_mask_dict:
                        binary_mask = (self.inner_mask_dict[name] > 0).float()
                    else:
                        binary_mask = (self.mask_dict[name] > 0).float()
                    param.data.sub_(lr * param.grad * binary_mask)
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

        # When skip_layers is set, include ALL layers except those in the skip list.
        # Split each layer into attn/mlp sub-blocks for memory efficiency.
        # Otherwise, fall back to the original last_n_layers behavior.
        if self.implicit_block_skip_layers:
            blocks = []
            for layer_id in sorted(by_layer.keys()):
                if layer_id in self.implicit_block_skip_layers:
                    continue
                entries = by_layer[layer_id]
                if len(entries) == 0:
                    continue
                # Split into attn and mlp sub-blocks to reduce per-block memory
                attn_entries = [(n, p) for n, p in entries if ".self_attn." in n]
                mlp_entries = [(n, p) for n, p in entries if ".mlp." in n]
                other_entries = [(n, p) for n, p in entries if ".self_attn." not in n and ".mlp." not in n]
                if attn_entries:
                    blocks.append((f"layer_{layer_id}_attn", attn_entries))
                if mlp_entries:
                    blocks.append((f"layer_{layer_id}_mlp", mlp_entries))
                if other_entries:
                    blocks.append((f"layer_{layer_id}_other", other_entries))
            return blocks

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
        L_alm=None,
        L_inner=None,
        outer_iter: Optional[int] = None,
        inner_loss_fn=None,
        alm_loss_fn=None,
    ):
        """
        Apply implicit correction block-by-block to avoid full-model flatten OOM.
        Updates g_alm_dict in-place.
        Pass either (L_alm, L_inner) for double-backward HVP, or
        (inner_loss_fn, alm_loss_fn) for FD-HVP compatible with any attention backend.
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
                    inner_loss_fn=inner_loss_fn,
                    alm_loss_fn=alm_loss_fn,
                )
                corr_parts = self.unflatten_params(g_corr_flat, [g_alm_dict[name] for name in names])
                for name, corr in zip(names, corr_parts):
                    if self.implicit_trace_guided and name in self.mask_dict:
                        # Blend: forget-dominant neurons keep original gradient,
                        # retain/contested neurons get implicit-corrected gradient
                        keep_orig = (self.mask_dict[name] > self.implicit_trace_threshold).float()
                        g_alm_dict[name] = keep_orig * g_alm_dict[name] + (1.0 - keep_orig) * corr
                    else:
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
                    if inner_loss_fn is not None:
                        hvp = self.compute_hvp_fd(inner_loss_fn, params, vec_masked)
                    else:
                        hvp = self.compute_hvp(L_inner, params, vec_masked)
                    return hvp * mask_block + self.cg_damping * vec

                h = self.conjugate_gradient(hvp_func, v_block)
                if alm_loss_fn is not None:
                    corr = self.compute_hvp_fd(alm_loss_fn, params, h)
                else:
                    corr = self.compute_hvp(L_alm, params, h)
                corr_parts = self.unflatten_params(corr, [g_alm_dict[name] for name in names])
                for name, corr_part in zip(names, corr_parts):
                    g_alm_dict[name] = g_alm_dict[name] - corr_part

                lin_res = 0.0
                cond_proxy = {}
                if self.debug_implicit:
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
        """Compute Hessian-vector product H*v via double backward (requires create_graph=True).
        NOTE: This fails with flash_attention_2/sdpa. Use compute_hvp_fd() for those backends.
        """
        grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
        flat_grad = torch.cat([g.reshape(-1) for g in grads])

        grad_v = (flat_grad * v).sum()

        hvp_grads = torch.autograd.grad(grad_v, params, retain_graph=True)
        flat_hvp = torch.cat([h.reshape(-1) for h in hvp_grads])

        return flat_hvp

    def compute_hvp_fd(self, loss_fn, params, v, eps=1e-3):
        """Finite-difference HVP: H*v ≈ (grad(θ+εv̂) - grad(θ-εv̂)) / (2ε) * ||v||
        Works with any attention backend (no create_graph=True needed).

        Normalizes v before perturbation to ensure the step size is always ε regardless
        of ||v||, then scales back. This avoids numerical issues with large-norm v.

        params: list of parameter tensors (the block to differentiate w.r.t.)
        v: flat vector of same size as flattened params
        loss_fn: callable() -> scalar loss (recomputes from current model state)
        """
        v_norm = v.norm().clamp(min=1e-12)
        v_unit = v / v_norm  # unit direction
        v_unit_list = self.unflatten_params(v_unit, params)

        # +ε perturbation in unit direction
        with torch.no_grad():
            for p, dv in zip(params, v_unit_list):
                p.data.add_(eps * dv.to(p.dtype))
        loss_p = loss_fn()
        grads_p = torch.autograd.grad(loss_p, params, allow_unused=True)
        grads_p_list = [
            g.detach().clone() if g is not None else torch.zeros_like(p)
            for g, p in zip(grads_p, params)
        ]
        del loss_p

        # -ε perturbation in unit direction (from +ε → -ε, so subtract 2ε)
        with torch.no_grad():
            for p, dv in zip(params, v_unit_list):
                p.data.sub_(2.0 * eps * dv.to(p.dtype))
        loss_m = loss_fn()
        grads_m = torch.autograd.grad(loss_m, params, allow_unused=True)
        grads_m_list = [
            g.detach().clone() if g is not None else torch.zeros_like(p)
            for g, p in zip(grads_m, params)
        ]
        del loss_m

        # Restore params to original state
        with torch.no_grad():
            for p, dv in zip(params, v_unit_list):
                p.data.add_(eps * dv.to(p.dtype))

        # HVP w.r.t. unit direction, then scale by ||v||:  H*(v) = ||v|| * H*(v̂)
        hvp_unit_list = [(gp - gm) / (2.0 * eps) for gp, gm in zip(grads_p_list, grads_m_list)]
        hvp_flat = torch.cat([h.reshape(-1) for h in hvp_unit_list])
        return hvp_flat * v_norm.item()

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
        L_alm=None,
        L_inner=None,
        inner_loss_fn=None,
        alm_loss_fn=None,
    ):
        """
        Truncated Neumann implicit correction: approximate h ≈ (H_inner + μI)^{-1} v
        with v = masked outer gradient, then g_corr = v - H_outer(h_correct).

        Supports two HVP backends:
        - Double backward (L_alm, L_inner): requires create_graph=True; fails with flash_attention_2.
        - Finite difference (inner_loss_fn, alm_loss_fn): callables that recompute the loss;
          works with any attention backend.

        Returns (g_corr_flat, status_string, meta_dict).
        """
        use_fd = (inner_loss_fn is not None)
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
            if use_fd:
                Hv = self.compute_hvp_fd(inner_loss_fn, params_list, x_act)
            else:
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
        if use_fd:
            c = self.compute_hvp_fd(alm_loss_fn, params_list, h_act)
        else:
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

        # 7) Solve quality and condition proxy (expensive HVPs, only when debugging)
        lin_res = 0.0
        cond_proxy = {}
        if self.debug_implicit:
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

    def _forward_with_hooks(self, batch, layer_indices):
        """Forward pass capturing hidden states at specified layers.

        Returns:
            activations: Dict[layer_idx → Tensor(batch, seq, hidden)]
            outputs: Model outputs
        """
        caches = {}
        hooks = []
        for layer_idx in layer_indices:
            module = self.model.model.layers[layer_idx]
            def make_hook(idx):
                def hook_fn(mod, inp, out):
                    caches[idx] = out[0] if isinstance(out, tuple) else out
                return hook_fn
            hooks.append(module.register_forward_hook(make_hook(layer_idx)))

        input_ids = batch['input_ids'].to(self.args.device)
        attention_mask = batch['attention_mask'].to(self.args.device)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        for h in hooks:
            h.remove()
        return caches, outputs

    def _compute_steering_loss(self, forget_batch, retain_batch=None):
        """Compute activation steering loss: push forget representations at target layers
        toward random control vectors (RMU-style) or toward retain representations.

        This operates in REPRESENTATION SPACE, bypassing weight-space gradient entanglement.
        """
        labels = forget_batch.get('labels', forget_batch['input_ids']).to(self.args.device)
        mask = (labels != -100).float()

        # Forward pass on forget data, capturing activations at steering layers
        forget_caches, _ = self._forward_with_hooks(forget_batch, self.steering_layers)

        # Optionally forward pass on retain data for retain-matching mode
        retain_caches = {}
        if self.steering_retain_match and retain_batch is not None:
            with torch.no_grad():
                retain_caches, _ = self._forward_with_hooks(retain_batch, self.steering_layers)

        total_loss = torch.tensor(0.0, device=self.args.device)

        for layer_idx, activation in forget_caches.items():
            hidden_dim = activation.shape[-1]

            if self.steering_retain_match and layer_idx in retain_caches:
                # Target: match retain activations (make forget look like retain)
                target = retain_caches[layer_idx].detach()
                # Handle sequence length mismatch between forget and retain
                min_seq = min(activation.shape[1], target.shape[1])
                activation = activation[:, :min_seq, :]
                target = target[:, :min_seq, :]
                layer_mask = mask[:, :min_seq] if mask.shape[1] > min_seq else mask
            else:
                # Target: random control vector (RMU-style misdirection)
                if layer_idx not in self._steering_vectors:
                    rv = torch.randn(1, 1, hidden_dim, device=activation.device, dtype=activation.dtype)
                    self._steering_vectors[layer_idx] = rv / rv.norm() * self.steering_coeff
                target = self._steering_vectors[layer_idx].expand_as(activation)
                layer_mask = mask

            # MSE loss with label masking + optional neuron mask
            diff = (activation - target) ** 2  # (batch, seq, hidden)
            mask_exp = layer_mask.unsqueeze(-1).expand_as(diff)  # (batch, seq, hidden)
            if layer_idx in self._steering_neuron_masks:
                # Apply neuron mask: compute MSE only over forget-dominant hidden dims
                nm = self._steering_neuron_masks[layer_idx].to(activation.device)
                nm_exp = nm.unsqueeze(0).unsqueeze(0).expand_as(diff).float()  # (batch, seq, hidden)
                combined = mask_exp * nm_exp
                n_active = nm.sum().clamp(min=1).float()
                per_sample = (diff * combined).sum(dim=2).sum(dim=1) / (
                    (layer_mask.sum(dim=1).clamp(min=1).float() * n_active)
                )
                layer_loss = per_sample.mean()
            else:
                per_sample = (diff * mask_exp).mean(dim=2).sum(dim=1)
                n_tokens = layer_mask.sum(dim=1).clamp(min=1)
                layer_loss = (per_sample / n_tokens).mean()
            total_loss = total_loss + layer_loss

        return total_loss / max(len(forget_caches), 1)

    def _apply_gradient_projection(self, g_alm_dict, retain_batch,
                                     g_forget_dict=None, al_retain_coeff=0.0):
        """Project the forget component of the gradient orthogonal to the retain gradient.

        Instead of projecting the full ALM gradient (which removes AL retain protection),
        this decomposes G_alm = G_fgt + c*G_ret, projects only G_fgt orthogonally,
        then reconstructs: G_update = G_fgt_proj + c*G_ret.

        Args:
            g_alm_dict: Full ALM gradient dict (name → tensor)
            retain_batch: Retain data batch (to compute retain gradient)
            g_forget_dict: Pre-computed forget gradient dict (if available)
            al_retain_coeff: λ + ρ*(L_ret - ε) coefficient for AL retain terms
        """
        # Compute retain gradient — if projection_layers is set, only compute grads
        # for params in those layers to save memory (skips grad alloc for other layers).
        L_ret_proj = self.compute_retain_loss(retain_batch)
        if self.projection_layers is not None:
            # Only compute retain gradients for params in projection_layers (memory efficient)
            target_named_params = [
                (name, param) for name, param in self.model.named_parameters()
                if param.requires_grad and
                (self._layer_id_from_param_name(name) in self.projection_layers)
            ]
            if target_named_params:
                grads = torch.autograd.grad(
                    L_ret_proj,
                    [p for _, p in target_named_params],
                    retain_graph=False,
                    allow_unused=True,
                )
                g_retain_dict = {
                    name: g.detach() for (name, _), g in zip(target_named_params, grads)
                    if g is not None
                }
            else:
                g_retain_dict = {}
            del L_ret_proj
        else:
            L_ret_proj.backward()
            g_retain_dict = {}
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    g_retain_dict[name] = param.grad.clone()
                param.grad = None
            self.model.zero_grad()
        torch.cuda.empty_cache()

        scope = self.gradient_projection_scope
        eps = 1e-10

        # If we have separate forget gradient, use it for projection
        # Otherwise, extract forget component: G_fgt = G_alm - al_retain_coeff * G_ret
        if g_forget_dict is None:
            g_forget_dict = {}
            for name in g_alm_dict:
                if name in g_retain_dict:
                    g_forget_dict[name] = (
                        g_alm_dict[name].float() - al_retain_coeff * g_retain_dict[name].float()
                    )
                else:
                    g_forget_dict[name] = g_alm_dict[name].float()

        def _project_layer(names):
            """Project forget gradient orthogonal to retain gradient (memory-efficient, no concat)."""
            valid_names = [n for n in names if n in g_retain_dict and n in g_forget_dict]
            if not valid_names:
                return
            # Accumulate dot products per-param without concatenating large tensors
            dev = g_retain_dict[valid_names[0]].device
            dot = torch.zeros(1, device=dev, dtype=torch.float32)
            g_r_norm_sq = torch.zeros(1, device=dev, dtype=torch.float32)
            g_f_norm_sq = torch.zeros(1, device=dev, dtype=torch.float32) if self.projection_rescale else None
            for name in valid_names:
                gf = g_forget_dict[name].float()
                gr = g_retain_dict[name].float()
                dot += (gf * gr).sum()
                g_r_norm_sq += (gr * gr).sum()
                if g_f_norm_sq is not None:
                    g_f_norm_sq += (gf * gf).sum()
            g_r_norm_sq = g_r_norm_sq + eps
            coeff = self.projection_strength * (dot / g_r_norm_sq).item()
            # Compute rescale factor without materializing projected concat
            rescale = 1.0
            if self.projection_rescale and g_f_norm_sq is not None:
                g_f_proj_norm_sq = (g_f_norm_sq - 2 * coeff * dot + coeff ** 2 * (g_r_norm_sq - eps)).clamp(min=0.0)
                g_f_proj_norm = g_f_proj_norm_sq.sqrt().item()
                g_f_norm = g_f_norm_sq.sqrt().item()
                rescale = (g_f_norm / (g_f_proj_norm + eps)) if g_f_proj_norm > eps else 1.0
            # Reconstruct: G_update = rescale * G_fgt_proj + al_retain_coeff * G_ret
            for name in valid_names:
                gf = g_forget_dict[name].float()
                gr = g_retain_dict[name].float()
                g_fgt_proj = gf - coeff * gr
                g_alm_dict[name] = (
                    rescale * g_fgt_proj + al_retain_coeff * gr
                ).to(g_alm_dict[name].dtype)

        def _layer_allowed(layer_id):
            """Return True if this layer should have projection applied."""
            if self.projection_layers is None:
                return True
            return layer_id in self.projection_layers

        if scope == "param":
            for name in list(g_alm_dict.keys()):
                lid = self._layer_id_from_param_name(name)
                if lid is None or _layer_allowed(lid):
                    _project_layer([name])
        elif scope == "layer":
            by_layer = {}
            for name in g_alm_dict:
                layer_id = self._layer_id_from_param_name(name)
                if layer_id is None:
                    layer_id = -1
                if layer_id not in by_layer:
                    by_layer[layer_id] = []
                by_layer[layer_id].append(name)
            for layer_id, names in by_layer.items():
                if _layer_allowed(layer_id):
                    _project_layer(names)
        elif scope == "global":
            _project_layer(list(g_alm_dict.keys()))
        else:
            raise ValueError(f"Unknown gradient_projection_scope: {scope}")

        # Clean up
        del g_retain_dict, g_forget_dict
        torch.cuda.empty_cache()

        return g_alm_dict

    def outer_step(self, forget_batch, retain_batch, outer_iter: Optional[int] = None,
                   forget_batches: Optional[list] = None, retain_batches: Optional[list] = None):
        """Outer loop: Update parameters to forget while respecting budget.

        Supports gradient accumulation: if forget_batches/retain_batches are provided,
        gradients are averaged across all mini-batches before the implicit correction
        and weight update. forget_batch/retain_batch are used as the representative
        batch for implicit correction closures (last mini-batch by convention).
        """
        self.model.train()

        # Build list of (forget, retain) mini-batch pairs for gradient accumulation.
        # Falls back to single-batch behaviour when no lists provided.
        if forget_batches is not None and retain_batches is not None:
            batch_pairs = list(zip(forget_batches, retain_batches))
        else:
            batch_pairs = [(forget_batch, retain_batch)]

        n_accum = len(batch_pairs)

        # Accumulate g_alm and scalar losses across mini-batches.
        g_alm_dict = {}
        L_fgt_accum = 0.0
        L_ret_accum = 0.0
        r_accum = 0.0

        for fb, rb in batch_pairs:
            # Compute forget loss
            if self.use_steering:
                L_steer = self._compute_steering_loss(fb, rb)
                if self.steering_only:
                    L_fgt = self.steering_alpha * L_steer
                else:
                    L_logit = self.compute_forget_loss(fb)
                    L_fgt = L_logit + self.steering_alpha * L_steer
            else:
                L_fgt = self.compute_forget_loss(fb)

            # Compute retain loss (CE or KL(pretrained || model))
            retain_ids = rb['input_ids'].to(self.args.device)
            retain_mask = rb['attention_mask'].to(self.args.device)
            retain_labels = rb.get('labels', retain_ids).to(self.args.device)
            retain_outputs = self.model(
                input_ids=retain_ids,
                attention_mask=retain_mask,
                labels=retain_labels
            )
            if self.outer_retain_loss_type == "kl_pretrained" and self.ref_model is not None:
                L_ret = self._kl_loss_from_ref(retain_outputs.logits, retain_ids, retain_mask, retain_labels)
            else:
                L_ret = retain_outputs.loss

            if not torch.isfinite(L_fgt) or not torch.isfinite(L_ret):
                logger.warning(
                    "Outer step received non-finite loss: "
                    f"L_fgt={L_fgt.item() if torch.isfinite(L_fgt) else 'nan/inf'} "
                    f"L_ret={L_ret.item() if torch.isfinite(L_ret) else 'nan/inf'}. "
                    "Skipping outer update."
                )
                return float("nan"), float("nan"), float("nan")

            r_tensor = L_ret - self.epsilon
            L_alm = L_fgt + self.lambda_dual * L_ret + 0.5 * self.rho * (r_tensor ** 2)

            # Scale by 1/n_accum so averaged gradient matches single-batch scale
            (L_alm / n_accum).backward(retain_graph=False)

            for name, param in self.model.named_parameters():
                g = param.grad.clone() if param.grad is not None else torch.zeros_like(param.data)
                if name in g_alm_dict:
                    g_alm_dict[name] = g_alm_dict[name] + g
                else:
                    g_alm_dict[name] = g

            self.model.zero_grad()

            L_fgt_accum += L_fgt.item()
            L_ret_accum += L_ret.item()
            r_accum += r_tensor.item()

        # Average scalar losses for logging / dual update
        L_fgt_val = L_fgt_accum / n_accum
        L_ret_val = L_ret_accum / n_accum
        r = r_accum / n_accum

        # Representative batch for implicit correction closures (last mini-batch)
        forget_batch = batch_pairs[-1][0]
        retain_batch = batch_pairs[-1][1]

        # Implicit correction (if enabled): use outer gradient as RHS (masked g_alm)
        if self.use_implicit:
            # FD-HVP loss functions: recompute from current model state (no create_graph needed)
            # These are closures over retain_batch / forget_batch captured here.
            def _inner_loss_fn():
                """Recompute inner loss (CE or KL) + regularization."""
                if self.inner_loss_type == "kl_pretrained" and self.ref_model is not None:
                    _ids = retain_batch['input_ids'].to(self.args.device)
                    _mask = retain_batch['attention_mask'].to(self.args.device)
                    _labels = retain_batch.get('labels', _ids).to(self.args.device)
                    _out = self.model(input_ids=_ids, attention_mask=_mask, labels=_labels)
                    l = self._kl_loss_from_ref(_out.logits, _ids, _mask, _labels)
                else:
                    l = self.compute_retain_loss(retain_batch)
                l = l + self.compute_sparsity_regularizer()
                return l

            def _alm_loss_fn():
                """Recompute ALM loss: forget + λ*retain + ρ/2*(retain-ε)²."""
                if self.use_steering:
                    l_steer = self._compute_steering_loss(forget_batch, retain_batch)
                    if self.steering_only:
                        l_f = self.steering_alpha * l_steer
                    else:
                        l_f = self.compute_forget_loss(forget_batch) + self.steering_alpha * l_steer
                else:
                    l_f = self.compute_forget_loss(forget_batch)
                if self.outer_retain_loss_type == "kl_pretrained" and self.ref_model is not None:
                    _ids = retain_batch['input_ids'].to(self.args.device)
                    _mask = retain_batch['attention_mask'].to(self.args.device)
                    _labels = retain_batch.get('labels', _ids).to(self.args.device)
                    _out = self.model(input_ids=_ids, attention_mask=_mask, labels=_labels)
                    l_r = self._kl_loss_from_ref(_out.logits, _ids, _mask, _labels)
                else:
                    l_r = self.compute_retain_loss(retain_batch)
                r_t = l_r - self.epsilon
                return l_f + self.lambda_dual * l_r + 0.5 * self.rho * (r_t ** 2)

            # Keep legacy double-backward path available for non-flash-attention runs.
            # Use FD unless L_alm.backward was already called without retain_graph=True.
            # We detect flash/sdpa by checking if retain_graph was set (proxy: always use FD).
            use_fd_hvp = True  # Always use FD: works with flash_attn_2, sdpa, and eager

            L_ret_v = self.compute_retain_loss(retain_batch)
            R_theta_v = self.compute_sparsity_regularizer()
            L_inner = L_ret_v + R_theta_v
            # If inner repr anchor is enabled, add its term to L_inner so that the
            # implicit correction uses the CORRECT inner Hessian (matching actual inner loss).
            if self.inner_repr_anchor and self._repr_anchor_model is not None:
                ref_caches_imp = self._forward_with_hooks_on_model(
                    self._repr_anchor_model, retain_batch, self.inner_repr_layers
                )
                cur_caches_imp, _ = self._forward_with_hooks(retain_batch, self.inner_repr_layers)
                repr_labels = retain_batch.get('labels', retain_batch['input_ids']).to(self.args.device)
                repr_token_mask = (repr_labels != -100).float()
                repr_loss_imp = torch.tensor(0.0, device=self.args.device)
                for layer_idx in self.inner_repr_layers:
                    if layer_idx not in ref_caches_imp or layer_idx not in cur_caches_imp:
                        continue
                    ref_act = ref_caches_imp[layer_idx].detach()
                    cur_act = cur_caches_imp[layer_idx]
                    min_seq = min(cur_act.shape[1], ref_act.shape[1])
                    cur_act = cur_act[:, :min_seq, :]
                    ref_act = ref_act[:, :min_seq, :]
                    lmask = repr_token_mask[:, :min_seq]
                    diff = (cur_act - ref_act) ** 2
                    lmask_exp = lmask.unsqueeze(-1).expand_as(diff)
                    per_sample = (diff * lmask_exp).mean(dim=2).sum(dim=1) / lmask.sum(dim=1).clamp(min=1)
                    repr_loss_imp = repr_loss_imp + per_sample.mean()
                repr_loss_imp = repr_loss_imp / max(len(self.inner_repr_layers), 1)
                L_inner = L_inner + self.inner_repr_alpha * repr_loss_imp
            if self.implicit_blockwise:
                self._apply_blockwise_implicit_correction(
                    g_alm_dict=g_alm_dict,
                    L_alm=None if use_fd_hvp else L_alm,
                    L_inner=None if use_fd_hvp else L_inner,
                    outer_iter=outer_iter,
                    inner_loss_fn=_inner_loss_fn if use_fd_hvp else None,
                    alm_loss_fn=_alm_loss_fn if use_fd_hvp else None,
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
                        L_alm=None if use_fd_hvp else L_alm,
                        L_inner=None if use_fd_hvp else L_inner,
                        inner_loss_fn=_inner_loss_fn if use_fd_hvp else None,
                        alm_loss_fn=_alm_loss_fn if use_fd_hvp else None,
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
                            if self.implicit_trace_guided and name in self.mask_dict:
                                keep_orig = (self.mask_dict[name] > self.implicit_trace_threshold).float()
                                g_alm_dict[name] = keep_orig * g_alm_dict[name] + (1.0 - keep_orig) * g_update
                            else:
                                g_alm_dict[name] = g_update
                else:
                    # CG path: solve (H_inner + damping)*h = v with v = masked g_alm (outer gradient)
                    v = g_alm_flat * mask_flat

                    def hvp_func(vec):
                        vec_masked = vec * mask_flat
                        if use_fd_hvp:
                            hvp = self.compute_hvp_fd(_inner_loss_fn, params_list, vec_masked)
                        else:
                            hvp = self.compute_hvp(L_inner, params_list, vec_masked)
                        return hvp * mask_flat + self.cg_damping * vec

                    h = self.conjugate_gradient(hvp_func, v)
                    if use_fd_hvp:
                        hvp_correction = self.compute_hvp_fd(_alm_loss_fn, params_list, h)
                    else:
                        hvp_correction = self.compute_hvp(L_alm, params_list, h)
                    lin_res = 0.0
                    cond_proxy = {}
                    if self.debug_implicit:
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

        # Gradient disentanglement: project forget component orthogonal to retain gradient
        if self.gradient_projection:
            # al_retain_coeff: how much of the AL retain gradient to add back
            # When 0: project full ALM gradient (aggressive forget, less retain protection)
            # When λ+ρr: preserve AL constraint terms (conservative, better retain)
            if self.gradient_projection_scope == "aggressive":
                # Special mode: project full gradient without AL decomposition
                al_retain_coeff = 0.0
                # Use regular layer-based projection
                self.gradient_projection_scope = "layer"
                g_alm_dict = self._apply_gradient_projection(
                    g_alm_dict, retain_batch, al_retain_coeff=al_retain_coeff
                )
                self.gradient_projection_scope = "aggressive"
            else:
                al_retain_coeff = self.lambda_dual + self.rho * r
                g_alm_dict = self._apply_gradient_projection(
                    g_alm_dict, retain_batch, al_retain_coeff=al_retain_coeff
                )

        # Primal update: mask values encode LR scale; outer_freeze_layers skipped
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.mask_dict and name in g_alm_dict:
                    # Skip outer update for frozen layers (they still get inner retain updates)
                    if self.outer_freeze_layers:
                        layer_id = self._layer_id_from_param_name(name)
                        if layer_id is not None and layer_id in self.outer_freeze_layers:
                            param.grad = None
                            continue
                    mask = self.mask_dict[name]
                    param.data.sub_(self.eta_theta * g_alm_dict[name] * mask)
                param.grad = None

        # Two-phase ALM: boost λ and ρ at specified step
        if (self.lambda_boost_step > 0 and not self._lambda_boosted
                and outer_iter is not None and outer_iter >= self.lambda_boost_step):
            self.lambda_dual = max(self.lambda_dual, self.lambda_boost_value)
            self.rho = self.rho_boost_value
            self._lambda_boosted = True
            logger.info(f"[Step {outer_iter}] ALM boost: λ→{self.lambda_dual:.2f}, ρ→{self.rho:.2f} "
                       f"(two-phase transition to retain recovery)")

        # Dual update
        self.lambda_dual = max(0.0, self.lambda_dual + self.rho * r)

        return L_fgt_val, L_ret_val, r

    def train(self):
        """Override the train method to implement custom S-BiAL training loop."""
        # Initialize mask
        if self.mask_dict is None:
            self._initialize_mask()

        # Build inverted inner mask when bitmap is loaded and invert_inner_mask is enabled.
        # Inner loop updates RETAIN-dominant neurons (1 - bitmap), outer keeps bitmap as-is.
        if self.invert_inner_mask and self.neuron_bitmap_path and self.inner_mask_dict is None:
            self.inner_mask_dict = {}
            bitmap_params = getattr(self, '_bitmap_params', set())
            for name, mask in self.mask_dict.items():
                if name in bitmap_params:
                    self.inner_mask_dict[name] = (1.0 - mask)
                else:
                    # Non-bitmap params (embeddings, layernorm, lm_head) stay frozen
                    self.inner_mask_dict[name] = mask.clone()
            n_inner = sum((m > 0).sum().item() for m in self.inner_mask_dict.values())
            n_total = sum(m.numel() for m in self.inner_mask_dict.values())
            logger.info(f"Inverted inner mask: {n_inner:,}/{n_total:,} active ({100*n_inner/max(n_total,1):.2f}%) "
                        f"— inner loop updates RETAIN-dominant neurons only")

        # After building inner_mask_dict, optionally reset outer mask to full model.
        # This gives G1-level forget (NPO on all params) + targeted inner retain (inverted bitmap).
        if self.outer_full_model and self.invert_inner_mask and self.inner_mask_dict is not None:
            bitmap_params = getattr(self, '_bitmap_params', set())
            for name, param in self.model.named_parameters():
                if name in bitmap_params:
                    self.mask_dict[name] = torch.ones_like(param.data).to(self.args.device)
                # Non-bitmap params stay frozen (zeros) — consistent with bitmap behavior
            logger.info("Outer full model: outer mask reset to ones for bitmap params (full NPO like G1)")

        # Eagerly initialize NPO reference model so it snapshots the ORIGINAL model
        # (before any inner/outer steps). Lazy init risks snapshotting a post-inner model.
        # Also needed when inner_loss_type or post_inner_loss_type is kl_pretrained.
        needs_ref = (self.forget_loss_type == "npo"
                     or self.inner_loss_type == "kl_pretrained"
                     or self.post_inner_loss_type == "kl_pretrained"
                     or self.outer_retain_loss_type == "kl_pretrained")
        if needs_ref and self.ref_model is None:
            self._prepare_ref_model()

        # Initialize inner representation anchor model if needed
        if self.inner_repr_anchor and self._repr_anchor_model is None:
            # Reuse NPO ref_model if available (same frozen copy), else create new
            if self.ref_model is not None:
                self._repr_anchor_model = self.ref_model
                logger.info(f"Inner repr anchor: reusing NPO reference model, layers={self.inner_repr_layers}")
            else:
                logger.info("Creating frozen reference model for inner repr anchor...")
                self._repr_anchor_model = copy.deepcopy(self.model)
                self._repr_anchor_model.eval()
                for p in self._repr_anchor_model.parameters():
                    p.requires_grad = False
                self._repr_anchor_model = self._repr_anchor_model.to(self.args.device)
                logger.info(f"Inner repr anchor ready, layers={self.inner_repr_layers}")

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

        # Gradient accumulation: number of mini-batches per outer step
        accum_steps = max(1, self.args.gradient_accumulation_steps)

        # Determine total outer steps: honour self.T if set explicitly (legacy),
        # otherwise derive from num_train_epochs × steps_per_epoch.
        steps_per_epoch = max(1, len(train_dataloader) // accum_steps)
        num_epochs = int(self.args.num_train_epochs) if self.args.num_train_epochs > 0 else 1
        total_outer_steps = num_epochs * steps_per_epoch
        # If T was explicitly configured (non-default), honour it as an override.
        # Default T=10 in the original code was the entire "training"; now it's per-epoch steps.
        # We only use self.T if it differs from the dataloader-derived count, as an explicit cap.
        if self.T != total_outer_steps:
            logger.info(
                f"S-BiAL: self.T={self.T} overridden by epoch-derived total "
                f"({num_epochs} epochs × {steps_per_epoch} steps/epoch = {total_outer_steps} outer steps). "
                f"Set trainer.method_args.T explicitly to cap."
            )
        effective_T = total_outer_steps

        # Single data iterator that advances through ALL epochs
        data_iter = iter(train_dataloader)
        epoch_step = 0  # counts mini-batches consumed to track epoch boundaries
        current_epoch = 0

        logger.info(f"\nStarting S-BiAL unlearning for {effective_T} outer steps "
                    f"({num_epochs} epochs, {steps_per_epoch} steps/epoch, accum={accum_steps})...")
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

        if self.gradient_projection:
            sched_str = f", schedule={self.projection_schedule}" if self.projection_schedule != "constant" else ""
            logger.info(f"Gradient projection: enabled (scope={self.gradient_projection_scope}, strength={self.projection_strength}{sched_str})")

        if self.use_steering:
            mode = "retain_match" if self.steering_retain_match else "random"
            mix = "steering_only" if self.steering_only else f"mixed (α={self.steering_alpha})"
            logger.info(f"Activation steering: enabled (layers={self.steering_layers}, coeff={self.steering_coeff}, mode={mode}, {mix})")

        if self.inner_loss_type != "ce":
            logger.info(f"Inner loop loss: {self.inner_loss_type} (T_kl={self.inner_kl_temperature})")
        if self.outer_retain_loss_type != "ce":
            logger.info(f"Outer retain loss: {self.outer_retain_loss_type} (T_kl={self.inner_kl_temperature})")
        if self.invert_inner_mask:
            logger.info("Inverted inner mask: inner loop updates RETAIN neurons, outer updates FORGET neurons")
        if self.implicit_trace_guided:
            logger.info(f"Trace-guided implicit: blending threshold={self.implicit_trace_threshold}")

        for t in range(effective_T):
            t_start = time.time()

            # Collect accum_steps mini-batches for this outer step
            forget_batches = []
            retain_batches = []
            for _ in range(accum_steps):
                try:
                    combined_batch = next(data_iter)
                    epoch_step += 1
                except StopIteration:
                    # End of epoch — restart iterator
                    data_iter = iter(train_dataloader)
                    epoch_step = 0
                    try:
                        combined_batch = next(data_iter)
                        epoch_step += 1
                    except StopIteration:
                        break
                forget_batches.append(combined_batch['forget'])
                retain_batches.append(combined_batch['retain'])

            if not forget_batches:
                logger.error(f"No batches available at outer step {t}, stopping.")
                break

            # Inner loop: K steps over the accumulated retain batches (cycling)
            retain_loader = (retain_batches * ((self.K // len(retain_batches)) + 1))[:self.K]
            self.inner_loop(retain_loader)

            # Update projection strength based on schedule
            if self.gradient_projection and self.projection_schedule == "linear_decay":
                self.projection_strength = self._projection_strength_base * (1.0 - t / max(effective_T - 1, 1))

            # Outer step with gradient accumulation across all mini-batches
            L_fgt, L_ret, r = self.outer_step(
                forget_batch=forget_batches[-1],
                retain_batch=retain_batches[-1],
                outer_iter=t,
                forget_batches=forget_batches,
                retain_batches=retain_batches,
            )

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
            if t % max(1, effective_T // 20) == 0 or t == effective_T - 1:
                proj_str = f" | α={self.projection_strength:.2f}" if self.gradient_projection and self.projection_schedule != "constant" else ""
                logger.info(
                    f"[{t:3d}/{effective_T}] L_fgt={L_fgt:.4f} | L_ret={L_ret:.4f} | "
                    f"r={r:+.4f} | λ={self.lambda_dual:.3f}{proj_str} | t={t_elapsed:.2f}s"
                )

            # Update training state
            self.state.global_step = t + 1

            # Per-step checkpoint saving (for trajectory analysis)
            if self.checkpoint_every_steps > 0 and (t + 1) % self.checkpoint_every_steps == 0:
                ckpt_dir = os.path.join(self.args.output_dir, f"checkpoint-step-{t+1}")
                logger.info(f"Saving step checkpoint to {ckpt_dir}")
                self.model.save_pretrained(ckpt_dir)

            if (
                self.debug_stop_after_outer is not None
                and t >= self.debug_stop_after_outer
            ):
                logger.info(
                    f"Debug stop requested at outer_iter={t} "
                    f"(debug_stop_after_outer={self.debug_stop_after_outer})"
                )
                break

            # Epoch-boundary evaluation
            if steps_per_epoch > 0 and (t + 1) % steps_per_epoch == 0:
                current_epoch += 1
                if current_epoch < num_epochs:  # skip final epoch — full eval runs after post-inner
                    logger.info(f"=== Epoch {current_epoch}/{num_epochs} complete — running mid-training eval ===")
                    _orig_output_dir = self.args.output_dir
                    epoch_eval_dir = os.path.join(_orig_output_dir, f"evals", f"epoch_{current_epoch}")
                    os.makedirs(epoch_eval_dir, exist_ok=True)
                    self.args.output_dir = epoch_eval_dir
                    try:
                        self.evaluate()
                    except Exception as e:
                        logger.warning(f"Mid-training eval failed at epoch {current_epoch}: {e}")
                    finally:
                        self.args.output_dir = _orig_output_dir

        # Post-unlearning retention recovery: pure CE inner steps to close the retain gap
        # The last outer step often leaves L_ret elevated with no recovery pass.
        # Running N pure inner steps here restores retention without disrupting forgetting.
        if self.post_unlearn_inner_steps > 0:
            logger.info(f"Post-unlearning inner recovery: {self.post_unlearn_inner_steps} pure CE steps...")
            # Temporarily disable repr anchor so recovery is pure CE only
            _saved_anchor = self.inner_repr_anchor
            self.inner_repr_anchor = False

            # Free NPO reference model — no longer needed (unless KL post-inner needs it)
            if self.ref_model is not None and self.post_inner_loss_type != "kl_pretrained":
                logger.info("Freeing NPO reference model before post-inner recovery (saves ~14GB GPU memory)...")
                del self.ref_model
                self.ref_model = None
                torch.cuda.empty_cache()
            elif self.post_inner_loss_type == "kl_pretrained" and self.ref_model is not None:
                logger.info("Keeping NPO reference model for KL post-inner recovery")

            # Determine recovery mask mode:
            #   1. DGA soft mask (post_inner_soft_mask_path): continuous α = σ(-β*s)
            #      — contested neurons get partial updates, no binary cliff edge.
            #   2. Binary bitmap inversion (post_inner_retain_only): exact G3 behavior.
            #   3. No mask change: standard inner_step with existing mask.
            _saved_mask_dict = None
            _use_soft_mask = False

            if self.post_inner_soft_mask_path is not None:
                # DGA soft mask: overrides binary inversion even if post_inner_retain_only=True
                logger.info(f"Post-inner: DGA soft mask (β={self.post_inner_soft_mask_beta}) "
                            f"from {self.post_inner_soft_mask_path}")
                selectivity = torch.load(
                    self.post_inner_soft_mask_path, map_location="cpu", weights_only=False
                )
                _soft_mask = self._build_dga_soft_mask(selectivity, self.post_inner_soft_mask_beta)
                _use_soft_mask = True
                # NOTE: we do NOT replace self.mask_dict here — we apply soft_mask inline below
                # so that inner_step's binary binarization is bypassed.

            elif self.post_inner_retain_only and self.mask_dict is not None:
                logger.info("Post-inner: inverted mask active — updating only retain-dominant neurons (bitmap=0)")
                _saved_mask_dict = self.mask_dict
                _bitmap_params = getattr(self, '_bitmap_params', set())
                self.mask_dict = {
                    # Only invert params that are actually in the bitmap.
                    # bitmap=1 (forget-dominant) → 0 (frozen); bitmap=0 (retain-dominant) → 1 (update).
                    # Params not in bitmap (embeddings, layernorm, lm_head) stay frozen (zeros).
                    name: (1.0 - mask).clamp(min=0.0, max=1.0) if name in _bitmap_params else torch.zeros_like(mask)
                    for name, mask in self.mask_dict.items()
                }

            # Temporarily swap inner_loss_type if post_inner_loss_type differs
            _saved_inner_loss_type = self.inner_loss_type
            if self.post_inner_loss_type != "ce" and self.post_inner_loss_type != self.inner_loss_type:
                logger.info(f"Post-inner: overriding inner_loss_type {self.inner_loss_type} → {self.post_inner_loss_type}")
                self.inner_loss_type = self.post_inner_loss_type

            retain_loader_post = self.get_train_dataloader()
            data_iter_post = iter(retain_loader_post)
            for k in range(self.post_unlearn_inner_steps):
                try:
                    combined = next(data_iter_post)
                except StopIteration:
                    data_iter_post = iter(retain_loader_post)
                    combined = next(data_iter_post)
                retain_batch = combined['retain']

                if _use_soft_mask:
                    # Soft-masked update: apply α continuously (NOT binarized).
                    # inner_step binarizes mask → bypassed here.
                    self.model.train()
                    input_ids = retain_batch['input_ids'].to(self.args.device)
                    attention_mask = retain_batch['attention_mask'].to(self.args.device)
                    labels = retain_batch.get('labels', input_ids).to(self.args.device)
                    outputs = self.model(
                        input_ids=input_ids, attention_mask=attention_mask, labels=labels
                    )
                    # KL or CE loss for recovery
                    if self.inner_loss_type == "kl_pretrained" and self.ref_model is not None:
                        recovery_loss = self._kl_loss_from_ref(outputs.logits, input_ids, attention_mask, labels)
                    else:
                        recovery_loss = outputs.loss
                    recovery_loss.backward()
                    with torch.no_grad():
                        for name, param in self.model.named_parameters():
                            if param.grad is not None:
                                alpha = _soft_mask.get(name)
                                if alpha is not None:
                                    param.data.sub_(self.eta_in * param.grad * alpha)
                            param.grad = None
                else:
                    self.inner_step(retain_batch)

                if (k + 1) % 5 == 0 or k == self.post_unlearn_inner_steps - 1:
                    logger.info(f"  Post-inner step {k+1}/{self.post_unlearn_inner_steps} done")

            # Restore inner_loss_type
            self.inner_loss_type = _saved_inner_loss_type

            # Restore original mask
            if _saved_mask_dict is not None:
                self.mask_dict = _saved_mask_dict

            self.inner_repr_anchor = _saved_anchor
            # Free ref_model now if it was kept for KL post-inner
            if self.ref_model is not None and self.post_inner_loss_type == "kl_pretrained":
                del self.ref_model
                self.ref_model = None
                torch.cuda.empty_cache()
            logger.info("Post-unlearning retention recovery complete.")

        self.save_model()
        self.evaluate()
        if self.debug_implicit and len(self._debug_records) > 0:
            summary_path = os.path.join(self.debug_dir, "implicit_debug_summary.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(self._debug_records, f, indent=2)
            logger.info(f"Saved implicit debug summary to {summary_path}")
        logger.info("Unlearning complete!")

        # Return training output
        return type('TrainOutput', (), {'global_step': effective_T, 'training_loss': L_ret})()

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