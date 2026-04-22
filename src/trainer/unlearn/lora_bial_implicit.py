"""FD-HVP + Truncated Neumann implicit correction for LoRA-BiAL."""

import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)


def unflatten_params(flat_vec, params_list):
    out = []
    offset = 0
    for p in params_list:
        n = p.numel()
        out.append(flat_vec[offset:offset + n].reshape(p.shape))
        offset += n
    return out


def compute_hvp_fd(loss_fn, params, v, eps=0.01):
    """Finite-difference HVP: H*v ≈ (∇L(θ+εv̂) - ∇L(θ-εv̂)) / (2ε) * ||v||"""
    v_norm = v.norm().clamp(min=1e-12)
    v_unit = v / v_norm
    v_list = unflatten_params(v_unit, params)

    with torch.no_grad():
        for p, dv in zip(params, v_list):
            p.data.add_(eps * dv.to(p.dtype))
    loss_p = loss_fn()
    grads_p = torch.autograd.grad(loss_p, params, allow_unused=True)
    grads_p = [g.detach().clone() if g is not None else torch.zeros_like(p) for g, p in zip(grads_p, params)]
    del loss_p

    with torch.no_grad():
        for p, dv in zip(params, v_list):
            p.data.sub_(2.0 * eps * dv.to(p.dtype))
    loss_m = loss_fn()
    grads_m = torch.autograd.grad(loss_m, params, allow_unused=True)
    grads_m = [g.detach().clone() if g is not None else torch.zeros_like(p) for g, p in zip(grads_m, params)]
    del loss_m

    with torch.no_grad():
        for p, dv in zip(params, v_list):
            p.data.add_(eps * dv.to(p.dtype))

    hvp_flat = torch.cat([(gp - gm) / (2.0 * eps) for gp, gm in zip(grads_p, grads_m)])
    return hvp_flat * v_norm.item()


def truncated_neumann(params_list, v, inner_loss_fn, outer_loss_fn, cfg):
    """Truncated Neumann implicit correction.

    cfg should have: neumann_steps, neumann_mu, neumann_alpha_default,
    neumann_alpha_min, neumann_alpha_max, neumann_use_probe_alpha,
    neumann_max_growth_ratio, fd_hvp_eps, implicit_offload_cpu
    """
    device = v.device
    offload = cfg.get("implicit_offload_cpu", False)

    if not torch.isfinite(v).all():
        logger.warning("Neumann: non-finite v, skipping correction")
        return v, "fallback_nonfinite_v"

    def H_in(x):
        x_gpu = x.to(device) if offload else x
        Hv = compute_hvp_fd(inner_loss_fn, params_list, x_gpu, eps=cfg["fd_hvp_eps"])
        result = Hv + cfg["neumann_mu"] * x_gpu
        return result.cpu() if offload else result

    alpha = cfg["neumann_alpha_default"]
    if cfg.get("neumann_use_probe_alpha", True):
        u = torch.randn_like(v)
        u = u / u.norm().clamp(min=1e-12)
        Hu = H_in(u)
        L_est = Hu.norm().clamp(min=1e-12).item()
        alpha = 0.5 / (L_est + 1e-12)
        if not np.isfinite(alpha) or alpha <= 0:
            alpha = cfg["neumann_alpha_default"]
    alpha = float(np.clip(alpha, cfg["neumann_alpha_min"], cfg["neumann_alpha_max"]))

    work_v = v.cpu() if offload else v
    h = torch.zeros_like(work_v)
    for j in range(cfg["neumann_steps"]):
        residual = work_v - H_in(h)
        if not torch.isfinite(residual).all():
            logger.warning(f"Neumann step {j}: non-finite residual, fallback")
            return v, "fallback_nonfinite"
        h = h + alpha * residual
        if not torch.isfinite(h).all():
            logger.warning(f"Neumann step {j}: non-finite h, fallback")
            return v, "fallback_nonfinite"

    h_gpu = h.to(device) if offload else h
    c = compute_hvp_fd(outer_loss_fn, params_list, h_gpu, eps=cfg["fd_hvp_eps"])
    g_corr = v - c

    v_norm = v.norm().clamp(min=1e-12).item()
    g_corr_norm = g_corr.norm().item()
    max_growth = cfg["neumann_max_growth_ratio"]
    if g_corr_norm > max_growth * v_norm:
        logger.warning(f"Neumann: correction exploded ||g_corr||={g_corr_norm:.4f} > {max_growth}*||v||={v_norm:.4f}, fallback")
        return v, "fallback_exploded"

    logger.debug(f"Neumann: α={alpha:.6f} ||v||={v_norm:.4f} ||h||={h_gpu.norm().item():.4f} ||g_corr||={g_corr_norm:.4f}")
    return g_corr, "ok"
