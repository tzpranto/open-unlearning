#!/usr/bin/env python
"""
Debug-only implicit capture on a parameter subset.

This script runs a minimal SIBL step and captures matrices/vectors for a selected
parameter block to avoid full-model flatten OOM.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import List, Tuple

import numpy as np
import torch
from hydra import compose, initialize

from data import get_collators, get_data
from model import get_model
from trainer import load_trainer
from trainer.utils import seed_everything


def _flatten_tensors(ts: List[torch.Tensor]) -> torch.Tensor:
    return torch.cat([t.reshape(-1) for t in ts])


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _save(path: str, tensor: torch.Tensor) -> None:
    np.save(path, tensor.detach().float().cpu().numpy())


def build_cfg(overrides: List[str]):
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="unlearn.yaml", overrides=overrides)
    return cfg


def pick_params(
    trainer,
    regex: str,
    max_params: int,
) -> List[Tuple[str, torch.nn.Parameter]]:
    pat = re.compile(regex)
    selected = []
    for name, p in trainer.model.named_parameters():
        if not p.requires_grad:
            continue
        if pat.search(name):
            selected.append((name, p))
            if len(selected) >= max_params:
                break
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-name",
        type=str,
        default="muse_Llama-2-7b-hf_News_SIBL_debug_block_capture",
    )
    parser.add_argument("--data-split", type=str, default="News")
    parser.add_argument("--model", type=str, default="Llama-2-7b-hf")
    parser.add_argument(
        "--param-regex",
        type=str,
        default=r"model\.layers\.(30|31)\..*(q_proj|k_proj|v_proj|o_proj|down_proj)\.weight",
    )
    parser.add_argument("--max-params", type=int, default=8)
    parser.add_argument("--out-dir", type=str, default="debug/captured_block")
    parser.add_argument("--neumann-mu", type=float, default=0.01)
    parser.add_argument("--neumann-steps", type=int, default=4)
    parser.add_argument("--neumann-alpha-default", type=float, default=0.1)
    parser.add_argument("--neumann-use-probe-alpha", action="store_true")
    parser.add_argument("--cg-damping", type=float, default=0.001)
    parser.add_argument("--cg-iters", type=int, default=10)
    args = parser.parse_args()

    _ensure_dir(args.out_dir)

    overrides = [
        "experiment=unlearn/muse/sibl.yaml",
        f"model={args.model}",
        f"data_split={args.data_split}",
        "trainer=SIBL",
        f"task_name={args.task_name}",
        "trainer.args.per_device_train_batch_size=1",
        "trainer.args.gradient_accumulation_steps=1",
        "trainer.args.gradient_checkpointing=true",
        "trainer.args.ddp_find_unused_parameters=true",
        "trainer.args.do_eval=false",
        "trainer.method_args.use_implicit=false",
        "trainer.method_args.T=1",
        "trainer.method_args.K=1",
        f"trainer.method_args.neumann_mu={args.neumann_mu}",
        f"trainer.method_args.neumann_steps={args.neumann_steps}",
        f"trainer.method_args.neumann_alpha_default={args.neumann_alpha_default}",
        f"trainer.method_args.neumann_use_probe_alpha={str(args.neumann_use_probe_alpha).lower()}",
        f"trainer.method_args.cg_damping={args.cg_damping}",
        f"trainer.method_args.cg_iters={args.cg_iters}",
    ]

    cfg = build_cfg(overrides)
    seed_everything(cfg.trainer.args.seed)

    model, tokenizer = get_model(cfg.model)
    data = get_data(
        cfg.data,
        mode=cfg.get("mode", "unlearn"),
        tokenizer=tokenizer,
        template_args=cfg.model.template_args,
    )
    collator = get_collators(cfg.collator, tokenizer=tokenizer)

    trainer, _ = load_trainer(
        trainer_cfg=cfg.trainer,
        model=model,
        train_dataset=data.get("train", None),
        eval_dataset=data.get("eval", None),
        tokenizer=tokenizer,
        data_collator=collator,
        evaluators=None,
        template_args=cfg.model.template_args,
    )

    # Build mask and get one batch.
    trainer._initialize_mask()
    train_dataloader = trainer.get_train_dataloader()
    combined_batch = next(iter(train_dataloader))
    forget_batch = combined_batch["forget"]
    retain_batch = combined_batch["retain"]

    # One short inner update.
    trainer.inner_loop([retain_batch])

    # Compute ALM pieces exactly as training outer-step, but stay debug-local.
    trainer.model.train()
    L_fgt = trainer.compute_forget_loss(forget_batch)
    L_ret = trainer.compute_retain_loss(retain_batch)
    r_tensor = L_ret - trainer.epsilon
    L_alm = L_fgt + trainer.lambda_dual * L_ret + 0.5 * trainer.rho * (r_tensor ** 2)
    L_alm.backward(retain_graph=True)

    g_alm_dict = {}
    for name, p in trainer.model.named_parameters():
        if p.grad is not None:
            g_alm_dict[name] = p.grad.detach().clone()
        else:
            g_alm_dict[name] = torch.zeros_like(p.data)
    trainer.model.zero_grad()

    # Inner objective for HVPs.
    L_ret_v = trainer.compute_retain_loss(retain_batch)
    R_theta_v = trainer.compute_sparsity_regularizer()
    L_inner = L_ret_v + R_theta_v

    selected = pick_params(trainer, args.param_regex, args.max_params)
    if len(selected) == 0:
        raise RuntimeError(f"No params matched regex: {args.param_regex}")

    names = [n for n, _ in selected]
    params_list = [p for _, p in selected]
    g_list = [g_alm_dict[n] for n in names]
    m_list = [trainer.mask_dict[n] for n in names]
    g_flat = _flatten_tensors(g_list)
    mask_flat = _flatten_tensors(m_list)
    v_flat = g_flat * mask_flat

    # Neumann (Richardson) debug on selected block.
    old_variant = trainer.neumann_variant
    trainer.neumann_variant = "richardson"
    g_corr_flat, neumann_status, neumann_meta = trainer._truncated_neumann_correction(
        params_list=params_list,
        g_alm_flat=g_flat,
        mask_flat=mask_flat,
        L_alm=L_alm,
        L_inner=L_inner,
    )
    trainer.neumann_variant = old_variant

    # CG debug on selected block.
    def hvp_func(vec):
        vec_masked = vec * mask_flat
        hv = trainer.compute_hvp(L_inner, params_list, vec_masked)
        return hv * mask_flat + trainer.cg_damping * vec

    h_cg = trainer.conjugate_gradient(hvp_func, v_flat)
    cg_lin_res = ((hvp_func(h_cg) - v_flat).norm() / (v_flat.norm().clamp(min=1e-12))).item()

    # Persist artifacts.
    _save(os.path.join(args.out_dir, "g_flat.npy"), g_flat)
    _save(os.path.join(args.out_dir, "mask_flat.npy"), mask_flat)
    _save(os.path.join(args.out_dir, "v_flat.npy"), v_flat)
    _save(os.path.join(args.out_dir, "g_corr_neumann.npy"), g_corr_flat)
    _save(os.path.join(args.out_dir, "h_cg.npy"), h_cg)

    summary = {
        "task_name": args.task_name,
        "data_split": args.data_split,
        "model": args.model,
        "selected_param_names": names,
        "selected_total_dim": int(v_flat.numel()),
        "losses": {
            "L_fgt": float(L_fgt.item()),
            "L_ret": float(L_ret.item()),
            "r": float(r_tensor.item()),
            "L_alm": float(L_alm.item()),
        },
        "neumann": {
            "status": neumann_status,
            "meta": neumann_meta,
            "v_norm": float(v_flat.norm().item()),
            "g_corr_norm": float(g_corr_flat.norm().item()),
        },
        "cg": {
            "linear_residual": float(cg_lin_res),
            "h_norm": float(h_cg.norm().item()),
            "v_norm": float(v_flat.norm().item()),
        },
        "paths": {
            "g_flat": os.path.join(args.out_dir, "g_flat.npy"),
            "mask_flat": os.path.join(args.out_dir, "mask_flat.npy"),
            "v_flat": os.path.join(args.out_dir, "v_flat.npy"),
            "g_corr_neumann": os.path.join(args.out_dir, "g_corr_neumann.npy"),
            "h_cg": os.path.join(args.out_dir, "h_cg.npy"),
        },
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    md_lines = [
        "# Captured Implicit Block",
        "",
        f"- Selected params: `{len(names)}`",
        f"- Selected dim: `{summary['selected_total_dim']}`",
        f"- L_fgt: `{summary['losses']['L_fgt']:.6f}`",
        f"- L_ret: `{summary['losses']['L_ret']:.6f}`",
        f"- L_alm: `{summary['losses']['L_alm']:.6f}`",
        f"- Neumann status: `{neumann_status}`",
        f"- Neumann linear residual: `{neumann_meta.get('linear_residual')}`",
        f"- CG linear residual: `{cg_lin_res:.6e}`",
        "",
        "## Selected parameters",
    ]
    md_lines.extend([f"- `{n}`" for n in names])
    with open(os.path.join(args.out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"Wrote artifacts to {args.out_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
