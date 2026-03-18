#!/usr/bin/env python
"""
Debug-only blockwise implicit experiment for SIBL.

Purpose:
- avoid full-model flatten in implicit correction
- run implicit correction block-by-block (layer-wise approximation)
- save a compact report for decision making
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
from hydra import compose, initialize

from data import get_collators, get_data
from model import get_model
from trainer import load_trainer
from trainer.utils import seed_everything


def _flatten_tensors(ts: List[torch.Tensor]) -> torch.Tensor:
    if len(ts) == 0:
        return torch.empty(0)
    return torch.cat([t.reshape(-1) for t in ts])


def _unflatten_like(flat: torch.Tensor, refs: List[torch.Tensor]) -> List[torch.Tensor]:
    out = []
    offset = 0
    for r in refs:
        n = r.numel()
        out.append(flat[offset : offset + n].reshape_as(r))
        offset += n
    return out


def _layer_id(name: str) -> Optional[int]:
    m = re.search(r"model\.layers\.(\d+)\.", name)
    if m is None:
        return None
    return int(m.group(1))


def build_cfg(overrides: List[str]):
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="unlearn.yaml", overrides=overrides)
    return cfg


def build_layer_blocks(
    trainer,
    last_n_layers: int,
    include_mlp: bool,
    include_attn: bool,
) -> List[Tuple[str, List[Tuple[str, torch.nn.Parameter]]]]:
    by_layer: Dict[int, List[Tuple[str, torch.nn.Parameter]]] = defaultdict(list)
    max_layer = -1

    for name, p in trainer.model.named_parameters():
        if not p.requires_grad:
            continue
        if name not in trainer.mask_dict:
            continue
        lid = _layer_id(name)
        if lid is None:
            continue
        is_attn = ".self_attn." in name
        is_mlp = ".mlp." in name
        if (is_attn and not include_attn) or (is_mlp and not include_mlp):
            continue
        by_layer[lid].append((name, p))
        max_layer = max(max_layer, lid)

    if max_layer < 0:
        return []

    min_keep = max(0, max_layer - last_n_layers + 1)
    blocks = []
    for lid in sorted(by_layer.keys()):
        if lid < min_keep:
            continue
        entries = by_layer[lid]
        if len(entries) == 0:
            continue
        blocks.append((f"layer_{lid}", entries))
    return blocks


def run_blockwise_outer_step(
    trainer,
    forget_batch,
    retain_batch,
    blocks: List[Tuple[str, List[Tuple[str, torch.nn.Parameter]]]],
    solver: str,
    neumann_variant: str,
) -> Tuple[float, float, float, List[Dict]]:
    trainer.model.train()
    diagnostics: List[Dict] = []

    # Core losses (same structure as SIBL)
    L_fgt = trainer.compute_forget_loss(forget_batch)
    L_ret = trainer.compute_retain_loss(retain_batch)
    r_tensor = L_ret - trainer.epsilon
    r = r_tensor.item()
    L_alm = L_fgt + trainer.lambda_dual * L_ret + 0.5 * trainer.rho * (r_tensor ** 2)

    # Raw gradients
    L_alm.backward(retain_graph=True)
    g_alm_dict: Dict[str, torch.Tensor] = {}
    for name, p in trainer.model.named_parameters():
        if p.grad is not None:
            g_alm_dict[name] = p.grad.detach().clone()
        else:
            g_alm_dict[name] = torch.zeros_like(p.data)
    trainer.model.zero_grad()

    # Inner objective for HVP
    L_ret_v = trainer.compute_retain_loss(retain_batch)
    R_theta_v = trainer.compute_sparsity_regularizer()
    L_inner = L_ret_v + R_theta_v

    # Blockwise implicit correction
    if solver == "neumann":
        old_variant = trainer.neumann_variant
        trainer.neumann_variant = neumann_variant

    for block_name, entries in blocks:
        names = [n for n, _ in entries]
        params = [p for _, p in entries]
        g_block = _flatten_tensors([g_alm_dict[n] for n in names]).to(trainer.args.device)
        m_block = _flatten_tensors([trainer.mask_dict[n] for n in names]).to(trainer.args.device)
        v_block = g_block * m_block

        if v_block.numel() == 0:
            continue

        if solver == "neumann":
            g_corr_flat, status, meta = trainer._truncated_neumann_correction(
                params_list=params,
                g_alm_flat=g_block,
                mask_flat=m_block,
                L_alm=L_alm,
                L_inner=L_inner,
            )
            corr_parts = _unflatten_like(g_corr_flat, [g_alm_dict[n] for n in names])
            for name, cp in zip(names, corr_parts):
                g_alm_dict[name] = cp
            diagnostics.append(
                {
                    "block": block_name,
                    "solver": "neumann",
                    "status": status,
                    "dim": int(v_block.numel()),
                    "v_norm": float(v_block.norm().item()),
                    "meta": meta,
                }
            )
        else:
            # Blockwise CG
            def hvp_func(vec):
                vec_masked = vec * m_block
                hv = trainer.compute_hvp(L_inner, params, vec_masked)
                return hv * m_block + trainer.cg_damping * vec

            h = trainer.conjugate_gradient(hvp_func, v_block)
            corr = trainer.compute_hvp(L_alm, params, h)
            corr_parts = _unflatten_like(corr, [g_alm_dict[n] for n in names])
            for name, cp in zip(names, corr_parts):
                g_alm_dict[name] = g_alm_dict[name] - cp

            lin_res = ((hvp_func(h) - v_block).norm() / (v_block.norm().clamp(min=1e-12))).item()
            diagnostics.append(
                {
                    "block": block_name,
                    "solver": "cg",
                    "status": "ok",
                    "dim": int(v_block.numel()),
                    "v_norm": float(v_block.norm().item()),
                    "h_norm": float(h.norm().item()),
                    "linear_residual": float(lin_res),
                }
            )

    if solver == "neumann":
        trainer.neumann_variant = old_variant

    # Primal update
    with torch.no_grad():
        for name, p in trainer.model.named_parameters():
            if name in trainer.mask_dict and name in g_alm_dict:
                p.data.sub_(trainer.eta_theta * g_alm_dict[name] * trainer.mask_dict[name])
            p.grad = None

    # Dual update
    trainer.lambda_dual = max(0.0, trainer.lambda_dual + trainer.rho * r)

    return L_fgt.item(), L_ret.item(), r, diagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-name", type=str, default="muse_Llama-2-7b-hf_News_SIBL_debug_blockwise")
    parser.add_argument("--model", type=str, default="Llama-2-7b-hf")
    parser.add_argument("--data-split", type=str, default="News")
    parser.add_argument("--solver", type=str, default="neumann", choices=["neumann", "cg"])
    parser.add_argument("--neumann-variant", type=str, default="richardson", choices=["legacy", "richardson"])
    parser.add_argument("--last-n-layers", type=int, default=2)
    parser.add_argument("--include-attn", action="store_true")
    parser.add_argument("--include-mlp", action="store_true")
    parser.add_argument("--T", type=int, default=2)
    parser.add_argument("--K", type=int, default=1)
    parser.add_argument("--neumann-mu", type=float, default=1.0)
    parser.add_argument("--neumann-steps", type=int, default=2)
    parser.add_argument("--neumann-alpha-default", type=float, default=0.01)
    parser.add_argument("--neumann-use-probe-alpha", action="store_true")
    parser.add_argument("--cg-damping", type=float, default=0.1)
    parser.add_argument("--cg-iters", type=int, default=20)
    parser.add_argument("--out-dir", type=str, default="debug/blockwise_run")
    args = parser.parse_args()

    if not args.include_attn and not args.include_mlp:
        # sensible default: include both
        args.include_attn = True
        args.include_mlp = True

    os.makedirs(args.out_dir, exist_ok=True)

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
        f"trainer.method_args.T={args.T}",
        f"trainer.method_args.K={args.K}",
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

    trainer._initialize_mask()
    if getattr(trainer.args, "gradient_checkpointing", False):
        kwargs = getattr(trainer.args, "gradient_checkpointing_kwargs", None) or {}
        kwargs = dict(kwargs)
        kwargs["use_reentrant"] = False
        trainer.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=kwargs)

    blocks = build_layer_blocks(
        trainer,
        last_n_layers=args.last_n_layers,
        include_mlp=args.include_mlp,
        include_attn=args.include_attn,
    )
    if len(blocks) == 0:
        raise RuntimeError("No blocks selected. Adjust layer/mode settings.")

    report: Dict = {
        "config": {
            "solver": args.solver,
            "neumann_variant": args.neumann_variant,
            "last_n_layers": args.last_n_layers,
            "include_attn": args.include_attn,
            "include_mlp": args.include_mlp,
            "T": args.T,
            "K": args.K,
            "neumann_mu": args.neumann_mu,
            "neumann_steps": args.neumann_steps,
            "neumann_alpha_default": args.neumann_alpha_default,
            "neumann_use_probe_alpha": args.neumann_use_probe_alpha,
            "cg_damping": args.cg_damping,
            "cg_iters": args.cg_iters,
        },
        "blocks": [
            {
                "name": bname,
                "num_params": len(entries),
                "total_dim": int(sum(p.numel() for _, p in entries)),
            }
            for bname, entries in blocks
        ],
        "iters": [],
    }

    train_dataloader = trainer.get_train_dataloader()

    for t in range(args.T):
        t0 = time.time()
        batch = next(iter(train_dataloader))
        forget_batch = batch["forget"]
        retain_batch = batch["retain"]
        trainer.inner_loop([retain_batch] * args.K)

        L_fgt, L_ret, r, diag = run_blockwise_outer_step(
            trainer=trainer,
            forget_batch=forget_batch,
            retain_batch=retain_batch,
            blocks=blocks,
            solver=args.solver,
            neumann_variant=args.neumann_variant,
        )
        report["iters"].append(
            {
                "iter": t,
                "L_fgt": float(L_fgt),
                "L_ret": float(L_ret),
                "r": float(r),
                "lambda": float(trainer.lambda_dual),
                "elapsed_sec": float(time.time() - t0),
                "diag": diag,
            }
        )

    # Save report
    json_path = os.path.join(args.out_dir, "blockwise_report.json")
    md_path = os.path.join(args.out_dir, "blockwise_report.md")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    md = []
    md.append("# Blockwise Implicit Experiment")
    md.append("")
    md.append("## Config")
    for k, v in report["config"].items():
        md.append(f"- {k}: `{v}`")
    md.append("")
    md.append("## Selected Blocks")
    for b in report["blocks"]:
        md.append(f"- {b['name']}: params={b['num_params']}, dim={b['total_dim']}")
    md.append("")
    md.append("## Iteration Summary")
    md.append("")
    md.append("| iter | L_fgt | L_ret | r | lambda | sec |")
    md.append("|---:|---:|---:|---:|---:|---:|")
    for it in report["iters"]:
        md.append(
            f"| {it['iter']} | {it['L_fgt']:.4f} | {it['L_ret']:.4f} | "
            f"{it['r']:+.4f} | {it['lambda']:.4f} | {it['elapsed_sec']:.2f} |"
        )
    md.append("")
    md.append(f"- JSON report: `{json_path}`")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()

