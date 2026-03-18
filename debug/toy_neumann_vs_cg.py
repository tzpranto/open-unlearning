#!/usr/bin/env python
"""
Toy linear-system benchmark for implicit correction solvers.

We compare:
- CG solve for Hh=v
- Legacy Neumann (current historical implementation shape)
- Richardson/Neumann (correct fixed-point for H^{-1}v)
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List

import numpy as np


@dataclass
class SolverStats:
    solver: str
    dim: int
    cond_target: float
    rel_residual: float
    rel_error: float
    h_norm: float


def make_spd_matrix(dim: int, cond_target: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
    eigs = np.geomspace(1.0, cond_target, dim)
    return q @ np.diag(eigs) @ q.T


def cg_solve(H: np.ndarray, v: np.ndarray, iters: int, tol: float) -> np.ndarray:
    x = np.zeros_like(v)
    r = v - H @ x
    p = r.copy()
    rs_old = float(r @ r)
    for _ in range(iters):
        Hp = H @ p
        alpha = rs_old / (float(p @ Hp) + 1e-12)
        x = x + alpha * p
        r = r - alpha * Hp
        rs_new = float(r @ r)
        if np.sqrt(rs_new) < tol:
            break
        beta = rs_new / (rs_old + 1e-12)
        p = r + beta * p
        rs_old = rs_new
    return x


def legacy_neumann(H: np.ndarray, v: np.ndarray, alpha: float, steps: int) -> np.ndarray:
    # Mirrors previous implementation style:
    # h = sum_k (alpha H)^k v ; return alpha * h
    h = np.zeros_like(v)
    p = v.copy()
    for _ in range(steps + 1):
        h = h + p
        p = alpha * (H @ p)
    return alpha * h


def richardson_neumann(H: np.ndarray, v: np.ndarray, alpha: float, steps: int) -> np.ndarray:
    # h_{k+1} = h_k + alpha * (v - H h_k)
    h = np.zeros_like(v)
    for _ in range(steps + 1):
        h = h + alpha * (v - H @ h)
    return h


def solve_stats(
    solver: str, H: np.ndarray, v: np.ndarray, h: np.ndarray, h_true: np.ndarray, dim: int, cond_target: float
) -> SolverStats:
    rel_residual = float(np.linalg.norm(H @ h - v) / (np.linalg.norm(v) + 1e-12))
    rel_error = float(np.linalg.norm(h - h_true) / (np.linalg.norm(h_true) + 1e-12))
    return SolverStats(
        solver=solver,
        dim=dim,
        cond_target=cond_target,
        rel_residual=rel_residual,
        rel_error=rel_error,
        h_norm=float(np.linalg.norm(h)),
    )


def run_case(dim: int, cond_target: float, steps: int, cg_iters: int, seed: int) -> List[SolverStats]:
    H = make_spd_matrix(dim, cond_target=cond_target, seed=seed)
    rng = np.random.default_rng(seed + 17)
    v = rng.standard_normal(dim)
    h_true = np.linalg.solve(H, v)

    # Conservative alpha from spectral upper bound proxy.
    eig_max = float(np.linalg.eigvalsh(H).max())
    alpha = 0.5 / (eig_max + 1e-12)

    h_cg = cg_solve(H, v, iters=cg_iters, tol=1e-10)
    h_legacy = legacy_neumann(H, v, alpha=alpha, steps=steps)
    h_rich = richardson_neumann(H, v, alpha=alpha, steps=steps)

    return [
        solve_stats("cg", H, v, h_cg, h_true, dim, cond_target),
        solve_stats("legacy_neumann", H, v, h_legacy, h_true, dim, cond_target),
        solve_stats("richardson_neumann", H, v, h_rich, h_true, dim, cond_target),
    ]


def write_markdown(path: str, rows: List[SolverStats]) -> None:
    lines = [
        "# Toy Neumann vs CG",
        "",
        "Lower `rel_residual` and `rel_error` are better.",
        "",
        "| solver | dim | cond_target | rel_residual | rel_error | h_norm |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r.solver} | {r.dim} | {r.cond_target:.1f} | "
            f"{r.rel_residual:.4e} | {r.rel_error:.4e} | {r.h_norm:.4e} |"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dims", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--conds", type=float, nargs="+", default=[10.0, 100.0, 1000.0])
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--cg-iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="debug/results")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows: List[SolverStats] = []

    for dim in args.dims:
        for cond_target in args.conds:
            rows.extend(
                run_case(
                    dim=dim,
                    cond_target=cond_target,
                    steps=args.steps,
                    cg_iters=args.cg_iters,
                    seed=args.seed + dim + int(cond_target),
                )
            )

    md_path = os.path.join(args.out_dir, "toy_neumann_vs_cg.md")
    json_path = os.path.join(args.out_dir, "toy_neumann_vs_cg.json")
    write_markdown(md_path, rows)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in rows], f, indent=2)

    print(f"Wrote {md_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
