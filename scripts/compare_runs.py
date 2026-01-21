#!/usr/bin/env python
import argparse
from pathlib import Path

import numpy as np


def _import_deps():
    try:
        import anndata as ad
    except ImportError as exc:
        raise SystemExit("Missing dependency: anndata") from exc
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:
        raise SystemExit("Missing dependency: scipy") from exc
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("Missing dependency: matplotlib") from exc
    return ad, linear_sum_assignment, plt


def _corr_matrix(a, b):
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    a /= np.linalg.norm(a, axis=0, keepdims=True) + 1e-8
    b /= np.linalg.norm(b, axis=0, keepdims=True) + 1e-8
    return a.T @ b


def _match_corr(corr, linear_sum_assignment):
    cost = -corr
    row_ind, col_ind = linear_sum_assignment(cost)
    matched = corr[row_ind, col_ind]
    return matched, row_ind, col_ind


def _get_matrix(adata, key, axis="obsm"):
    if axis == "obsm":
        mat = adata.obsm[key]
    else:
        mat = adata.varm[key]
    if hasattr(mat, "to_numpy"):
        mat = mat.to_numpy()
    return np.asarray(mat)


def main():
    parser = argparse.ArgumentParser(description="Compare NMF runs by matching patterns.")
    parser.add_argument("run_a", help="Path to first .h5ad")
    parser.add_argument("run_b", help="Path to second .h5ad")
    parser.add_argument("--p-key", default="best_P")
    parser.add_argument("--a-key", default="best_A")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    ad, linear_sum_assignment, plt = _import_deps()

    a_path = Path(args.run_a)
    b_path = Path(args.run_b)
    adata_a = ad.read_h5ad(a_path)
    adata_b = ad.read_h5ad(b_path)

    P_a = _get_matrix(adata_a, args.p_key, axis="obsm")
    P_b = _get_matrix(adata_b, args.p_key, axis="obsm")
    A_a = _get_matrix(adata_a, args.a_key, axis="varm")
    A_b = _get_matrix(adata_b, args.a_key, axis="varm")

    if P_a.shape[1] != P_b.shape[1]:
        raise SystemExit(f"Pattern count mismatch in P: {P_a.shape} vs {P_b.shape}")
    if A_a.shape[1] != A_b.shape[1]:
        raise SystemExit(f"Pattern count mismatch in A: {A_a.shape} vs {A_b.shape}")

    corr_P = _corr_matrix(P_a, P_b)
    matched_P, rows_P, cols_P = _match_corr(corr_P, linear_sum_assignment)

    corr_A = _corr_matrix(A_a, A_b)
    matched_A, rows_A, cols_A = _match_corr(corr_A, linear_sum_assignment)

    print("P correlation summary:")
    print(f"  mean: {matched_P.mean():.3f}  median: {np.median(matched_P):.3f}  min: {matched_P.min():.3f}")
    print("A correlation summary:")
    print(f"  mean: {matched_A.mean():.3f}  median: {np.median(matched_A):.3f}  min: {matched_A.min():.3f}")

    print("\nTop 5 lowest P matches (corr, idx_a -> idx_b):")
    worst_idx = np.argsort(matched_P)[:5]
    for i in worst_idx:
        print(f"  {matched_P[i]:.3f}  {rows_P[i]} -> {cols_P[i]}")

    print("\nTop 5 lowest A matches (corr, idx_a -> idx_b):")
    worst_idx = np.argsort(matched_A)[:5]
    for i in worst_idx:
        print(f"  {matched_A[i]:.3f}  {rows_A[i]} -> {cols_A[i]}")

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = a_path.parent
    else:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    def _save_heatmap(mat, title, fname):
        fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
        im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=-1, vmax=1)
        ax.set_title(title)
        ax.set_xlabel("run_b patterns")
        ax.set_ylabel("run_a patterns")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out_dir / fname)
        plt.close(fig)

    def _save_hist(vals, title, fname):
        fig, ax = plt.subplots(figsize=(5, 3.5), dpi=150)
        ax.hist(vals, bins=20, color="#4477AA", alpha=0.85)
        ax.set_title(title)
        ax.set_xlabel("correlation")
        ax.set_ylabel("count")
        fig.tight_layout()
        fig.savefig(out_dir / fname)
        plt.close(fig)

    def _save_scatter(mat, rows, cols, title, fname):
        fig, ax = plt.subplots(figsize=(4.5, 4.5), dpi=150)
        diag_vals = mat[rows, cols]
        ax.scatter(np.arange(len(diag_vals)), diag_vals, s=20, color="#228833", alpha=0.8)
        ax.axhline(0.0, color="#444444", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("matched pattern index")
        ax.set_ylabel("correlation")
        ax.set_ylim(-1.0, 1.0)
        fig.tight_layout()
        fig.savefig(out_dir / fname)
        plt.close(fig)

    _save_heatmap(corr_P, f"P corr: {a_path.name} vs {b_path.name}", "corr_P_heatmap.png")
    _save_heatmap(corr_A, f"A corr: {a_path.name} vs {b_path.name}", "corr_A_heatmap.png")
    _save_hist(matched_P, "Matched P correlations", "corr_P_matched_hist.png")
    _save_hist(matched_A, "Matched A correlations", "corr_A_matched_hist.png")
    _save_scatter(corr_P, rows_P, cols_P, "Matched P correlations", "corr_P_matched_scatter.png")
    _save_scatter(corr_A, rows_A, cols_A, "Matched A correlations", "corr_A_matched_scatter.png")
    print(f"\nSaved correlation plots to: {out_dir}")


if __name__ == "__main__":
    main()
