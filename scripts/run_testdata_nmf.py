#!/usr/bin/env python
import argparse
import math
import os
from datetime import datetime

import anndata as ad
import numpy as np
import torch
import pyro

from pyroNMF.run_inference import run_nmf_unsupervised
from pyroNMF.utils import plot_results


def _ensure_spatial(adata, flip_y=False):
    if "spatial" in adata.obsm:
        if flip_y:
            coords = adata.obsm["spatial"]
            coords = coords.copy()
            coords[:, 1] *= -1
            adata.obsm["spatial"] = coords
        return True
    if {"x", "y"}.issubset(adata.obs.columns):
        coords = adata.obs.loc[:, ["x", "y"]].to_numpy()
        if flip_y:
            coords[:, 1] *= -1
        adata.obsm["spatial"] = coords
        return True
    return False


def _choose_pattern_key(adata):
    for key in ("best_P", "last_P", "loc_P", "P_total"):
        if key in adata.obsm:
            return key
    return None


def _resolve_plot_dims(patterns, rows, cols):
    if rows is not None and cols is not None:
        return rows, cols
    n_patterns = patterns.shape[1]
    cols = math.ceil(math.sqrt(n_patterns))
    rows = math.ceil(n_patterns / cols)
    return rows, cols


def _auto_point_size(num_points, min_size=0.2, max_size=4.0):
    if num_points <= 0:
        return 1.0
    size = 4000.0 / float(num_points)
    return float(min(max_size, max(min_size, size)))


def main():
    parser = argparse.ArgumentParser(
        description="Run NMF on test_data .h5ad with tensorboard logging and output plots."
    )
    parser.add_argument(
        "--path",
        default=os.path.join("test_data", "20251031_loyal_annotations_and_figures_for_manuscript.h5ad"),
        help="Path to .h5ad file",
    )
    parser.add_argument("--num-patterns", type=int, default=20)
    parser.add_argument("--num-steps", type=int, default=10000)
    parser.add_argument("--model-family", choices=["gamma", "exponential"], default="exponential")
    parser.add_argument("--device", default="mps", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--clip-norm", type=float, default=None)
    parser.add_argument("--use-chisq", action="store_true")
    parser.add_argument("--use-pois", action="store_true")
    parser.add_argument("--max-cells", type=int, default=1000)
    parser.add_argument("--max-genes", type=int, default=None)
    parser.add_argument("--plot-rows", type=int, default=None)
    parser.add_argument("--plot-cols", type=int, default=None)
    parser.add_argument("--point-size", type=float, default=None)
    parser.add_argument("--tb-max-points", type=int, default=5000)
    parser.add_argument("--post-full-p-steps", type=int, default=200)
    parser.add_argument("--param-P", dest="param_p", action="store_true")
    parser.add_argument("--no-param-P", dest="param_p", action="store_false")
    parser.set_defaults(param_p=False)
    parser.add_argument("--flip-y", action="store_true", default=True)
    parser.add_argument("--no-flip-y", dest="flip_y", action="store_false")
    parser.add_argument("--output-dir", default="runs/test_data")
    parser.add_argument("--tensorboard-tag", default="_test_data_run")

    args = parser.parse_args()

    device = None if args.device == "auto" else args.device

    print(f"Loading data from {args.path}")
    adata = ad.read_h5ad(args.path)

    if args.max_cells is not None and args.max_cells > 0 and adata.n_obs > args.max_cells:
        adata = adata[: args.max_cells].copy()
        print(f"Subsampled to {adata.n_obs} cells")

    if args.max_genes is not None and adata.n_vars > args.max_genes:
        adata = adata[:, : args.max_genes].copy()
        print(f"Subsampled to {adata.n_vars} genes")

    has_spatial = _ensure_spatial(adata, flip_y=args.flip_y)
    if has_spatial:
        print("Using spatial coordinates for plotting")
    else:
        print("No spatial coordinates found; disabling spatial plots")

    os.makedirs(args.output_dir, exist_ok=True)
    os.chdir(args.output_dir)

    pyro.set_rng_seed(0)

    nmf_res = run_nmf_unsupervised(
        adata,
        num_patterns=args.num_patterns,
        num_steps=args.num_steps,
        device=device,
        NB_probs=0.5,
        use_chisq=args.use_chisq,
        use_pois=args.use_pois,
        use_tensorboard_id=args.tensorboard_tag,
        spatial=has_spatial,
        plot_dims=None,
        scale=None,
        model_family=args.model_family,
        batch_size=args.batch_size,
        lr=args.lr,
        clip_norm=args.clip_norm,
        tb_max_points=args.tb_max_points,
        post_full_P_steps=args.post_full_p_steps,
        param_P=args.param_p,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_h5ad = f"nmf_{args.model_family}_k{args.num_patterns}_steps{args.num_steps}_{timestamp}.h5ad"
    nmf_res.write_h5ad(out_h5ad)
    print(f"Saved results to {out_h5ad}")

    pattern_key = _choose_pattern_key(nmf_res)
    if has_spatial and pattern_key is not None:
        patterns = nmf_res.obsm[pattern_key]
        rows, cols = _resolve_plot_dims(patterns, args.plot_rows, args.plot_cols)
        point_size = args.point_size
        if point_size is None:
            point_size = _auto_point_size(nmf_res.n_obs)
        plot_path = f"patterns_{pattern_key}_{timestamp}.png"
        plot_results(
            nmf_res,
            nrows=rows,
            ncols=cols,
            which=pattern_key,
            s=point_size,
            a=1,
            scale_alpha=False,
            scale_values=False,
            savename=plot_path,
            title=f"{pattern_key} ({args.model_family}, k={args.num_patterns})",
        )
        print(f"Saved plot to {plot_path}")
    else:
        print("Skipping pattern plot (no spatial coords or patterns missing)")

    pyro.clear_param_store()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
