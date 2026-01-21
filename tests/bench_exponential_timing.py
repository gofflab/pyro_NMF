#!/usr/bin/env python
import argparse
import sys
import time


def _import_deps():
    try:
        import torch
    except ImportError as exc:
        print("Missing dependency: torch. Install torch to run this script.")
        raise SystemExit(1) from exc
    try:
        import anndata as ad
    except ImportError as exc:
        print("Missing dependency: anndata. Install anndata to run this script.")
        raise SystemExit(1) from exc
    return torch, ad


def _resolve_device(torch, device_str):
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _run_once(adata, torch, device, num_patterns, num_steps, batch_size):
    from pyroNMF.run_inference import prepare_tensors, setup_model_and_optimizer, run_inference_loop

    keep_on_cpu = batch_size is not None and batch_size < adata.shape[0]
    D, U, scale, _ = prepare_tensors(adata, device, keep_on_cpu=keep_on_cpu)
    model, guide, svi = setup_model_and_optimizer(
        D,
        num_patterns=num_patterns,
        scale=scale,
        NB_probs=0.5,
        use_chisq=False,
        use_pois=False,
        device=device,
        model_type="exponential_unsupervised",
        batch_size=batch_size,
    )
    _reset_peak_memory(torch, device)
    start = time.perf_counter()
    run_inference_loop(
        svi,
        model,
        D,
        U,
        num_steps=num_steps,
        use_tensorboard_id=None,
        spatial=False,
    )
    elapsed = time.perf_counter() - start
    mem_mb = _get_peak_memory_mb(torch, device)
    return elapsed, mem_mb


def _reset_peak_memory(torch, device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    elif device.type == "mps":
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()


def _get_peak_memory_mb(torch, device):
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    if device.type == "mps":
        # Apple MPS provides current/driver allocated memory, not peak.
        if hasattr(torch.mps, "current_allocated_memory"):
            return torch.mps.current_allocated_memory() / (1024 ** 2)
        return None
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Sanity check timing for exponential NMF (full-batch vs minibatch)."
    )
    parser.add_argument("--path", required=True, help="Path to .h5ad file")
    parser.add_argument("--num-patterns", type=int, default=20)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--warmup", type=int, default=0, help="Warmup steps before timing")

    args = parser.parse_args()

    torch, ad = _import_deps()
    device = _resolve_device(torch, args.device)

    print("Loading", args.path)
    adata = ad.read_h5ad(args.path)
    print("adata shape", adata.shape)
    print("device", device)
    print("num_patterns", args.num_patterns)
    print("num_steps", args.num_steps)

    if args.warmup > 0:
        print(f"warmup steps: {args.warmup} (full batch)")
        _run_once(adata, torch, device, args.num_patterns, args.warmup, batch_size=None)
        if args.batch_size is not None:
            print(f"warmup steps: {args.warmup} (batch_size={args.batch_size})")
            _run_once(adata, torch, device, args.num_patterns, args.warmup, batch_size=args.batch_size)

    full_time, full_mem = _run_once(
        adata, torch, device, args.num_patterns, args.num_steps, batch_size=None
    )
    if full_mem is None:
        print(f"full_batch_s ({args.num_steps} steps): {full_time:.3f}")
    else:
        print(f"full_batch_s ({args.num_steps} steps): {full_time:.3f} | mem_mb: {full_mem:.1f}")

    if args.batch_size is not None:
        mb_time, mb_mem = _run_once(
            adata, torch, device, args.num_patterns, args.num_steps, batch_size=args.batch_size
        )
        if mb_mem is None:
            print(f"minibatch_s ({args.num_steps} steps, batch_size={args.batch_size}): {mb_time:.3f}")
        else:
            print(
                f"minibatch_s ({args.num_steps} steps, batch_size={args.batch_size}): {mb_time:.3f} | mem_mb: {mb_mem:.1f}"
            )


if __name__ == "__main__":
    main()
