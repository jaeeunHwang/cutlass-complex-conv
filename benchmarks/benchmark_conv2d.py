import argparse
from dataclasses import dataclass

import torch
import torch.nn as nn

from complex_conv2d import ComplexConv2d


from benchmarks.baseline import ReferenceComplexConv2d


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    n: int
    c: int
    h: int
    w: int
    k: int
    r: int
    s: int
    stride: int = 1
    padding: int = 1


DEFAULT_CASES = [
    BenchmarkCase("conv_32_32", 1, 32, 32, 32, 32, 3, 3),
    BenchmarkCase("conv_32_64", 1, 32, 56, 56, 64, 3, 3),
    BenchmarkCase("conv_64_128", 1, 64, 28, 28, 128, 3, 3),
    BenchmarkCase("conv_128_256", 1, 128, 14, 14, 256, 3, 3),
    BenchmarkCase("conv_256_512", 1, 256, 7, 7, 512, 3, 3),
]


def benchmark_cuda(fn, warmup=30, iters=200):
    with torch.no_grad():
        for _ in range(warmup):
            fn()

        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(iters):
            fn()
        end.record()

        torch.cuda.synchronize()

    return start.elapsed_time(end) / iters


def make_layers(case, device):
    baseline = ReferenceComplexConv2d(
        case.c,
        case.k,
        (case.r, case.s),
        stride=case.stride,
        padding=case.padding,
    ).to(device).eval()

    cutlass = ComplexConv2d(
        case.c,
        case.k,
        (case.r, case.s),
        stride=case.stride,
        padding=case.padding,
        bias=False,
        device=device,
    ).eval()

    with torch.no_grad():
        cutlass.weight_real.copy_(baseline.conv_real.weight)
        cutlass.weight_imag.copy_(baseline.conv_imag.weight)

    return baseline, cutlass


def run_case(case, device, warmup, iters):
    baseline, cutlass = make_layers(case, device)

    x = torch.randn(
        case.n,
        case.c,
        case.h,
        case.w,
        device=device,
        dtype=torch.complex64,
    )

    with torch.no_grad():
        y_ref = baseline(x)
        y_cutlass = cutlass(x)

    max_error = (y_ref - y_cutlass).abs().max().item()
    mean_error = (y_ref - y_cutlass).abs().mean().item()
    close = torch.allclose(y_ref, y_cutlass, atol=1e-3, rtol=1e-3)

    if not close:
        raise RuntimeError(
            f"{case.name}: correctness check failed "
            f"(max_error={max_error:.6e}, mean_error={mean_error:.6e})"
        )

    # Public API: complex NCHW -> planar NHWC -> CUTLASS -> complex NCHW
    baseline_ms = benchmark_cuda(lambda: baseline(x), warmup, iters)
    cutlass_ms = benchmark_cuda(lambda: cutlass(x), warmup, iters)

    # Planar fast path: pre-transform input and keep output planar.
    x_real = x.real.permute(0, 2, 3, 1).contiguous()
    x_imag = x.imag.permute(0, 2, 3, 1).contiguous()

    # Trigger lazy weight packing before timing the fast path.
    with torch.no_grad():
        cutlass.forward_planar(x_real, x_imag)

    planar_ms = benchmark_cuda(
        lambda: cutlass.forward_planar(x_real, x_imag),
        warmup,
        iters,
    )

    return {
        "case": case.name,
        "shape": (
            case.n,
            case.h,
            case.w,
            case.c,
            case.k,
            case.r,
            case.s,
        ),
        "baseline_ms": baseline_ms,
        "cutlass_e2e_ms": cutlass_ms,
        "planar_ms": planar_ms,
        "e2e_speedup": baseline_ms / cutlass_ms,
        "planar_speedup": baseline_ms / planar_ms,
        "max_error": max_error,
        "mean_error": mean_error,
    }


def print_results(results):
    header = (
        f"{'Case':<16}"
        f"{'Baseline (ms)':>15}"
        f"{'CUTLASS E2E':>15}"
        f"{'Planar':>12}"
        f"{'E2E x':>10}"
        f"{'Planar x':>11}"
        f"{'Max err':>13}"
    )

    print()
    print(header)
    print("-" * len(header))

    for result in results:
        print(
            f"{result['case']:<16}"
            f"{result['baseline_ms']:>15.6f}"
            f"{result['cutlass_e2e_ms']:>15.6f}"
            f"{result['planar_ms']:>12.6f}"
            f"{result['e2e_speedup']:>10.3f}"
            f"{result['planar_speedup']:>11.3f}"
            f"{result['max_error']:>13.3e}"
        )

    avg_e2e = sum(r["e2e_speedup"] for r in results) / len(results)
    avg_planar = sum(r["planar_speedup"] for r in results) / len(results)

    print()
    print(f"Average E2E speedup   : {avg_e2e:.3f}x")
    print(f"Average planar speedup: {avg_planar:.3f}x")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark CUTLASS ComplexConv2d against a PyTorch reference."
    )
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this benchmark")

    device = torch.device(args.device)

    print("PyTorch :", torch.__version__)
    print("CUDA    :", torch.version.cuda)
    print("GPU     :", torch.cuda.get_device_name(device))

    results = [
        run_case(case, device, args.warmup, args.iters)
        for case in DEFAULT_CASES
    ]

    print_results(results)


if __name__ == "__main__":
    main()
