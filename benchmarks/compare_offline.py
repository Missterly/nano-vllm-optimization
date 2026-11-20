"""Compare offline performance of the development and baseline packages."""

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from benchmarks.bench_offline import BENCHMARK_DIR, REPO_ROOT, THROUGHPUTS, parse_args as parse_benchmark_args


BASELINE_DIR = REPO_ROOT / "nanovllm-base"
METRICS = {
    "output_throughput_tok_s": THROUGHPUTS["output_throughput_tok_s"],
    "request_throughput_req_s": THROUGHPUTS["request_throughput_req_s"],
    "total_throughput_tok_s": THROUGHPUTS["total_throughput_tok_s"],
    "elapsed_s": "Elapsed time (s)",
    "peak_gpu_memory_allocated_mib": "Peak CUDA allocated (MiB)",
    "peak_gpu_memory_reserved_mib": "Peak CUDA reserved (MiB)",
}


def parse_args(argv=None):
    return parse_comparison_args(__doc__, "offline", parse_benchmark_args, argv)


def parse_comparison_args(description, mode, parse_benchmark, argv):
    parser = argparse.ArgumentParser(
        description=description,
        epilog=f"Other arguments are forwarded to both benchmarks. See python -m benchmarks.bench_{mode} --help.",
        allow_abbrev=False,
    )
    parser.add_argument("--baseline-dir", type=Path, default=BASELINE_DIR, help="Baseline package (default: nanovllm-base)")
    default_output = "ab" if mode == "offline" else "online_ab"
    parser.add_argument("--output-dir", type=Path, default=BENCHMARK_DIR / "results" / default_output)
    parser.add_argument("--optimized-first", action="store_true", help="Reverse the measurement order")
    args, benchmark_args = parser.parse_known_args(argv)
    benchmark = parse_benchmark(benchmark_args)
    model = benchmark.model.expanduser().resolve()
    if not model.is_dir():
        parser.error(f"Model directory not found: {model} (use --model)")
    args.baseline_dir = args.baseline_dir.expanduser().resolve()
    args.optimized_dir = benchmark.engine_dir.expanduser().resolve()
    for source in (args.baseline_dir, args.optimized_dir):
        if not (source / "__init__.py").is_file():
            parser.error(f"nano-vLLM package not found: {source}")
    return args, [*benchmark_args, "--model", str(model)]


def source_info(source):
    digest = hashlib.sha256()
    for path in sorted(source.rglob("*.py")):
        digest.update(path.relative_to(source).as_posix().encode("utf-8") + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return {"path": str(source), "python_sha256": digest.hexdigest()}


def run_variant(mode, label, source, output_dir, benchmark_args):
    print(f"\n=== {label.capitalize()} ===", flush=True)
    subprocess.run(
        [sys.executable, "-B", "-m", f"benchmarks.bench_{mode}", *benchmark_args,
         "--engine-dir", str(source), "--output-dir", str(output_dir)],
        check=True,
    )
    reports = list(output_dir.glob(f"{mode}_*/summary.json"))
    if len(reports) != 1:
        raise RuntimeError(f"Expected one benchmark report in {output_dir}, found {len(reports)}")
    return reports[0]


def compare_reports(baseline, optimized, metrics=METRICS):
    baseline_config, optimized_config = baseline["config"], optimized["config"]
    for key in sorted((baseline_config.keys() | optimized_config.keys()) - {"output_dir", "engine_dir", "label"}):
        if baseline_config.get(key) != optimized_config.get(key):
            raise ValueError(f"Mismatched benchmark configuration: {key}")
    if not baseline["runs"] or len(baseline["runs"]) != len(optimized["runs"]):
        raise ValueError("Mismatched or empty run counts")
    checked_fields = ("run", "seed", "workload_sha256", "total_requests", "total_input_tokens", "total_output_tokens")
    for baseline_run, optimized_run in zip(baseline["runs"], optimized["runs"], strict=True):
        if not baseline_run.get("valid", True) or not optimized_run.get("valid", True):
            raise ValueError(f"Run {baseline_run['run']}: incomplete or failed benchmark")
        for key in checked_fields:
            if baseline_run[key] != optimized_run[key]:
                raise ValueError(f"Run {baseline_run['run']}: mismatched {key}")

    comparison = {}
    for key in metrics:
        original = baseline["across_runs"][key]
        improved = optimized["across_runs"][key]
        change = None
        if original["mean"] not in (None, 0) and improved["mean"] is not None:
            change = (improved["mean"] / original["mean"] - 1) * 100
        comparison[key] = {"baseline": original, "optimized": improved, "change_pct": change}
    return comparison


def print_comparison(metrics, labels, mode):
    def format_stats(stats):
        mean, stddev = stats["mean"], stats["stddev"]
        if mean is None:
            return "N/A"
        return f"{mean:.2f}" if stddev is None else f"{mean:.2f} +/- {stddev:.2f}"

    print(f"\n=== {mode.capitalize()} A/B (mean across runs +/- sample SD) ===")
    print(f"{'Metric':<28} {'Baseline':>22} {'Optimized':>22} {'Change':>10}")
    for key, label in labels.items():
        result = metrics[key]
        change = result["change_pct"]
        delta = "N/A" if change is None else f"{change:+.2f}%"
        print(f"{label:<28} {format_stats(result['baseline']):>22} {format_stats(result['optimized']):>22} {delta:>10}")


def run_comparison(args, benchmark_args, *, mode, metrics):
    output_root = args.output_dir.expanduser().resolve()
    order = ("optimized", "baseline") if args.optimized_first else ("baseline", "optimized")
    sources = {"baseline": args.baseline_dir, "optimized": args.optimized_dir}
    try:
        engines = {label: source_info(source) for label, source in sources.items()}
        scripts = ["bench_offline.py"]
        if mode == "online":
            scripts.append("bench_online.py")
        benchmark_hashes = {name: hashlib.sha256((BENCHMARK_DIR / name).read_bytes()).hexdigest() for name in scripts}
        output_root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = Path(tempfile.mkdtemp(prefix=f"ab_{stamp}_", dir=output_root))
        print(f"Baseline: {args.baseline_dir}\nOptimized: {args.optimized_dir}\nResults: {output_dir}", flush=True)
        reports = {}
        for label in order:
            variant_args = benchmark_args
            if mode == "online":
                variant_args = [*benchmark_args, "--label", label]
                if reports:
                    trace = reports[order[0]].with_name("workload.json")
                    variant_args.extend(("--workload-file", str(trace)))
            reports[label] = run_variant(mode, label, sources[label], output_dir / label, variant_args)

        summaries = {label: json.loads(path.read_text(encoding="utf-8")) for label, path in reports.items()}
        results = compare_reports(summaries["baseline"], summaries["optimized"], metrics)
        comparison = {
            "engines": engines,
            "benchmark_sha256": benchmark_hashes,
            "order": order,
            "reports": {label: path.relative_to(output_dir).as_posix() for label, path in reports.items()},
            "metrics": results,
        }
        comparison_path = output_dir / "comparison.json"
        comparison_path.write_text(json.dumps(comparison, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (OSError, subprocess.CalledProcessError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"{mode.capitalize()} comparison failed: {exc}") from exc
    print_comparison(results, metrics, mode)
    print(f"\nComparison: {comparison_path}")


def main():
    args, benchmark_args = parse_args()
    run_comparison(args, benchmark_args, mode="offline", metrics=METRICS)


if __name__ == "__main__":
    main()
