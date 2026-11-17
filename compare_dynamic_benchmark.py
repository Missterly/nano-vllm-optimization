"""Compare benchmark summary.json files from baseline and hybrid checkouts."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("baseline", type=Path, help="path to baseline summary.json")
    p.add_argument("hybrid", type=Path, help="path to hybrid summary.json")
    args = p.parse_args()
    a = json.loads(args.baseline.read_text(encoding="utf-8"))
    b = json.loads(args.hybrid.read_text(encoding="utf-8"))
    if a["workload_sha256"] != b["workload_sha256"]:
        p.error("Workload hashes differ; results are not comparable")
    if a["arrival_mode"] != b["arrival_mode"]:
        p.error("Arrival modes differ")
    for field in ("model", "max_model_len", "max_seqs", "token_budget",
                  "temperature", "enforce_eager", "tensor_parallel_size"):
        if a["config"][field] != b["config"][field]:
            p.error(f"Config differs: {field}")
    if a["variant"] != "baseline" or b["variant"] != "hybrid":
        p.error("Expected a baseline summary first and a hybrid summary second")
    if a["system"]["gpu"] != b["system"]["gpu"]:
        print("WARNING: different GPUs, results may not be comparable")
    print(f"Workload SHA256: {a['workload_sha256'][:16]}")
    print(f"Arrival mode: {a['arrival_mode']}")
    print("Lower latency is better; higher throughput is better.")
    print(f"{'Metric':<28} {'Baseline':>13} {'Hybrid':>13} {'Change':>12}")
    print("-" * 70)
    latency_keys = (
        ("TTFT mean (ms)", "ttft_ms", "mean"),
        ("TTFT P95 (ms)", "ttft_ms", "p95"),
        ("ITL mean (ms)", "itl_ms", "mean"),
        ("ITL P95 (ms)", "itl_ms", "p95"),
        ("ITL P99 (ms)", "itl_ms", "p99"),
        ("TPOT mean (ms)", "tpot_ms", "mean"),
        ("Request latency P95 (ms)", "request_latency_ms", "p95"),
    )
    if a["arrival_mode"] == "time":
        latency_keys += (
            ("Scheduled TTFT P95 (ms)", "scheduled_ttft_ms", "p95"),
            ("Arrival lag P95 (ms)", "arrival_lag_ms", "p95"),
        )
    for label, metric, stat in latency_keys:
        x, y = a["metrics"][metric][stat], b["metrics"][metric][stat]
        ch = (y / x - 1) * 100 if x else float("nan")
        print(f"{label:<28} {x:>13.2f} {y:>13.2f} {ch:>+11.2f}%")
    for label, metric in (
        ("Output throughput (tok/s)", "output_throughput_tps"),
        ("Total throughput (tok/s)", "total_throughput_tps"),
    ):
        x, y = a["metrics"][metric], b["metrics"][metric]
        ch = (y / x - 1) * 100 if x else float("nan")
        print(f"{label:<28} {x:>13.2f} {y:>13.2f} {ch:>+11.2f}%")
    print("\nBaseline / Hybrid mixed steps:", a["totals"]["mixed_steps"], '/', b["totals"]["mixed_steps"])
    print("Baseline / Hybrid preemptions:", a["totals"]["preemptions"], '/', b["totals"]["preemptions"])
    print("NOTE: a single run is not a statistically stable performance result.")


if __name__ == "__main__":
    main()
