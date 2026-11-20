"""Replay identical timed requests against the development and baseline packages."""

from benchmarks.bench_online import parse_args as parse_benchmark_args
from benchmarks.compare_offline import METRICS as OFFLINE_METRICS, parse_comparison_args, run_comparison


METRICS = OFFLINE_METRICS | {
    f"{metric}_{stat}_ms": f"{label} {stat.upper()} (ms)"
    for metric, label in (
        ("ttft", "TTFT"),
        ("tpot", "TPOT"),
        ("itl", "ITL"),
        ("e2e_latency", "End-to-end latency"),
    )
    for stat in ("mean", "p50", "p95", "p99")
}


def main():
    args, benchmark_args = parse_comparison_args(__doc__, "online", parse_benchmark_args, None)
    run_comparison(args, benchmark_args, mode="online", metrics=METRICS)


if __name__ == "__main__":
    main()
