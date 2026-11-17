"""Replay two scheduler policies sequentially without editing core sources."""
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
SCRIPT = PROJECT / "bench_dynamic_arrivals_saved_baseline.py"


def run(variant, label, workload):
    output = ROOT / label
    command = [
        sys.executable, "-B", str(SCRIPT),
        "--model", str(Path.home() / "huggingface/Qwen3-0.6B"),
        "--variant", variant, "--arrival-mode", "step",
        "--workload", str(workload), "--token-budget", "256",
        "--max-seqs", "16", "--max-model-len", "2048", "--seed", "1",
        "--output-dir", str(output),
    ]
    if variant == "baseline":
        command += ["--original-scheduler", str(PROJECT / "nanovllm/copy/scheduler.py")]
    (ROOT / (label + "_command.json")).write_text(json.dumps(command, indent=2))
    print(f"START {label}", flush=True)
    log = ROOT / (label + ".log")
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    with log.open("w") as handle:
        result = subprocess.run(command, cwd=PROJECT, env=env, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        print(log.read_text()[-10000:], flush=True)
        raise RuntimeError(f"{label} failed ({result.returncode}); see {log}")
    summary = json.loads((output / "summary.json").read_text())
    m, t = summary["metrics"], summary["totals"]
    print(f"DONE {label}: {m['output_throughput_tps']:.2f} tok/s; "
          f"TTFT P95 {m['ttft_ms']['p95']:.2f} ms; ITL P95 {m['itl_ms']['p95']:.2f} ms; "
          f"mixed={t['mixed_steps']}; preemptions={t['preemptions']}", flush=True)
    return summary


def main():
    run("baseline", "pilot_baseline", ROOT / "pilot_workload.json")
    run("hybrid", "pilot_hybrid", ROOT / "pilot_workload.json")
    summaries = {"baseline": [], "hybrid": []}
    for repeat in range(1, 6):
        order = ("baseline", "hybrid") if repeat % 2 else ("hybrid", "baseline")
        for variant in order:
            summaries[variant].append(run(variant, f"run{repeat}_{variant}", ROOT / "main_workload.json"))

    all_runs = summaries["baseline"] + summaries["hybrid"]
    assert len({s["workload_sha256"] for s in all_runs}) == 1
    assert all(s["config"] == all_runs[0]["config"] for s in all_runs)
    assert len({s["totals"]["output_tokens"] for s in all_runs}) == 1
    fields = {
        "output_throughput_tps": ("output_throughput_tps",),
        "ttft_mean_ms": ("ttft_ms", "mean"),
        "ttft_p95_ms": ("ttft_ms", "p95"),
        "itl_mean_ms": ("itl_ms", "mean"),
        "itl_p95_ms": ("itl_ms", "p95"),
        "itl_p99_ms": ("itl_ms", "p99"),
        "request_latency_p95_ms": ("request_latency_ms", "p95"),
    }
    aggregate = {}
    for variant, runs in summaries.items():
        aggregate[variant] = {}
        for label, path in fields.items():
            values = []
            for summary in runs:
                value = summary["metrics"]
                for key in path:
                    value = value[key]
                values.append(value)
            aggregate[variant][label] = {
                "values": values, "median": statistics.median(values),
                "min": min(values), "max": max(values),
            }
    changes = {
        label: (aggregate["hybrid"][label]["median"] / aggregate["baseline"][label]["median"] - 1) * 100
        for label in fields
    }
    hashes = json.loads((ROOT / "source_hashes_before.json").read_text())
    unchanged = all(hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() == digest for name, digest in hashes.items())
    report = {
        "baseline_kind": "prefill_first_policy_on_current_runtime",
        "arrival_mode": "step", "warmup": "script default plus excluded pilot runs",
        "repeats": 5, "source_unchanged": unchanged,
        "workload_sha256": all_runs[0]["workload_sha256"],
        "config": all_runs[0]["config"], "system": all_runs[0]["system"],
        "aggregate": aggregate, "median_change_percent": changes,
        "run_totals": {k: [s["totals"] for s in runs] for k, runs in summaries.items()},
    }
    (ROOT / "comparison.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"source_unchanged": unchanged, "median_change_percent": changes}, indent=2), flush=True)
    assert unchanged, "Core source changed during benchmark"


if __name__ == "__main__":
    main()
