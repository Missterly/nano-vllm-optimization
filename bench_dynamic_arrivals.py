"""Reproducible dynamic-arrival benchmark for nano-vllm (baseline or hybrid).

Run this SAME script in two code checkouts; it does not modify nano-vllm.
Supports old LLMEngine.step() -> (outputs, signed_token_count) and
hybrid LLMEngine.step() -> (outputs, prefill_tokens, decode_tokens).

Two arrival policies:
  step: inject at the same logical scheduling iteration in both variants;
        useful for isolating scheduler interference, NOT a fixed QPS load.
  time: replay the same wall-clock arrival offsets; injection happens at the
        NEXT step boundary, since LLMEngine.step() is synchronous.

Outputs: workload.json, summary.json, requests.csv, token_itl.csv,
         steps.csv, preemptions.csv (all standard-library formats).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch

from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.sampling_params import SamplingParams


def percentile(values: list[float], p: float) -> float | None:
    """Linear-interpolated percentile, 0 <= p <= 100."""
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * p / 100
    lo, hi = math.floor(rank), math.ceil(rank)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


def describe(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": statistics.mean(values) if values else None,
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": max(values) if values else None,
    }


def generate_workload(args: argparse.Namespace) -> dict[str, Any]:
    rng = random.Random(args.seed)
    # First short prompt enters Decode before the later long prompts arrive.
    lengths = [64, 768, 1024, 384, 512, 896, 256, 640]
    requests = []
    for i in range(args.num_requests):
        prompt_len = lengths[i % len(lengths)]
        requests.append({
            "request_id": i,
            "arrival_step": i * args.step_gap,
            "arrival_offset_s": round(i * args.arrival_interval_ms / 1000, 6),
            "prompt_token_ids": [rng.randint(1000, 30000) for _ in range(prompt_len)],
            "output_tokens": args.output_tokens,
        })
    return {"schema_version": 1, "seed": args.seed, "requests": requests}


def validate_workload(workload: dict[str, Any], max_model_len: int) -> None:
    requests = workload["requests"]
    if not requests:
        raise ValueError("Workload must contain at least one request")
    if workload.get("schema_version") != 1:
        raise ValueError("Unsupported workload schema_version")
    seen: set[int] = set()
    last_step, last_offset = -1, -1.0
    for request in requests:
        rid = request["request_id"]
        ids = request["prompt_token_ids"]
        out = request["output_tokens"]
        step = request["arrival_step"]
        offset = request["arrival_offset_s"]
        if rid in seen or not isinstance(ids, list) or not ids or out < 2:
            raise ValueError(f"Invalid request {rid}; require unique IDs and >=2 output tokens")
        if len(ids) + out > max_model_len:
            raise ValueError(f"Request {rid}: {len(ids)}+{out} > max_model_len {max_model_len}")
        if step < last_step or offset < last_offset or step < 0 or offset < 0:
            raise ValueError("Requests must be sorted in nondecreasing arrival order")
        seen.add(rid)
        last_step, last_offset = step, offset


def workload_hash(workload: dict[str, Any]) -> str:
    encoded = json.dumps(workload["requests"], sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="~/huggingface/Qwen3-0.6B/")
    p.add_argument("--variant", choices=("baseline", "hybrid", "auto"), default="auto",
                   help="Expected engine step interface. 'auto' detects it.")
    p.add_argument("--arrival-mode", choices=("step", "time"), default="step")
    p.add_argument("--num-requests", type=int, default=12)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--step-gap", type=int, default=4)
    p.add_argument("--arrival-interval-ms", type=float, default=75.0)
    p.add_argument("--output-tokens", type=int, default=64)
    p.add_argument("--token-budget", type=int, default=256)
    p.add_argument("--max-seqs", type=int, default=8)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--workload", type=Path, default=None, help="Read a previously saved workload.json")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--max-steps", type=int, default=20000)
    p.add_argument("--skip-warmup", action="store_true")
    p.add_argument("--verbose-steps", action="store_true")
    args = p.parse_args()
    if args.num_requests < 1 or args.step_gap < 1 or args.arrival_interval_ms <= 0:
        p.error("num-requests/step-gap must be >=1 and arrival-interval-ms > 0")
    if args.token_budget < 1 or args.max_seqs < 1 or args.max_steps < 1:
        p.error("token-budget, max-seqs and max-steps must be positive")
    if not args.temperature > 1e-10:
        p.error("nano-vllm requires temperature > 1e-10")
    return args


def benchmark(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    workload = json.loads(args.workload.read_text(encoding="utf-8")) if args.workload else generate_workload(args)
    validate_workload(workload, args.max_model_len)
    jobs = workload["requests"]
    digest = workload_hash(workload)

    # NOTE: ModelRunner initialization already has GPU warmup for memory planning.
    # An additional request-level warmup keeps compilation/cache setup outside timing.
    model_path = str(Path(args.model).expanduser())
    llm = LLMEngine(
        model_path,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.token_budget,
        max_num_seqs=args.max_seqs,
    )
    if not args.skip_warmup:
        llm.generate(
            [[100, 101, 102, 103, 104, 105, 106, 107]],
            SamplingParams(temperature=args.temperature, ignore_eos=True, max_tokens=8),
            use_tqdm=False,
        )
        # Pilot a dynamic overlap once OUTSIDE the timed section. This helps
        # warm the distinct pure-Decode, pure-Prefill and mixed-forward shapes
        # (nano-vllm's RMSNorm uses torch.compile internally).
        pilot_a = min(64, args.max_model_len - 8)
        pilot_b = min(384, args.max_model_len - 8)
        if pilot_a >= 1 and pilot_b >= 1:
            pilot_params = SamplingParams(
                temperature=args.temperature, ignore_eos=True, max_tokens=4,
            )
            llm.add_request(list(range(30001, 30001 + pilot_a)), pilot_params)
            llm.step()  # A performs Prefill, then enters Decode.
            llm.add_request(list(range(40001, 40001 + pilot_b)), pilot_params)
            for _ in range(2000):
                if llm.is_finished():
                    break
                llm.step()
            else:
                raise RuntimeError("Dynamic pilot warmup failed to finish")
    if not llm.is_finished():
        raise RuntimeError("Warmup left unfinished requests")
    torch.manual_seed(args.seed)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    job_state: dict[int, dict[str, Any]] = {}
    trace: list[dict[str, Any]] = []
    preemptions: list[dict[str, Any]] = []
    token_itl_rows: list[dict[str, Any]] = []
    current_step = -1
    t_start = time.perf_counter()

    real_schedule = llm.scheduler.schedule
    schedule_observation: dict[str, Any] = {}
    detected_variant: str | None = None

    def observed_schedule():
        nonlocal detected_variant, schedule_observation
        result = real_schedule()
        if isinstance(result, tuple) and len(result) == 2:
            seqs, is_prefill = result
            prefill = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else 0
            decode = 0 if is_prefill else len(seqs)
            actual_variant = "baseline"
        elif hasattr(result, "scheduled_seqs"):
            seqs = result.scheduled_seqs
            prefill = result.num_prefill_tokens
            decode = result.num_decode_tokens
            actual_variant = "hybrid"
        else:
            raise TypeError(f"Unrecognized Scheduler.schedule result: {type(result)}")
        detected_variant = actual_variant
        if args.variant != "auto" and args.variant != actual_variant:
            raise RuntimeError(f"Expected {args.variant}, got {actual_variant}; check git branch/worktree")
        schedule_observation = {
            "prefill_tokens": prefill,
            "decode_tokens": decode,
            "scheduled_requests": len(seqs),
            "sequences": "|".join(
                f"{s.seq_id}:{'P' if s.is_prefill else 'D'}:{s.num_scheduled_tokens}"
                for s in seqs
            ),
        }
        return result

    real_preempt = llm.scheduler.preempt

    def observed_preempt(seq):
        preemptions.append({
            "step": current_step,
            "seq_id": seq.seq_id,
            "time_since_start_s": time.perf_counter() - t_start,
            "cached_tokens_before": seq.num_cached_tokens,
            "completion_tokens_before": seq.num_completion_tokens,
        })
        return real_preempt(seq)

    llm.scheduler.schedule = observed_schedule
    llm.scheduler.preempt = observed_preempt
    step_idx = 0
    next_job = 0
    try:
        while next_job < len(jobs) or not llm.is_finished():
            if step_idx >= args.max_steps:
                raise RuntimeError(f"Exceeded {args.max_steps} engine steps: potential scheduler starvation")
            elapsed = time.perf_counter() - t_start
            while next_job < len(jobs):
                j = jobs[next_job]
                due = (j["arrival_step"] <= step_idx) if args.arrival_mode == "step" \
                    else (j["arrival_offset_s"] <= elapsed)
                if not due:
                    break
                params = SamplingParams(
                    temperature=args.temperature,
                    ignore_eos=True,
                    max_tokens=j["output_tokens"],
                )
                llm.add_request(j["prompt_token_ids"], params)
                # LLMEngine.add_request currently returns None; the latest WAITING
                # sequence is the one just appended (both baseline and hybrid).
                seq = llm.scheduler.waiting[-1]
                job_state[j["request_id"]] = {
                    "seq": seq,
                    "job": j,
                    "actual_arrival_offset_s": time.perf_counter() - t_start,
                    "actual_arrival_step": step_idx,
                }
                next_job += 1
                elapsed = time.perf_counter() - t_start

            if llm.is_finished():
                # Nothing to compute until the next arrival event.
                if next_job < len(jobs):
                    if args.arrival_mode == "step":
                        step_idx = max(step_idx, jobs[next_job]["arrival_step"])
                    else:
                        sleep_for = jobs[next_job]["arrival_offset_s"] - (time.perf_counter() - t_start)
                        if sleep_for > 0:
                            time.sleep(min(sleep_for, 0.02))
                    continue
                break

            current_step = step_idx
            before = time.perf_counter()
            results = llm.step()
            torch.cuda.synchronize()  # Ensure GPU work is included in step latency.
            after = time.perf_counter()

            if not isinstance(results, tuple) or len(results) not in (2, 3):
                raise RuntimeError(f"Unexpected LLMEngine.step() result: {type(results)}")
            if len(results) == 2:
                _outputs, signed_tokens = results
                step_p, step_d = (signed_tokens, 0) if signed_tokens > 0 else (0, -signed_tokens)
            else:
                _outputs, step_p, step_d = results
            obs = schedule_observation
            if (step_p, step_d) != (obs["prefill_tokens"], obs["decode_tokens"]):
                raise RuntimeError(f"Scheduler/Engine token-count mismatch at step={step_idx}")

            trace.append({
                "step": step_idx,
                "start_offset_s": before - t_start,
                "latency_ms": (after - before) * 1000,
                "prefill_tokens": step_p,
                "decode_tokens": step_d,
                "mixed": int(step_p > 0 and step_d > 0),
                "scheduled_requests": obs["scheduled_requests"],
                "waiting_after": len(llm.scheduler.waiting),
                "running_after": len(llm.scheduler.running),
                "free_kv_blocks_after": len(llm.scheduler.block_manager.free_block_ids),
                "sequences": obs["sequences"],
            })
            if args.verbose_steps or (step_idx + 1) % 50 == 0:
                print(f"  step={step_idx:<5} P={step_p:<4} D={step_d:<3} "
                      f"latency={(after-before)*1000:.1f}ms  "
                      f"waiting={len(llm.scheduler.waiting)}  running={len(llm.scheduler.running)}")
            step_idx += 1

        torch.cuda.synchronize()
        t_end = time.perf_counter()
    finally:
        llm.scheduler.schedule = real_schedule
        llm.scheduler.preempt = real_preempt

    requests_out = []
    for rid in sorted(job_state):
        entry = job_state[rid]
        seq = entry["seq"]
        job = entry["job"]
        if not seq.is_finished or seq.first_token_time is None or seq.finish_time is None:
            raise RuntimeError(f"Request {rid} did not finish or has missing metric timestamps")
        if seq.num_completion_tokens != job["output_tokens"]:
            raise RuntimeError(f"Request {rid} produced {seq.num_completion_tokens}, expected {job['output_tokens']}")
        itls = list(seq.inter_token_latencies)
        if len(itls) != job["output_tokens"] - 1:
            raise RuntimeError(f"Request {rid}: expected {job['output_tokens'] - 1} ITLs, got {len(itls)}")
        for index, itl in enumerate(itls, start=2):
            token_itl_rows.append({"request_id": rid, "token_index": index, "itl_ms": itl})
        planned = job["arrival_offset_s"]
        actual = entry["actual_arrival_offset_s"]
        requests_out.append({
            "request_id": rid,
            "seq_id": seq.seq_id,
            "prompt_tokens": len(job["prompt_token_ids"]),
            "output_tokens": seq.num_completion_tokens,
            "planned_arrival_step": job["arrival_step"],
            "actual_arrival_step": entry["actual_arrival_step"],
            "planned_arrival_s": planned,
            "actual_arrival_s": actual,
            "arrival_lag_ms": (actual - planned) * 1000 if args.arrival_mode == "time" else None,
            "ttft_ms": seq.time_to_first_token,
            # For time-based replay, include the benchmark's enqueue lag
            # relative to the predeclared arrival schedule.
            "scheduled_ttft_ms": (
                (seq.first_token_time - t_start - planned) * 1000
                if args.arrival_mode == "time" else None
            ),
            "tpot_ms": seq.time_per_output_token,
            "itl_p50_ms": percentile(itls, 50),
            "itl_p95_ms": percentile(itls, 95),
            "latency_ms": (seq.finish_time - seq.arrival_time) * 1000,
            "scheduled_latency_ms": (
                (seq.finish_time - t_start - planned) * 1000
                if args.arrival_mode == "time" else None
            ),
        })

    duration = t_end - t_start
    all_itls = [row["itl_ms"] for row in token_itl_rows]
    input_tokens = sum(row["prompt_tokens"] for row in requests_out)
    output_tokens = sum(row["output_tokens"] for row in requests_out)
    peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
    stats = {
        "ttft_ms": describe([r["ttft_ms"] for r in requests_out]),
        "tpot_ms": describe([r["tpot_ms"] for r in requests_out]),
        "itl_ms": describe(all_itls),
        "request_latency_ms": describe([r["latency_ms"] for r in requests_out]),
        "step_latency_ms": describe([r["latency_ms"] for r in trace]),
        "arrival_lag_ms": describe([r["arrival_lag_ms"] for r in requests_out if r["arrival_lag_ms"] is not None]),
        "scheduled_ttft_ms": describe([r["scheduled_ttft_ms"] for r in requests_out if r["scheduled_ttft_ms"] is not None]),
        "scheduled_latency_ms": describe([r["scheduled_latency_ms"] for r in requests_out if r["scheduled_latency_ms"] is not None]),
        "output_throughput_tps": output_tokens / duration,
        "total_throughput_tps": (input_tokens + output_tokens) / duration,
        "scheduled_prefill_tps": sum(t["prefill_tokens"] for t in trace) / duration,
        "scheduled_decode_tps": sum(t["decode_tokens"] for t in trace) / duration,
    }
    summary = {
        "variant": detected_variant,
        "arrival_mode": args.arrival_mode,
        "workload_sha256": digest,
        "config": {
            "seed": args.seed,
            "num_requests": len(jobs),
            "temperature": args.temperature,
            "max_model_len": args.max_model_len,
            "max_seqs": args.max_seqs,
            "token_budget": args.token_budget,
            "enforce_eager": True,
            "tensor_parallel_size": 1,
            "model": model_path,
        },
        "system": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "totals": {
            "duration_s": duration,
            "steps": len(trace),
            "requests": len(requests_out),
            "prompt_tokens": input_tokens,
            "output_tokens": output_tokens,
            "scheduled_prefill_tokens": sum(t["prefill_tokens"] for t in trace),
            "scheduled_decode_tokens": sum(t["decode_tokens"] for t in trace),
            "mixed_steps": sum(t["mixed"] for t in trace),
            "preemptions": len(preemptions),
            "gpu_peak_allocated_gb": peak_gb,
        },
        "metrics": stats,
    }

    out_dir = args.output_dir or Path("results") / f"dynamic_{detected_variant}_{args.arrival_mode}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "workload.json").write_text(json.dumps(workload, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(out_dir / "requests.csv", requests_out, list(requests_out[0]))
    write_csv(out_dir / "token_itl.csv", token_itl_rows, ["request_id", "token_index", "itl_ms"])
    write_csv(out_dir / "steps.csv", trace, list(trace[0]))
    write_csv(out_dir / "preemptions.csv", preemptions,
              ["step", "seq_id", "time_since_start_s", "cached_tokens_before", "completion_tokens_before"])
    print(f"\nVariant: {detected_variant} | Arrival mode: {args.arrival_mode} | Workload SHA256: {digest[:12]}")
    print(f"Requests={len(requests_out)}  steps={len(trace)}  mixed_steps={summary['totals']['mixed_steps']} "
          f"preemptions={len(preemptions)}  elapsed={duration:.3f}s")
    labels = ["ttft_ms", "tpot_ms", "itl_ms", "request_latency_ms"]
    if args.arrival_mode == "time":
        labels.extend(["arrival_lag_ms", "scheduled_ttft_ms"])
    for label in labels:
        m = stats[label]
        print(f"{label:>20}: mean={m['mean']:.2f}  p50={m['p50']:.2f}  "
              f"p95={m['p95']:.2f}  p99={m['p99']:.2f} ms")
    print(f"Output throughput={stats['output_throughput_tps']:.1f} tok/s  "
          f"Total throughput={stats['total_throughput_tps']:.1f} tok/s")
    print(f"Results written to {out_dir.resolve()}")
    if args.arrival_mode == "step":
        print("NOTE: step-based arrivals are NOT fixed real-time arrivals; do not interpret as fixed QPS.")
    else:
        print("NOTE: arrivals are injected at step boundaries; inspect arrival_lag_ms for timing fidelity.")
    if len(requests_out) < 30:
        print("NOTE: P95/P99 are exploratory with small request counts; repeat with larger workloads.")
    return summary, out_dir


def main():
    args = parse_args()
    benchmark(args)


if __name__ == "__main__":
    main()
