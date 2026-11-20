"""Replay wall-clock request arrivals against nano-vLLM without an HTTP server."""

import argparse
import hashlib
import inspect
import json
import math
import random
import tempfile
import threading
import time
from bisect import bisect_right
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, SimpleQueue

from benchmarks.bench_offline import (
    LATENCIES,
    across_runs,
    add_benchmark_arguments,
    describe,
    generate_workload,
    load_engine,
    print_run,
    run_batch,
    validate_benchmark_args,
    validate_lengths,
    write_csv,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_benchmark_arguments(parser, num_requests=64, repeat=1)
    parser.add_argument("--arrival", choices=("constant", "poisson"), default="poisson")
    parser.add_argument("--request-rate", type=float, default=1.0, help="Target arrivals per second")
    parser.add_argument("--workload-file", type=Path, help="Replay every run from a saved workload.json")
    parser.add_argument("--timeout", type=float, default=300.0, help="Deadline in seconds from traffic start, checked between steps")
    parser.add_argument("--label", default="optimized")
    args = parser.parse_args(argv)
    validate_benchmark_args(parser, args, replay=args.workload_file is not None)
    if not math.isfinite(args.timeout) or not 0 < args.timeout <= threading.TIMEOUT_MAX:
        parser.error("--timeout must be finite, positive and within the platform wait limit")
    if args.workload_file is None and not (math.isfinite(args.request_rate) and args.request_rate > 0):
        parser.error("--request-rate must be finite and positive")
    return args


def arrival_offsets(count, mode, rate, seed):
    rng = random.Random(f"online-arrivals:{seed}")
    offsets = [0.0]
    for index in range(1, count):
        offset = index / rate if mode == "constant" else offsets[-1] + rng.expovariate(rate)
        offsets.append(offset)
    return offsets


def make_trace(args, vocab_size):
    runs = []
    for seed in range(args.seed, args.seed + args.repeat):
        workload = generate_workload(args, args.num_requests, seed, vocab_size)
        offsets = arrival_offsets(len(workload), args.arrival, args.request_rate, seed)
        requests = [
            {"prompt_token_ids": prompt, "output_len": limit, "scheduled_arrival_time": offset}
            for (prompt, limit), offset in zip(workload, offsets, strict=True)
        ]
        runs.append({"seed": seed, "requests": requests})
    return {"version": 1, "workload": args.workload, "arrival": args.arrival, "request_rate": args.request_rate, "runs": runs}


def validate_trace(trace, context_limit, vocab_size=None):
    if not isinstance(trace, dict) or trace.get("version") != 1 or trace.get("arrival") not in ("constant", "poisson"):
        raise ValueError("expected an Online Benchmark workload (schema version 1)")
    rate = trace.get("request_rate")
    runs = trace.get("runs")
    if type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0 or not isinstance(runs, list) or not runs:
        raise ValueError("workload requires a positive request_rate and at least one run")
    for run in runs:
        if not isinstance(run, dict) or type(run.get("seed")) is not int or not 0 <= run["seed"] < 2**64:
            raise ValueError("invalid run seed in workload")
        jobs = run.get("requests")
        if not isinstance(jobs, list) or not jobs:
            raise ValueError("each run must contain requests")
        previous = 0.0
        for request_id, job in enumerate(jobs):
            if not isinstance(job, dict):
                raise ValueError(f"invalid request {request_id}")
            offset, prompt, limit = (job.get(key) for key in ("scheduled_arrival_time", "prompt_token_ids", "output_len"))
            if (type(offset) not in (int, float) or not math.isfinite(offset)
                    or offset < previous or (request_id == 0 and offset != 0)):
                raise ValueError("arrival offsets must start at zero and be finite and nondecreasing")
            if not isinstance(prompt, list) or not prompt or type(limit) is not int or limit < 1:
                raise ValueError(f"request {request_id}: expected nonempty token IDs and a positive output length")
            if len(prompt) + limit > context_limit:
                raise ValueError(f"request {request_id}: input + output exceeds context limit ({context_limit})")
            if any(type(token) is not int or token < 0 or (vocab_size is not None and token >= vocab_size) for token in prompt):
                raise ValueError(f"request {request_id}: invalid token ID")
            previous = offset


@dataclass
class RequestState:
    job: dict
    arrival_time: float | None = None
    admitted_time: float | None = None
    sequence: object = None
    first_token_time: float | None = None
    finish_time: float | None = None
    last_token_time: float | None = None
    observed_tokens: int = 0
    itls_ms: list = field(default_factory=list)


def produce(requests, ready, stop, start, deadline):
    for request_id, request in enumerate(requests):
        scheduled = start + request.job["scheduled_arrival_time"]
        if stop.wait(max(0.0, min(scheduled, deadline) - time.perf_counter())):
            return
        arrival = time.perf_counter()
        if arrival >= deadline:
            return
        request.arrival_time = arrival
        ready.put(request_id)


def observe_tokens(postprocess, admitted):
    def observe(seqs, *args, **kwargs):
        token_ready = time.perf_counter()
        postprocess(seqs, *args, **kwargs)
        for seq in seqs:
            request = admitted[seq.seq_id]
            count = seq.num_completion_tokens
            if count == request.observed_tokens:
                continue
            if request.last_token_time is None:
                request.first_token_time = token_ready
            else:
                request.itls_ms.append((token_ready - request.last_token_time) * 1000)
            request.last_token_time = token_ready
            request.observed_tokens = count
            if seq.is_finished:
                request.finish_time = token_ready
    return observe


def run_traffic(llm, jobs, params, timeout):
    requests = [RequestState(job) for job in jobs]
    ready, stop = SimpleQueue(), threading.Event()
    admitted = {}
    original_postprocess = llm.scheduler.postprocess
    native_timing = "token_ready_time" in inspect.signature(original_postprocess).parameters

    def admit(request_id):
        request = requests[request_id]
        llm.add_request(request.job["prompt_token_ids"], params[request_id])
        request.admitted_time = time.perf_counter()
        request.sequence = llm.scheduler.waiting[-1]
        admitted[request.sequence.seq_id] = request

    decode_starts, decode_ends = [], []
    mixed_steps = 0
    finished = 0
    error = None
    start = time.perf_counter()
    deadline = start + timeout
    producer = threading.Thread(target=produce, args=(requests, ready, stop, start, deadline), name="online-arrivals")
    producer.start()
    if not native_timing:
        llm.scheduler.postprocess = observe_tokens(original_postprocess, admitted)
    try:
        while finished < len(jobs) and (remaining := deadline - time.perf_counter()) > 0:
            if llm.is_finished():
                if len(admitted) == len(jobs):
                    raise RuntimeError(f"engine drained after reporting {finished}/{len(jobs)} completions")
                try:
                    incoming = ready.get(timeout=remaining)
                except Empty:
                    break
                if time.perf_counter() >= deadline:
                    break
                admit(incoming)

            while time.perf_counter() < deadline:
                try:
                    incoming = ready.get_nowait()
                except Empty:
                    break
                admit(incoming)

            step_start = time.perf_counter()
            if step_start >= deadline:
                break
            outputs, *counts = llm.step()
            step_end = time.perf_counter()
            finished += len(outputs)
            if len(counts) == 2:
                prefill, decode = counts
            else:
                (signed_tokens,) = counts
                prefill, decode = max(signed_tokens, 0), max(-signed_tokens, 0)
            if decode:
                decode_starts.append(step_start)
                decode_ends.append(step_end)
                mixed_steps += bool(prefill)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        end = time.perf_counter()
        stop.set()
        producer.join()
        if not native_timing:
            llm.scheduler.postprocess = original_postprocess

    if native_timing:
        for request in admitted.values():
            seq = request.sequence
            request.first_token_time = seq.first_token_time
            request.finish_time = seq.finish_time
            request.itls_ms = seq.inter_token_latencies

    return summarize_traffic(requests, start, deadline, end, error, decode_starts, decode_ends, mixed_steps)


def summarize_traffic(requests, start, deadline, end, error, decode_starts, decode_ends, mixed_steps):
    all_itls = []
    rows = []

    def offset(timestamp):
        return timestamp - start if timestamp is not None else None

    def delta_ms(later, earlier):
        return (later - earlier) * 1000 if later is not None and earlier is not None else None

    for request_id, request in enumerate(requests):
        arrival = request.arrival_time
        first = request.first_token_time
        finish = request.finish_time
        count = request.sequence.num_completion_tokens if request.sequence is not None else 0
        completed = finish is not None and finish <= deadline and count == request.job["output_len"]
        if completed:
            status = "completed"
        elif (finish is not None and finish <= deadline) or error is not None:
            status = "failed"
        else:
            status = "timed_out"
        itls = request.itls_ms if completed and count >= 2 else []
        all_itls.extend(itls)
        step_index = bisect_right(decode_starts, arrival) - 1 if arrival is not None else -1
        rows.append({
            "request_id": request_id,
            "input_tokens": len(request.job["prompt_token_ids"]),
            "requested_output_tokens": request.job["output_len"],
            "output_tokens": count,
            "scheduled_arrival_time": request.job["scheduled_arrival_time"],
            "arrival_time": offset(arrival),
            "admitted_time": offset(request.admitted_time),
            "first_token_time": offset(first),
            "finish_time": offset(finish),
            "arrival_lag_ms": delta_ms(arrival, start + request.job["scheduled_arrival_time"]),
            "admission_delay_ms": delta_ms(request.admitted_time, arrival),
            "arrived_during_decode_step": step_index >= 0 and arrival < decode_ends[step_index],
            "ttft_ms": delta_ms(first, arrival) if completed else None,
            "tpot_ms": delta_ms(finish, first) / (count - 1) if completed and count >= 2 else None,
            "mean_itl_ms": sum(itls) / len(itls) if itls else None,
            "max_itl_ms": max(itls) if itls else None,
            "e2e_latency_ms": delta_ms(finish, arrival) if completed else None,
            "status": status,
        })

    successful = [row for row in rows if row["status"] == "completed"]
    timed_out = sum(row["status"] == "timed_out" for row in rows)
    failed = len(rows) - len(successful) - timed_out
    valid = not failed and not timed_out and error is None
    first_arrival = next((request.arrival_time for request in requests if request.arrival_time is not None), None)
    cutoff = max(request.finish_time for request in requests) if valid else min(end, deadline)
    elapsed = max(0.0, cutoff - first_arrival) if first_arrival is not None else 0.0
    total_input = sum(row["input_tokens"] for row in successful)
    total_output = sum(row["output_tokens"] for row in successful)
    summary = {
        "valid": valid,
        "error": error,
        "elapsed_s": elapsed,
        "total_requests": len(rows),
        "completed_requests": len(successful),
        "failed_requests": failed,
        "timed_out_requests": timed_out,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "request_throughput_req_s": len(successful) / elapsed if elapsed else None,
        "output_throughput_tok_s": total_output / elapsed if elapsed else None,
        "total_throughput_tok_s": (total_input + total_output) / elapsed if elapsed else None,
        "mixed_steps": mixed_steps,
        "arrivals_during_decode_steps": sum(row["arrived_during_decode_step"] for row in rows),
    }
    for metric in (*LATENCIES, "arrival_lag_ms", "admission_delay_ms"):
        population = rows if metric in ("arrival_lag_ms", "admission_delay_ms") else successful
        samples = all_itls if metric == "itl_ms" else [row[metric] for row in population if row[metric] is not None]
        summary.update({f"{metric.removesuffix('_ms')}_{stat}_ms": value for stat, value in describe(samples).items()})
    return summary, rows


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    model = args.model.expanduser().resolve()
    if not model.is_dir():
        raise SystemExit(f"Model directory not found: {model} (use --model)")
    trace = None
    if args.workload_file is not None:
        try:
            trace = json.loads(args.workload_file.expanduser().read_text(encoding="utf-8"))
            validate_trace(trace, args.max_model_len)
        except (OSError, ValueError) as exc:
            raise SystemExit(f"Invalid workload file: {exc}") from exc

    import torch

    LLM, SamplingParams = load_engine(args.engine_dir)

    print(f"========== Online Benchmark: {args.label} ==========", flush=True)
    print(f"Model: {model.name} | Max seqs: {args.max_num_seqs} | Token budget: {args.max_num_batched_tokens}", flush=True)
    llm = LLM(
        str(model),
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        tensor_parallel_size=1,
    )
    engine = llm.model_runner.config
    vocab_size = engine.hf_config.vocab_size
    try:
        if trace is None:
            validate_lengths(args, engine.max_model_len)
            trace = make_trace(args, vocab_size)
        validate_trace(trace, engine.max_model_len, vocab_size)
    except ValueError as exc:
        raise SystemExit(f"Invalid workload: {exc}") from exc
    print(f"Arrival: {trace['arrival']} | Target rate: {trace['request_rate']} req/s", flush=True)

    def sampling_params(limit):
        return SamplingParams(temperature=args.temperature, max_tokens=limit, ignore_eos=True)

    output_root = args.output_dir.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(tempfile.mkdtemp(prefix=f"online_{stamp}_", dir=output_root))
    save_json(output_dir / "workload.json", trace)
    first_run = trace["runs"][0]
    if args.warmup_requests:
        print(f"Warming up with {args.warmup_requests} requests...", flush=True)
        rng = random.Random(f"online-warmup:{first_run['seed']}")
        warmup = []
        for index in range(args.warmup_requests):
            job = first_run["requests"][index % len(first_run["requests"])]
            warmup.append(([rng.randrange(vocab_size) for _ in job["prompt_token_ids"]], job["output_len"]))
        torch.manual_seed(first_run["seed"])
        run_batch(llm, warmup, [sampling_params(limit) for _, limit in warmup])

    summaries, requests = [], []
    for run_number, run in enumerate(trace["runs"], start=1):
        jobs = run["requests"]
        params = [sampling_params(job["output_len"]) for job in jobs]
        fingerprint = hashlib.sha256(json.dumps(jobs, sort_keys=True, separators=(",", ":")).encode("ascii")).hexdigest()
        torch.manual_seed(run["seed"])
        input_lengths = [len(job["prompt_token_ids"]) for job in jobs]
        output_lengths = [job["output_len"] for job in jobs]
        ranges = {
            "observed_min_input_len": min(input_lengths), "observed_max_input_len": max(input_lengths),
            "observed_min_output_len": min(output_lengths), "observed_max_output_len": max(output_lengths),
        }
        print(f"Running traffic {run_number}/{len(trace['runs'])}: {len(jobs)} requests | "
              f"Input: {min(input_lengths)}-{max(input_lengths)} | Output: {min(output_lengths)}-{max(output_lengths)}", flush=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        summary, rows = run_traffic(llm, jobs, params, args.timeout)
        summary = {"run": run_number, "seed": run["seed"], "workload_sha256": fingerprint} | ranges | summary | {
            "peak_gpu_memory_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "peak_gpu_memory_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        }
        summaries.append(summary)
        requests.extend({"run": run_number} | row for row in rows)
        print_run(summary, len(trace["runs"]))
        print(f"Completed: {summary['completed_requests']} | Failed: {summary['failed_requests']} | "
              f"Timed out: {summary['timed_out_requests']} | Arrivals during Decode steps: {summary['arrivals_during_decode_steps']}")
        if not summary["valid"]:
            print(f"Invalid run; remaining experiments skipped. {summary['error'] or 'Deadline or incomplete output.'}", flush=True)
            break

    config = {
        "label": args.label,
        "model": str(model),
        "engine_dir": str(args.engine_dir.expanduser().resolve()),
        "dtype": str(engine.hf_config.dtype),
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_model_len": engine.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": args.enforce_eager,
        "temperature": args.temperature,
        "ignore_eos": True,
        "workload": trace.get("workload"),
        "arrival": trace["arrival"],
        "request_rate": trace["request_rate"],
        "timeout_s": args.timeout,
        "warmup_requests": args.warmup_requests,
        "requested_runs": len(trace["runs"]),
    }
    valid_runs = [summary for summary in summaries if summary["valid"]]
    write_csv(output_dir / "summary.csv", [config | summary for summary in summaries])
    write_csv(output_dir / "requests.csv", requests)
    report = {
        "config": config,
        "runs": summaries,
        "across_runs": across_runs(valid_runs) if valid_runs else {},
        "timestamp_reference": "perf_counter seconds relative to traffic start",
        "elapsed_time_scope": "first actual arrival to last token-ready finish; invalid runs end at cutoff",
    }
    save_json(output_dir / "summary.json", report)
    print(f"\nResults: {output_dir}")
    return 0 if len(valid_runs) == len(trace["runs"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
