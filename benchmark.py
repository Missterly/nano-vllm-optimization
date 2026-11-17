import os
import time
import numpy as np
from random import randint, seed
from nanovllm import LLM, SamplingParams


def main():
    seed(0)
    num_seqs = 128
    max_input_len = 1024
    max_ouput_len = 1024

    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    llm = LLM(path, enforce_eager=False, max_model_len=4096)

    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)) for _ in range(num_seqs)]

    llm.generate(["Benchmark: "], SamplingParams(), use_tqdm=False)
    start = time.perf_counter()
    outputs = llm.generate(prompt_token_ids, sampling_params, use_tqdm=True)
    elapsed = time.perf_counter() - start

    total_output_tokens = sum(
        len(output["token_ids"])
        for output in outputs
    )
    # 计算吞吐
    throughtput = total_output_tokens / elapsed
    
    ttfts = [
        output["metrics"]["time_to_first_token"]
        for output in outputs
    ]

    tpots = [
        output["metrics"]["time_per_output_token"]
        for output in outputs
        if output["metrics"]["time_per_output_token"] is not None
    ]

    itls = [
        itl
        for output in outputs
        for itl in output["metrics"]["inter_token_latencies"]
    ]

    statistics = {
        "TTFT": summarize_metrics(ttfts),
        "TPOT": summarize_metrics(tpots),
        "ITL": summarize_metrics(itls),
    }

    print(f"Throughput: {throughtput:.2f} tokens/s")
    for name, metrics in statistics.items():
        if not metrics:
            print("No valid samples")
            continue
        
        for key, value in metrics.items():
            print(f"{key.upper():<5} {name}: {value:.3f} ms")

def summarize_metrics(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    
    mean = np.mean(values)
    p50, p95, p99 = np.percentile(values, [50, 95, 99])

    return {
        "mean": float(mean),
        "p50": float(p50),
        "p95": float(p95),
        "p99": float(p99),
    }


if __name__ == "__main__":
    main()