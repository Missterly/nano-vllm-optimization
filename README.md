# Nano-vLLM Optimization

A lightweight LLM inference engine based on [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm), featuring hybrid scheduling, kernel fusion, and offline/online benchmarking.

## Key Features

- 🚀 **Hybrid Scheduling:** Implemented Decode-priority scheduling with Chunked Prefill, supporting mixed Prefill/Decode execution through FlashAttention Varlen while retaining CUDA Graph acceleration for pure Decode.
- ⚡ **Kernel Fusion:** Implemented a fused Q/K RMSNorm and RoPE kernel using Triton.
- 📊 **Performance Benchmarking:** Provides offline and online benchmarks with throughput, TTFT, TPOT, and ITL measurements.

## Quick Start

See [`nanovllm/example.py`](nanovllm/example.py) for a runnable example. The API mirrors vLLM's interface with minor differences in the `LLM.generate` method:

```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH")
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Introduce the AI Infra."]
outputs = llm.generate(prompts, sampling_params)
print(outputs[0]["text"])
```

## Benchmark

Run the benchmark commands from the repository root. Reports from the offline and online benchmarks are written to `benchmarks/results/`.

**Benchmark Setup:**
- Hardware: RTX 4060 Laptop (8GB)
- Environment: Ubuntu 24.04 (WSL2)
- Model: Qwen3-0.6B

### Offline Comparison

```bash
python -m benchmarks.compare_offline --workload fixed --num-requests 96 --input-len 512 --output-len 128 --max-num-seqs 8 --max-num-batched-tokens 1024 --max-model-len 4096 --warmup-requests 8 --repeat 3
```

**Performance Results (Output throughput):**

| Input Length | Baseline (tok/s) | Optimized (tok/s) | Change      |
|:------------:|-----------------:|------------------:|------------:|
| 512          | 849.14           | 866.54            | +2.05%      |
| 1024         | 591.08           | 636.55            | +7.69%      |

### Online Comparison

```bash
python -m benchmarks.compare_online --workload fixed --num-requests 64 --input-len 512 --output-len 128 --arrival poisson --request-rate 2.0 --max-num-seqs 8 --max-num-batched-tokens 1024 --max-model-len 4096 --warmup-requests 8 --repeat 3
```

**Performance Results:**

| Metric                    | Baseline | Optimized | Change      |
|---------------------------|---------:|----------:|------------:|
| Output throughput (tok/s) | 240.64   | 240.82    | +0.08%      |
| P99 TTFT (ms)             | 66.60    | 54.44     | -18.26%     |
| P99 TPOT (ms)             | 8.85     | 8.51      | -3.81%      |
| P99 ITL (ms)              | 29.99    | 25.87     | -13.76%     |
| P99 E2E latency (ms)      | 1163.23  | 1125.17   | -3.27%      |
