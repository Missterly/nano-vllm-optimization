# nano-vLLM Optimization

A lightweight LLM inference engine based on [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm), with mixed-batch scheduling and tools for measuring throughput and request latency.

## Features

- Decode-first scheduling with chunked prefill in the same batch.
- Paged KV cache with prefix reuse, unified attention for mixed batches, and CUDA Graph replay for decode-only batches.
- Offline and online benchmarks with per-request latency metrics and reproducible workload comparisons.

## Quick Start

Use Python 3.10–3.12 in a CUDA-enabled environment. The example and benchmarks default to a local Qwen3-0.6B checkpoint at `~/huggingface/Qwen3-0.6B`; benchmarks accept `--model` to override it.

```bash
pip install -e .
python example.py
```

Tested on Ubuntu 24.04 (WSL2) with an NVIDIA RTX 4060 Laptop GPU (8 GB).

## Benchmark

**Fixed lengths** — 1,024 input tokens and 128 output tokens per request:

```bash
python bench_offline.py --workload fixed --num-requests 16 --input-len 1024 --output-len 128
```

**Random lengths** — sample input and output lengths per request:

```bash
python bench_offline.py --workload random --num-requests 16 --min-input-len 128 --max-input-len 1024 --min-output-len 32 --max-output-len 128
```

Reports are saved under `benchmarks/results/`. For timed arrivals or upstream comparisons, see [`bench_online.py`](bench_online.py), [`compare_offline.py`](compare_offline.py), and [`compare_online.py`](compare_online.py); each supports `--help`.
