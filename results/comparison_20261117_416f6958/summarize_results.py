"""Build a readable report and plot from the saved benchmark results."""
import csv
import json
from pathlib import Path


root = Path(__file__).resolve().parent
report = json.loads((root / "comparison.json").read_text())
aggregate = report["aggregate"]
labels = {
    "output_throughput_tps": "输出吞吐（tok/s）",
    "ttft_mean_ms": "TTFT mean（ms）",
    "ttft_p95_ms": "TTFT P95（ms）",
    "itl_mean_ms": "ITL mean（ms）",
    "itl_p95_ms": "ITL P95（ms）",
    "itl_p99_ms": "ITL P99（ms）",
    "request_latency_p95_ms": "请求总延迟 P95（ms）",
}
lines = [
    "# 当前版本性能对比：初步实验", "",
    "对比对象：保存的原 Prefill 优先调度 vs 当前 Decode 优先混合调度。",
    "两边共用当前模型执行、KV 管理、按需 LM Head / 采样和指标统计。",
    "这不是完整上游 nano-vLLM 对照，也不是 Attention 融合的独立消融实验。", "",
    "## 设置", "",
    f"- GPU：{report['system']['gpu']}。",
    f"- PyTorch：{report['system']['torch']}；CUDA：{report['system']['cuda']}。",
    "- 模型：本地 Qwen3-0.6B；单 GPU；enforce_eager=True（不测 CUDA Graph）。",
    "- 128 个请求；输入长度循环为 64、768、1024、384、512、896、256、640。",
    "- 每请求固定生成 64 tokens，ignore_eos=True；共 72,704 输入、8,192 输出 tokens。",
    "- token budget=256；每轮最多 16 请求；max_model_len=2048；seed=1。",
    "- 每 4 个调度轮次注入一个请求（step 模式，不是固定 QPS）。",
    "- 同一 workload.json，独立进程运行，5 轮交替顺序：A/B、B/A、A/B、B/A、A/B。",
    "- 保留 benchmark 默认预热；另做不纳入汇总的 16 请求试跑。",
    "- 默认预热未系统覆盖所有 batch 形状，结果属于初步性能观察。",
    f"- 负载 SHA256：`{report['workload_sha256']}`。",
    f"- 实验后核心源码哈希一致：{report['source_unchanged']}。", "",
    "## 五次运行的中位数", "",
    "延迟行汇总的是每次运行对应统计量的中位数，没有把五轮 token 混池重新计算分位数。", "",
    "| 指标 | 原策略 | 当前策略 | 相对变化 |",
    "| --- | ---: | ---: | ---: |",
]
for field, label in labels.items():
    a = aggregate["baseline"][field]["median"]
    b = aggregate["hybrid"][field]["median"]
    lines.append(f"| {label} | {a:.2f} | {b:.2f} | {report['median_change_percent'][field]:+.2f}% |")
lines += ["", "延迟下降、吞吐上升表示改善。", "", "## 每次运行与波动", "",
          "| 轮次 | 原策略吞吐 | 当前吞吐 | 原 TTFT P95 | 当前 TTFT P95 | 原 ITL P95 | 当前 ITL P95 |",
          "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
for i in range(5):
    values = [aggregate[v][field]["values"][i] for field in (
        "output_throughput_tps", "ttft_p95_ms", "itl_p95_ms") for v in ("baseline", "hybrid")]
    lines.append(f"| {i + 1} | " + " | ".join(f"{v:.2f}" for v in values) + " |")
lines += ["", "## 调度与资源观察", "",
          "| 方法 | 总轮数（每次） | 混合轮数（每次） | 抢占次数（每次） | 驻留请求峰值 | 等待请求峰值 | KV 块池容量（每次） |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
for variant, title in (("baseline", "原策略"), ("hybrid", "当前策略")):
    totals = report["run_totals"][variant]
    peaks, waits, capacities = [], [], []
    for i in range(1, 6):
        with (root / f"run{i}_{variant}" / "steps.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        peaks.append(max(int(r["running_after"]) for r in rows))
        waits.append(max(int(r["waiting_after"]) for r in rows))
        capacities.append(int(rows[-1]["free_kv_blocks_after"]))
    cells = [str([t[key] for t in totals]) for key in ("steps", "mixed_steps", "preemptions")]
    cells += [str(peaks), str(waits), str(capacities)]
    lines.append(f"| {title} | " + " | ".join(cells) + " |")
lines += ["", "每轮最多 16 请求并不保证两种策略的总驻留请求数相同。",
          "收益包含批次组织、排队、KV 占用、抢占与重算差异，不能全部归因于统一 Attention。", "",
          "## 小规模试跑（各一次，未纳入五轮汇总）", "",
          "| 方法 | 输出吞吐 tok/s | TTFT P95 ms | ITL P95 ms |",
          "| --- | ---: | ---: | ---: |"]
for variant, title in (("baseline", "原策略"), ("hybrid", "当前策略")):
    pilot = json.loads((root / f"pilot_{variant}" / "summary.json").read_text())["metrics"]
    lines.append(f"| {title} | {pilot['output_throughput_tps']:.2f} | {pilot['ttft_ms']['p95']:.2f} | {pilot['itl_ms']['p95']:.2f} |")
lines += ["", "小规模试跑未观察到吞吐提升；它只有一次，不能据此给出稳定的小负载结论。", "",
          "## 复现与文件", "",
          "- 使用同一 conda 环境执行 `python -B run_comparison.py` 可重跑（会覆盖本目录同名运行结果）。",
          "- `comparison.json`：五轮统计和逐次原始指标。",
          "- `runN_baseline/`、`runN_hybrid/`：逐请求、逐 token、逐 step CSV 与 summary。",
          "- `*_command.json`、`*.log`：每次实际命令和完整日志。",
          "- `source_hashes_before.json`：核心源码指纹。", ""]
(root / "report.md").write_text("\n".join(lines))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
for ax, field, title in zip(axes, ("output_throughput_tps", "ttft_p95_ms", "itl_p95_ms"),
                           ("Output throughput (tok/s)", "TTFT P95 (ms)", "ITL P95 (ms)")):
    medians = [aggregate[v][field]["median"] for v in ("baseline", "hybrid")]
    ax.bar([0, 1], medians, color=["#7c8798", "#2874a6"], width=0.55, alpha=0.8)
    for x, variant in enumerate(("baseline", "hybrid")):
        ax.scatter([x] * 5, aggregate[variant][field]["values"], color="#17202a", s=20, zorder=3)
        ax.text(x, medians[x], f"{medians[x]:.1f}", ha="center", va="bottom")
    ax.set_xticks([0, 1], ["Prefill-first", "Mixed"])
    ax.set_title(title)
    ax.set_ylim(0, max(aggregate[v][field]["max"] for v in ("baseline", "hybrid")) * 1.2)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
fig.suptitle("Qwen3-0.6B | 128 requests | step arrivals | eager | 5 runs (dots)")
fig.savefig(root / "comparison.png", dpi=180)
print(root / "report.md")
