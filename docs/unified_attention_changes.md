# 统一混合 Attention 与按请求采样：本轮修改说明

本次相对开始实施前的工作区文件生成差异，并非相对上游 Git HEAD。
完整逐行差异见 [unified_attention_changes.patch](unified_attention_changes.patch)：
`-` 为本轮删除，`+` 为本轮新增。

后续按要求删除 `run()` 的 `try/finally`，该次独立差异见
[run_control_flow_changes.patch](run_control_flow_changes.patch)。

开始实施前，工作区已经删除了 `prepare_mixed()` 和 Attention 的混合拆分分支；
但旧 `run()` 仍调用 `prepare_mixed()`。下文区分已存在的改动与本轮完成的内容。

## 执行和结果约定

| batch 类型 | 模型与 Attention 路径 | 输出选择 |
| --- | --- | --- |
| 纯 Prefill | 一次模型 forward，逐层一次 varlen Attention | 仅已完成 Prefill 的请求 |
| Prefill + Decode | 一次打包模型 forward，逐层一次统一分页 varlen Attention | Decode 和完成 Prefill 的请求 |
| 纯 Decode | 保留专用 Attention / CUDA Graph 重放 | 所有 Decode 请求 |

输入、hidden states 的 query 分段和返回结果均保持调度器提供的请求顺序。
每个请求占一个结果位置。未完成 Prefill chunk 的结果为 `None`，只推进 KV，
不产生输出 token，也不更新 TTFT/ITL 的输出时间。

## 逐文件增删

### nanovllm/engine/model_runner.py

新增或修改：

- `prepare_prefill()` 同时准备纯 Prefill 和混合 batch，补充用途说明。
- Decode 使用 `append(seq.last_token)`。原工作区的 `extend(seq.last_token)` 会对整数
  进行迭代而报错；使用 `last_token` 也兼容 TP worker 不持有完整历史的情况。
- `run_model()` 的路径参数改名为 `use_varlen`；eager 和图重放均只返回 hidden states。
- `run()` 每轮先计算 `sample_indices`：Decode，或本轮计算终点等于当前序列长度的 Prefill。
- 统一模型 forward 后，根据 query 分段末位置选择这些请求的 hidden states，再执行 LM Head。
- 所有 TP rank 选择同样的行、参与 LM Head；仅 rank 0 准备 temperature、执行采样。
- 按原请求位置回填 token，未采样位置保留 `None`。
- 整轮无请求需要采样时，仍执行模型和 KV 写入，并同步当前 CUDA stream。
  原先 `tolist()` 隐含的同步在该分支不再存在，需显式保证提交进度和计时时计算已完成。
- 按后续要求移除 `try/finally`，在正常返回前清理 Context；异常会直接向上传播，
  此时不会自动清理 Context。

删除：

- 对已经不存在的 `prepare_mixed()` 的调用。
- 旧混合分支的请求重排、结果顺序还原及单独的 hidden-state 选择。
- 按 Prefill / Decode 遍历的阶段执行循环，以及重复的采样结果拼装。
- `run_model()` 内无条件计算所有请求 logits 的逻辑。
- 已失效的注释形式旧输入语句。

纯 Decode 的图捕获结构、图缓冲区更新和重放操作继续沿用。

### nanovllm/layers/attention.py

- 明确标注 `context.is_prefill` 对应纯 Prefill / 混合 batch 的 varlen 路径。
- 整理统一 Attention 调用，修正 Q/K 的歧义注释；分页 KV 表覆盖整个混合 batch。
- KV 写入 kernel 和纯 Decode 调用沿用当前实现。
- 原先的双 Attention 混合分支在本轮实施前已经被删除；本轮没有重复删除。

### nanovllm/layers/embed_head.py

- 删除 `get_context` 导入。
- 删除 `context.is_prefill` 判断及末 query 的隐式切片。
- LM Head 只对 runner 已选好的 hidden states 做词表投影，保留 TP gather。

### nanovllm/utils/context.py

- 标注 `is_prefill=True` 表示 varlen 执行路径，不是每个请求的业务阶段。
- 删除 `is_mixed`、Context 内的 `num_decode_tokens`、`prefill_block_tables`。
- 同步删除 `set_context()` 的对应参数和构造字段。
- 调度器/引擎里的 Prefill、Decode token 统计继续保留。

### nanovllm/engine/scheduler.py

- 仅修改 `postprocess()`，不修改排队、准入和抢占策略。
- 结果类型调整为 `list[int | None]`，验证与请求数量一致。
- 在任何 KV 进度提交前检查：只有未完成 Prefill 的位置允许且必须为 `None`。
- 对 `None` 仅提交 KV 进度，不追加 token、不修改输出时间指标。

## 验证入口

```bash
python -B -m unittest discover -s tests -p test_unified_forward.py -v
python -B -m unittest discover -s tests -p test_unified_forward_gpu.py -v
python -B test_mixed_forward_gpu.py
```

GPU 测试需要 CUDA 和本地 `~/huggingface/Qwen3-0.6B` 权重。

- CPU 回归共 8 项：元数据、请求顺序、跨块位置、TP 序列化后的 Decode 输入、
  采样筛选、全 partial 分支、工作 rank、KV/结果提交。移除自动异常清理后，
  对应的旧契约测试同步删除。
- 新增 GPU 回归共 2 项：分页 GQA Attention 与独立 dense causal 计算对照；
  Qwen3 混合 logits、共享前缀、多轮 chunk、按需 LM Head 和纯 Decode CUDA Graph。
- 模型数值对照使用相同打包 GEMM 形状、拆分 Attention 的参考路径，隔离本次融合改动；
  logits 相对 L2 阈值为 2%，同时检查 argmax 一致。没有放宽阈值。
- 曾尝试将每个请求分别运行整模型作参考，相对 L2 约 2.12%，超过该阈值。
  这种参考还改变了 BF16 GEMM 批次形状；本轮验证不据此声称逐请求独立前向数值等价。
- 现有 `test_mixed_forward_gpu.py` 检查实际随机采样和动态到达请求的端到端完成。

实际验证环境：RTX 4060 Laptop GPU，PyTorch 2.8.0+cu126，FlashAttention 2.8.3.post1。
没有进行性能 benchmark，也没有多 GPU 硬件上的 TP 集成验证；TP 输入和分支在 CPU 回归中检查。

## 旧内部诊断接口的兼容性

`run_model()` 现在返回 hidden states；`prepare_mixed()` 和 `Context.is_mixed` 不再存在。
原有 `test_hybrid_logits_gpu.py`、`test_hybrid_logits_diagnostic.py`、
`test_hybrid_layer_probe.py`、`test_qkv_batch_shape.py` 依赖旧内部接口，不能直接用来验证新版本。
旧 CPU 脚本若给未完成 chunk 填入虚构整数 token，也不再符合新的结果约定。
本轮保留这些脚本原文，新增上述回归入口覆盖新契约。

不在本轮范围：aging、KV 分配策略、已有抢占队列推进问题，以及其它配置边界修复。
