"""Real CUDA regressions: paged Attention numerics and Qwen3 mixed execution.

Run: python -B -m unittest discover -s tests -p test_unified_forward_gpu.py -v
The model test uses ~/huggingface/Qwen3-0.6B and one GPU; it is not a benchmark.
"""
import atexit
from pathlib import Path
from time import perf_counter
import unittest
from unittest.mock import patch

import torch

from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.layers import attention as attention_module
from nanovllm.layers.attention import Attention
from nanovllm.sampling_params import SamplingParams
from nanovllm.utils.context import get_context, reset_context, set_context


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class UnifiedForwardGPUTests(unittest.TestCase):
    @torch.inference_mode()
    def test_paged_mixed_attention_matches_dense_causal_reference(self):
        torch.manual_seed(17)
        device, dtype = "cuda", torch.float16
        q_lens, cached = [1, 3, 4], [256, 255, 0]
        kv_lens = [c + q for c, q in zip(cached, q_lens)]
        tables = [[3, 1], [4, 0], [2, -1]]
        q = torch.randn(8, 4, 32, device=device, dtype=dtype)
        k = torch.randn(8, 2, 32, device=device, dtype=dtype)
        v = torch.randn_like(k)
        attn = Attention(4, 32, 32 ** -0.5, 2)
        attn.k_cache = torch.randn(5, 256, 2, 32, device=device, dtype=dtype)
        attn.v_cache = torch.randn_like(attn.k_cache)
        slots = [
            table[p // 256] * 256 + p % 256
            for table, start, length in zip(tables, cached, q_lens)
            for p in range(start, start + length)
        ]
        set_context(
            True,
            cu_seqlens_q=torch.tensor([0, 1, 4, 8], dtype=torch.int32, device=device),
            cu_seqlens_k=torch.tensor([0, 257, 515, 519], dtype=torch.int32, device=device),
            max_seqlen_q=4, max_seqlen_k=258,
            slot_mapping=torch.tensor(slots, dtype=torch.int32, device=device),
            block_tables=torch.tensor(tables, dtype=torch.int32, device=device),
        )
        try:
            with patch.object(attention_module, "flash_attn_varlen_func", wraps=attention_module.flash_attn_varlen_func) as varlen, \
                 patch.object(attention_module, "flash_attn_with_kvcache", wraps=attention_module.flash_attn_with_kvcache) as decode:
                actual = attn(q, k, v)
            self.assertEqual(varlen.call_count, 1)
            self.assertEqual(decode.call_count, 0)

            expected = []
            offset = 0
            for table, start, q_len, kv_len in zip(tables, cached, q_lens, kv_lens):
                physical = torch.tensor([
                    table[p // 256] * 256 + p % 256 for p in range(kv_len)
                ], device=device)
                keys = attn.k_cache.flatten(0, 1)[physical].repeat_interleave(2, dim=1).float()
                values = attn.v_cache.flatten(0, 1)[physical].repeat_interleave(2, dim=1).float()
                queries = q[offset:offset + q_len].float()
                scores = torch.einsum("qhd,khd->hqk", queries, keys) * attn.scale
                mask = torch.arange(kv_len, device=device)[None, :] > (
                    start + torch.arange(q_len, device=device)[:, None]
                )
                scores.masked_fill_(mask.unsqueeze(0), float("-inf"))
                expected.append(torch.einsum("hqk,khd->qhd", scores.softmax(-1), values))
                offset += q_len
            torch.testing.assert_close(actual.float(), torch.cat(expected), atol=3e-3, rtol=3e-3)
            print("PASS: one paged varlen call agrees with dense GQA causal Attention")
        finally:
            reset_context()

    @torch.inference_mode()
    def test_qwen_logits_sampling_kv_progress_and_decode_graph(self):
        model_path = Path.home() / "huggingface/Qwen3-0.6B"
        if not model_path.is_dir():
            self.skipTest("local Qwen3-0.6B weights required")
        llm = LLMEngine(
            str(model_path), enforce_eager=False, tensor_parallel_size=1,
            max_model_len=1024, max_num_batched_tokens=128, max_num_seqs=8,
            gpu_memory_utilization=0.5,
        )
        runner, scheduler = llm.model_runner, llm.scheduler
        self.addCleanup(atexit.unregister, llm.exit)
        self.addCleanup(llm.exit)

        def add(tokens, max_tokens):
            llm.add_request(list(tokens), SamplingParams(max_tokens=max_tokens, ignore_eos=True))
            return scheduler.waiting[-1]

        def step(check_reference=False, expected_samples=None):
            output = scheduler.schedule()
            seqs = output.scheduled_seqs
            ready = [i for i, s in enumerate(seqs) if not s.is_prefill or s.num_cached_tokens + s.num_scheduled_tokens == len(s)]
            use_varlen = any(s.is_prefill for s in seqs)
            reference = {}
            if check_reference:
                slots = sorted({
                    s.block_table[p // runner.block_size] * runner.block_size + p % runner.block_size
                    for s in seqs for p in range(s.num_cached_tokens, s.num_cached_tokens + s.num_scheduled_tokens)
                })
                indices = torch.tensor(slots, device="cuda", dtype=torch.long)
                cache = runner.kv_cache.flatten(2, 3)
                saved = cache.index_select(2, indices).clone()
                # Match the pre-change architecture: packed GEMMs, split Attention.
                # Keeping GEMM shapes identical isolates Attention fusion from BF16
                # roundoff introduced by running each request as a separate batch.
                varlen_func = attention_module.flash_attn_varlen_func
                decode_func = attention_module.flash_attn_with_kvcache

                def split_attention(q, k, v, **kwargs):
                    rows = []
                    cursor = 0
                    for row, seq in enumerate(seqs):
                        nq = seq.num_scheduled_tokens
                        nk = seq.num_cached_tokens + nq
                        query = q[cursor:cursor + nq]
                        table = kwargs["block_table"][row:row + 1]
                        if seq.is_prefill:
                            rows.append(varlen_func(
                                query, k, v,
                                cu_seqlens_q=torch.tensor([0, nq], dtype=torch.int32, device=q.device),
                                cu_seqlens_k=torch.tensor([0, nk], dtype=torch.int32, device=q.device),
                                max_seqlen_q=nq, max_seqlen_k=nk,
                                softmax_scale=kwargs["softmax_scale"], causal=True,
                                block_table=table,
                            ))
                        else:
                            rows.append(decode_func(
                                query.unsqueeze(1), k, v,
                                cache_seqlens=torch.tensor([nk], dtype=torch.int32, device=q.device),
                                block_table=table, softmax_scale=kwargs["softmax_scale"], causal=True,
                            ).squeeze(1))
                        cursor += nq
                    return torch.cat(rows)

                try:
                    ids, pos = runner.prepare_prefill(seqs) if use_varlen else runner.prepare_decode(seqs)
                    with patch.object(attention_module, "flash_attn_varlen_func", side_effect=split_attention):
                        hidden = runner.run_model(ids, pos, use_varlen)
                    last = []
                    cursor = 0
                    for i, seq in enumerate(seqs):
                        cursor += seq.num_scheduled_tokens
                        if i in ready:
                            last.append(cursor - 1)
                    logits = runner.model.compute_logits(hidden[last]).float().cpu()
                    reference = {i: logits[row:row + 1] for row, i in enumerate(ready)}
                finally:
                    reset_context()
                cache.index_copy_(2, indices, saved)

            head_rows, actual_logits, forwards = [], [], []
            compute_logits = runner.model.compute_logits

            def capture_logits(hidden):
                head_rows.append(hidden.shape[0])
                logits = compute_logits(hidden)
                actual_logits.append(logits.float().cpu())
                return logits

            hook = runner.model.register_forward_hook(lambda mod, args, result: forwards.append(args[0].numel()))
            try:
                with patch.object(attention_module, "flash_attn_varlen_func", wraps=attention_module.flash_attn_varlen_func) as varlen, \
                     patch.object(attention_module, "flash_attn_with_kvcache", wraps=attention_module.flash_attn_with_kvcache) as decode, \
                     patch.object(runner.model, "compute_logits", side_effect=capture_logits), \
                     patch.object(runner.sampler, "forward", side_effect=lambda logits, temps: logits.argmax(-1)):
                    tokens = runner.run(seqs)
                    if use_varlen:
                        self.assertEqual(varlen.call_count, runner.config.hf_config.num_hidden_layers)
                        self.assertEqual(decode.call_count, 0)
                    else:
                        # Replay does not re-enter Python model/Attention forwards.
                        self.assertEqual(forwards, [])
                        self.assertEqual(varlen.call_count, 0)
                        self.assertEqual(decode.call_count, 0)
            finally:
                hook.remove()

            self.assertEqual([i for i, token in enumerate(tokens) if token is not None], ready)
            self.assertEqual(head_rows, [len(ready)] if ready else [])
            if expected_samples is not None:
                self.assertEqual(len(ready), expected_samples)
            for row, i in enumerate(ready):
                if i in reference:
                    ref, got = reference[i][0], actual_logits[0][row]
                    relative = (got - ref).norm() / ref.norm().clamp_min(1e-6)
                    self.assertLess(relative.item(), 0.02)
                    self.assertEqual(got.argmax().item(), ref.argmax().item())
            scheduler.postprocess(seqs, tokens, perf_counter())
            return output

        # No-output chunks must still complete GPU KV writes and commit progress.
        prime_tokens = [100 + i % 300 for i in range(600)]
        prime = add(prime_tokens, 1)
        step(expected_samples=0)
        self.assertEqual(prime.num_cached_tokens, 128)
        self.assertIsNone(prime.first_token_time)
        while not scheduler.is_finished():
            step()

        a = add(range(10, 18), 8)
        step(expected_samples=1)
        b = add(prime_tokens[:512] + [900 + i for i in range(200)], 2)
        self.assertGreaterEqual(scheduler.block_manager.can_allocate(b), 1)
        step(check_reference=True, expected_samples=1)
        self.assertEqual(b.num_completion_tokens, 0)
        step(check_reference=True, expected_samples=2)
        self.assertEqual(b.num_completion_tokens, 1)
        rounds = 0
        while not scheduler.is_finished():
            step(check_reference=True)
            rounds += 1
            self.assertLess(rounds, 16)
        self.assertEqual((a.num_completion_tokens, b.num_completion_tokens), (8, 2))
        self.assertFalse(scheduler.block_manager.used_block_ids)
        print("PASS: Qwen3 mixed logits, prefix chunks, selected LM Head rows and Decode CUDA Graph")


if __name__ == "__main__":
    unittest.main()
