"""CPU regressions for unified input packing, sampling and KV progress commits.

Run: python -B -m unittest discover -s tests -p test_unified_forward.py -v
CUDA transfers are replaced with CPU transfers; the real runner methods execute.
"""
import pickle
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.utils.context import get_context, reset_context


def sequence(tokens, cached=0, scheduled=1, prefill=True, blocks=()):
    seq = Sequence(list(tokens))
    seq.num_cached_tokens = cached
    seq.num_scheduled_tokens = scheduled
    seq.is_prefill = prefill
    seq.block_table = list(blocks)
    return seq


class UnifiedForwardTests(unittest.TestCase):
    def setUp(self):
        self.runner = ModelRunner.__new__(ModelRunner)
        self.runner.block_size = 256
        self.runner.rank = 0
        tensor = torch.tensor

        def cpu_tensor(*args, **kwargs):
            kwargs.pop("pin_memory", None)
            return tensor(*args, **kwargs)

        self.tensor_patch = patch("torch.tensor", side_effect=cpu_tensor)
        self.cuda_patch = patch.object(torch.Tensor, "cuda", lambda value, **kw: value)
        self.tensor_patch.start()
        self.cuda_patch.start()
        self.addCleanup(self.tensor_patch.stop)
        self.addCleanup(self.cuda_patch.stop)
        self.addCleanup(reset_context)

    def test_packing_preserves_order_and_cross_block_slots(self):
        prefill = sequence(range(270), cached=255, scheduled=3, blocks=[2, 4])
        decode = sequence(range(257), cached=256, prefill=False, blocks=[1, 3])
        ids, positions = self.runner.prepare_prefill([prefill, decode])
        ctx = get_context()
        self.assertEqual(ids.tolist(), [255, 256, 257, 256])
        self.assertEqual(positions.tolist(), [255, 256, 257, 256])
        self.assertEqual(ctx.cu_seqlens_q.tolist(), [0, 3, 4])
        self.assertEqual(ctx.cu_seqlens_k.tolist(), [0, 258, 515])
        self.assertEqual(ctx.slot_mapping.tolist(), [767, 1024, 1025, 768])
        self.assertEqual(ctx.block_tables.tolist(), [[2, 4], [1, 3]])
        self.assertEqual((ctx.max_seqlen_q, ctx.max_seqlen_k), (3, 258))

    def test_tp_worker_decode_uses_last_token_without_history(self):
        decode = pickle.loads(pickle.dumps(
            sequence(range(257), cached=256, prefill=False, blocks=[1, 3])
        ))
        self.assertEqual(decode.token_ids, [])
        prefill = pickle.loads(pickle.dumps(sequence([9, 10], scheduled=2, blocks=[4])))
        ids, _ = self.runner.prepare_prefill([decode, prefill])
        self.assertEqual(ids.tolist(), [256, 9, 10])

    def fake_forward(self, ids, positions, use_varlen):
        # Row values make wrong last-query selection or result order observable.
        return torch.arange(ids.numel(), dtype=torch.float32).unsqueeze(1)

    def setup_model(self):
        self.runner.run_model = Mock(side_effect=self.fake_forward)
        self.runner.model = SimpleNamespace(compute_logits=Mock(side_effect=lambda x: x))
        self.runner.sampler = Mock(side_effect=lambda logits, temps: logits[:, 0].long())

    def test_only_ready_requests_reach_lm_head_in_original_order(self):
        self.setup_model()
        partial = sequence([10, 11, 12], scheduled=2, blocks=[0])
        decode = sequence([20, 21], cached=1, prefill=False, blocks=[1])
        final = sequence([30, 31], scheduled=2, blocks=[2])
        tokens = self.runner.run([partial, decode, final])
        self.assertEqual(tokens, [None, 2, 4])
        self.runner.run_model.assert_called_once()
        self.assertTrue(self.runner.run_model.call_args.args[2])
        self.assertEqual(self.runner.model.compute_logits.call_args.args[0].flatten().tolist(), [2, 4])
        self.assertEqual(self.runner.sampler.call_args.args[1].numel(), 2)
        self.assertIsNone(get_context().cu_seqlens_q)

    def test_all_partial_chunks_still_forward_and_synchronize(self):
        self.setup_model()
        stream = Mock()
        with patch("torch.cuda.current_stream", return_value=stream):
            tokens = self.runner.run([sequence([1, 2, 3], scheduled=2, blocks=[0])])
        self.assertEqual(tokens, [None])
        self.runner.run_model.assert_called_once()
        self.runner.model.compute_logits.assert_not_called()
        self.runner.sampler.assert_not_called()
        stream.synchronize.assert_called_once()

    def test_pure_decode_selects_decode_execution_path(self):
        self.setup_model()
        tokens = self.runner.run([
            sequence([1, 2], cached=1, prefill=False, blocks=[0]),
            sequence([3, 4], cached=1, prefill=False, blocks=[1]),
        ])
        self.assertEqual(tokens, [0, 1])
        self.assertFalse(self.runner.run_model.call_args.args[2])

    def test_tp_worker_runs_lm_head_without_sampling(self):
        self.setup_model()
        self.runner.rank = 1
        decode = pickle.loads(pickle.dumps(sequence([1, 2], cached=1, prefill=False, blocks=[0])))
        final = pickle.loads(pickle.dumps(sequence([3, 4], scheduled=2, blocks=[1])))
        self.assertIsNone(self.runner.run([decode, final]))
        self.runner.model.compute_logits.assert_called_once()
        self.runner.sampler.assert_not_called()

    def test_postprocess_commits_partial_kv_without_output_metrics(self):
        scheduler = Scheduler(SimpleNamespace(
            max_num_seqs=4, max_num_batched_tokens=16, eos=-1,
            num_kvcache_blocks=4, kvcache_block_size=256,
        ))
        partial = sequence([1, 2, 3], scheduled=2)
        final = sequence([4, 5], scheduled=2)
        scheduler.postprocess([partial, final], [None, 42], 10.0)
        self.assertEqual(partial.num_cached_tokens, 2)
        self.assertEqual(partial.num_completion_tokens, 0)
        self.assertIsNone(partial.first_token_time)
        self.assertIsNone(partial.last_token_time)
        self.assertEqual(final.completion_token_ids, [42])
        self.assertEqual(final.first_token_time, 10.0)

    def test_invalid_result_is_rejected_before_any_commit(self):
        scheduler = Scheduler(SimpleNamespace(
            max_num_seqs=4, max_num_batched_tokens=16, eos=-1,
            num_kvcache_blocks=4, kvcache_block_size=256,
        ))
        ready = sequence([1, 2], scheduled=2)
        partial = sequence([3, 4, 5], scheduled=2)
        with self.assertRaises(AssertionError):
            scheduler.postprocess([ready, partial], [42, 43], 10.0)
        self.assertEqual(ready.num_cached_tokens, 0)
        self.assertEqual(ready.num_completion_tokens, 0)


if __name__ == "__main__":
    unittest.main()
