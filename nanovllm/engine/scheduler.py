from collections import deque
from dataclasses import dataclass

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager

@dataclass
class ScheduleOutput:
    scheduled_seqs: list[Sequence]
    num_decode_tokens: int
    num_prefill_tokens: int

    @property
    def num_scheduled_tokens(self) -> int:
        return self.num_decode_tokens + self.num_prefill_tokens


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    # 11.11 Hybrid scheduling
    def schedule(self) -> ScheduleOutput:
        scheduled_seqs = []
        num_decode_tokens = 0
        num_prefill_tokens = 0
        
        # Phase 1: Decode
        decode_seqs = []
        preempted_ids = set()
        while (
            self.running 
            and len(scheduled_seqs) < self.max_num_seqs 
            and num_decode_tokens < self.max_num_batched_tokens
        ):
            # 从 running 队列中取出一个请求
            seq = self.running.popleft()

            while not self.block_manager.can_append(seq):
                # 如果还有其他运行中的请求 优先抢占"队尾"请求
                if self.running:
                    victim = self.running.pop()
                    self.preempt(victim)
                    preempted_ids.add(id(victim))
                else:
                    self.preempt(seq)
                    preempted_ids.add(id(seq))
                    break
            
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                # 按需为新增的 token 分配 Block
                self.block_manager.may_append(seq)
                
                decode_seqs.append(seq)
                scheduled_seqs.append(seq)
                num_decode_tokens += 1

        # 将刚解码完成的序列按原顺序放回运行队列的队首 (维持 FCFS 调度顺序)
        self.running.extendleft(reversed(decode_seqs))

        # Phase 2: Prefill
        # 在进入 Prefill 循环之前 将 Waiting 队首那些本轮刚被抢占的请求暂时取出
        deferred = []
        while self.waiting and id(self.waiting[0]) in preempted_ids:
            deferred.append(self.waiting.popleft())
        
        num_prefill_seqs = 0
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            remaining = self.max_num_batched_tokens - num_decode_tokens - num_prefill_tokens
            if remaining == 0:
                break
            
            seq = self.waiting[0]
            if not seq.block_table:
                # 返回可复用的前缀 Block 数量 -1 表示空间不足
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                    
                # 扣除可直接复用的前缀 Token 得到剩余待计算 Token 数
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            
            # 当前请求已分配 KV Cache Blocks (Chunked prefill)
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens

            # if remaining < num_tokens and num_prefill_seqs > 0:
            #     break   # Keep the original one-chunked-Prefill-per-step rule.

            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)

            seq.is_prefill = True
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_prefill_tokens += seq.num_scheduled_tokens
            num_prefill_seqs += 1
            scheduled_seqs.append(seq)

            if seq.num_scheduled_tokens + seq.num_cached_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            # Otherwise the incomplete Prefill remains at the head of waiting.
            
        self.waiting.extendleft(reversed(deferred))
        
        if not scheduled_seqs:
            raise RuntimeError("No sequence schedulable (check KV cache capacity)")
        
        return ScheduleOutput(scheduled_seqs, num_decode_tokens, num_prefill_tokens)

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)

        # 被强占的请求直接放回 waiting 队首
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int | None], token_ready_time: float):
        assert len(seqs) == len(token_ids), "Each scheduled request must have one result slot"
        # 在提交任何 KV 进度前校验结果 避免缺失输出或把占位值写进序列
        for seq, token_id in zip(seqs, token_ids):
            partial_prefill = (
                seq.is_prefill
                and seq.num_cached_tokens + seq.num_scheduled_tokens < seq.num_tokens
            )
            assert (token_id is None) == partial_prefill, "Invalid sampling result for request stage"
        for seq, token_id in zip(seqs, token_ids):
            # 将新计算完成的 KV Cache Blocks 注册到哈希表中
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if token_id is None:  # 未完成 chunk 仅提交 KV 进度 不更新输出和时间指标
                continue
            seq.append_token(token_id)
            
            # 计算 ITL
            if seq.last_token_time is not None:
                inter_token_latencie = (token_ready_time - seq.last_token_time) * 1000
                seq.inter_token_latencies.append(inter_token_latencie)
            
            # 更新
            seq.last_token_time = token_ready_time

            if seq.num_completion_tokens == 1:  # 记录 first token 时刻 (2026.10.24)
                seq.first_token_time = token_ready_time

            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.finish_time = token_ready_time  # 记录完成时刻 (2026.10.24)

                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
