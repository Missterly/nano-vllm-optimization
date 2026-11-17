from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


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

    # 10.30 本轮需要处理的token数: num_new_tokens = num_tokens - num_cached_tokens
    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break

            # 当前请求尚未分配 KV Cache Blocks
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

            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:  # Prefill优先
            return scheduled_seqs, True

        # decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()

            # while...else: 如果 while 循环正常结束 没有通过 break 退出 就执行 else
            # 检查是否有足够的 KV Cache 空间继续执行
            while not self.block_manager.can_append(seq):

                # 如果还有其他运行中的请求 优先抢占"队尾"请求
                if self.running:
                    self.preempt(self.running.pop())

                # 如果没有其他请求可以抢占 则抢占当前请求
                else:
                    self.preempt(seq)
                    break
            # 如果 KV Cache 空间足够 则调度当前请求
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False

                # 必要时分配新的 KV Cache Block
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)

        # 若本轮没有请求调度成功 会抛异常(后续可以优化)
        assert scheduled_seqs

        # 将本轮选中的请求重新放回 running 队列
        self.running.extendleft(reversed(scheduled_seqs))

        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool, token_ready_time: float):
        for seq, token_id in zip(seqs, token_ids):
            # 将新计算完成的 KV Cache Blocks 注册到哈希表中
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
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
