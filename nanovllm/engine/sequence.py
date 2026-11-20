from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    block_size = 256
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0

        self.num_scheduled_tokens = 0  # 本轮计划计算的 token 数
        
        self.is_prefill = True
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

        # 用于统计 TTFT TPOT 等指标 (2026.10.24)
        self.arrival_time: float | None = None
        self.first_token_time: float | None = None
        self.finish_time: float | None = None

        self.last_token_time: float | None = None
        self.inter_token_latencies: list[float] = []


    # 2026.10.24
    @property 
    def time_to_first_token(self) -> float | None:
        if self.arrival_time is None or self.first_token_time is None:
            return None
        
        return (self.first_token_time - self.arrival_time) * 1000
    
    @property 
    def time_per_output_token(self) -> float | None:
        if self.first_token_time is None or self.finish_time is None or self.num_completion_tokens == 1:
            return None
        
        return (self.finish_time - self.first_token_time) / (self.num_completion_tokens - 1)  * 1000

    @property
    def metrics(self) -> dict:
        return {
            "time_to_first_token": self.time_to_first_token,
            "time_per_output_token": self.time_per_output_token,
            "inter_token_latencies": self.inter_token_latencies.copy()
        }

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        last_state = self.last_token if not self.is_prefill else self.token_ids
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state)

    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state = state
        self.is_prefill = isinstance(last_state, list)
        if isinstance(last_state, list):
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else:
            self.token_ids = []
            self.last_token = last_state
