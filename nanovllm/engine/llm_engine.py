import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        arrival_time = perf_counter()   # 记录 seq 到达时刻 (2026.10.24)

        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        seq.arrival_time = arrival_time # 记录 seq 到达时刻 (2026.10.24)
        
        self.scheduler.add(seq)

    def step(self):
        # Each schedule step may contain both Decode and Prefill requests.
        schedule_output = self.scheduler.schedule()
        seqs = schedule_output.scheduled_seqs
        num_prefill_tokens = schedule_output.num_prefill_tokens
        num_decode_tokens = schedule_output.num_decode_tokens

        token_ids = self.model_runner.call("run", seqs)

        token_ready_time = perf_counter()   # 记录本轮推理结果就绪时间

        # 模型计算结束后 更新请求的状态
        self.scheduler.postprocess(seqs, token_ids, token_ready_time) # 记录 first token 完成时刻 (2026.10.24)

        outputs = [
            (seq.seq_id, seq.completion_token_ids, seq.metrics)
            for seq in seqs if seq.is_finished
        ]
        return outputs, num_prefill_tokens, num_decode_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)    
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_prefill_tokens, num_decode_tokens = self.step()
            elapsed = perf_counter() - t

            if num_prefill_tokens:
                prefill_throughput = num_prefill_tokens / elapsed
            if num_decode_tokens:
                decode_throughput = num_decode_tokens / elapsed
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })

            # ouput 只记录已经完成的 seq
            for seq_id, token_ids, metrics in output:
                outputs[seq_id] = (token_ids, metrics)
                pbar.update(1)

        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]    # 按照 seq_id 排序
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids, "metrics": metrics} for token_ids, metrics in outputs]
        return outputs
