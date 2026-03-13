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


# LLMEngine 类实现 LLM 引擎，负责模型推理和请求管理
class LLMEngine:

    def __init__(self, model, **kwargs):
        # 1. 解析配置参数
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        # 2. 初始化模型运行器, 并启动其他进程, 每个进程运行一个模型副本,张量并行
        self.ps = []    #子进程列表
        self.events = []    # 事件列表, 用于进程间通信
        ctx = mp.get_context("spawn")   # 创建子进程上下文，使用spawn方法
        # 3. 启动主进程模型运行器, 并等待其他进程初始化完成, 启动子进程（张量并行）
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        # 4. 主进程启动模型运行器
        self.model_runner = ModelRunner(config, 0, self.events)
        # 5. 分词器初始化
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id     # 配置结束符ID
        # 6. 调度器初始化
        self.scheduler = Scheduler(config)
        # 7. 注册退出函数, 确保在程序结束时正确退出模型运行器和子进程
        atexit.register(self.exit)

    def exit(self):
        # 通知模型运行器退出
        self.model_runner.call("exit")
        # 删除主进程运行器
        del self.model_runner
        # 等待所有子进程结束
        for p in self.ps:
            p.join()

    # 将用户请求添加到调度器
    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            # 文本转token ID
            prompt = self.tokenizer.encode(prompt)
        # 创建序列对象
        seq = Sequence(prompt, sampling_params)
        # 将序列添加到调度器
        self.scheduler.add(seq)

    # 执行模型推理步骤
    def step(self):
        # 1. 调度：从调度器获取待处理序列
        seqs, is_prefill = self.scheduler.schedule()
        # 2. 模型运行：调用模型运行器执行推理
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        # 3. 后处理：更新序列状态和调度器
        self.scheduler.postprocess(seqs, token_ids)
        # 4. 收集输出：返回已完成序列的ID和生成的token ID
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        # 5. 返回结果：已完成序列ID和生成的token ID列表, 以及处理的token数量。计算token数量用于性能统计。
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    # 检查所有序列是否完成
    def is_finished(self):
        return self.scheduler.is_finished()

    # 生成文本序列，批量处理多个请求
    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        # 1. 初始化进度条
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        # 2. 处理输入参数：确保采样参数列表与提示列表长度匹配
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        # 3. 添加请求到调度器
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        # 4. 主循环：持续执行推理步骤，直到所有序列完成
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            # 执行推理步骤，开始计时
            t = perf_counter()
            # 执行单步推理
            output, num_tokens = self.step()
            # 5. 更新进度条和统计信息
            if use_tqdm:
                if num_tokens > 0:  # 仅在预填充阶段更新预填充吞吐量
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:   # 仅在解码阶段更新解码吞吐量
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            # 6. 处理输出：将已完成序列的输出存储到字典中
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)  # 更新进度条
        # 7. 结果整理和返回：根据序列ID排序，将token ID转换为文本输出
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        # 8. 关闭进度条
        if use_tqdm:
            pbar.close()
        return outputs
