from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


# 调度器类，负责管理等待队列和运行队列中的序列
class Scheduler:

    def __init__(self, config: Config):
        # 最大并行序列数
        self.max_num_seqs = config.max_num_seqs
        # 最大批处理token数
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    # 调度等待队列中的序列到运行队列
    def schedule(self) -> tuple[list[Sequence], bool]:
        # prefill
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            # 从等待队列中取出第一个序列
            seq = self.waiting[0]
            # 检查资源是否足够
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break   # 资源不足，跳出循环
            num_seqs += 1
            self.block_manager.allocate(seq)    # 分配缓存块
            num_batched_tokens += len(seq) - seq.num_cached_tokens  # 累加批处理token数
            seq.status = SequenceStatus.RUNNING # 标记为运行中
            self.waiting.popleft()  # 从等待队列中移除
            self.running.append(seq)    # 加入运行队列
            scheduled_seqs.append(seq)  # 加入已调度序列列表
        if scheduled_seqs:
            return scheduled_seqs, True # 返回预填充序列和标志

        # decode
        while self.running and num_seqs < self.max_num_seqs:
            # 从运行队列中取出第一个序列
            seq = self.running.popleft()
            # 检查是否可以追加token
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())    # 抢占其他序列
                else:
                    self.preempt(seq)    # 抢占当前序列
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)    # 尝试追加token
                scheduled_seqs.append(seq)  # 加入已调度序列列表
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))    # 将已调度序列逆序加入运行队列
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        # 状态回退到等待
        seq.status = SequenceStatus.WAITING
        # 释放缓存块
        self.block_manager.deallocate(seq)
        # 加入等待队列
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)  # 添加生成的token
            # 检查是否完成
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
