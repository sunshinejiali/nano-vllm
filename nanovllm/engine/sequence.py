from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


# 定义序列状态枚举类，用于表示序列在生成过程中的不同状态
class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


# 负责管理语言模型生成过程中的单个序列，包括输入提示、生成结果、缓存状态等信息。
class Sequence:
    # 定义序列的最大缓存块大小，用于管理缓存中的token块
    block_size = 256
    # 定义序列计数器，用于为每个序列分配唯一的ID
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        # 初始化序列ID，从计数器中获取下一个唯一ID
        self.seq_id = next(Sequence.counter)
        # 初始化序列状态为等待运行
        self.status = SequenceStatus.WAITING
        # 初始化序列的token ID列表，深拷贝输入的token ID列表
        self.token_ids = copy(token_ids)
        # 初始化序列的最后一个token ID，即输入提示的最后一个token
        self.last_token = token_ids[-1]
        # 初始化序列的token数量
        self.num_tokens = len(self.token_ids)
        # 初始化序列的提示token数量
        self.num_prompt_tokens = len(token_ids)
        # 初始化序列的缓存token数量，初始为0
        self.num_cached_tokens = 0
        # 初始化序列的缓存块表，初始为空列表. 块表（存储缓存块ID）
        self.block_table = []
        # 初始化序列的温度参数，用于控制生成的随机性
        self.temperature = sampling_params.temperature
        # 初始化序列的最大生成token数量
        self.max_tokens = sampling_params.max_tokens
        # 初始化序列的忽略EOS参数，用于控制是否忽略生成的EOS token
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        # 返回序列的token数量
        return self.num_tokens

    def __getitem__(self, key):
        # 返回序列的token ID列表中指定索引的token ID
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        # 返回序列的完成token数量，即生成的token数量减去提示token数量
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self):
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    # 返回序列的缓存块表中指定索引的缓存块
    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    # 在解码阶段逐个添加生成的token。
    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    # 如果还没有生成token，保存完整token列表；否则只保存最后一个token。
    # 序列状态为运行时
    def __getstate__(self):
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
                self.token_ids if self.num_completion_tokens == 0 else self.last_token)

    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1]
        if self.num_completion_tokens == 0:
            # 恢复完整token列表
            self.token_ids = state[-1]
        else:
            # 恢复最后一个生成的token
            self.last_token = state[-1]
