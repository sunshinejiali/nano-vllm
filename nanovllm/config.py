import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    # model 模型路径，指向 Hugging Face 模型仓库或本地模型目录
    model: str
    # max_num_batched_tokens 最大批量 token 数量，即模型每次处理的最大 token 数量
    max_num_batched_tokens: int = 16384
    # max_num_seqs 最大序列数量，即模型同时处理的最大输入提示数量
    max_num_seqs: int = 512
    # max_model_len 最大模型长度，即模型可以处理的最大 token 数量
    max_model_len: int = 4096
    # gpu_memory_utilization GPU 内存利用率，即模型占用的 GPU 内存比例
    gpu_memory_utilization: float = 0.9
    # tensor_parallel_size 张量并行大小，即模型并行处理的 GPU 数量
    tensor_parallel_size: int = 1
    # enforce_eager 是否强制使用 eager 模式，使用 torch.compile 优化
    enforce_eager: bool = False
    # hf_config Hugging Face 模型配置，自动从模型路径加载
    hf_config: AutoConfig | None = None
    # eos 结束 token 索引，默认值为 -1 表示不使用结束 token
    eos: int = -1
    # kvcache_block_size KV 缓存块大小，即 KV 缓存中每个缓存块的 token 数量
    kvcache_block_size: int = 256
    # num_kvcache_blocks KV 缓存块数量，默认值为 -1 表示自动计算
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        # 检查模型路径是否存在
        assert os.path.isdir(self.model)
        # 检查 KV 缓存块大小是否为 256 的倍数
        # 原因：KV 缓存块大小必须是 256 的倍数，以确保对齐
        assert self.kvcache_block_size % 256 == 0
        # 检查张量并行大小是否在 [1, 8] 之间
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        # 检查最大批量 token 数量是否大于等于最大模型长度
        assert self.max_num_batched_tokens >= self.max_model_len
