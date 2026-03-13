import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context


# 并行词汇表嵌入层类，用于 Qwen3 模型的并行词汇表嵌入，支持张量并行
class VocabParallelEmbedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        # 当前GPU在张量并行中的排名
        self.tp_rank = dist.get_rank()
        # 张量并行GPU总数
        self.tp_size = dist.get_world_size()
        # 确保词汇表可均匀分割
        assert num_embeddings % self.tp_size == 0
        # 计算每个GPU负责的词汇表范围
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        # 初始化权重矩阵（每个GPU只负责部分词汇表）
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        # 当前分区的词汇表大小
        shard_size = param_data.size(0)
        # 全局起始索引
        start_idx = self.tp_rank * shard_size
        # 切片加载
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        # 复制到当前分区
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        # 张量并行处理：掩码和索引重映射
        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx)
        # 嵌入查找
        y = F.embedding(x, self.weight)
        # 张量并行：掩码处理和全局归约
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y  # 应用掩码
            dist.all_reduce(y)  # 全局求和
        return y


# 并行语言模型头层类，用于 Qwen3 模型的并行语言模型头，支持张量并行
class ParallelLMHead(VocabParallelEmbedding):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        context = get_context()
        # 预填充阶段：提取序列最后一个token的隐藏状态
        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        # 线性投影：隐藏状态 → 词汇表logits
        logits = F.linear(x, self.weight)
        # 张量并行：收集所有GPU的logits
        if self.tp_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, 0)
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits
