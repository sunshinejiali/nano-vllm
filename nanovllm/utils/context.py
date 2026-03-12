from dataclasses import dataclass
import torch

# Context 上下文类，用于存储模型推理时的上下文信息
@dataclass
class Context:
    # is_prefill 是否为预填充阶段，True 表示预填充阶段，False 表示解码阶段
    is_prefill: bool = False
    # cu_seqlens_q 查询序列长度的累积和，用于计算查询序列的起始位置
    cu_seqlens_q: torch.Tensor | None = None
    # cu_seqlens_k 键序列长度的累积和，用于计算键序列的起始位置
    cu_seqlens_k: torch.Tensor | None = None
    # max_seqlen_q 查询序列的最大长度
    max_seqlen_q: int = 0
    # max_seqlen_k 键序列的最大长度
    max_seqlen_k: int = 0
    # slot_mapping 槽位映射，用于将查询序列和键序列映射到 KV 缓存槽位
    slot_mapping: torch.Tensor | None = None
    # context_lens 上下文长度，记录每个序列的有效上下文长度
    context_lens: torch.Tensor | None = None
    # block_tables 块表，记录每个序列的 KV 缓存块索引
    block_tables: torch.Tensor | None = None

_CONTEXT = Context()

# get_context 获取上下文实例
def get_context():
    return _CONTEXT

# set_context 设置上下文实例
def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)

# reset_context 重置上下文实例
def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
