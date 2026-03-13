import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context

# 存储键值缓存的 Triton ，内核函数 Triton 即时编译装饰器
@triton.jit # 应用 Triton JIT 编译， 将 Python 代码编译为优化的 GPU 汇编代码
def store_kvcache_kernel(
    # 输入key张量的指针
    key_ptr,
    # 输入key张量的步幅
    key_stride,
    # 输入value张量的指针
    value_ptr,
    # 输入value张量的步幅
    value_stride,
    # 输出key缓存张量的指针
    k_cache_ptr,
    # 输出value缓存张量的指针
    v_cache_ptr,
    # 槽映射指针，用于指定每个样本的缓存位置
    slot_mapping_ptr,
    # 头维度，即每个头的特征维度
    D: tl.constexpr,
):
    # 每个线程处理一个样本的键值对
    # 获取当前线程处理的全局索引
    idx = tl.program_id(0)
    # 加载槽位映射，确定当前样本的缓存位置
    slot = tl.load(slot_mapping_ptr + idx)
    # 如果槽位无效，直接返回
    if slot == -1: return
    # 计算输入key/value的内存偏移
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    # 加载输入key/value
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    # 计算输出缓存的内存偏移
    cache_offsets = slot * D + tl.arange(0, D)
    # 存储key/value到缓存
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


# 存储键值缓存的 Triton 内核函数
def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


# 注意力层类，实现 Qwen3 模型的自注意力机制
class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        # 注意力头数量
        self.num_heads = num_heads
        # 每个头的特征维度
        self.head_dim = head_dim
        # 缩放因子，用于控制注意力权重的缩放（通常为1/head_dim的平方根）
        self.scale = scale
        # KV 头数量，通常与 Q 头数量相同 支持分组查询注意力机制
        self.num_kv_heads = num_kv_heads
        # 初始化空的 KV 缓存张量
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context() #获取当前上下文
        k_cache, v_cache = self.k_cache, self.v_cache
        # 1. 存储键值对到缓存
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        # 2. 执行注意力计算，预填充阶段
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            # 使用分组查询注意力机制，进行预填充阶段的注意力计算
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            # 使用带kv缓存的flash attention，进行解码阶段的注意力计算
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables,
                                        softmax_scale=self.scale, causal=True)
        return o
