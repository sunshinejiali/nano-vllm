from functools import lru_cache
import torch
from torch import nn


# 应用旋转嵌入到输入张量, 旋转位置编码
def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    # 将向量分成两部分
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    # 应用旋转嵌入
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    # 合并两部分
    # (x1 + ix2) * (cosθ + isinθ)
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        # 旋转维度等于头维度
        self.head_size = head_size
        assert rotary_dim == head_size
        # 计算频率：θ_i = base^(-2i/d)
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        # 生成位置序列的频率
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        # 计算频率矩阵：θ = t * inv_freq
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        # 计算余弦和正弦值
        cos = freqs.cos()
        sin = freqs.sin()
        # 缓存cos和sin值
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 根据位置获取cos/sin值
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        # 应用旋转嵌入到查询和键
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    assert rope_scaling is None
    # 使用LRU缓存创建RotaryEmbedding实例，避免重复创建。
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
