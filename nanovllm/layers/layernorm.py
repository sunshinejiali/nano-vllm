import torch
from torch import nn


# 根均方归一化层类，用于 Qwen3 模型的归一化层，支持张量并行
class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps  # 数值稳定性常数, 防止除0错误
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        # 保存原始数据类型, 后续操作需转换为float32
        orig_dtype = x.dtype
        # 转换为float32提高数值稳定性
        x = x.float()
        # 计算均方值
        var = x.pow(2).mean(dim=-1, keepdim=True)
        # 归一化：x = x / sqrt(mean(x^2) + eps)
        x.mul_(torch.rsqrt(var + self.eps))
        # 缩放：x = x * weight
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    # 带残差连接的归一化前向传播
    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        # 添加残差连接
        x = x.float().add_(residual.float())
        # 保存新的残差
        residual = x.to(orig_dtype)
        # 计算归一化后的均方值
        var = x.pow(2).mean(dim=-1, keepdim=True)
        # 归一化：x = x / sqrt(mean(x^2) + eps)
        x.mul_(torch.rsqrt(var + self.eps))
        # 缩放：x = x * weight
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    # 前向传播，根据是否有残差连接选择不同的归一化方法
    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
