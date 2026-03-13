import torch
from torch import nn
import torch.nn.functional as F


# SiluAndMul 类实现 SiLU 激活函数和元素乘法
class SiluAndMul(nn.Module):

    def __init__(self):
        super().__init__()

    # 前向传播函数，输入张量 x 进行 SiLU 激活函数和元素乘法
    @torch.compile  # 应用编译装饰器
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y
