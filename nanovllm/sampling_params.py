from dataclasses import dataclass


@dataclass
class SamplingParams:
    # temperature 温度参数，控制生成文本的随机性
    # max_tokens 每个生成文本的最大 token 数量
    # ignore_eos 是否忽略生成文本中的结束 token（如 <|endoftext|>）
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self):
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
