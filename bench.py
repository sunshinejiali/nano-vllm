import os
import time
from random import randint, seed
from nanovllm import LLM, SamplingParams
# from vllm import LLM, SamplingParams


def main():
    seed(0)
    # 基准测试参数
    # num_seqs 序列数量，即同时处理的输入提示数量
    num_seqs = 256
    # max_input_len 最大输入长度，即每个输入提示的最大 token 数量
    max_input_len = 1024
    # max_ouput_len 最大输出长度，即每个生成文本的最大 token 数量
    max_ouput_len = 1024

    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    # 加载 Qwen3-0.6B 模型
    # enforce_eager=False 不强制使用 eager 模式，使用 torch.compile 优化
    # max_model_len=4096 模型最大长度，即模型可以处理的最大 token 数量
    llm = LLM(path, enforce_eager=False, max_model_len=4096)

    # 生成随机输入提示的 token 序列
    # 每个序列的长度在 [100, max_input_len] 之间
    # 每个 token 是一个随机整数，范围在 [0, 10000] 之间
    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    # 生成随机采样参数
    # temperature=0.6 温度参数，控制生成文本的随机性
    # ignore_eos=True 忽略生成文本中的结束 token（如 <|endoftext|>）
    # max_tokens=randint(100, max_ouput_len) 每个生成文本的最大 token 数量，在 [100, max_ouput_len] 之间随机选择
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)) for _ in range(num_seqs)]
    # uncomment the following line for vllm
    # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    # 预热模型，确保模型加载完成
    llm.generate(["Benchmark: "], SamplingParams())
    t = time.time()
    # 调用模型生成文本
    # prompt_token_ids 输入提示的 token 序列列表
    # sampling_params 采样参数列表
    # use_tqdm=False 不使用进度条显示生成进度
    outputs = llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    # 模型生成文本的总时间
    t = (time.time() - t)
    # 计算总 token 数量
    # 每个序列的最大 token 数量之和
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    # 计算吞吐量
    # 总 token 数量除以生成时间，单位为 token/秒
    throughput = total_tokens / t
    print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")


if __name__ == "__main__":
    main()
