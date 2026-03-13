import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    # 用于解析路径中的用户主目录简写符号（如 ~ 或 ~user）
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    # 分词器 用于将文本转换为模型可以理解的 token 序列
    tokenizer = AutoTokenizer.from_pretrained(path)
    # 加载 Qwen3-0.6B 模型
    # enforce_eager=True 强制使用 eager 模式，不使用 torch.compile 优化
    # tensor_parallel_size=1 表示不使用张量并行，单卡运行
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    # 采样参数 用于控制生成文本的随机性和长度
    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    # 输入提示列表 包含两个示例：自我介绍和列出100以内的所有质数
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    # 对每个提示应用聊天模板，生成模型可以理解的输入格式
    # tokenize=False 表示不进行分词，直接返回字符串
    # add_generation_prompt=True 表示在生成时添加生成提示
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    # 调用模型生成文本
    # prompts 输入提示列表
    # sampling_params 采样参数
    outputs = llm.generate(prompts, sampling_params)

    # 打印每个提示和对应的生成文本
    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
