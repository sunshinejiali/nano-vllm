# -*- coding: utf-8 -*-
import os
import torch
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer
from torch.profiler import profile, record_function, ProfilerActivity
from torch.profiler import tensorboard_trace_handler

PROFILE_DIR = "./nanovlm_test_profiler"
MEMORY_SNAPSHOT_PATH = "./nanovlm_test_memory.pickle"
LOG_FILE = "./nanovlm_test.log"
os.makedirs(PROFILE_DIR, exist_ok=True)

if os.path.exists(MEMORY_SNAPSHOT_PATH):
    os.remove(MEMORY_SNAPSHOT_PATH)

path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
tokenizer = AutoTokenizer.from_pretrained(path)
llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)

prompt = "Hello, who are you?"

with torch.no_grad():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=tensorboard_trace_handler(PROFILE_DIR),
        profile_memory=True,
        with_stack=True,
        with_modules=True,
    ) as prof:

        with record_function("NANO_VLLM_GENERATE"):
            outputs = llm.generate(
                prompts=prompt,
                sampling_params=sampling_params
            )

        prof.step()

    torch.cuda.memory._snapshot(MEMORY_SNAPSHOT_PATH)

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("===== Performance Stats =====\n")
        f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
        f.write("\n\n===== Memory Usage =====\n")
        f.write(f"Peak memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB\n")
        f.write(f"Current memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB\n")
        f.write("\n\n===== Output =====\n")
        f.write(str(outputs))

print(f"✅ Run finished! Results saved to: {LOG_FILE}")
