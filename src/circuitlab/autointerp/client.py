import os
import torch.multiprocessing as mp
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from pathlib import Path

# Set environment variables before any other imports
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

mp.set_start_method("spawn", force=True)

def run_client(prompts: list[Path], out_dir: Path, vllm_model: str, vllm_max_tokens: int):
    torch.cuda.empty_cache()
    
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(vllm_model)
    
    messages = [
        tokenizer.apply_chat_template([
                {"role": "user", "content": f"You are an expert at summarizing neuron behaviors.\n\n{prompt.read_text()}"}
            ], 
            tokenize=False, 
            add_generation_prompt=True
        )
        for prompt in prompts
    ]

    # Configuration for 70B model - requires multiple GPUs
    gpu_count = torch.cuda.device_count()
    print(f"Available GPUs: {gpu_count}")

    # Optimized configuration
    llm_config = {
        "model": vllm_model,
        "hf_token": "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",
        "gpu_memory_utilization": 0.90,  # Can go higher if not sharing GPU
        "max_model_len": 3500,
        "disable_custom_all_reduce": True,
        "max_num_seqs": 8, 
        "use_v2_block_manager": False
    }
    
    # Only use tensor parallelism if you have multiple GPUs
    if gpu_count > 1:
        llm_config["tensor_parallel_size"] = gpu_count
    else:
        # Single GPU optimizations
        llm_config["tensor_parallel_size"] = 1
        # Remove enforce_eager for better performance on single GPU
        llm_config["enforce_eager"] = False
        # Enable some optimizations
        llm_config["enable_prefix_caching"] = True
        
    llm = LLM(**llm_config)

    sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=vllm_max_tokens)

    # Debug: Print first prompt to check formatting
    if messages:
        print(f"\n📝 First prompt being sent to model:\n{messages[0][:500]}...\n")

    results = llm.generate(messages, sampling_params)

    print("👉 Raw results object:", results)

    for path, result in zip(prompts, results):
        output = result.outputs[0].text.strip()

        # Strip <|assistant|> if present
        if output.startswith("<|assistant|>"):
            output = output[len("<|assistant|>"):].lstrip()
            
        # Strip "LLM ANSWER:" prefix if present
        if output.startswith("LLM ANSWER:"):
            output = output[len("LLM ANSWER:"):].lstrip()
            
        (out_dir / path.name).write_text(output)
        print(f"\n{'='*60}")
        print(f"🔍 EXPLANATION FOR {path.name}:")
        print(f"{'='*60}")
        print(output)
        print(f"{'='*60}\n")
