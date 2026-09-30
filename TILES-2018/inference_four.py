import json
import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import os
import gc
import torch

def inference_with_vllm(llm, tokenizer, input_jsonl, output_csv, debug_num=None):
    data = []
    if not os.path.exists(input_jsonl):
        print(f"Can't find input file: {input_jsonl}")
        return
        
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
            
    if debug_num is not None:
        print(f"\nDebug mode, only extracting the first {debug_num} records for inference testing...\n")
        data = data[:debug_num]

    formatted_prompts = []
    
    system_prompt = (
        "You are an expert analyst in occupational health and organizational behavior. "
        "Based purely on the provided neutral baseline profile and today's HRV metrics, "
        "infer the individual's current physiological and psychological state. "
        "Directly provide the final analysis in a single cohesive paragraph. "
        "DO NOT show your reasoning, thinking process, or step-by-step analysis."
    )

    for item in data:
        user_prompt = ""
        if 'llm_prompt' in item:
            user_prompt = item['llm_prompt']
        elif 'messages' in item:
            for msg in item['messages']:
                if msg['role'] == 'user':
                    user_prompt = msg['content']
                    break
        
        formatted = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\nThis individual's"
        )
        formatted_prompts.append(formatted)
 
    stop_tokens = ["<|im_end|>", "<|endoftext|>", "Analysis:", "Thinking Process:"]
    sampling_params = SamplingParams(
        temperature=0.1,      
        max_tokens=1500,     
        top_p=0.8,
        presence_penalty=1.1, 
        stop=stop_tokens
    )
    
    print(f"Starting vLLM inference generation (total {len(formatted_prompts)} records)...")
    outputs = llm.generate(formatted_prompts, sampling_params)
 
    results = []
    for i, (item, output) in enumerate(zip(data, outputs)):
        generated_text = output.outputs[0].text.strip()
        full_analysis = "This individual's " + generated_text
        clean_text = full_analysis
        
        if "<think>" in clean_text or "</think>" in clean_text:
            clean_text = clean_text.split("</think>")[-1].strip()
        
        if "1. " in clean_text and "2. " in clean_text:
            segments = clean_text.split("\n")
            clean_text = segments[-1] if len(segments) > 1 else clean_text

        if debug_num is not None:
            print(f"\n{'='*20} Preview {i+1} {'='*20}")
            print(f"🔹 [ID]: {item.get('record_id')}")
            print(f"[Final Output]:\n{clean_text}\n")

        results.append({
            "record_id": item.get("record_id"),
            "user_id": item.get("user_id"),
            "date": item.get("date"),
            "stress_label": item.get("binarized_stress"),
            "generated_analysis": clean_text
        })
  
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\nCompletion processing and saving to: {output_csv}\n")
    
    del data, formatted_prompts, outputs, results, df
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    merged_model_path = "Qwen/Qwen3.5-9B"

    tasks = [
       (
           './data/hrv_time_full.jsonl',
           './data/hrv_time_description_full.csv'
       ),
       (
           './data/hrv_freq_full.jsonl',
           './data/hrv_freq_description_full.csv'
       ),
        (
            './data/hrv_ms_rr_full.jsonl',
            './data/hrv_ms_rr_description_full.csv'
        ),
        (
            './data/hrv_ms_drr_full.jsonl',
            './data/hrv_ms_drr_description_full.csv'
        )
    ]
    
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(merged_model_path, trust_remote_code=True)
    
    print("Loading vLLM engine (global load only)...")
    llm = LLM(
        model=merged_model_path, 
        gpu_memory_utilization=0.9,
        max_model_len=8192,
        tensor_parallel_size=1,
        trust_remote_code=True,
        dtype="bfloat16",
        max_num_seqs=1
    )

    for input_jsonl, output_csv in tasks:
        print(f"\n{'='*50}")
        print(f"Starting task: {os.path.basename(input_jsonl)}")
        print(f"{'='*50}")
        
        inference_with_vllm(
            llm=llm, 
            tokenizer=tokenizer, 
            input_jsonl=input_jsonl, 
            output_csv=output_csv,
            debug_num=None
        )
        
    print("\nCompletion processing for all 4 files is done")