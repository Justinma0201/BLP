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
        print(f"File not found: {input_jsonl}")
        return

    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip(): data.append(json.loads(line))

    if debug_num is not None:
        print(f"\nDebug mode: Only using the first {debug_num} records\n")
        data = data[:debug_num]

    print(f"Loading complete, total {len(data)} records")

    system_prompt = (
        "You are an expert analyst in occupational health, psychological well-being, and stress-related behavioral factors. "
        "Based only on the provided baseline profile, infer the individual's physiological and psychological state. "
        "Provide a single cohesive analytical paragraph. "
        "Do not directly classify or conclude that the individual is stressed or not stressed, and do not assign any stress label, category, probability, or score. "
        "Do not introduce personal facts that are not explicitly provided. "
        "Directly provide the final analysis without showing reasoning or step-by-step analysis."
    )

    formatted_prompts = []

    for item in data:
        user_prompt = ""

        if "llm_prompt" in item:
            user_prompt = item["llm_prompt"]

        elif "messages" in item:
            for msg in item["messages"]:
                if msg.get("role") == "user":
                    user_prompt = msg.get("content", "")
                    break

        if not user_prompt: raise RuntimeError(f"Cannot find user prompt，record_id={item.get('record_id')}")

        formatted = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        formatted_prompts.append(formatted)

    stop_tokens = ["<|im_end|>", "<|endoftext|>"]

    sampling_params = SamplingParams(
        temperature=0.1,
        max_tokens=1500,
        top_p=0.8,
        presence_penalty=1.1,
        stop=stop_tokens
    )

    print(f"Starting vLLM inference (total {len(formatted_prompts)} records)...")

    outputs = llm.generate(formatted_prompts, sampling_params)

    results = []

    for i, (item, output) in enumerate(zip(data, outputs)):
        generated_text = output.outputs[0].text
        print(f"\n[DEBUG RAW OUTPUT] record_id={item.get('record_id')}")
        print(f"finish_reason={output.outputs[0].finish_reason}")
        print(f"stop_reason={getattr(output.outputs[0], 'stop_reason', None)}")
        print(f"raw_text={repr(generated_text)}")

        clean_text = generated_text.strip()

        if "</think>" in clean_text:
            clean_text = clean_text.rsplit("</think>", 1)[-1].strip()
        if "<think>" in clean_text or not clean_text:
            raise RuntimeError(
                f"Model did not generate valid text, record_id={item.get('record_id')}，"
                f"finish_reason={output.outputs[0].finish_reason}，"
                f"stop_reason={getattr(output.outputs[0], 'stop_reason', None)}。"
                "Please check if the model's chat template supports enable_thinking=False；"
                "If finish_reason=length, please check the generated token budget."
            )

        if debug_num is not None:
            print(f"\n{'=' * 20} Preview {i + 1} {'=' * 20}")
            print(f"[record_id]: {item.get('record_id')}")
            print(f"[user_id]: {item.get('user_id')}")
            print(f"\n[generated_description]\n{clean_text}\n")

        results.append({
            "record_id": item.get("record_id"),
            "user_id": item.get("user_id"),
            "generated_description": clean_text
        })

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"\nCompletion, total {len(df)} records:")
    print(output_csv)

    del data, formatted_prompts, outputs, results, df
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    merged_model_path = "Qwen/Qwen3.5-9B"

    INPUT_JSONL = "./data/baseline_rewrite.jsonl"

    OUTPUT_CSV = "./data/baseline_rewrite.csv"

    print("=" * 100)
    print("BASELINE DESCRIPTION REWRITE")
    print("=" * 100)
    print(f"Model : {merged_model_path}")
    print(f"Input : {INPUT_JSONL}")
    print(f"Output: {OUTPUT_CSV}")

    print("\nLoading Tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        merged_model_path,
        trust_remote_code=True
    )

    print("Loading vLLM engine...")

    llm = LLM(
        model=merged_model_path,
        gpu_memory_utilization=0.9,
        max_model_len=4096,
        tensor_parallel_size=1,
        trust_remote_code=True,
        dtype="bfloat16",
        max_num_seqs=128
    )

    inference_with_vllm(
        llm=llm,
        tokenizer=tokenizer,
        input_jsonl=INPUT_JSONL,
        output_csv=OUTPUT_CSV,
        debug_num=None
    )

    print("\nFinished")
