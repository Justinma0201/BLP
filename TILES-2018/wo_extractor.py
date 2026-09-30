"""Extract Firefighter embeddings without BiPO steering or fold filtering.

Uses the same JSONL records/user messages and CSV metadata as llm_direct_embedding.
Chat formatting follows the reference inference scripts (enable_thinking=False).
"""
import argparse
import gc
import json
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "Qwen/Qwen3.5-9B"

TASKS = [
    (
        "./data/hrv_time_full.jsonl",
        "./data/hrv_time_qwen_prompt_last_embedding.csv",
    ),
    (
        "./data/hrv_freq_full.jsonl",
        "./data/hrv_freq_qwen_prompt_last_embedding.csv",
    ),
    (
        "./data/hrv_ms_rr_full.jsonl",
        "./data/hrv_ms_rr_qwen_prompt_last_embedding.csv",
    ),
    (
        "./data/hrv_ms_drr_full.jsonl",
        "./data/hrv_ms_drr_qwen_prompt_last_embedding.csv",
    ),
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default=MODEL_PATH)
    parser.add_argument("--input_jsonl", help="Process one JSONL instead of the four default tasks")
    parser.add_argument("--output_csv", help="Output for --input_jsonl")
    parser.add_argument("--output_dir", help="Override the default output directory")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_prompt_length", type=int, default=8192)
    parser.add_argument("--max_new_tokens", type=int, default=1500)
    parser.add_argument("--debug_num", type=int)
    args = parser.parse_args()
    if bool(args.input_jsonl) != bool(args.output_csv):
        parser.error("--input_jsonl and --output_csv must be supplied together")
    if min(args.batch_size, args.max_prompt_length, args.max_new_tokens) <= 0:
        parser.error("Batch size and token limits must be positive")
    if args.debug_num is not None and args.debug_num <= 0:
        parser.error("--debug_num must be positive")
    return args


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def extract_user_prompt(item):
    for message in item.get("messages", []):
        if message.get("role") == "user":
            return message["content"]
    return ""


def tokenize_batch(tokenizer, prompts, max_prompt_length, device):
    texts = [tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    ) for prompt in prompts]
    batch = tokenizer(texts, add_special_tokens=False, padding=True,
                      truncation=True, max_length=max_prompt_length,
                      return_tensors="pt")
    return {key: value.to(device) for key, value in batch.items()}


@torch.no_grad()
def extract_embeddings(model, tokenizer, batch, args, record_ids):
    output = model(**batch, output_hidden_states=True, use_cache=False,
                   return_dict=True)
    embeddings = output.hidden_states[-1][:, -1, :].float().cpu().numpy()
    return embeddings, None



def process_jsonl_to_embeddings(model, tokenizer, input_jsonl, output_csv, args):
    data = load_jsonl(input_jsonl)
    if args.debug_num is not None:
        data = data[:args.debug_num]
    if not data:
        print(f"No records: {input_jsonl}")
        return
    rows = []
    for start in range(0, len(data), args.batch_size):
        items = data[start:start + args.batch_size]
        batch = tokenize_batch(tokenizer, [extract_user_prompt(item) for item in items],
                               args.max_prompt_length, model.device)
        embeddings, analyses = extract_embeddings(model, tokenizer, batch, args,
                                                  [item.get("record_id") for item in items])
        for i, item in enumerate(items):
            row = {
                "record_id": item.get("record_id"),
                "user_id": item.get("user_id"),
                "date": item.get("date"),
                "stress_label": item.get("binarized_stress"),
                "feature_group": item.get("feature_group"),
            }
            if analyses is not None:
                row["generated_analysis"] = analyses[i]
            row.update({f"emb_{k}": float(value) for k, value in enumerate(embeddings[i])})
            rows.append(row)
        del batch, embeddings
        print(f"{input_jsonl}: {min(start + args.batch_size, len(data))}/{len(data)}")
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"Saved: {output_csv}")


def main():
    args = parse_args()
    tasks = [(args.input_jsonl, args.output_csv)] if args.input_jsonl else [
        (src, dst.replace("_direct_embedding.csv", "_prompt_last_embedding.csv")) for src, dst in TASKS
    ]
    if args.output_dir:
        tasks = [(src, str(Path(args.output_dir) / Path(dst).name)) for src, dst in tasks]
    for src, _ in tasks:
        if not Path(src).is_file():
            raise FileNotFoundError(src)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    ).to("cuda")
    model.eval()
    try:
        for src, dst in tasks:
            process_jsonl_to_embeddings(model, tokenizer, src, dst, args)
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
