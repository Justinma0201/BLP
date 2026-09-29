"""Extract WESAD prompt-last-token embeddings from hrv_jsonlcopy.py JSONL.

Chat formatting follows the reference inference scripts (enable_thinking=False).
Outputs L2-normalized final-layer hidden states.
"""
import argparse
import gc
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "Qwen/Qwen3.5-9B"

TASKS = [
    (
        f"./data/hrv_full_raw_{suffix}.jsonl",
        f"./data/hrv_full_raw_{suffix}_qwen_prompt_last_embedding_wesad.csv",
    )
    for suffix in ("time", "freq", "ms_rr", "ms_drr")
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default=MODEL_PATH)
    parser.add_argument("--input_jsonl", help="Override the default WESAD JSONL")
    parser.add_argument("--output_csv", help="Output for --input_jsonl")
    parser.add_argument("--output_dir", help="Override the default output directory")
    parser.add_argument("--batch_size", type=int, default=8)
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
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content
            break
    raise ValueError(f"Missing nonempty user prompt: record_id={item.get('record_id')}")


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
    # Equivalent to generate().hidden_states[0][-1][:, -1, :].
    # With left padding, -1 is the final token of the chat-formatted prompt,
    # including the assistant generation prefix (not the raw user text).
    output = model(**batch, output_hidden_states=True, use_cache=False,
                   return_dict=True)
    last_hidden = output.hidden_states[-1][:, -1, :].float()
    embeddings = F.normalize(last_hidden, p=2, dim=-1).cpu().numpy()
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
                "window_start_sec": item.get("window_start_sec"),
                "window_end_sec": item.get("window_end_sec"),
                "condition": item.get("condition"),
                "protocol_label": item.get("protocol_label"),
                "binarized_stress": item.get("binarized_stress"),
                "stress_label": item.get("binarized_stress"),
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
    tasks = [(args.input_jsonl, args.output_csv)] if args.input_jsonl else list(TASKS)
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
