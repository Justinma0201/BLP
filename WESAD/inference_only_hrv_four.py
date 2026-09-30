import os
import json
import gc
import argparse
import pandas as pd
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText


SOURCE_NAMES = {
    "Time": "time",
    "Freq": "freq",
    "MS_RR": "ms_rr",
    "MS_dRR": "ms_drr",
}


def inference_with_transformers(model, processor, input_jsonl, output_csv, batch_size, source, debug_num=None):
    data = []

    if not os.path.exists(input_jsonl): raise FileNotFoundError(f"Can't find input file: {input_jsonl}")

    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip(): data.append(json.loads(line))

    if debug_num is not None:
        print(f"\n⚠️ Enabling Debug mode, only extracting first {debug_num} records for inference testing...\n")
        data = data[:debug_num]

    print(f"Total datas: {len(data)}")
    print(f"Batch size = {batch_size}")

    wrong_groups = [item.get("record_id") for item in data if item.get("feature_group") != source]
    if wrong_groups: raise RuntimeError(f"Input feature_group does not match {source}; example record_id: {wrong_groups[:10]}")

    system_prompt = (
        "You are an expert analyst in physiological stress and autonomic nervous system dynamics. "
        "Based purely on the provided HRV metrics from the current physiological window, "
        "infer the individual's current physiological and psychological state. "
        "Directly provide the final analysis in a single cohesive paragraph. "
        "DO NOT show your reasoning, thinking process, or step-by-step analysis."
    )

    results = []

    for batch_start in range(0, len(data), batch_size):
        batch_data = data[batch_start:batch_start + batch_size]
        batch_messages = []

        for item in batch_data:
            if "llm_prompt" in item:
                user_prompt = item["llm_prompt"]
            elif "messages" in item:
                user_prompt = ""
                for msg in item["messages"]:
                    if msg.get("role") == "user":
                        user_prompt = msg.get("content", "")
                        break
            else:
                user_prompt = ""

            if not user_prompt: raise RuntimeError(f"No valid user prompt found for record_id={item.get('record_id')}")

            messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": system_prompt
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_prompt
                        }
                    ]
                }
            ]

            batch_messages.append(messages)

        inputs = processor.apply_chat_template(batch_messages, add_generation_prompt=True, enable_thinking=False, tokenize=True, padding=True, return_dict=True, return_tensors="pt")
        input_length = inputs["input_ids"].shape[1]

        batch_end = batch_start + len(batch_data)

        print(
            f"[{batch_start + 1}-{batch_end}/{len(data)}] "
            f"Batch={len(batch_data)} | "
            f"Padded input tokens={input_length}"
        )

        inputs = inputs.to(model.device)

        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=1500, do_sample=True, temperature=0.1, top_p=0.8, use_cache=True)

        generated_ids = generated_ids[:, input_length:]
        generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

        for item, generated_text in zip(batch_data, generated_texts):
            clean_text = generated_text.strip()

            if "<think>" in clean_text or "</think>" in clean_text: clean_text = clean_text.split("</think>")[-1].strip()

            if debug_num is not None:
                print(f"\n{'=' * 20} Preview {'=' * 20}")
                print(f"🔹 [ID]: {item.get('record_id')}")
                print(f"[Final Output]:\n{clean_text}\n")

            results.append({
                "record_id": item.get("record_id"),
                "user_id": item.get("user_id"),
                "window_start_sec": item.get("window_start_sec"),
                "window_end_sec": item.get("window_end_sec"),
                "condition": item.get("condition"),
                "protocol_label": item.get("protocol_label"),
                "stress_label": item.get("binarized_stress"),
                "feature_group": source,
                "generated_analysis": clean_text
            })

        del inputs, generated_ids, generated_texts
        gc.collect()
        torch.cuda.empty_cache()

    output_dir = os.path.dirname(output_csv)
    if output_dir: os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame(results)

    if len(df) != len(data): raise RuntimeError(f"Output row count error: Input={len(data)}, Output={len(df)}")

    if "record_id" in df.columns and df["record_id"].duplicated().any():
        duplicated = df.loc[df["record_id"].duplicated(keep=False), "record_id"].tolist()
        raise RuntimeError(f"Found duplicate record_id, e.g.: {duplicated[:10]}")

    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 70)
    print("🎉 WESAD HRV inference completed")
    print(f"Input rows : {len(data)}")
    print(f"Output rows: {len(df)}")
    print(f"Output     : {output_csv}")
    print("=" * 70)

    del data, results, df
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--debug_num", type=int, default=None)
    parser.add_argument("--source", choices=SOURCE_NAMES, required=True)
    parser.add_argument("--input_dir", default=".")
    parser.add_argument("--output_dir", default="./data")
    args = parser.parse_args()

    merged_model_path = "Qwen/Qwen3.5-9B"

    suffix = SOURCE_NAMES[args.source]
    input_jsonl = os.path.join(args.input_dir, f"hrv_full_only_hrv_{suffix}.jsonl")
    output_csv = os.path.join(args.output_dir, f"hrv_generated_only_hrv_{suffix}.csv")

    if not os.path.isfile(input_jsonl): raise FileNotFoundError(f"Can't find input file: {input_jsonl}")
    if torch.cuda.device_count() != 1: raise RuntimeError(f"{args.source} expects each process to see only one GPU, but found {torch.cuda.device_count()}")

    print("=" * 70)
    print("WESAD Qwen3.5 HRV Transformers Inference")
    print("=" * 70)
    print(f"Model      : {merged_model_path}")
    print(f"Input      : {input_jsonl}")
    print(f"Output     : {output_csv}")
    print(f"Source     : {args.source}")
    print(f"Batch size : {args.batch_size}")
    print(f"GPU count  : {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()): print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    print("\nLoading Processor...")

    processor = AutoProcessor.from_pretrained(merged_model_path, trust_remote_code=True)

    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
        if processor.tokenizer.pad_token_id is None: processor.tokenizer.pad_token = processor.tokenizer.eos_token

    print("Loading Transformers model...")

    model = AutoModelForImageTextToText.from_pretrained(merged_model_path, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.eval()

    print(f"Model loaded successfully, current device: {model.device}")

    inference_with_transformers(model=model, processor=processor, input_jsonl=input_jsonl, output_csv=output_csv, batch_size=args.batch_size, source=args.source, debug_num=args.debug_num)

    print("\nWESAD HRV inference completed!")
