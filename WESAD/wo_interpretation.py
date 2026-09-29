import os
import sys
os.environ["USE_TORCH"] = "1"
os.environ["USE_TF"] = "0"
os.environ["USE_JAX"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.modules["tensorflow"] = None
sys.modules["keras"] = None
import json
import gc
import csv
import argparse

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

DATA_DIR = "./data"
SOURCE_NAMES = {"Time": "time", "Freq": "freq", "MS_RR": "ms_rr", "MS_dRR": "ms_drr"}
TASKS = [
    (
        source,
        os.path.join(DATA_DIR, f"hrv_full_raw_{suffix}.jsonl"),
        os.path.join(DATA_DIR, f"hrv_full_raw_{suffix}_qwen3_0.6b_direct_embedding.csv"),
    )
    for source, suffix in SOURCE_NAMES.items()
]

METADATA_FIELDS = (
    "record_id", "user_id", "window_start_sec", "window_end_sec",
    "condition", "protocol_label", "binarized_stress", "feature_group",
)


def load_jsonl(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def extract_user_prompt(item):
    if "messages" in item:
        for msg in item["messages"]:
            if msg.get("role") == "user":
                content = msg.get("content")
                if isinstance(content, str) and content.strip():
                    return content
    raise ValueError(f"Missing user prompt for record_id={item.get('record_id')}")


def process_jsonl_to_embeddings(model, source, input_jsonl, output_csv, batch_size, debug_num=None):
    print(f"\n{'='*50}\nReading JSONL: {input_jsonl}")
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    count = 0
    pending = []
    writer = None

    with open(output_csv, "w", encoding="utf-8-sig", newline="") as target:
        def write_batch(items):
            nonlocal writer
            embeddings = model.encode(
                [extract_user_prompt(item) for item in items],
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            if len(embeddings) != len(items):
                raise RuntimeError(f"Embedding count mismatch: input={len(items)}, output={len(embeddings)}")
            if writer is None:
                emb_cols = [f"emb_{i}" for i in range(len(embeddings[0]))]
                writer = csv.writer(target)
                writer.writerow([*METADATA_FIELDS, "stress_label", *emb_cols])
            for item, embedding in zip(items, embeddings):
                if item.get("feature_group") != source:
                    raise ValueError(f"feature_group mismatch: expected {source}, record_id={item.get('record_id')}")
                writer.writerow([*(item.get(field) for field in METADATA_FIELDS),
                                 item.get("binarized_stress"), *embedding])

        for item in load_jsonl(input_jsonl):
            if item.get("feature_group") != source:
                raise ValueError(f"feature_group mismatch: expected {source}, record_id={item.get('record_id')}")
            pending.append(item)
            count += 1
            if len(pending) == batch_size:
                write_batch(pending)
                pending.clear()
                print(f"{source}: {count} records embedded", flush=True)
            if debug_num is not None and count >= debug_num:
                break
        if pending:
            write_batch(pending)
    print(f"{source}: {count} records saved to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=SOURCE_NAMES, help="Process one WESAD domain; default processes all four sequentially")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--debug_num", type=int)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--input_dir", default=DATA_DIR)
    parser.add_argument("--output_dir", default=DATA_DIR)
    args = parser.parse_args()
    if args.batch_size < 1 or (args.debug_num is not None and args.debug_num < 1):
        parser.error("--batch_size and --debug_num must be positive")

    from sentence_transformers import SentenceTransformer

    print(f"Loading SentenceTransformer Embedding model: {args.model}")
    model = SentenceTransformer(args.model, trust_remote_code=True, device="cuda")
    for source, _, _ in TASKS:
        if args.source is None or args.source == source:
            suffix = SOURCE_NAMES[source]
            input_jsonl = os.path.join(args.input_dir, f"hrv_full_raw_{suffix}.jsonl")
            output_csv = os.path.join(args.output_dir, f"hrv_full_{suffix}_qwen3_0.6b_direct_embedding.csv")
            process_jsonl_to_embeddings(model, source, input_jsonl, output_csv,
                                        batch_size=args.batch_size, debug_num=args.debug_num)

    del model
    gc.collect()
    print("\nAll files have been processed!")
