import os
import sys

os.environ["USE_TORCH"] = "1"
os.environ["USE_TF"] = "0"
os.environ["USE_JAX"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.modules['tensorflow'] = None
sys.modules['keras'] = None

import gc
import torch
import re
import pandas as pd
from sentence_transformers import SentenceTransformer

def clean_text(text):
    if not isinstance(text, str):
        return str(text)
    text = re.sub(r"[’‘`´]", "'", text)
    text = re.sub(r"[‑–—−]", "-", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


BASE_DIR = "./data"
model_name_or_path = "Qwen/Qwen3-Embedding-0.6B"
tasks = [
    (
        "./data/hrv_generated_only_hrv_time.csv",
        "./data/hrv_generated_only_hrv_time_embedding.csv",
    ),
    (
        "./data/hrv_generated_only_hrv_freq.csv",
        "./data/hrv_generated_only_hrv_freq_embedding.csv",
    ),
    (
        "./data/hrv_generated_only_hrv_ms_rr.csv",
        "./data/hrv_generated_only_hrv_ms_rr_embedding.csv",
    ),
    (
        "./data/hrv_generated_only_hrv_ms_drr.csv",
        "./data/hrv_generated_only_hrv_ms_drr_embedding.csv",
    )
]

print(f"\nLoading: {model_name_or_path}")
model = SentenceTransformer(model_name_or_path, trust_remote_code=True, device='cuda')

for input_file, output_file in tasks:
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print(f"\n{'='*70}")
    print(f"Processing: {input_file}")
    print(f"{'='*70}")

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"File not found: {input_file}")

    df_all = pd.read_csv(input_file)
    text_col = None
    for col in ['Description', 'generated_analysis']:
        if col in df_all.columns:
            text_col = col
            break

    if text_col is None or 'record_id' not in df_all.columns:
        raise ValueError("File is missing required columns 'record_id' or text column!")

    print(f"Cleaning text data (using column: {text_col})...")
    df_all['cleaned_text'] = df_all[text_col].apply(clean_text)
    texts = df_all['cleaned_text'].tolist()

    print(f"Computing Embeddings (total {len(texts)} samples)...")
    embeddings = model.encode(texts, batch_size=256, show_progress_bar=True, convert_to_numpy=True)
    embedding_dim = embeddings.shape[1]

    print("Combining data and saving...")
    emb_cols = [f"emb_{i}" for i in range(embedding_dim)]
    emb_df = pd.DataFrame(embeddings, columns=emb_cols)
    result_df = pd.concat([df_all[['record_id']], emb_df], axis=1)
    result_df.to_csv(output_file, index=False)
    print(f"Conversion complete! Data saved to: {output_file} (Total samples: {len(result_df)}, Dimensionality: {embedding_dim})")

    del embeddings, emb_df, result_df, df_all, texts
    gc.collect()
    torch.cuda.empty_cache()

del model
gc.collect()
torch.cuda.empty_cache()
print("\nFinished")
