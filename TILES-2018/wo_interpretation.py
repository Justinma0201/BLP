import os
import sys
import json
os.environ["USE_TORCH"] = "1"
os.environ["USE_TF"] = "0"
os.environ["USE_JAX"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

sys.modules['tensorflow'] = None
sys.modules['keras'] = None

import gc
import re
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
import pdb

HRV_DATASET_PATH = './data/df_unique_time_label_total793.csv'
DESC_PATH = './data/handmade_description.csv'
MAPPING_PATH = './data/feature_mapping_793.csv'

OUTPUT_CSVS = {
    'Time':   './data/hrv_time_qwen-direct_embedding.csv',
    'Freq':   './data/hrv_freq_qwen-direct_embedding.csv',
    'MS_RR':  './data/hrv_ms_rr_qwen-direct_embedding.csv',
    'MS_dRR': './data/hrv_ms_drr_qwen-direct_embedding.csv',
}

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
#MODEL_NAME = "sentence-transformers/all-roberta-large-v1"

def classify_feature(name):
    n = name.lower()
    if name.startswith('stdNN') or any(n.startswith(p) for p in [
        'meannn', 'cvnn', 'pnn50', 'mean_diff1', 'std_abs_diff1', 'rmssd', 'nmae1_diff1'
    ]):
        return 'Time'
    elif any(n.startswith(p) for p in ['hf', 'lf', 'vlf']):
        return 'Freq'
    elif n.startswith('pe_dw'):
        return 'MS_RR'
    elif n.startswith('dpe_dw'):
        return 'MS_dRR'
    elif any(n.startswith(p) for p in ['dmspe', 'dmsmpe', 'dds1', 'dds2', 'diff_dds', 'sum_dds']):
        return 'MS_dRR'
    elif any(n.startswith(p) for p in ['mspe', 'msmpe', 'ds1', 'ds2', 'diff_ds', 'sum_ds']):
        return 'MS_RR'
    elif 'asym' in n:
        return 'MS_dRR'
    return None

def clean_text(text):
    if not isinstance(text, str):
        return str(text)
    text = re.sub(r"[’‘`´]", "'", text)
    text = re.sub(r"[‑–—−]", "-", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def process_and_embed():
    print(f"\n{'='*70}\nReading Data and Feature Mappings\n{'='*70}")
    
    df_hrv = pd.read_csv(HRV_DATASET_PATH)
    df_hrv.columns = df_hrv.columns.astype(str)

    desc_df = pd.read_csv(DESC_PATH)
    desc_df = desc_df.rename(columns={'participant_id': 'ID'})

    df_merged = pd.merge(df_hrv, desc_df[['ID', 'Description']], on='ID', how='left')
    print(f"Finished merging data, total {len(df_merged)} records (Many-to-One)")

    mapping_df = pd.read_csv(MAPPING_PATH)
    mapping_df['Category'] = mapping_df['Feature_Name'].apply(classify_feature)

    group_mappings = {}
    for cat in ['Time', 'Freq', 'MS_RR', 'MS_dRR']:
        sub = mapping_df[mapping_df['Category'] == cat]
        group_mappings[cat] = dict(zip(sub['Index'].astype(str), sub['Feature_Name']))
        print(f"Feature {cat}: Corresponding to {len(sub)} features")

    print(f"\n🧠 Loading Embedding Model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True, device='cuda')

    df_cols = set(df_merged.columns)

    for cat, out_path in OUTPUT_CSVS.items():
        print(f"\n{'='*50}\nProcessing Domain: {cat}\n{'='*50}")

        if os.path.exists(out_path):
            print(f"Found existing file, skipping {cat} domain")
            continue

        cat_mapping = group_mappings[cat]
        valid_cols = [col for col in cat_mapping.keys() if col in df_cols]

        if not valid_cols:
            print(f"{cat}: No corresponding features found in the dataset, skipping.")
            continue

        texts = []
        meta_data = []

        for _, row in df_merged.iterrows():
            record_id = f"{row['ID']}_{row['date']}" if 'date' in row else f"{row['ID']}"
 
            user_baseline = str(row['Description']) if pd.notna(row.get('Description')) else "No baseline description available."
            pid_str = str(row['ID'])
            user_baseline = user_baseline.replace(f"{pid_str} is a", "This individual is a")
            user_baseline = user_baseline.replace(pid_str, "This individual")
            user_baseline = clean_text(user_baseline)

            hrv_str_list = []
            for col_num in valid_cols:
                if pd.notna(row[col_num]):
                    val = row[col_num]
                    real_name = cat_mapping[col_num]
                    formatted_val = f"{round(val, 4)}" if isinstance(val, (int, float)) else str(val)
                    hrv_str_list.append(f"{real_name}: {formatted_val}")
            
            hrv_text = ", ".join(hrv_str_list)
            combined_text = f"HRV {cat} Metrics: {hrv_text} | Baseline Description: {user_baseline}"
            texts.append(combined_text)
            
            meta_data.append({
                "record_id": record_id,
                "user_id": row['ID'],
                "date": row.get('date'),
                "label_binarized": row.get('label_binarized')
            })

        print(f"Preview of Combined Results (1st Entry):\n{texts[0]}\n")

        print(f"Starting Embedding Conversion for {cat} (Total: {len(texts)} Entries)...")
 
        embeddings = model.encode(texts, batch_size=8, show_progress_bar=True, convert_to_numpy=True)
        embedding_dim = embeddings.shape[1]

        meta_df = pd.DataFrame(meta_data)
        emb_cols = [f"emb_{i}" for i in range(embedding_dim)]
        emb_df = pd.DataFrame(embeddings, columns=emb_cols)
        
        result_df = pd.concat([meta_df, emb_df], axis=1)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        result_df.to_csv(out_path, index=False)
        
        print(f"{cat} Conversion Complete! Saved to: {out_path}")

        del texts, meta_data, embeddings, meta_df, emb_df, result_df
        gc.collect()
        torch.cuda.empty_cache()

    del model, df_merged, df_hrv, desc_df
    gc.collect()
    torch.cuda.empty_cache()
    print("\nFinished")

if __name__ == "__main__":
    process_and_embed()