import os
import sys

# 1. 阻斷 Keras/TF 干擾 & 環境設定（必須在模型套件匯入前）
os.environ["USE_TORCH"] = "1"
os.environ["USE_TF"] = "0"
os.environ["USE_JAX"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.modules['tensorflow'] = None
sys.modules['keras'] = None

# 2. 匯入模型與其他套件
import gc
import torch
import re
import pandas as pd
from sentence_transformers import SentenceTransformer

# 3. 基礎函數與路徑設定
def clean_text(text):
    if not isinstance(text, str):
        return str(text)
    text = re.sub(r"[’‘`´]", "'", text)
    text = re.sub(r"[‑–—−]", "-", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


BASE_DIR = "/mnt/sdb/justin/Stress-Detection/ICASSP2027/LLM_Rewrite/data"
model_name_or_path = "Qwen/Qwen3-Embedding-0.6B"
tasks = [
    (
        "/mnt/sdb/justin/Stress-Detection/ICASSP2027/LLM_Rewrite/WESAD/Features/hrv_generated_only_hrv_time.csv",
        "/mnt/sdb/justin/Stress-Detection/ICASSP2027/LLM_Rewrite/WESAD/Features/hrv_generated_only_hrv_time_embedding.csv",
    ),
    (
        "/mnt/sdb/justin/Stress-Detection/ICASSP2027/LLM_Rewrite/WESAD/Features/hrv_generated_only_hrv_freq.csv",
        "/mnt/sdb/justin/Stress-Detection/ICASSP2027/LLM_Rewrite/WESAD/Features/hrv_generated_only_hrv_freq_embedding.csv",
    ),
    (
        "/mnt/sdb/justin/Stress-Detection/ICASSP2027/LLM_Rewrite/WESAD/Features/hrv_generated_only_hrv_ms_rr.csv",
        "/mnt/sdb/justin/Stress-Detection/ICASSP2027/LLM_Rewrite/WESAD/Features/hrv_generated_only_hrv_ms_rr_embedding.csv",
    ),
    (
        "/mnt/sdb/justin/Stress-Detection/ICASSP2027/LLM_Rewrite/WESAD/Features/hrv_generated_only_hrv_ms_drr.csv",
        "/mnt/sdb/justin/Stress-Detection/ICASSP2027/LLM_Rewrite/WESAD/Features/hrv_generated_only_hrv_ms_drr_embedding.csv",
    )
]

# 四個檔案共用同一個模型。
print(f"\n🧠 載入模型: {model_name_or_path}")
model = SentenceTransformer(model_name_or_path, trust_remote_code=True, device='cuda')

for input_file, output_file in tasks:
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print(f"\n{'='*70}")
    print(f"🚀 開始處理檔案: {input_file}")
    print(f"{'='*70}")

    # 4. 讀取資料與清理
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"⚠️ 找不到檔案: {input_file}")

    df_all = pd.read_csv(input_file)
    text_col = None
    for col in ['Description', 'generated_analysis']:
        if col in df_all.columns:
            text_col = col
            break

    if text_col is None or 'record_id' not in df_all.columns:
        raise ValueError("⚠️ 檔案缺必要欄位 'record_id' 或文本欄位！")

    print(f"🧹 正在清理文本資料 (使用欄位: {text_col})...")
    df_all['cleaned_text'] = df_all[text_col].apply(clean_text)
    texts = df_all['cleaned_text'].tolist()

    # 5. Embedding 轉換
    print(f"⚡ 開始轉換 Embedding (共 {len(texts)} 筆資料)...")
    embeddings = model.encode(texts, batch_size=256, show_progress_bar=True, convert_to_numpy=True)
    embedding_dim = embeddings.shape[1]

    # 6. 組合並儲存結果
    print("💾 正在組合資料與存檔...")
    emb_cols = [f"emb_{i}" for i in range(embedding_dim)]
    emb_df = pd.DataFrame(embeddings, columns=emb_cols)
    result_df = pd.concat([df_all[['record_id']], emb_df], axis=1)
    result_df.to_csv(output_file, index=False)
    print(f"🎉 轉換完成！已儲存至: {output_file} (總筆數: {len(result_df)}，維度: {embedding_dim})")

    # 每個檔案完成後釋放資料，模型保留給下一個檔案。
    del embeddings, emb_df, result_df, df_all, texts
    gc.collect()
    torch.cuda.empty_cache()

# 全部完成後釋放模型。
del model
gc.collect()
torch.cuda.empty_cache()
print("\n✅ 四個 Qwen embeddings 皆已完成！")
