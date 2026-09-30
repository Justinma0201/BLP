import pandas as pd
import json
import os

# ==========================================
# 1. 檔案路徑設定
# ==========================================
HRV_DATASET_PATH = './data/df_unique_time_label_total793.csv'
DESC_PATH = './data/handmade_description.csv'
MAPPING_PATH = './data/feature_mapping_793.csv'

OUTPUT_JSONL = {
    'Time':   './data/hrv_time_full.jsonl',
    'Freq':   './data/hrv_freq_full.jsonl',
    'MS_RR':  './data/hrv_ms_rr_full.jsonl',
    'MS_dRR': './data/hrv_ms_drr_full.jsonl',
}

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

FEATURE_GROUP_DESC = {
    'Time': "Time-domain HRV features (e.g., meanNN, sdNN, RMSSD, pNN50, cvNN)",
    'Freq': "Frequency-domain HRV features (e.g., HF, LF, VLF, LF/HF ratio, normalized HF/LF)",
    'MS_RR': "Multi-scale RR interval features (e.g., MSPE, MSmPE, PE_dw, motif distribution distances)",
    'MS_dRR': "Multi-scale differential RR features and asymmetry index (e.g., dMSPE, dMSmPE, dPE_dw, total asymmetry index)",
}

def export_grouped_jsonl(df, mapping_path, output_paths):
    print(f"Reading Feature Mappings: {mapping_path}")
    mapping_df = pd.read_csv(mapping_path)
    mapping_df['Category'] = mapping_df['Feature_Name'].apply(classify_feature)

    group_mappings = {}
    for cat in ['Time', 'Freq', 'MS_RR', 'MS_dRR']:
        sub = mapping_df[mapping_df['Category'] == cat]
        group_mappings[cat] = dict(zip(sub['Index'].astype(str), sub['Feature_Name']))
        print(f"  {cat}: {len(sub)} features mapped")

    df_cols = set(df.columns)

    for cat, out_path in output_paths.items():
        cat_mapping = group_mappings[cat]
        valid_cols = [col for col in cat_mapping.keys() if col in df_cols]

        if not valid_cols:
            print(f"{cat}: No corresponding columns found, skipping.")
            continue

        output_data = []
        for _, row in df.iterrows():
            record_id = f"{row['ID']}_{row['date']}" if 'date' in row else f"{row['ID']}"

            user_baseline = str(row['Description']) if pd.notna(row.get('Description')) else "No baseline description available."
            pid_str = str(row['ID'])
            user_baseline = user_baseline.replace(f"{pid_str} is a", "This individual is a")
            user_baseline = user_baseline.replace(pid_str, "This individual")

            hrv_lines = []
            for col_num in valid_cols:
                if pd.notna(row[col_num]):
                    val = row[col_num]
                    real_name = cat_mapping[col_num]
                    formatted_val = f"{round(val, 4)}" if isinstance(val, (int, float)) else str(val)
                    hrv_lines.append(f"- {real_name}: {formatted_val}")

            hrv_text = "\n".join(hrv_lines)

            user_content = f"""[Neutral Baseline Profile]
{user_baseline}

[Data Interpretation Context]
Note: The following HRV metrics are not instantaneous readings. They are strictly calculated based on rigorous daily aggregation:
- Feature Group: {FEATURE_GROUP_DESC[cat]}
- Base calculation: Extracted from 5-minute rolling windows of Normal-to-Normal (NN) intervals. Time-domain differences and Frequency domains were normalized.
- Suffixes meaning: Metric ending in '_mean' is the daily average; '_std' is the daily standard deviation; '_cv' (Coefficient of Variation) shows daily volatility; '_skew'/'_kurt' show distribution shape; '_1quartile'/'_3quartile' are the 25th/75th percentiles.
Please interpret these metrics as the overall autonomic stability and allostatic load for the entire day.

[Current Physiological Markers]
{hrv_text}

[Task]
Based on the individual's baseline profile and today's HRV metrics, write a daily analytical log assessing their current stress state.

[Output Constraints]
- Combine your final analysis into a SINGLE, cohesive paragraph.
- Focus strictly on physiological and psychological interpretation.
- DO NOT directly state, conclude, or predict whether the individual is "stressed" or "not stressed".
- DO NOT list, quote, or repeat the raw HRV numbers provided above. Use qualitative descriptions (e.g., "highly volatile," "elevated parasympathetic activity") instead.
- If there's 'Unknown' in the description, it means the data is missing and you can ignore it."""

            output_data.append({
                "record_id": record_id,
                "user_id": row['ID'],
                "date": row.get('date'),
                "binarized_stress": row.get('label_binarized'),
                "feature_group": cat,
                "messages": [
                    {"role": "user", "content": user_content}
                ]
            })

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            for item in output_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"{cat}: Output {len(output_data)} entries to {out_path}")

if __name__ == "__main__":
    print("Reading data...")
    df_hrv = pd.read_csv(HRV_DATASET_PATH)
    df_hrv.columns = df_hrv.columns.astype(str)

    desc_df = pd.read_csv(DESC_PATH)
    desc_df = desc_df.rename(columns={'participant_id': 'ID'})

    df_merged = pd.merge(df_hrv, desc_df[['ID', 'Description']], on='ID', how='left')
    print(f"Merge completed, total {len(df_merged)} records")

    export_grouped_jsonl(df_merged, MAPPING_PATH, OUTPUT_JSONL)
    print("\nAll done")