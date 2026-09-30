#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import pandas as pd
HRV_DATASET_PATH = "./data/df_unique_time_label_total793.csv"
MAPPING_PATH = "./data/feature_mapping_793.csv"
OUTPUT_JSONL = {
    "Time": "./data/hrv_full_only_hrv_time.jsonl",
    "Freq": "./data/hrv_full_only_hrv_freq.jsonl",
    "MS_RR": "./data/hrv_full_only_hrv_ms_rr.jsonl",
    "MS_dRR": "./data/hrv_full_only_hrv_ms_drr.jsonl",
}
FEATURE_GROUP_DESC = {
    "Time": "Time-domain HRV features (e.g., meanNN, sdNN, RMSSD, pNN50, cvNN)",
    "Freq": "Frequency-domain HRV features (e.g., HF, LF, VLF, LF/HF ratio, normalized HF/LF)",
    "MS_RR": "Multi-scale RR interval features (e.g., MSPE, MSmPE, PE_dw, motif distribution distances)",
    "MS_dRR": "Multi-scale differential RR features and asymmetry index (e.g., dMSPE, dMSmPE, dPE_dw, total asymmetry index)",
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
def load_feature_mapping(mapping_path):
    print(f"Reading HRV feature mapping: {mapping_path}")
    mapping_df = pd.read_csv(mapping_path)
    required_cols = ["Index", "Feature_Name"]
    for col in required_cols:
        if col not in mapping_df.columns: raise ValueError(f"Mapping file missing '{col}'，current columns: {mapping_df.columns.tolist()}")
    mapping_df = mapping_df.rename(columns={"Index": "feature_column", "Feature_Name": "feature_name"})
    mapping_df["feature_column"] = mapping_df["feature_column"].astype(str)
    mapping_df["feature_name"] = mapping_df["feature_name"].astype(str)
    mapping_df["category"] = mapping_df["feature_name"].apply(classify_feature)
    unclassified = mapping_df.loc[mapping_df["category"].isna(), "feature_name"].tolist()
    if unclassified: raise RuntimeError(f"Unclassified HRV features: {unclassified}")
    feature_mapping = dict(zip(mapping_df["feature_column"], mapping_df["feature_name"]))
    group_mappings = {
        cat: dict(zip(sub["feature_column"], sub["feature_name"]))
        for cat in OUTPUT_JSONL
        for sub in [mapping_df[mapping_df["category"] == cat]]
    }
    empty_groups = [cat for cat, mapping in group_mappings.items() if not mapping]
    if empty_groups: raise RuntimeError(f"No mapped features for domains: {empty_groups}")
    print(f"Mapping 共 {len(feature_mapping)} 個 features")
    return feature_mapping, group_mappings
def build_hrv_text(row, feature_mapping):
    hrv_lines = []
    for feature_col, feature_name in feature_mapping.items():
        if feature_col not in row.index: continue
        value = row[feature_col]
        if pd.isna(value): continue
        try: formatted_value = f"{float(value):.4f}"
        except (TypeError, ValueError): formatted_value = str(value)
        hrv_lines.append(f"- {feature_name}: {formatted_value}")
    return "\n".join(hrv_lines)
def build_user_content(row, feature_mapping, cat):
    hrv_text = build_hrv_text(row, feature_mapping)
    user_content = f"""
[Data Interpretation Context]
Note: The following HRV metrics are not instantaneous readings. They are strictly calculated based on rigorous daily aggregation:
- Feature Group: {FEATURE_GROUP_DESC[cat]}
- Base calculation: Extracted from 5-minute rolling windows of Normal-to-Normal (NN) intervals. Time-domain differences and Frequency domains were normalized.
- Suffixes meaning: Metric ending in '_mean' is the daily average; '_std' is the daily standard deviation; '_cv' (Coefficient of Variation) shows daily volatility; '_skew'/'_kurt' show distribution shape; '_1quartile'/'_3quartile' are the 25th/75th percentiles.
Please interpret these metrics as the overall autonomic stability and allostatic load for the entire day.
[Current Physiological Markers]
{hrv_text}
[Task]
Based only on today's HRV metrics, write a daily analytical log assessing the individual's current physiological and psychological state.
[Output Constraints]
- Combine your final analysis into a SINGLE, cohesive paragraph.
- Focus strictly on physiological and psychological interpretation.
- DO NOT directly state, conclude, or predict whether the individual is "stressed" or "not stressed".
- DO NOT list, quote, or repeat the raw HRV numbers provided above. Use qualitative descriptions (e.g., "highly volatile," "elevated parasympathetic activity") instead.
"""
    return user_content
def export_jsonl(df, group_mappings, output_paths):
    for cat, output_path in output_paths.items():
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        print(f"\nBuilding {cat} JSONL: {output_path} ({len(group_mappings[cat])} features)")
        with open(output_path, "w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                subject_id = str(row["ID"])
                if "date" in row.index and pd.notna(row["date"]): record_id = f"{subject_id}_{row['date']}"
                else: record_id = subject_id
                item = {
                    "record_id": record_id,
                    "user_id": subject_id,
                    "date": row.get("date"),
                    "binarized_stress": row.get("label_binarized"),
                    "feature_group": cat,
                    "messages": [{"role": "user", "content": build_user_content(row, group_mappings[cat], cat)}],
                }
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Exported {len(df)} records to: {output_path}")
if __name__ == "__main__":
    print("=" * 100)
    print("TILES HRV ONLY -> FOUR DOMAIN JSONL")
    print("=" * 100)
    print(f"\nReading raw HRV data: {HRV_DATASET_PATH}")
    df_hrv = pd.read_csv(HRV_DATASET_PATH)
    df_hrv.columns = df_hrv.columns.astype(str)
    df_hrv["ID"] = df_hrv["ID"].astype(str)
    print(f"HRV rows: {len(df_hrv)}")
    print(f"Subjects: {df_hrv['ID'].nunique()}")
    feature_mapping, group_mappings = load_feature_mapping(MAPPING_PATH)
    available_features = [col for col in feature_mapping.keys() if col in df_hrv.columns]
    missing_features = [col for col in feature_mapping.keys() if col not in df_hrv.columns]
    print(f"\nHRV features found: {len(available_features)}")
    if missing_features: raise RuntimeError(f"Missing HRV features: {missing_features}")
    if "date" in df_hrv.columns:
        duplicated = df_hrv.duplicated(subset=["ID", "date"], keep=False)
        print(f"Duplicated ID + date rows: {duplicated.sum()}")
        if duplicated.any():
            print(df_hrv.loc[duplicated, ["ID", "date"]].head(20))
            raise RuntimeError("Found duplicate ID + date, please check the data")
    export_jsonl(df_hrv, group_mappings, OUTPUT_JSONL)
    print("\n" + "=" * 100)
    print("COMPLETE")
    print("=" * 100)
