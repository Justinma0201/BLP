#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import pandas as pd

HRV_DATASET_PATH = "./data/wesad_ecg_hrv_60sec_step0.25sec_61features.csv"

PERSONAL_DESC_PATH = "./data/wesad_personal_attributes.csv"

MAPPING_PATH = "./data/hrv_feature_mapping_ecg_61features.csv"

OUTPUT_JSONL = "./data/hrv_full_only_word.jsonl"

def load_feature_mapping(mapping_path):
    print(f"Reading HRV feature mapping: {mapping_path}")

    mapping_df = pd.read_csv(mapping_path)

    required_cols = ["feature_column", "feature_name"]

    for col in required_cols:
        if col not in mapping_df.columns: raise ValueError(f"Mapping file 缺少欄位 '{col}'，目前欄位為: {mapping_df.columns.tolist()}")

    feature_mapping = dict(zip(mapping_df["feature_column"].astype(str), mapping_df["feature_name"].astype(str)))

    print(f"Mapping {len(feature_mapping)} features")

    return feature_mapping

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

def build_user_content(row, feature_mapping):
    if pd.notna(row.get("personal_description")): user_baseline = str(row["personal_description"])
    else: user_baseline = "No baseline description available."

    hrv_text = build_hrv_text(row, feature_mapping)

    user_content = f"""[Neutral Baseline Profile]
{user_baseline}
[Task]
Based on the individual's baseline profile, write an analytical log assessing their current physiological and psychological state.

[Output Constraints]
- Combine your final analysis into a SINGLE, cohesive paragraph.
- Focus strictly on physiological and psychological interpretation.
- DO NOT directly state, conclude, or predict whether the individual is "stressed" or "not stressed".
- If there's 'Unknown' in the description, it means the data is missing and you can ignore it."""

    return user_content

def export_jsonl(df, feature_mapping, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"\nCreating JSONL: {output_path}")

    with open(output_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            subject_id = str(row["ID"])

            if "window_start_sec" in row.index and pd.notna(row["window_start_sec"]): record_id = f"{subject_id}_{row['window_start_sec']}"
            else: record_id = subject_id

            item = {
                "record_id": record_id,
                "user_id": subject_id,
                "window_start_sec": row.get("window_start_sec"),
                "window_end_sec": row.get("window_end_sec"),
                "condition": row.get("condition"),
                "protocol_label": row.get("protocol_label"),
                "binarized_stress": row.get("label_binary"),
                "messages": [{"role": "user", "content": build_user_content(row, feature_mapping)}],
            }

            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Successfully exported {len(df)} entries to:")
    print(output_path)

if __name__ == "__main__":
    print("=" * 100)
    print("WESAD RAW HRV -> SINGLE JSONL")
    print("=" * 100)

    print(f"\nReading raw HRV data: {HRV_DATASET_PATH}")

    df_hrv = pd.read_csv(HRV_DATASET_PATH)
    df_hrv.columns = df_hrv.columns.astype(str)
    df_hrv["ID"] = df_hrv["ID"].astype(str)

    print(f"HRV rows: {len(df_hrv)}")
    print(f"Subjects: {df_hrv['ID'].nunique()}")

    print(f"\nReading personal descriptions: {PERSONAL_DESC_PATH}")

    df_personal = pd.read_csv(PERSONAL_DESC_PATH)
    df_personal["ID"] = df_personal["ID"].astype(str)

    if "personal_description" not in df_personal.columns: raise ValueError("personal description CSV 中找不到 personal_description 欄位")

    print(f"Personal description subjects: {df_personal['ID'].nunique()}")

    df_merged = pd.merge(df_hrv, df_personal[["ID", "personal_description"]], on="ID", how="left")

    print(f"\nMerge rows: {len(df_merged)}")

    missing_desc = df_merged["personal_description"].isna().sum()

    print(f"Missing personal descriptions: {missing_desc}")

    feature_mapping = load_feature_mapping(MAPPING_PATH)

    if len(feature_mapping) != 61: raise RuntimeError(f"Expected 61 mapped features, but mapping contains {len(feature_mapping)}")

    available_features = [col for col in feature_mapping.keys() if col in df_merged.columns]
    missing_features = [col for col in feature_mapping.keys() if col not in df_merged.columns]

    print(f"\nHRV features found: {len(available_features)}")

    if missing_features: raise RuntimeError(f"Missing HRV features: {missing_features}")
    if len(available_features) != 61: raise RuntimeError(f"Expected 61 features, but found {len(available_features)}")

    if "window_start_sec" in df_merged.columns:
        duplicated = df_merged.duplicated(subset=["ID", "window_start_sec"], keep=False)

        print(f"Duplicated ID + window_start_sec rows: {duplicated.sum()}")

        if duplicated.any():
            print(df_merged.loc[duplicated, ["ID", "window_start_sec", "window_end_sec"]].head(20))
            raise RuntimeError("Found duplicate ID + window_start_sec")
    export_jsonl(df_merged, feature_mapping, OUTPUT_JSONL)

    print("\n" + "=" * 100)
    print("COMPLETE")
    print("=" * 100)