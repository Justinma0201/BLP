#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import pandas as pd

HRV_DATASET_PATH = "./data/wesad_ecg_hrv_60sec_step0.25sec_61features.csv"

PERSONAL_DESC_PATH = "./data/wesad_personal_attributes.csv"

MAPPING_PATH = "./data/hrv_feature_mapping_ecg_61features.csv"

OUTPUT_JSONL = {
    "Time": "./data/hrv_full_raw_time.jsonl",
    "Freq": "./data/hrv_full_raw_freq.jsonl",
    "MS_RR": "./data/hrv_full_raw_ms_rr.jsonl",
    "MS_dRR": "./data/hrv_full_raw_ms_drr.jsonl",
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

    required_cols = ["feature_column", "feature_name"]

    for col in required_cols:
        if col not in mapping_df.columns: raise ValueError(f"Mapping file missing column '{col}'，current columns: {mapping_df.columns.tolist()}")

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

    print(f"Mapping total {len(feature_mapping)} features")

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
    if pd.notna(row.get("personal_description")): user_baseline = str(row["personal_description"])
    else: user_baseline = "No baseline description available."

    hrv_text = build_hrv_text(row, feature_mapping)

    user_content = f"""[Neutral Baseline Profile]
{user_baseline}

[Data Interpretation Context]
Note: The following HRV metrics are calculated from a 60-second ECG segment.
- Feature Group: {FEATURE_GROUP_DESC[cat]}
- Base calculation: R-peaks were detected from the ECG signal, converted into RR intervals, cleaned into Normal-to-Normal (NN) intervals, and used to derive time-domain, frequency-domain, and multi-scale HRV features.
- Window setting: Each sample represents one 60-second physiological window.
- Feature values: The HRV metrics below are provided in their original feature scale without fold-specific normalization.
Please interpret these metrics as the individual's short-term autonomic dynamics during this physiological window.

[Current Physiological Markers]
{hrv_text}

[Task]
Based on the individual's baseline profile and the HRV metrics observed in this physiological window, write an analytical log assessing their current physiological and psychological state.

[Output Constraints]
- Combine your final analysis into a SINGLE, cohesive paragraph.
- Focus strictly on physiological and psychological interpretation.
- DO NOT directly state, conclude, or predict whether the individual is "stressed" or "not stressed".
- DO NOT list, quote, or repeat the raw HRV numbers provided above. Use qualitative descriptions (e.g., "highly volatile," "elevated parasympathetic activity") instead.
- If there's 'Unknown' in the description, it means the data is missing and you can ignore it."""

    return user_content

def export_jsonl(df, group_mappings, output_paths):
    for cat, output_path in output_paths.items():
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        print(f"\nCreating {cat} JSONL: {output_path} ({len(group_mappings[cat])} features)")

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
                    "feature_group": cat,
                    "messages": [{"role": "user", "content": build_user_content(row, group_mappings[cat], cat)}],
                }

                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"Exported {len(df)} records to: {output_path}")

if __name__ == "__main__":
    print("=" * 100)
    print("WESAD RAW HRV -> FOUR DOMAIN JSONL")
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

    if "personal_description" not in df_personal.columns: raise ValueError("personal_description column not found")

    print(f"Personal description subjects: {df_personal['ID'].nunique()}")

    df_merged = pd.merge(df_hrv, df_personal[["ID", "personal_description"]], on="ID", how="left")

    print(f"\nMerge rows: {len(df_merged)}")

    missing_desc = df_merged["personal_description"].isna().sum()

    print(f"Missing personal descriptions: {missing_desc}")

    feature_mapping, group_mappings = load_feature_mapping(MAPPING_PATH)

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
            raise RuntimeError("Found duplicate ID + window_start_sec, please check the data")

    export_jsonl(df_merged, group_mappings, OUTPUT_JSONL)

    print("\n" + "=" * 100)
    print("COMPLETE")
    print("=" * 100)
