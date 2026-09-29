#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import pickle
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import butter, filtfilt, find_peaks
from numpy.lib.stride_tricks import sliding_window_view
from hrvanalysis import get_nn_intervals, get_time_domain_features, get_frequency_domain_features

warnings.filterwarnings("ignore")

WESAD_DIR = "./WESAD"
OUTPUT_DIR = "./data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "wesad_ecg_hrv_60sec_step0.25sec_61features.csv")
MAPPING_FILE = os.path.join(OUTPUT_DIR, "hrv_feature_mapping_ecg_61features.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

ECG_FS = 700.0
LABEL_FS = 700.0

WINDOW_SEC = 60.0
STRIDE_SEC = 0.25
WINDOW_SAMPLES = int(WINDOW_SEC * ECG_FS)
STRIDE_SAMPLES = int(STRIDE_SEC * ECG_FS)

MIN_RR_COVERAGE = 0.80
MIN_RR_MS = 300.0
MAX_RR_MS = 2000.0
MIN_NN_COUNT = 30

VALID_PROTOCOL_LABELS = [1, 2, 3]
CONDITION_NAMES = {1: "baseline", 2: "stress", 3: "amusement"}
THREE_CLASS_MAP = {1: 0, 2: 1, 3: 2}
BINARY_MAP = {1: 0, 2: 1, 3: 0}

VALID_FEATURE_IDX = [
    0,1,2,3,4,5,6,8,
    11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,
    31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,
    51,52,53,54,55,56,57,58,59,60,61,62,63
]

FULL_FEATURE_NAMES = [
    "meanNN",                 # 0
    "stdNN",                  # 1
    "cvNN",                   # 2
    "pNN50",                  # 3
    "mean_diff1",             # 4
    "std_abs_diff1",          # 5
    "RMSSD",                  # 6
    "mean_abs_diff1",         # 7  removed
    "nMAE1_diff1",            # 8
    "nMAE2_diff1",            # 9  removed
    "total_power",            # 10 removed
    "HF",                     # 11
    "HF_norm",                # 12
    "LF",                     # 13
    "LF_norm",                # 14
    "LF_HF_ratio",            # 15
    "VLF",                    # 16

    "MSPE_scale1",            # 17
    "MSPE_scale2",            # 18
    "MSPE_scale3",            # 19
    "MSPE_scale4",            # 20
    "MSPE_scale5",            # 21

    "MSMPE_scale1",           # 22
    "MSMPE_scale2",           # 23
    "MSMPE_scale3",           # 24
    "MSMPE_scale4",           # 25
    "MSMPE_scale5",           # 26

    "PE_dw",                  # 27

    "DS1_scale3",             # 28
    "DS1_scale4",             # 29
    "DS1_scale5",             # 30

    "DS2_scale3",             # 31
    "DS2_scale4",             # 32
    "DS2_scale5",             # 33

    "DS1_diff1_1",            # 34
    "DS1_diff1_2",            # 35
    "DS2_diff1_1",            # 36
    "DS2_diff1_2",            # 37

    "DS1_sum",                # 38
    "DS2_sum",                # 39

    "dMSPE_scale1",           # 40
    "dMSPE_scale2",           # 41
    "dMSPE_scale3",           # 42
    "dMSPE_scale4",           # 43
    "dMSPE_scale5",           # 44

    "dMSMPE_scale1",          # 45
    "dMSMPE_scale2",          # 46
    "dMSMPE_scale3",          # 47
    "dMSMPE_scale4",          # 48
    "dMSMPE_scale5",          # 49

    "dPE_dw",                 # 50

    "dDS1_scale3",            # 51
    "dDS1_scale4",            # 52
    "dDS1_scale5",            # 53

    "dDS2_scale3",            # 54
    "dDS2_scale4",            # 55
    "dDS2_scale5",            # 56

    "dDS1_diff1_1",           # 57
    "dDS1_diff1_2",           # 58
    "dDS2_diff1_1",           # 59
    "dDS2_diff1_2",           # 60

    "dDS1_sum",               # 61
    "dDS2_sum",               # 62
    "total_asym_idx",         # 63
]


def save_feature_mapping():
    rows = [{"feature_column": f"feature_{final_idx}", "final_index": final_idx, "original_index": original_idx, "feature_name": FULL_FEATURE_NAMES[original_idx]} for final_idx, original_idx in enumerate(VALID_FEATURE_IDX)]
    df = pd.DataFrame(rows)
    df.to_csv(MAPPING_FILE, index=False)
    print(f"Feature mapping saved to: {MAPPING_FILE}")
    return df

def scale(rr, s):
    rr = np.asarray(rr, dtype=float)
    if len(rr) < s: return []
    windows = sliding_window_view(rr, s)
    return (windows.sum(axis=1) / s).tolist()


def motif_series(rr, m=3, l=1):
    rr = np.asarray(rr, dtype=float)
    if len(rr) < m: return []

    a, b, c = rr[:-2], rr[1:-1], rr[2:]

    conditions = [
        (a < b) & (b < c) & (a < c),
        (a < b) & (b > c) & (a < c),
        (a > b) & (b < c) & (a < c),
        (a < b) & (b > c) & (a > c),
        (a > b) & (b > c) & (a > c),
        (a > b) & (b < c) & (a > c),
        (a == b) & (b < c),
        (a == b) & (b > c),
        (a > b) & (a == c),
        (a < b) & (a == c),
    ]

    codes = np.select(conditions, list(range(10)), default=-1)
    return codes[codes != -1].tolist()


def d_s_r(ps, pr):
    return np.sqrt(6 / 5) * np.sqrt(np.sum([(ps_ - pr_) ** 2 for ps_, pr_ in zip(ps, pr)]))


def multi_scale_feats(rr):
    rr = np.asarray(rr, dtype=float)

    xs = [scale(rr, s) for s in [1,2,3,4,5]]
    xs_ord = [motif_series(xs_) for xs_ in xs]

    xs_p = [[np.sum(np.array(x) == j) / np.sum(np.array(x) < 6) for j in range(6)] for x in xs_ord]
    xs_mp = [[np.sum(np.array(x) == j) / np.sum(np.array(x) < 10) for j in range(10)] for x in xs_ord]

    mspe = [-np.sum([p * np.log(p + 1e-20) for p in xs_p[s]]) for s in range(5)]
    msmpe = [-np.sum([p * np.log(p + 1e-20) for p in xs_mp[s]]) for s in range(5)]

    ws = [np.sum([d_s_r(xs_p[i], xs_p[s]) for i in range(5)]) / 4 for s in range(5)]
    PE_dw = np.sum([ws[s] * mspe[s] for s in range(5)]) / 5

    ds1 = [d_s_r(xs_p[0], xs_p[k]) for k in [2,3,4]]
    ds2 = [d_s_r(xs_p[1], xs_p[k]) for k in [2,3,4]]
    ds1_diff1 = np.diff(ds1).tolist()
    ds2_diff1 = np.diff(ds2).tolist()
    ds1_sum = np.sum(ds1)
    ds2_sum = np.sum(ds2)

    drr = np.abs(np.diff(rr))

    dxs = [scale(drr, s) for s in [1,2,3,4,5]]
    dxs_ord = [motif_series(dxs_) for dxs_ in dxs]

    dxs_p = [[np.sum(np.array(x) == j) / np.sum(np.array(x) < 6) for j in range(6)] for x in dxs_ord]
    dxs_mp = [[np.sum(np.array(x) == j) / np.sum(np.array(x) < 10) for j in range(10)] for x in dxs_ord]

    dmspe = [-np.sum([p * np.log(p + 1e-20) for p in dxs_p[s]]) for s in range(5)]
    dmsmpe = [-np.sum([p * np.log(p + 1e-20) for p in dxs_mp[s]]) for s in range(5)]

    dws = [np.sum([d_s_r(dxs_p[i], dxs_p[s]) for i in range(5)]) / 4 for s in range(5)]
    dPE_dw = np.sum([dws[s] * dmspe[s] for s in range(5)]) / 5

    dds1 = [d_s_r(dxs_p[0], dxs_p[k]) for k in [2,3,4]]
    dds2 = [d_s_r(dxs_p[1], dxs_p[k]) for k in [2,3,4]]
    dds1_diff1 = np.diff(dds1).tolist()
    dds2_diff1 = np.diff(dds2).tolist()
    dds1_sum = np.sum(dds1)
    dds2_sum = np.sum(dds2)

    asym_idx = [(np.sum(np.heaviside(-np.array(dxs[s]), 0.5)) - np.sum(np.heaviside(np.array(dxs[s]), 0.5))) / len(dxs[s]) for s in range(5)]
    total_asym_idx = np.sum(asym_idx)

    return [*mspe, *msmpe, PE_dw, *ds1, *ds2, *ds1_diff1, *ds2_diff1, ds1_sum, ds2_sum, *dmspe, *dmsmpe, dPE_dw, *dds1, *dds2, *dds1_diff1, *dds2_diff1, dds1_sum, dds2_sum, total_asym_idx]

def bandpass_ecg(ecg, fs=ECG_FS, lowcut=5.0, highcut=20.0, order=3):
    ecg = np.asarray(ecg, dtype=float)
    nyq = fs / 2.0
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, ecg)


def detect_r_peaks(ecg, fs=ECG_FS):
    ecg = np.asarray(ecg, dtype=float)
    if len(ecg) < int(5 * fs): return None
    if not np.all(np.isfinite(ecg)): return None

    try: filtered = bandpass_ecg(ecg, fs)
    except Exception: return None

    min_distance = int(0.30 * fs)
    prominence = max(np.std(filtered) * 0.5, 1e-8)

    peaks_pos, props_pos = find_peaks(filtered, distance=min_distance, prominence=prominence)
    peaks_neg, props_neg = find_peaks(-filtered, distance=min_distance, prominence=prominence)

    pos_prominence = np.median(props_pos["prominences"]) if len(peaks_pos) > 0 else 0.0
    neg_prominence = np.median(props_neg["prominences"]) if len(peaks_neg) > 0 else 0.0

    peaks = peaks_pos if pos_prominence >= neg_prominence else peaks_neg

    if len(peaks) < 2: return None
    return np.asarray(peaks, dtype=int)

def get_window_rr(global_rpeaks, start_idx, end_idx):
    left = np.searchsorted(global_rpeaks, start_idx, side="left")
    right = np.searchsorted(global_rpeaks, end_idx, side="left")
    peaks = global_rpeaks[left:right]

    if len(peaks) < MIN_NN_COUNT + 1: return None, None

    rr = np.diff(peaks) / ECG_FS * 1000.0
    rr = rr[np.isfinite(rr)]
    rr = rr[(rr >= MIN_RR_MS) & (rr <= MAX_RR_MS)]

    if len(rr) < MIN_NN_COUNT: return None, None

    rr_coverage = min(np.sum(rr) / (WINDOW_SEC * 1000.0), 1.0)
    return rr, rr_coverage

def extract_hrv_features(rr):
    rr = np.asarray(rr, dtype=float)
    rr = rr[np.isfinite(rr) & (rr > 0)]

    if len(rr) < MIN_NN_COUNT: return None

    try: nn = np.asarray(get_nn_intervals(rr, verbose=False), dtype=float)
    except Exception: return None

    nn = nn[np.isfinite(nn)]

    if len(nn) < MIN_NN_COUNT: return None
    if np.std(nn) == 0: return None
    if np.max(nn) - np.min(nn) == 0: return None

    try: time_features = get_time_domain_features(nn)
    except Exception: return None

    diff1 = np.diff(nn)
    norm_diff1 = np.diff((nn - np.mean(nn)) / np.std(nn))

    meanNN = time_features["mean_nni"]
    stdNN = time_features["sdnn"]
    cvNN = time_features["cvnni"]
    pnn50 = time_features["pnni_50"]
    rmssd = time_features["rmssd"]

    mean_diff1 = np.mean(diff1)
    std_abs_diff1 = np.std(np.abs(diff1))
    mean_abs_diff1 = np.mean(np.abs(norm_diff1))
    nmae1_diff1 = np.mean(np.abs(diff1)) / np.mean(nn)
    nmae2_diff1 = np.mean(np.abs(diff1)) / (np.max(nn) - np.min(nn))

    try: freq_features = get_frequency_domain_features(nn)
    except Exception: return None

    total_power = freq_features["total_power"]
    hf = freq_features["hf"]
    lf = freq_features["lf"]
    vlf = freq_features["vlf"]
    lf_hf_ratio = freq_features["lf_hf_ratio"]

    if not np.isfinite(total_power) or total_power == 0: return None

    hf_norm = hf / total_power
    lf_norm = lf / total_power

    try: ms_feats = multi_scale_feats(nn)
    except Exception: return None

    if len(ms_feats) != 47:
        print(f"[DEBUG] multi_scale_feats length={len(ms_feats)}, expected=47")
        return None

    full_features = np.asarray([
        meanNN, stdNN, cvNN, pnn50,
        mean_diff1, std_abs_diff1, rmssd,
        mean_abs_diff1, nmae1_diff1, nmae2_diff1,
        total_power, hf, hf_norm, lf, lf_norm, lf_hf_ratio, vlf,
        *ms_feats
    ], dtype=float)

    if len(full_features) != 64:
        print(f"[DEBUG] full_features length={len(full_features)}, expected=64")
        return None

    selected_features = full_features[VALID_FEATURE_IDX]

    if len(selected_features) != 61:
        print(f"[DEBUG] selected_features length={len(selected_features)}, expected=61")
        return None

    if not np.all(np.isfinite(selected_features)): return None
    return selected_features

def get_continuous_segments(labels):
    labels = np.asarray(labels)
    changes = np.where(np.diff(labels) != 0)[0] + 1
    boundaries = np.concatenate(([0], changes, [len(labels)]))
    segments = []

    for i in range(len(boundaries) - 1):
        start_idx, end_idx = boundaries[i], boundaries[i + 1]
        label = int(labels[start_idx])
        if label not in VALID_PROTOCOL_LABELS: continue
        segments.append((start_idx, end_idx, label))

    return segments

def process_subject(subject_dir):
    subject = os.path.basename(subject_dir)
    pkl_path = os.path.join(subject_dir, f"{subject}.pkl")

    if not os.path.exists(pkl_path):
        print(f"[SKIP] {subject}: missing pkl")
        return []

    with open(pkl_path, "rb") as f: data = pickle.load(f, encoding="latin1")

    ecg = np.asarray(data["signal"]["chest"]["ECG"], dtype=float).squeeze()
    labels = np.asarray(data["label"]).squeeze()

    if len(ecg) != len(labels): raise RuntimeError(f"{subject}: ECG length={len(ecg)} != label length={len(labels)}")

    print(f"\n[{subject}] ECG samples={len(ecg)}, duration={len(ecg) / ECG_FS:.2f} sec")

    global_rpeaks = detect_r_peaks(ecg, ECG_FS)

    if global_rpeaks is None:
        print(f"[SKIP] {subject}: R-peak detection failed")
        return []

    print(f"[{subject}] detected R-peaks={len(global_rpeaks)}, mean HR={len(global_rpeaks) / (len(ecg) / ECG_FS) * 60:.2f} bpm")

    segments = get_continuous_segments(labels)
    rows = []

    for segment_start_idx, segment_end_idx, protocol_label in segments:
        if segment_end_idx - segment_start_idx < WINDOW_SAMPLES: continue

        condition = CONDITION_NAMES[protocol_label]
        start_idx = segment_start_idx

        while start_idx + WINDOW_SAMPLES <= segment_end_idx:
            end_idx = start_idx + WINDOW_SAMPLES
            rr, rr_coverage = get_window_rr(global_rpeaks, start_idx, end_idx)

            if rr is None:
                start_idx += STRIDE_SAMPLES
                continue

            if rr_coverage < MIN_RR_COVERAGE:
                start_idx += STRIDE_SAMPLES
                continue

            features = extract_hrv_features(rr)

            if features is None:
                start_idx += STRIDE_SAMPLES
                continue

            row = {
                "ID": subject,
                "window_start_sec": start_idx / ECG_FS,
                "window_end_sec": end_idx / ECG_FS,
                "condition": condition,
                "protocol_label": protocol_label,
                "label_3class": THREE_CLASS_MAP[protocol_label],
                "label_binary": BINARY_MAP[protocol_label],
                "num_rpeaks": len(rr) + 1,
                "num_rr": len(rr),
                "rr_coverage": rr_coverage,
            }

            for i, value in enumerate(features): row[f"feature_{i}"] = value

            rows.append(row)
            start_idx += STRIDE_SAMPLES

    print(f"[{subject}] valid windows={len(rows)}")
    return rows

def main():
    print("=" * 80)
    print("WESAD ECG-derived HRV extraction")
    print("=" * 80)
    print(f"ECG sampling rate : {ECG_FS} Hz")
    print(f"Window            : {WINDOW_SEC} sec")
    print(f"Window samples    : {WINDOW_SAMPLES}")
    print(f"Stride            : {STRIDE_SEC} sec")
    print(f"Stride samples    : {STRIDE_SAMPLES}")
    print(f"Overlap           : {WINDOW_SEC - STRIDE_SEC} sec")
    print(f"Min RR coverage   : {MIN_RR_COVERAGE}")
    print(f"Final HRV features: {len(VALID_FEATURE_IDX)}")
    print("=" * 80)

    subject_dirs = sorted([p for p in glob.glob(os.path.join(WESAD_DIR, "S*")) if os.path.isdir(p)])

    if len(subject_dirs) == 0: raise FileNotFoundError(f"No subjects found under {WESAD_DIR}")

    all_rows = []

    for subject_dir in tqdm(subject_dirs, desc="Subjects"):
        try: all_rows.extend(process_subject(subject_dir))
        except Exception as e: print(f"\n[ERROR] {subject_dir}: {e}")

    if len(all_rows) == 0: raise RuntimeError("No HRV samples generated.")

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 80)
    print("Sample distribution")
    print("=" * 80)
    print(pd.crosstab(df["ID"], df["condition"], margins=True))

    print("\n3-class:")
    print(df["condition"].value_counts())

    print("\nBinary:")
    print(df["label_binary"].value_counts().sort_index())

    print("\nRR coverage:")
    print(df["rr_coverage"].describe())

    print("\nR-peaks per window:")
    print(df["num_rpeaks"].describe())

    print("\nRR intervals per window:")
    print(df["num_rr"].describe())

    print("\n" + "=" * 80)
    print("DONE")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Samples: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print("=" * 80)

    save_feature_mapping()


if __name__ == "__main__":
    main()