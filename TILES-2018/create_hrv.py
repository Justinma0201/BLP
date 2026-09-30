import os
import glob
import time
import numpy as np
import pandas as pd
import scipy
from tqdm import tqdm
import warnings
from hrvanalysis import get_nn_intervals, get_time_domain_features, get_frequency_domain_features

warnings.filterwarnings("ignore")

HR_DIR = "./data/heartrate_feature" 
FEATURE_DIR = "./data/"  
SURVEY_FILE = "./data/stressd.csv" 
OUTPUT_FILE = "./data/df_unique_time_label_total793.csv"

os.makedirs(FEATURE_DIR, exist_ok=True)

def tsconvert(t): 
    struct_time = time.strptime(t, "%Y-%m-%dT%H:%M:%S.000")
    return int(time.mktime(struct_time))

from numpy.lib.stride_tricks import sliding_window_view

def scale(rr, s):
    rr = np.asarray(rr, dtype=float)
    if len(rr) < s:
        return []
    windows = sliding_window_view(rr, s)
    return (windows.sum(axis=1) / s).tolist()

def motif_series(rr, m=3, l=1):
    rr = np.asarray(rr, dtype=float)
    if len(rr) < m:
        return []
    a, b, c = rr[:-2], rr[1:-1], rr[2:]
    conditions = [
        (a < b) & (b < c) & (a < c),   # 0
        (a < b) & (b > c) & (a < c),   # 1
        (a > b) & (b < c) & (a < c),   # 2
        (a < b) & (b > c) & (a > c),   # 3
        (a > b) & (b > c) & (a > c),   # 4
        (a > b) & (b < c) & (a > c),   # 5
        (a == b) & (b < c),            # 6
        (a == b) & (b > c),            # 7
        (a > b) & (a == c),            # 8
        (a < b) & (a == c),            # 9
    ]
    codes = np.select(conditions, list(range(10)), default=-1)
    return codes[codes != -1].tolist()

def d_s_r(ps, pr):
    return np.sqrt(6/5) * np.sqrt(np.sum([(ps_ - pr_)**2 for ps_, pr_ in zip(ps, pr)]))

def multi_scale_feats(rr):
    xs = [scale(rr, s) for s in [1,2,3,4,5]]
    xs_ord = [motif_series(xs_) for xs_ in xs]
    
    xs_p = [[np.sum(np.array(x)==j)/np.sum(np.array(x)<6) for j in range(6)] for x in xs_ord]
    xs_mp = [[np.sum(np.array(x)==j)/np.sum(np.array(x)<10) for j in range(10)] for x in xs_ord]

    mspe = [-np.sum([p * np.log(p+1e-20) for p in xs_p[s]]) for s in range(5)]
    msmpe = [-np.sum([p * np.log(p+1e-20) for p in xs_mp[s]]) for s in range(5)]

    ws = [np.sum([d_s_r(xs_p[i], xs_p[s]) for i in range(5)])/4 for s in range(5)]
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

    dxs_p = [[np.sum(np.array(x)==j)/np.sum(np.array(x)<6) for j in range(6)] for x in dxs_ord]
    dxs_mp = [[np.sum(np.array(x)==j)/np.sum(np.array(x)<10) for j in range(10)] for x in dxs_ord]

    dmspe = [-np.sum([p * np.log(p+1e-20) for p in dxs_p[s]]) for s in range(5)]
    dmsmpe = [-np.sum([p * np.log(p+1e-20) for p in dxs_mp[s]]) for s in range(5)]

    dws = [np.sum([d_s_r(dxs_p[i], dxs_p[s]) for i in range(5)])/4 for s in range(5)]
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

print("=== Stage 1: Compute HRV Features ===")
filenames = glob.glob(os.path.join(HR_DIR, "*.csv"))

for filename in tqdm(filenames):
    ID = os.path.basename(filename).split('.')[0]
    out_path = os.path.join(FEATURE_DIR, f"{ID}.csv")

    if os.path.exists(out_path):
        continue
        
    df = pd.read_csv(filename)[["Timestamp", "HeartRate"]].dropna()
    if df.empty:
        continue
        
    df['Timestamp_c'] = (
        pd.to_datetime(df['Timestamp'], format="%Y-%m-%dT%H:%M:%S.000")
        .values.astype('datetime64[s]').astype(np.int64)
    )
    df = df.sort_values('Timestamp_c').reset_index(drop=True)

    features, ts = [], []
    ts_arr = df['Timestamp_c'].values
    hr_arr = df['HeartRate'].values
    ts_str_arr = df['Timestamp'].values
    n = len(df)
    idx = 0
    while idx < n:
        start_ts = ts_arr[idx]
        end_idx = idx + np.searchsorted(ts_arr[idx:], start_ts + 300, side='left')
        window_len = end_idx - idx
        rr_coverage = window_len / 60

        if window_len >= 48:
            rr = 60 / hr_arr[idx:end_idx] * 1000
            nn = get_nn_intervals(rr, verbose=False)

            time_features = get_time_domain_features(nn)
            diff1 = np.diff(nn)
            norm_diff1 = np.diff((nn - np.mean(nn)) / np.std(nn))

            meanNN, stdNN = time_features['mean_nni'], time_features['sdnn']
            cvNN, pnn50 = time_features['cvnni'], time_features['pnni_50']
            rmsdd = time_features['rmssd']
            mean_diff1 = np.mean(diff1)
            std_abs_diff1 = np.std(np.abs(diff1))
            mean_abs_diff1 = np.mean(np.abs(norm_diff1))
            nmae1_diff1 = np.mean(np.abs(diff1)) / np.mean(nn)
            nmae2_diff1 = np.mean(np.abs(diff1)) / (max(nn) - min(nn))

            try:
                freq_features = get_frequency_domain_features(nn)
            except:
                print(nn)
            total_power, hf, lf, vlf = freq_features['total_power'], freq_features['hf'], freq_features['lf'], freq_features['vlf']
            hf_norm, lf_norm, lf_hf_ratio = hf/total_power, lf/total_power, freq_features['lf_hf_ratio']

            ms_feats = multi_scale_feats(nn)

            features.append(np.array([
                meanNN, stdNN, cvNN, pnn50, mean_diff1, std_abs_diff1, rmsdd, mean_abs_diff1, nmae1_diff1, nmae2_diff1,
                total_power, hf, hf_norm, lf, lf_norm, lf_hf_ratio, vlf, *ms_feats, rr_coverage
            ]))
            ts.append(ts_str_arr[idx])

        idx = end_idx

    if not features:
        continue

    df_hrv = pd.DataFrame(features)
    df_hrv['Timestamp'] = ts
    df_hrv['date'] = df_hrv['Timestamp'].apply(lambda x: x[:10])
    
    daily_features = []
    for date in np.unique(df_hrv['date']):
        sub = df_hrv[df_hrv['date'] == date]
        daily = sub.iloc[:, :-3]
        weight = sub.iloc[:, -3]
        daily_weighted = daily.multiply(weight, axis=0)

        daily_std = daily.std(ddof=0)
        daily_weighted_std = daily_weighted.std(ddof=0)

        daily_features.append(np.hstack((
            daily.mean().values, daily_std.values, (daily_std / daily.mean()).values,
            daily.median().values, daily.min().values, daily.max().values,
            daily.quantile(0.25).values, daily.quantile(0.75).values,
            daily.apply(scipy.stats.skew).values, daily.apply(scipy.stats.kurtosis).values,
            daily_weighted.mean().values, daily_weighted_std.values, (daily_weighted_std / daily_weighted.mean()).values,
            date
        )).reshape(1, -1))

    output = pd.DataFrame(np.vstack(daily_features))
    output = output.rename(columns={832: 'date'})
    output.to_csv(out_path, index=False)

print("\n=== Stage 2: Merge Survey Labels and Features ===")
survey = pd.read_csv(SURVEY_FILE)

survey = survey[survey['Finished'] == 1].copy()
survey['date'] = survey['completed_ts'].astype(str).str[:10]
survey_df = survey[['participant_id', 'date', 'stressd']].dropna().drop_duplicates()
print(f"Found {len(survey_df)} valid survey records.")

valid_feature_idx = [0,1,2,3,4,5,6,8,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63]
idxs = [64*i + j for i in range(13) for j in valid_feature_idx]

daily_data = []
_feature_cache = {}
for ID, date, target in tqdm(survey_df.values):
    if ID not in _feature_cache:
        feature_file = os.path.join(FEATURE_DIR, f"{ID}.csv")
        _feature_cache[ID] = pd.read_csv(feature_file) if os.path.exists(feature_file) else None

    daily_df = _feature_cache[ID]
    if daily_df is None:
        continue

    data = daily_df[daily_df['date'] == date].iloc[:, idxs].values

    if data.shape[0] > 0:
        daily_data.append(np.hstack((ID, date, target, data[0])))

HRV = pd.DataFrame(daily_data)

HRV = HRV.rename(columns={0: 'ID', 1: 'date', 2: 'stressd'})
HRV.to_csv(OUTPUT_FILE, index=False)

print(f"\nProcessing complete! Final feature file saved to: {OUTPUT_FILE}")
print(f"Total of {len(HRV)} records successfully matched and output, including 3 basic columns + 793 feature columns.")