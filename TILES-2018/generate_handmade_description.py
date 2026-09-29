import pandas as pd

# 載入你的資料
df1 = pd.read_csv(
    "/mnt/sdb/justin/Stress-Detection/Features/Attributes/Tiles/attribute/part_one-demographics.csv"
)

df2 = pd.read_csv(
    "/mnt/sdb/justin/Stress-Detection/Features/Attributes/Tiles/attribute/part_two-demographics_timings.csv"
)

df = df1.merge(df2, on="participant_id", how="inner")

# 定義映射字典 (根據 Readme 內容)
# 註：假設 0, 1, 2 對應 Readme 中的順序或 A, B, C
mapping = {

    "age": {
        0: "<30",
        30: "30-35",
        35: "35-40",
        40: "40-45",
        45: "45-50",
        50: ">50",
    },

    "englyrs": {
        0: "<25 years",
        25: "25-35 years",
        35: ">=35 years",
    },

    "educ": {
        "A": "Some HS/College",
        "B": "College Degree",
        "C": "Graduate School",
    },

    "quantsup": {
        "A": "<5 people",
        "B": "5+ people",
    },

    "duration": {
        "A": "<1yr",
        "B": "1yr",
        "C": "2yr",
        "D": "3yr",
        "E": "4yr",
        "F": "5-9yrs",
        "G": "10+yrs",
    },

    "income": {
        "A": "<$50k",
        "B": "$50k-$75k",
        "C": "$75k-$100k",
        "D": "$100k-$125k",
        "E": "$125k-$150k",
        "F": ">$150k",
    },

    "children": {
        0: "0",
        1: "1",
        2: "2",
        "3+": "3+",
    },

    "nurseyears": {
        0: "<5yrs",
        5: "5-10yrs",
        10: "10-15yrs",
        15: "15+yrs",
    },

    "hours": {
        "A": "<=37.5h",
        "B": ">37.5h",
    },

    "overtime": {
        "A": "0h",
        "B": "1-10h",
        "C": "10-20h",
        "D": "20-40h",
        "E": ">=40h",
    },

    "race": {
        "A": "White",
        "B": "Asian",
        "C": "Black/Other/Multi",
        "D": "Prefer not to answer",
    },

    "relationship": {
        "A": "Never Married",
        "B": "Married/Civil",
        "C": "Divorced/Widowed",
    },

    "housing": {
        "A": "Own",
        "B": "Rent",
        "C": "Shared/Family",
    },

    "currentposition": {
        "A": "RN",
        "B": "CNA",
        "C": "Clinical Support",
        "D": "Non-Clinical Support",
    },

    "commute_time": {
        "A": "<30min",
        "B": "31-45min",
        "C": "46-60min",
        "D": ">60min",
    },

    "student": {
        "A": "None",
        "B": "BSN",
        "C": "Graduate Program",
    },

    "gender": {
        1: "Male",
        2: "Female",
    },

    "shift": {
        1: "Day Shift",
        2: "Night Shift",
    },
}

def generate_description_en(row):
    # 1. 根據性別動態決定代名詞 (避免使用複數的 they/their)
    gender = str(row['gender']).strip()
    if gender == 'Male':
        p_subj = "he"
        p_poss = "His"
        p_poss_lower = "his"
    elif gender == 'Female':
        p_subj = "she"
        p_poss = "Her"
        p_poss_lower = "her"
    else:
        # 防呆機制：若有其他選項或缺失值，改用單數 participant
        p_subj = "the participant"
        p_poss = "The participant's"
        p_poss_lower = "the participant's"

    # 2. 處理直接填空會語句不順的欄位
    
    # 處理 Student 欄位 (避免出現 "identified as None")
    student_val = str(row['student'])
    if student_val == 'None':
        student_desc = "is not currently enrolled in a degree program"
    else:
        student_desc = f"is currently enrolled in a {student_val} program"

    # 處理 Housing 欄位 (避免出現 "reside in Own" 或 "reside in Rent")
    housing_val = str(row['housing'])
    if housing_val == 'Own':
        housing_desc = f"owns {p_poss_lower} home"
    elif housing_val == 'Rent':
        housing_desc = "rents a home"
    elif housing_val == 'Shared/Family':
        housing_desc = "lives in a shared or family housing arrangement"
    else:
        housing_desc = f"resides in {housing_val}"
        
    # 處理 Current Position (避免 a RN 這種母音開頭沒用 an 的文法問題)
    position = str(row['currentposition'])
    

    # 3. 組裝 Template
    # 注意：row['englyrs'] 的 mapping 已經包含 'years'，所以把 template 裡的 years 拿掉
    # 注意：row['nurseyears'] 同理，把後面的 years 拿掉避免重複
    template = (
        f"{row['participant_id']} is a {gender.lower()} participant in the {row['age']} age group, "
        f"and {student_desc}. Professionally, {p_subj} works as a(n) {position}, "
        f"with {row['nurseyears']} of professional experience and {row['duration']} of tenure at the current organization. "
        f"Working the {row['shift'].lower()}, {p_subj} averages {row['hours']} per week with approximately "
        f"{row['overtime']} of monthly overtime, and supervises {row['quantsup']}.\n\n"
        
        f"Regarding personal background, {p_subj} identifies as {row['race']} and has "
        f"{row['englyrs']} of English language experience. {p_poss} current relationship status is {row['relationship']}, "
        f"with an annual household income of {row['income']}. {p_subj.capitalize()} has {row['children']} child(ren) "
        f"under 18 in the household and {housing_desc}. {p_poss} typical daily commute is {row['commute_time']}."
    )
    return template

# 執行轉換
df_text = df.copy()
for col, trans_map in mapping.items():
    if col in df_text.columns:
        df_text[col] = df_text[col].map(trans_map).fillna(df_text[col])
df_text = df_text.fillna("Unknown")
# 儲存結果
df_text.to_csv('/mnt/sdb/justin/Stress-Detection/Features/Attributes/Tiles/attribute/attribute_readable.csv', index=False)
print("轉換完成！已儲存至 attribute_readable.csv")

df_text['Description'] = df_text.apply(generate_description_en, axis=1)
print(df_text['Description'].iloc[0])
df_text[['participant_id', 'Description']].to_csv('/mnt/sdb/justin/Stress-Detection/Final/data/handmade_description.csv', index=False)

# 查看前幾行
print(df_text.head())