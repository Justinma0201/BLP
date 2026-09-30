import pandas as pd
import json
import os

DESC_PATH = "./data/handmade_description.csv"

OUTPUT_JSONL = "./data/baseline_rewrite.jsonl"

def build_user_content(user_baseline):
    user_content = f"""[Original Baseline Profile]
{user_baseline}

[Task]
Based on the individual's baseline profile, write a cohesive description assessing their current stress state.
"""

    return user_content

def export_jsonl(df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"\nCreating JSONL: {output_path}")

    with open(output_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            participant_id = str(row["ID"])

            if pd.notna(row.get("Description")): user_baseline = str(row["Description"])
            else: user_baseline = "No baseline description available."

            user_baseline = user_baseline.replace(f"{participant_id} is a", "This individual is a")
            user_baseline = user_baseline.replace(participant_id, "This individual")

            item = {
                "record_id": participant_id,
                "user_id": participant_id,
                "messages": [
                    {
                        "role": "user",
                        "content": build_user_content(user_baseline)
                    }
                ]
            }

            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved {len(df)} entries to:")
    print(output_path)

if __name__ == "__main__":
    print("Reading baseline descriptions...")

    df = pd.read_csv(DESC_PATH)

    if "participant_id" in df.columns:
        df = df.rename(columns={"participant_id": "ID"})

    if "ID" not in df.columns:
        raise ValueError(f"Can not find ID / participant_id column, current columns: {df.columns.tolist()}")

    if "Description" not in df.columns:
        raise ValueError(f"Can not find Description column, current columns: {df.columns.tolist()}")

    df["ID"] = df["ID"].astype(str)

    print(f"Original rows: {len(df)}")
    print(f"Unique IDs: {df['ID'].nunique()}")

    duplicated_ids = df[df.duplicated(subset=["ID"], keep=False)]["ID"].unique()

    if len(duplicated_ids) > 0:
        print(f"Found {len(duplicated_ids)} duplicated IDs, keeping only the first occurrence")
        print(f"Duplicated IDs: {duplicated_ids.tolist()}")

    df = df.drop_duplicates(subset=["ID"], keep="first").reset_index(drop=True)

    print(f"Rows after deduplication: {len(df)}")
    print(f"Unique IDs after deduplication: {df['ID'].nunique()}")

    export_jsonl(df, OUTPUT_JSONL)

    print("\nFinished")