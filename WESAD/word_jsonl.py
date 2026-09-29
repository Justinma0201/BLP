#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
from pathlib import Path
import re


DESC_PATH = "./data/wesad_personal_attributes.csv"
OUTPUT_JSONL = "./data/wesad_baseline_only_word.jsonl"
EXPECTED_SUBJECTS = 15


def build_user_content(user_baseline):
    return f"""[Original Baseline Profile]
{user_baseline}

[Task]
Based on the individual's baseline profile, write a cohesive description assessing their current stress state.
"""


def load_records(desc_path):
    records = {}
    duplicate_count = 0
    with open(desc_path, encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required = {"ID", "personal_description"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Description table missing columns {sorted(required)}; current columns are {reader.fieldnames}")
        for line_number, row in enumerate(reader, 2):
            subject_id = (row["ID"] or "").strip()
            if not subject_id:
                raise ValueError(f"Row {line_number} is missing an ID")
            description = row["personal_description"]
            if description is None or not description.strip():
                description = "No baseline description available."
            if subject_id in records:
                duplicate_count += 1
                if records[subject_id] != description:
                    raise ValueError(f"ID={subject_id} has different personal_description, please confirm which one to use")
                continue
            records[subject_id] = description
    print(f"Original rows: {len(records) + duplicate_count}")
    print(f"Removed duplicate rows with same ID / description: {duplicate_count}")
    print(f"Unique IDs: {len(records)}")
    if len(records) != EXPECTED_SUBJECTS:
        raise ValueError(f"Expected {EXPECTED_SUBJECTS} subjects, but found {len(records)}; JSONL not exported")
    return records


def anonymize_description(description, subject_id):
    if subject_id.isdecimal():
        return re.sub(r"^" + re.escape(subject_id) + r"(?=\s+is\s+a\b)",
                      "This individual", description)
    return re.sub(r"(?<!\w)" + re.escape(subject_id) + r"(?!\w)",
                  "This individual", description)


def export_jsonl(records, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as target:
        for subject_id, description in records.items():
            item = {
                "record_id": subject_id,
                "user_id": subject_id,
                "messages": [{
                    "role": "user",
                    "content": build_user_content(anonymize_description(description, subject_id)),
                }],
            }
            target.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Exported {len(records)} prompts to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DESC_PATH, help="WESAD personal attributes CSV")
    parser.add_argument("--output", default=OUTPUT_JSONL, help="Export JSONL")
    args = parser.parse_args()
    if Path(args.input).resolve() == Path(args.output).resolve():
        raise ValueError("Input and output cannot be the same file")
    records = load_records(args.input)
    export_jsonl(records, args.output)


if __name__ == "__main__":
    main()
