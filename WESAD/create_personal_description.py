import os
import re
import glob
import pandas as pd

WESAD_DIR = "./WESAD"

OUTPUT_DIR = "./data"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "wesad_personal_attributes.csv",
)

def clean_value(value):
    if value is None:
        return None

    value = value.strip()

    if value == "" or value == "-":
        return None

    return value


def extract_field(text, pattern):
    match = re.search(
        pattern,
        text,
        flags=re.IGNORECASE,
    )

    if match is None:
        return None

    return clean_value(match.group(1))


def yes_no(value):
    if value is None:
        return None

    value = value.strip().upper()

    if value in ["YES", "Y"]:
        return True

    if value in ["NO", "N"]:
        return False

    return None

def parse_subject_readme(readme_path):
    with open(
        readme_path,
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as f:
        text = f.read()

    age = extract_field(
        text,
        r"Age:\s*([^\n\r]+)",
    )

    height = extract_field(
        text,
        r"Height\s*\(cm\):\s*([^\n\r]+)",
    )

    weight = extract_field(
        text,
        r"Weight\s*\(kg\):\s*([^\n\r]+)",
    )

    gender = extract_field(
        text,
        r"Gender:\s*([^\n\r]+)",
    )

    dominant_hand = extract_field(
        text,
        r"Dominant hand:\s*([^\n\r]+)",
    )

    coffee_today = extract_field(
        text,
        r"Did you drink coffee today\?\s*([^\n\r]+)",
    )

    coffee_last_hour = extract_field(
        text,
        r"Did you drink coffee within the last hour\?\s*([^\n\r]+)",
    )

    sports_today = extract_field(
        text,
        r"Did you do any sports today\?\s*([^\n\r]+)",
    )

    smoker = extract_field(
        text,
        r"Are you a smoker\?\s*([^\n\r]+)",
    )

    smoke_last_hour = extract_field(
        text,
        r"Did you smoke within the last hour\?\s*([^\n\r]+)",
    )

    ill_today = extract_field(
        text,
        r"Do you feel ill today\?\s*([^\n\r]+)",
    )

    return {
        "age": age,
        "height": height,
        "weight": weight,
        "gender": gender,
        "dominant_hand": dominant_hand,
        "coffee_today": coffee_today,
        "coffee_last_hour": coffee_last_hour,
        "sports_today": sports_today,
        "smoker": smoker,
        "smoke_last_hour": smoke_last_hour,
        "ill_today": ill_today,
    }

def build_personal_description(info):
    age = info["age"]
    height = info["height"]
    weight = info["weight"]
    gender = info["gender"]
    dominant_hand = info["dominant_hand"]

    coffee_today = yes_no(info["coffee_today"])
    coffee_last_hour = yes_no(info["coffee_last_hour"])
    sports_today = yes_no(info["sports_today"])
    smoker = yes_no(info["smoker"])
    smoke_last_hour = yes_no(info["smoke_last_hour"])
    ill_today = yes_no(info["ill_today"])

    sentences = []

    if age is not None and gender is not None:
        sentence = f"The individual is a {age}-year-old {gender.lower()}"

        if height is not None:
            sentence += f" who is {height} cm tall"

        if weight is not None:
            sentence += f" and weighs {weight} kg"

        sentence += "."

        sentences.append(sentence)

    else:
        basic_parts = []

        if age is not None:
            basic_parts.append(f"{age} years old")

        if gender is not None:
            basic_parts.append(gender.lower())

        if basic_parts:
            sentences.append(
                "The individual is " +
                " and ".join(basic_parts) +
                "."
            )

    if dominant_hand is not None:
        sentences.append(
            f"The individual is {dominant_hand.lower()}-hand dominant."
        )

    today_parts = []

    if coffee_today is True:
        today_parts.append("drank coffee")
    elif coffee_today is False:
        today_parts.append("did not drink coffee")

    if sports_today is True:
        today_parts.append("participated in sports")
    elif sports_today is False:
        today_parts.append("did not participate in sports")

    if smoker is True:
        today_parts.append("reported being a smoker")
    elif smoker is False:
        today_parts.append("reported not being a smoker")

    if ill_today is True:
        today_parts.append("reported feeling ill")
    elif ill_today is False:
        today_parts.append("reported not feeling ill")

    if today_parts:
        if len(today_parts) == 1:
            today_text = today_parts[0]

        elif len(today_parts) == 2:
            today_text = (
                today_parts[0] +
                " and " +
                today_parts[1]
            )

        else:
            today_text = (
                ", ".join(today_parts[:-1]) +
                ", and " +
                today_parts[-1]
            )

        sentences.append(
            f"On the day of data collection, the individual {today_text}."
        )

    last_hour_parts = []

    if coffee_last_hour is True:
        last_hour_parts.append("coffee consumption")
    elif coffee_last_hour is False:
        last_hour_parts.append("no coffee consumption")

    if smoke_last_hour is True:
        last_hour_parts.append("smoking")
    elif smoke_last_hour is False:
        last_hour_parts.append("no smoking")

    if last_hour_parts:
        if len(last_hour_parts) == 1:
            last_hour_text = last_hour_parts[0]
        else:
            last_hour_text = (
                last_hour_parts[0] +
                " and " +
                last_hour_parts[1]
            )

        sentences.append(
            f"The individual also reported {last_hour_text} within the hour prior to data collection."
        )

    return " ".join(sentences)

def main():
    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    readme_files = glob.glob(
        os.path.join(
            WESAD_DIR,
            "S*",
            "S*_readme.txt",
        )
    )

    readme_files = sorted(
        readme_files,
        key=lambda x: int(
            re.search(
                r"S(\d+)_readme\.txt",
                os.path.basename(x),
            ).group(1)
        ),
    )

    if len(readme_files) == 0:
        raise RuntimeError(
            f"No S*_readme.txt files found under: {WESAD_DIR}"
        )

    print("=" * 80)
    print("WESAD Personal Attribute Extraction")
    print("=" * 80)

    print(
        f"Found {len(readme_files)} subject readme files."
    )

    rows = []

    for readme_path in readme_files:
        filename = os.path.basename(
            readme_path
        )

        subject_match = re.search(
            r"(S\d+)_readme\.txt",
            filename,
        )

        if subject_match is None:
            continue

        subject_id = subject_match.group(1)

        info = parse_subject_readme(
            readme_path
        )

        description = build_personal_description(
            info
        )

        row = {
            "ID": subject_id,
            **info,
            "personal_description": description,
        }

        rows.append(row)

        print("\n" + "-" * 80)
        print(f"[{subject_id}]")

        print(
            f"Age           : {info['age']}"
        )

        print(
            f"Height        : {info['height']}"
        )

        print(
            f"Weight        : {info['weight']}"
        )

        print(
            f"Gender        : {info['gender']}"
        )

        print(
            f"Dominant hand : {info['dominant_hand']}"
        )

        print(
            f"Coffee today  : {info['coffee_today']}"
        )

        print(
            f"Coffee <1h    : {info['coffee_last_hour']}"
        )

        print(
            f"Sports today  : {info['sports_today']}"
        )

        print(
            f"Smoker        : {info['smoker']}"
        )

        print(
            f"Smoke <1h     : {info['smoke_last_hour']}"
        )

        print(
            f"Ill today     : {info['ill_today']}"
        )

        print(
            f"\nDescription:\n{description}"
        )

    df = pd.DataFrame(
        rows
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)

    print(
        f"Subjects: {len(df)}"
    )

    print(
        f"Saved to: {OUTPUT_FILE}"
    )

    print("\nMissing values:")

    print(
        df.isna().sum()
    )


if __name__ == "__main__":
    main()