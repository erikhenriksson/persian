import json
import re

import pandas as pd

# Load the CSV file with keep_default_na=False to prevent "NA" from being treated as NaN
df = pd.read_csv("data/persian.csv", keep_default_na=False, na_values=[""])

# Define valid labels
labels_structure = {
    "MT": [],
    "LY": [],
    "SP": ["it"],
    "ID": [],
    "NA": ["ne", "sr", "nb"],
    "HI": ["re"],
    "IN": ["en", "ra", "dtp", "fi", "lt"],
    "OP": ["rv", "ob", "rs", "av"],
    "IP": ["ds", "ed"],
}

# Create a set of all valid labels (main + sub)
valid_labels = set(labels_structure.keys())
for sub_labels in labels_structure.values():
    valid_labels.update(sub_labels)


# Function to clean and consolidate labels
def consolidate_labels(group):
    # Collect all unique values from Turku_NLP and Turku_NLP_sub
    all_labels = set()

    for val in group["Turku_NLP"]:
        if val and val != "":  # Check for empty strings
            # Split by semicolon and clean
            labels = [label.strip() for label in str(val).split(";")]
            all_labels.update(labels)

    for val in group["Turku_NLP_sub"]:
        if val and val != "":  # Check for empty strings
            # Split by semicolon and clean
            labels = [label.strip() for label in str(val).split(";")]
            all_labels.update(labels)

    # Remove non-alphabetic characters (keep only letters and spaces), then strip and filter empties
    all_labels = [re.sub(r"[^a-zA-Z\s]", "", label).strip() for label in all_labels]
    all_labels = [label for label in all_labels if label]

    # Filter to keep only valid labels
    all_labels = [label for label in all_labels if label in valid_labels]

    # Sort alphabetically
    all_labels = sorted(all_labels)

    # Join with spaces
    return " ".join(all_labels)


# Select only the relevant columns and drop duplicates based on 'id'
consolidated = df[["id", "text"]].drop_duplicates(subset=["id"], keep="first")

# Add the consolidated label column
consolidated["label"] = df.groupby("id").apply(consolidate_labels).values

# Create list of dictionaries for JSONL
records = consolidated.to_dict("records")

# Save to JSONL
output_file = "data/persian_consolidated.jsonl"
with open(output_file, "w", encoding="utf-8") as f:
    for record in records:
        json.dump(record, f, ensure_ascii=False)
        f.write("\n")

print(f"✓ Consolidated {len(df)} rows into {len(consolidated)} unique documents")
print(f"✓ Saved to {output_file}")
print(f"\nValid labels: {sorted(valid_labels)}")
print(f"\nSample of first 3 records:")
for i, record in enumerate(records[:3]):
    print(f"\nRecord {i + 1}:")
    print(f"  ID: {record['id']}")
    print(f"  Label: {record['label']}")
    print(f"  Text preview: {record['text'][:100]}...")
