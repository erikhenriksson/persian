import json
import os
import re
from collections import Counter

# Set cache directories to local
os.environ["TRANSFORMERS_CACHE"] = "./model_cache"
os.environ["HF_HOME"] = "./model_cache"

os.makedirs("./model_cache", exist_ok=True)

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, hamming_loss
from skmultilearn.model_selection import iterative_train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# Define valid labels structure
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

# Create ordered list of all valid labels
all_valid_labels = sorted(list(labels_structure.keys()))
for main_label in sorted(labels_structure.keys()):
    all_valid_labels.extend(sorted(labels_structure[main_label]))

print(f"Valid labels ({len(all_valid_labels)}): {all_valid_labels}")

# Load JSONL data
print("\nLoading data from data/persian_consolidated.jsonl...")
texts = []
labels = []

with open("data/persian_consolidated.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        record = json.loads(line)
        texts.append(record["text"])

        # Convert space-separated labels to binary vector
        label_list = record["label"].split()
        binary_vector = [1 if label in label_list else 0 for label in all_valid_labels]
        labels.append(binary_vector)

print(f"Loaded {len(texts)} documents")

# Check label distribution
label_counts = Counter()
for label_vec in labels:
    for i, val in enumerate(label_vec):
        if val == 1:
            label_counts[all_valid_labels[i]] += 1

print("\nLabel distribution:")
for label in all_valid_labels:
    print(f"  {label}: {label_counts[label]:,}")

# Convert to numpy arrays for scikit-multilearn
X = np.array(texts).reshape(-1, 1)  # reshape for iterative_train_test_split
y = np.array(labels)

print(f"\nData shape: X={X.shape}, y={y.shape}")

# Multi-label stratified split: train=70%, temp=30%
X_train, y_train, X_temp, y_temp = iterative_train_test_split(X, y, test_size=0.3)

# Split temp into dev=50% and test=50% (each 15% of total)
X_dev, y_dev, X_test, y_test = iterative_train_test_split(X_temp, y_temp, test_size=0.5)

# Extract texts from reshaped arrays
X_train = X_train.flatten().tolist()
X_dev = X_dev.flatten().tolist()
X_test = X_test.flatten().tolist()

# Convert labels to tensors
y_train = torch.tensor(
    y_train.toarray() if hasattr(y_train, "toarray") else y_train, dtype=torch.float32
)
y_dev = torch.tensor(
    y_dev.toarray() if hasattr(y_dev, "toarray") else y_dev, dtype=torch.float32
)
y_test = torch.tensor(
    y_test.toarray() if hasattr(y_test, "toarray") else y_test, dtype=torch.float32
)

print(f"\nMulti-label stratified split sizes:")
print(f"  Train: {len(X_train):,}")
print(f"  Dev:   {len(X_dev):,}")
print(f"  Test:  {len(X_test):,}")

# Verify label distribution in splits
print("\nLabel distribution per split:")
for split_name, split_labels in [("Train", y_train), ("Dev", y_dev), ("Test", y_test)]:
    print(f"\n{split_name}:")
    for i, label in enumerate(all_valid_labels):
        count = split_labels[:, i].sum().item()
        print(f"  {label}: {int(count):,}")

# Load ModernBERT
model_name = "answerdotai/ModernBERT-base"
print(f"\nLoading model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(all_valid_labels),
    problem_type="multi_label_classification",
    torch_dtype=torch.bfloat16,
)
print("Model loaded for multi-label classification")


# Dataset class
class MultiLabelDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": self.labels[idx],
        }


train_dataset = MultiLabelDataset(X_train, y_train, tokenizer)
dev_dataset = MultiLabelDataset(X_dev, y_dev, tokenizer)
test_dataset = MultiLabelDataset(X_test, y_test, tokenizer)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    bf16=True,
    report_to="none",
)


# Metric function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Apply sigmoid and threshold at 0.5
    predictions = torch.sigmoid(torch.tensor(predictions)) > 0.5
    predictions = predictions.float().numpy()

    # Hamming loss (fraction of wrong labels)
    h_loss = hamming_loss(labels, predictions)

    return {"hamming_loss": h_loss}


# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    compute_metrics=compute_metrics,
)

print("\n" + "=" * 60)
print("Training ModernBERT on Persian Register Classification...")
print("=" * 60)
trainer.train()

# Evaluate on test set
print("\n" + "=" * 60)
print("Evaluating on test set...")
print("=" * 60)
predictions = trainer.predict(test_dataset)
y_pred = torch.sigmoid(torch.tensor(predictions.predictions)) > 0.5
y_pred = y_pred.float().numpy()

print("\n" + "=" * 60)
print("TEST SET RESULTS")
print("=" * 60)
print(f"Hamming Loss: {hamming_loss(y_test.numpy(), y_pred):.4f}")
print()

print("Per-label metrics:")
for i, label in enumerate(all_valid_labels):
    y_true_label = y_test[:, i].numpy()
    y_pred_label = y_pred[:, i]

    if y_true_label.sum() > 0:  # Only show labels that appear in test set
        acc = accuracy_score(y_true_label, y_pred_label)
        print(f"  {label:6s}: accuracy={acc:.4f}, support={int(y_true_label.sum())}")

# Save model
print("\n" + "=" * 60)
print("Saving model to ./persian_register_model")
print("=" * 60)
trainer.save_model("./persian_register_model")
tokenizer.save_pretrained("./persian_register_model")

# Save label mapping
with open("./persian_register_model/label_mapping.json", "w") as f:
    json.dump({"labels": all_valid_labels}, f, indent=2)

print("Done!")
