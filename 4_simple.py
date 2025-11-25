import json
import os

# Set cache directories to local
os.environ["HF_HOME"] = "./model_cache"

import numpy as np
import torch
from datasets import Dataset
from scipy.special import expit as sigmoid
from sklearn.metrics import classification_report, f1_score
from skmultilearn.model_selection import iterative_train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

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

# Create ordered list of all valid labels (main labels + sublabels)
all_valid_labels = sorted(
    list(labels_structure.keys())
    + [sublabel for sublabels in labels_structure.values() for sublabel in sublabels]
)


def load_jsonl_data(filepath):
    """Load JSONL file and return texts and label vectors as numpy arrays"""
    texts = []
    labels = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            texts.append([record["text"]])

            # Convert space-separated labels to binary vector
            label_list = record["label"].split()
            binary_vector = [
                1.0 if label in label_list else 0.0 for label in all_valid_labels
            ]
            labels.append(binary_vector)

    X = np.array(texts)
    y = np.array(labels, dtype=np.float32)
    return X, y


# Load data
X, y = load_jsonl_data("data/persian_consolidated.jsonl")
print(f"Loaded {len(X)} documents")

# Check label distribution
label_counts = {label: 0 for label in all_valid_labels}
for label_vec in y:
    for i, val in enumerate(label_vec):
        if val == 1.0:
            label_counts[all_valid_labels[i]] += 1

print("\nLabel distribution:")
for label in all_valid_labels:
    print(f"  {label}: {label_counts[label]:,}")

# Multi-label stratified split: train=70%, temp=30%
X_train, y_train, X_temp, y_temp = iterative_train_test_split(X, y, test_size=0.3)

# Split temp into dev=50% and test=50% (each 15% of total)
X_dev, y_dev, X_test, y_test = iterative_train_test_split(X_temp, y_temp, test_size=0.5)


# Helper function to convert split data to HuggingFace Dataset
def create_dataset(X_split, y_split):
    return Dataset.from_dict(
        {
            "text": X_split.flatten().tolist(),
            "labels": y_split.tolist(),
        }
    )


# Convert to HuggingFace datasets
train_dataset = create_dataset(X_train, y_train)
dev_dataset = create_dataset(X_dev, y_dev)
test_dataset = create_dataset(X_test, y_test)

print(f"\nStratified multi-label split sizes:")
print(f"  Train: {len(train_dataset):,}")
print(f"  Dev:   {len(dev_dataset):,}")
print(f"  Test:  {len(test_dataset):,}")

# Load XLM-RoBERTa-large
model_name = "xlm-roberta-large"
print(f"\nLoading model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(all_valid_labels),
    problem_type="multi_label_classification",
)
print("Model loaded for multi-label classification")


# Tokenization function
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length",
    )


# Tokenize datasets
print("\nTokenizing datasets...")
train_dataset = train_dataset.map(tokenize_function, batched=True)
dev_dataset = dev_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Set format for PyTorch
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
dev_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])


# Metric function with threshold optimization
def compute_metrics(p):
    true_labels = p.label_ids
    predictions = sigmoid(p.predictions)

    # Find optimal threshold based on micro F1
    best_threshold, best_f1 = 0.5, 0
    for threshold in np.arange(0.3, 0.7, 0.05):
        binary_predictions = predictions > threshold
        f1 = f1_score(true_labels, binary_predictions, average="micro")
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    binary_predictions = predictions > best_threshold

    # Get classification report as dict
    report_dict = classification_report(
        true_labels,
        binary_predictions,
        target_names=all_valid_labels,
        output_dict=True,
    )

    # Print the report for visibility
    print("\n" + "=" * 60)
    print(
        classification_report(
            true_labels,
            binary_predictions,
            target_names=all_valid_labels,
        )
    )
    print("=" * 60)

    # Return dict with summary metrics (needed by Trainer)
    return {
        "f1_micro": report_dict["micro avg"]["f1-score"],
        "f1_macro": report_dict["macro avg"]["f1-score"],
        "precision_micro": report_dict["micro avg"]["precision"],
        "recall_micro": report_dict["micro avg"]["recall"],
        "threshold": best_threshold,
    }


# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=15,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    learning_rate=5e-5,
    load_best_model_at_end=True,
    tf32=True,
    save_total_limit=2,
    seed=42,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
)

# Initialize standard Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)

trainer.train()

# Evaluate on test set
print("\n" + "=" * 60)
print("FINAL EVALUATION ON TEST SET")
print("=" * 60)

# Use trainer.predict which will call compute_metrics automatically
test_results = trainer.predict(test_dataset)
test_metrics = test_results.metrics

# Save model
print("\n" + "=" * 60)
print("Saving model to ./persian_register_model")
print("=" * 60)
trainer.save_model("./persian_register_model")
tokenizer.save_pretrained("./persian_register_model")

# Save label mapping and results
with open("./persian_register_model/label_mapping.json", "w") as f:
    json.dump({"labels": all_valid_labels}, f, indent=2)

results = {
    "test_f1_micro": float(test_metrics["test_f1_micro"]),
    "test_f1_macro": float(test_metrics["test_f1_macro"]),
    "test_precision_micro": float(test_metrics["test_precision_micro"]),
    "test_recall_micro": float(test_metrics["test_recall_micro"]),
    "optimal_threshold": float(test_metrics["test_threshold"]),
}

with open("./persian_register_model/test_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Done!")
