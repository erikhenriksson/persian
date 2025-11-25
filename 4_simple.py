import json
import os

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

torch.manual_seed(42)
np.random.seed(42)

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

all_valid_labels = sorted(
    list(labels_structure.keys())
    + [sublabel for sublabels in labels_structure.values() for sublabel in sublabels]
)


def load_jsonl_data(filepath):
    texts, labels = [], []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            texts.append([record["text"]])
            label_list = record["label"].split()
            binary_vector = [
                1.0 if label in label_list else 0.0 for label in all_valid_labels
            ]
            labels.append(binary_vector)
    return np.array(texts), np.array(labels, dtype=np.float32)


def create_dataset(X_split, y_split):
    return Dataset.from_dict(
        {
            "text": X_split.flatten().tolist(),
            "labels": y_split.tolist(),
        }
    )


def compute_metrics(p):
    true_labels = p.label_ids
    predictions = sigmoid(p.predictions)

    # Find optimal threshold
    best_threshold, best_f1 = 0.5, 0
    for threshold in np.arange(0.3, 0.7, 0.05):
        binary_predictions = predictions > threshold
        f1 = f1_score(true_labels, binary_predictions, average="micro")
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    binary_predictions = predictions > best_threshold
    report_dict = classification_report(
        true_labels,
        binary_predictions,
        target_names=all_valid_labels,
        output_dict=True,
    )

    print(
        "\n"
        + classification_report(
            true_labels, binary_predictions, target_names=all_valid_labels
        )
    )

    return {
        "f1_micro": report_dict["micro avg"]["f1-score"],
        "f1_macro": report_dict["macro avg"]["f1-score"],
        "precision_micro": report_dict["micro avg"]["precision"],
        "recall_micro": report_dict["micro avg"]["recall"],
        "threshold": best_threshold,
    }


# Load and split data
X, y = load_jsonl_data("data/persian_consolidated.jsonl")
print(f"Loaded {len(X)} documents with {len(all_valid_labels)} labels")

X_train, y_train, X_temp, y_temp = iterative_train_test_split(X, y, test_size=0.3)
X_dev, y_dev, X_test, y_test = iterative_train_test_split(X_temp, y_temp, test_size=0.5)

train_dataset = create_dataset(X_train, y_train)
dev_dataset = create_dataset(X_dev, y_dev)
test_dataset = create_dataset(X_test, y_test)

print(
    f"Split: {len(train_dataset)} train, {len(dev_dataset)} dev, {len(test_dataset)} test"
)

# Load model
model_name = "xlm-roberta-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(all_valid_labels),
    problem_type="multi_label_classification",
)


# Tokenize
def tokenize_function(examples):
    return tokenizer(
        examples["text"], truncation=True, max_length=512, padding="max_length"
    )


train_dataset = train_dataset.map(tokenize_function, batched=True)
dev_dataset = dev_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
dev_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# Train
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./results",
        num_train_epochs=15,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        learning_rate=1e-5,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        tf32=True,
        save_total_limit=2,
        seed=42,
    ),
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)

trainer.train()

# Evaluate and save
test_results = trainer.predict(test_dataset)

model.config.label2id = {label: i for i, label in enumerate(all_valid_labels)}
model.config.id2label = {i: label for i, label in enumerate(all_valid_labels)}

trainer.save_model("./persian_register_model")
tokenizer.save_pretrained("./persian_register_model")

with open("./persian_register_model/test_results.json", "w") as f:
    json.dump(
        {k.replace("test_", ""): float(v) for k, v in test_results.metrics.items()},
        f,
        indent=2,
    )

print("Done!")
