import json
import os

os.environ["HF_HOME"] = "./hf_home"

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
    TrainerCallback,
    TrainingArguments,
)

torch.manual_seed(44)
np.random.seed(44)

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

print("=" * 80)
print("LABEL STRUCTURE ANALYSIS")
print("=" * 80)
print(f"Total labels: {len(all_valid_labels)}")
print(f"Labels: {all_valid_labels}")
print()


def load_jsonl_data(filepath):
    texts, labels = [], []
    raw_labels_list = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            texts.append(record["text"])  # Fixed: no nested list
            label_list = record["label"].split()
            raw_labels_list.append(label_list)

            binary_vector = [
                1.0 if label in label_list else 0.0 for label in all_valid_labels
            ]  # Must be float for BCEWithLogitsLoss
            labels.append(binary_vector)

    return np.array(texts), np.array(labels, dtype=np.float32), raw_labels_list


def analyze_label_distribution(y, raw_labels, split_name=""):
    print(f"\n{'=' * 80}")
    print(f"LABEL DISTRIBUTION ANALYSIS - {split_name}")
    print(f"{'=' * 80}")

    # Basic stats
    print(f"Total samples: {len(y)}")
    print(f"Total labels: {len(all_valid_labels)}")

    # Labels per sample
    labels_per_sample = y.sum(axis=1)
    print(f"\nLabels per sample:")
    print(f"  Mean: {labels_per_sample.mean():.2f}")
    print(f"  Median: {np.median(labels_per_sample):.2f}")
    print(f"  Min: {labels_per_sample.min()}")
    print(f"  Max: {labels_per_sample.max()}")
    print(f"  Samples with 0 labels: {(labels_per_sample == 0).sum()}")

    # Label frequency
    label_counts = y.sum(axis=0)
    print(f"\nLabel frequencies:")
    for i, (label, count) in enumerate(zip(all_valid_labels, label_counts)):
        percentage = (count / len(y)) * 100
        print(f"  {label:6s}: {int(count):5d} ({percentage:5.2f}%)")

    # Most imbalanced labels
    print(f"\nMost imbalanced labels:")
    sorted_indices = np.argsort(label_counts)
    print(
        f"  Rarest: {all_valid_labels[sorted_indices[0]]} ({int(label_counts[sorted_indices[0]])} samples)"
    )
    print(
        f"  Most common: {all_valid_labels[sorted_indices[-1]]} ({int(label_counts[sorted_indices[-1]])} samples)"
    )
    print(
        f"  Imbalance ratio: {label_counts[sorted_indices[-1]] / max(label_counts[sorted_indices[0]], 1):.1f}x"
    )

    # Check for labels that never appear
    missing_labels = [
        all_valid_labels[i] for i, count in enumerate(label_counts) if count == 0
    ]
    if missing_labels:
        print(f"\n⚠️  WARNING: Labels that never appear: {missing_labels}")

    # Sample label combinations
    print(f"\nMost common label combinations (top 10):")
    from collections import Counter

    combo_counter = Counter([tuple(labels) for labels in raw_labels])
    for combo, count in combo_counter.most_common(10):
        print(f"  {' '.join(combo):30s}: {count:4d}")

    print()


def create_dataset(X_split, y_split):
    return Dataset.from_dict(
        {
            "text": X_split.tolist(),  # Fixed: already flat
            "labels": y_split.tolist(),
        }
    )


# Custom callback for detailed logging
class DetailedLoggingCallback(TrainerCallback):
    def __init__(self):
        self.epoch_losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            print(f"\n[Step {state.global_step}]", end=" ")
            for key, value in logs.items():
                if isinstance(value, float):
                    print(f"{key}: {value:.4f}", end=" | ")
            print()

    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"\n{'=' * 60}")
        print(f"EPOCH {int(state.epoch)} COMPLETED")
        print(f"{'=' * 60}")


def compute_metrics(p):
    true_labels = p.label_ids
    predictions = sigmoid(p.predictions)

    print(f"\n{'=' * 80}")
    print("PREDICTION ANALYSIS")
    print(f"{'=' * 80}")

    # Analyze raw predictions
    print(f"Raw logits shape: {p.predictions.shape}")
    print(f"Raw logits stats:")
    print(f"  Mean: {p.predictions.mean():.4f}, Std: {p.predictions.std():.4f}")
    print(f"  Min: {p.predictions.min():.4f}, Max: {p.predictions.max():.4f}")

    print(f"\nSigmoid probabilities stats:")
    print(f"  Mean: {predictions.mean():.4f}, Std: {predictions.std():.4f}")
    print(f"  Min: {predictions.min():.4f}, Max: {predictions.max():.4f}")

    # Check if model is predicting anything
    at_05 = (predictions > 0.5).sum()
    at_03 = (predictions > 0.3).sum()
    at_07 = (predictions > 0.7).sum()
    total_predictions = predictions.shape[0] * predictions.shape[1]

    print(f"\nPredictions above threshold:")
    print(f"  >0.3: {at_03} ({at_03 / total_predictions * 100:.2f}%)")
    print(f"  >0.5: {at_05} ({at_05 / total_predictions * 100:.2f}%)")
    print(f"  >0.7: {at_07} ({at_07 / total_predictions * 100:.2f}%)")

    # Check per-label predictions
    print(f"\nPer-label prediction rates (at 0.5 threshold):")
    for i, label in enumerate(all_valid_labels):
        pred_count = (predictions[:, i] > 0.5).sum()
        true_count = true_labels[:, i].sum()
        print(
            f"  {label:6s}: predicted {int(pred_count):4d}, actual {int(true_count):4d}"
        )

    # Find optimal threshold
    best_threshold, best_f1 = 0.5, 0
    threshold_results = []
    for threshold in np.arange(0.1, 0.9, 0.05):
        binary_predictions = predictions > threshold
        f1 = f1_score(true_labels, binary_predictions, average="micro")
        threshold_results.append((threshold, f1))
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print(f"\nThreshold search results:")
    for thresh, f1 in threshold_results[:5]:
        print(f"  Threshold {thresh:.2f}: F1 = {f1:.4f}")
    print(f"  ...")
    print(f"  Best threshold: {best_threshold:.2f} (F1 = {best_f1:.4f})")

    binary_predictions = predictions > best_threshold

    # Check if model predicts all zeros
    predictions_per_sample = binary_predictions.sum(axis=1)
    zero_prediction_samples = (predictions_per_sample == 0).sum()
    print(
        f"\nSamples with zero predictions: {zero_prediction_samples} / {len(binary_predictions)}"
    )

    report_dict = classification_report(
        true_labels,
        binary_predictions,
        target_names=all_valid_labels,
        output_dict=True,
        zero_division=0,
    )

    print(
        "\n"
        + classification_report(
            true_labels,
            binary_predictions,
            target_names=all_valid_labels,
            zero_division=0,
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
print("=" * 80)
print("LOADING DATA")
print("=" * 80)

X, y, raw_labels = load_jsonl_data("data/persian_consolidated.jsonl")
print(f"Loaded {len(X)} documents with {len(all_valid_labels)} labels")

# Analyze full dataset
analyze_label_distribution(y, raw_labels, "FULL DATASET")

# Check for data quality issues
print("=" * 80)
print("DATA QUALITY CHECKS")
print("=" * 80)
print(f"Text samples (first 3):")
for i in range(min(3, len(X))):
    print(f"  [{i}] Length: {len(X[i])} chars, Labels: {raw_labels[i]}")
    print(f"      Preview: {X[i][:100]}...")
print()

empty_texts = sum(1 for text in X if len(text.strip()) == 0)
print(f"Empty texts: {empty_texts}")
print(f"Mean text length: {np.mean([len(text) for text in X]):.0f} chars")
print(f"Median text length: {np.median([len(text) for text in X]):.0f} chars")
print()

# Store raw labels mapping before reshaping
text_to_labels = {text: raw_labels[i] for i, text in enumerate(X)}

# Split data
# iterative_train_test_split requires 2D arrays
X_reshaped = X.reshape(-1, 1)
X_train, y_train, X_temp, y_temp = iterative_train_test_split(
    X_reshaped, y, test_size=0.3
)
X_dev, y_dev, X_test, y_test = iterative_train_test_split(X_temp, y_temp, test_size=0.5)

# Flatten X arrays back to 1D
X_train = X_train.flatten()
X_dev = X_dev.flatten()
X_test = X_test.flatten()

# Get corresponding raw labels for splits by matching texts
train_raw_labels = [text_to_labels[text] for text in X_train]
dev_raw_labels = [text_to_labels[text] for text in X_dev]
test_raw_labels = [text_to_labels[text] for text in X_test]

analyze_label_distribution(y_train, train_raw_labels, "TRAIN SET")
analyze_label_distribution(y_dev, dev_raw_labels, "DEV SET")
analyze_label_distribution(y_test, test_raw_labels, "TEST SET")

train_dataset = create_dataset(X_train, y_train)
dev_dataset = create_dataset(X_dev, y_dev)
test_dataset = create_dataset(X_test, y_test)

print(
    f"Split: {len(train_dataset)} train, {len(dev_dataset)} dev, {len(test_dataset)} test"
)

# Verify dataset format
print("\n" + "=" * 80)
print("DATASET FORMAT VERIFICATION")
print("=" * 80)
sample = train_dataset[0]
print(f"Sample keys: {sample.keys()}")
print(f"Text type: {type(sample['text'])}, length: {len(sample['text'])}")
print(f"Labels type: {type(sample['labels'])}, shape: {len(sample['labels'])}")
print(f"Labels dtype: {type(sample['labels'][0])}")
print(f"Sample labels: {sample['labels']}")
print(f"Sample text preview: {sample['text'][:100]}...")
print()

# Load model
print("=" * 80)
print("LOADING MODEL")
print("=" * 80)
model_name = "BAAI/bge-m3-retromae"
print(f"Model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
print(f"Tokenizer model max length: {tokenizer.model_max_length}")

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(all_valid_labels),
    problem_type="multi_label_classification",
)

print(f"Model num_labels: {model.config.num_labels}")
print(f"Model problem_type: {model.config.problem_type}")
print()


# Tokenize
def tokenize_function(examples):
    result = tokenizer(
        examples["text"], truncation=True, max_length=1024, padding="max_length"
    )
    return result


print("=" * 80)
print("TOKENIZING DATASETS")
print("=" * 80)

train_dataset = train_dataset.map(tokenize_function, batched=True)
dev_dataset = dev_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Verify tokenization
print("Tokenization verification:")
tokenized_sample = train_dataset[0]
print(f"  Input IDs shape: {len(tokenized_sample['input_ids'])}")
print(f"  Attention mask shape: {len(tokenized_sample['attention_mask'])}")
print(f"  Non-padding tokens: {sum(tokenized_sample['attention_mask'])}")
print()

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
dev_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# Final verification of torch tensors
print("Torch tensor verification:")
torch_sample = train_dataset[0]
print(
    f"  input_ids: shape={torch_sample['input_ids'].shape}, dtype={torch_sample['input_ids'].dtype}"
)
print(
    f"  attention_mask: shape={torch_sample['attention_mask'].shape}, dtype={torch_sample['attention_mask'].dtype}"
)
print(
    f"  labels: shape={torch_sample['labels'].shape}, dtype={torch_sample['labels'].dtype}"
)
print(f"  labels values: {torch_sample['labels']}")
print()

# Train
print("=" * 80)
print("STARTING TRAINING")
print("=" * 80)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=15,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,  # Changed to more standard LR
    weight_decay=0.01,  # Added weight decay
    warmup_ratio=0.1,  # Added warmup
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1_micro",
    greater_is_better=True,
    tf32=True,
    save_total_limit=2,
    seed=42,
)

print(f"Training arguments:")
print(f"  Learning rate: {training_args.learning_rate}")
print(f"  Batch size: {training_args.per_device_train_batch_size}")
print(f"  Epochs: {training_args.num_train_epochs}")
print(f"  Warmup ratio: {training_args.warmup_ratio}")
print(f"  Weight decay: {training_args.weight_decay}")
print()

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    compute_metrics=compute_metrics,
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=5),
        DetailedLoggingCallback(),
    ],
)

trainer.train()

# Evaluate and save
print("\n" + "=" * 80)
print("FINAL TEST SET EVALUATION")
print("=" * 80)

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

print("\n" + "=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)
print(f"Model saved to: ./persian_register_model")
print(f"Test results saved to: ./persian_register_model/test_results.json")
print("Done!")
