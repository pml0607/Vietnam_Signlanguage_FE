import sys
import os
import fire
import torch
import seaborn as sns
import numpy as np

from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

plt.switch_backend('Agg')

sys.path.append(os.path.dirname(__file__))

# Model & Trainer
from Model.model import VideoMAEForVideoClassification
from transformers import TrainingArguments, Trainer

# Dataset
from dataset import get_dataset, collate_fn
from dataset.utils import get_label_map

# Metric
from sklearn.metrics import confusion_matrix, classification_report


def get_predictions(trainer, dataset):
    outputs = trainer.predict(dataset)
    y_pred = np.argmax(outputs.predictions, axis=1)
    y_true = outputs.label_ids
    return y_pred, y_true


def plot_confusion_matrix(y_true, y_pred, labels, output_file="/work/21013187/linh/Vietnam_Signlanguage_FE/Transformer/rgb+landmark/confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    fig.tight_layout()
    fig.savefig(output_file)
    print(f"✅ Confusion matrix saved to {output_file}")
    plt.close(fig)
    return cm


def save_high_accuracy_classes(cm, id2label, output_path="high_accuracy_classes.txt", threshold=0.88):
    per_class_acc = {}
    high_acc_classes = []

    for i in range(len(cm)):
        correct = cm[i, i]
        total = cm[i].sum()
        acc = correct / total if total > 0 else 0.0
        label_name = id2label[i]
        per_class_acc[label_name] = acc
        if acc >= threshold:
            high_acc_classes.append((label_name, acc))

    with open(output_path, "w") as f:
        for label_name, acc in sorted(high_acc_classes, key=lambda x: -x[1]):
            f.write(f"{label_name}: {acc*100:.2f}%\n")

    print(f"✅ Saved {len(high_acc_classes)} classes with accuracy > {threshold*100:.0f}% to {output_path}")


def main(dataset_root_path="/work/21013187/SAM-SLR-v2/data/rgb",
         model_ckpt_path="/work/21013187/linh/Vietnam_Signlanguage_FE/Transformer/rgb+landmark/videomae-base-finetuned/checkpoint-4185",
         model_ckpt_name="MCG-NJU/videomae-base",
         batch_size=20):

    dataset_root_path = Path(dataset_root_path)
    label2id, id2label = get_label_map(dataset_root_path)

    # Load test dataset
    _, _, test_dataset = get_dataset(dataset_root_path)

    # Load model
    model = VideoMAEForVideoClassification.from_pretrained(
        model_ckpt_path,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True
    )

    args = TrainingArguments(
        output_dir="./tmp_test",
        per_device_eval_batch_size=batch_size,
        dataloader_drop_last=False,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        data_collator=collate_fn,
    )

    print("🔍 Running prediction on test set...")
    y_pred, y_true = get_predictions(trainer, test_dataset)

    labels = [id2label[i] for i in range(len(id2label))]

    # Plot confusion matrix and compute per-class accuracy
    cm = plot_confusion_matrix(y_true, y_pred, labels)

    # Save classes with accuracy > 88%
    save_high_accuracy_classes(cm, id2label, output_path="/work/21013187/linh/Vietnam_Signlanguage_FE/Transformer/rgb+landmark/high_accuracy_classes.txt", threshold=0.88)


if __name__ == "__main__":
    fire.Fire(main)
