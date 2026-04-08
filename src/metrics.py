from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score
)
import numpy as np


def compute_metrics(true_labels, pred_labels):
    """
    true_labels: 2D numpy array of shape (num_examples, num_labels)
    pred_labels: 2D numpy array of shape (num_examples, num_labels)
    both contain binary values (0 or 1)
    """
    micro_f1 = f1_score(true_labels, pred_labels, average="micro", zero_division=0)
    macro_f1 = f1_score(true_labels, pred_labels, average="macro", zero_division=0)
    precision = precision_score(true_labels, pred_labels, average="micro", zero_division=0)
    recall = recall_score(true_labels, pred_labels, average="micro", zero_division=0)

    return {
        "micro_f1": round(micro_f1, 4),
        "macro_f1": round(macro_f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4)
    }


def compute_per_label_f1(true_labels, pred_labels, label_names):
    """
    Returns a dict of {label_name: f1_score} for each of the 28 labels.
    """
    scores = f1_score(true_labels, pred_labels, average=None, zero_division=0)
    return {label_names[i]: round(scores[i], 4) for i in range(len(label_names))}