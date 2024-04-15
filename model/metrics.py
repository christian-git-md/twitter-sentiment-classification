from typing import List, Tuple, Dict

from datasets import load_metric
import evaluate
import numpy as np


def calculate_binary_accuracy(logits: np.ndarray, labels: List[int]) -> float:
    """
    Calculate the binary accuracy of only the positive / negative class, to make a comparison to binary sentiment
    models. For this:
    - We calculate our prediction as the argmax only within the first two logits (negative and positive)
    - We evaluate on labels only that are negative or positive

    This is equivalent to how we evaluate a binary model.
    """
    binary_logits = logits[:, :2]
    binary_preds = np.argmax(binary_logits, axis=-1)
    num_correct = 0
    num_total = 0
    for p, l in zip(binary_preds, labels):
        if l in (0, 1):
            num_total += 1
            if l == p:
                num_correct += 1
    if num_total == 0:
        return 0.0
    else:
        return num_correct / num_total


def compute_metrics(eval_pred: Tuple[np.ndarray, List]) -> Dict:
    """
    Computes accuracy, binary accuracy macro f1 and confusion matrix from the given predictions and labels.
    """
    load_accuracy = load_metric("accuracy")
    load_f1 = load_metric("f1")
    cfm_metric = evaluate.load("BucketHeadP65/confusion_matrix")

    logits, labels = eval_pred
    calculate_binary_accuracy(logits, labels)
    predictions = np.argmax(logits, axis=-1)

    accuracy = load_accuracy.compute(predictions=predictions, references=labels)[
        "accuracy"
    ]
    f1 = load_f1.compute(predictions=predictions, references=labels, average="macro")[
        "f1"
    ]
    confusion_matrix = cfm_metric.compute(predictions=predictions, references=labels)
    binary_accuracy = calculate_binary_accuracy(logits, labels)
    return {
        "accuracy": accuracy,
        "binary_accuracy": binary_accuracy,
        "f1": f1,
        "confusion_matrix": confusion_matrix["confusion_matrix"].tolist(),
    }
