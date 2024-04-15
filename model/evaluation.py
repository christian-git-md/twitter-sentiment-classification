from typing import Dict, List

import pandas as pd
from transformers import pipeline
import torch
import numpy as np

from model.config import val_file_name, label_definitions
from model.dataset import load_tweet_sentiment_csv_file, prepare_labels, clean_text
from model.metrics import compute_metrics
from model.utils import get_available_device


def eval_model(model_name: str) -> Dict:
    """
    Evaluates the given model name on the twitter sentiment analysis validation data in the following steps:
        - Load and preprocess the validation data
        - Load the model
        - Make predictions
        - Calculate metrics
        - Unload the model
    """
    val_df = load_tweet_sentiment_csv_file(val_file_name)
    val_df = prepare_labels(val_df, label_definitions)
    val_df["text"].apply(clean_text)
    text_data = list(val_df["text"])
    labels = list(val_df["label"])

    device = get_available_device()
    sentiment_model = pipeline(
        task="sentiment-analysis",
        model=model_name,
        return_all_scores=True,
        device=device,
    )

    preds = sentiment_model(text_data)

    raw_outputs = np.array(
        [[cls_output["score"] for cls_output in pred] for pred in preds]
    )
    metrics = compute_metrics((raw_outputs, labels))
    del sentiment_model
    torch.cuda.empty_cache()
    return metrics


def eval_model_summary(model_names: List[str]) -> pd.DataFrame:
    """
    Creates an evaluation summary of the list of given model names.
    """
    all_results = []
    column_names = ["model_name", "accuracy", "binary_accuracy", "f1"]
    for model_name in model_names:
        metrics = eval_model(model_name)
        del metrics["confusion_matrix"]
        metrics = {k: round(v, 3) for k, v in metrics.items()}
        metrics["model_name"] = model_name.split("/")[-1]
        all_results.append(metrics)
    results_table = pd.DataFrame(all_results, columns=column_names)
    return results_table
