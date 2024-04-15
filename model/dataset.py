from typing import Dict, Tuple

import pandas as pd
import re
from datasets import Dataset
from transformers import PreTrainedTokenizer


def load_tweet_sentiment_csv_file(file_name: str) -> pd.DataFrame:
    """
    Load a sentiment file for the coding challenge from the disk and add some fitting column names.
    """
    column_names = ["tweet_id", "tag", "sentiment", "text"]
    return pd.read_csv(file_name, header=None, names=column_names)


def clean_text(text: str):
    """
    Removes some patterns from the data, that have been already cleaned in the given training data, such that training
    and validation data are more consistent.
    """
    text = re.sub(r"\n", "", text)  # \n only appears in validation data
    text: re.sub(r"#\w+", "", text)  # Remove hashtags
    text: re.sub(r"http\S+|www\.\S+", "", text)  # Remove URLs
    return text


def prepare_text_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Does some basic cleaning on the data: removes invalid or too short samples and unwanted text
    segments.
    """
    df = df[~df["text"].isna()]
    df = df[df["text"].str.count("[a-zA-Z]") >= 5]
    df["text"] = df["text"].apply(clean_text)
    return df


def prepare_labels(df: pd.DataFrame, label_map: Dict[str, int]):
    """Creates the column for the labels, created by a map from the "sentiment" entry."""
    df["label"] = [label_map[label_name] for label_name in df["sentiment"]]
    return df


def remove_leaked_training_samples(
    train_df: pd.DataFrame, val_df: pd.DataFrame
) -> pd.DataFrame:
    """Removes the samples that appear in both training and validation data from the training data."""
    return train_df[~train_df["tweet_id"].isin(val_df["tweet_id"])]


def remove_augmented_training_samples(train_df: pd.DataFrame) -> pd.DataFrame:
    """Removes all augmented versions of a training sample, except the first one."""
    train_df = train_df[~train_df.duplicated(subset="tweet_id", keep="first")]
    return train_df


def create_datasets(
    train_file_name: str,
    val_file_name: str,
    label_definitions: Dict[str, int],
    tokenizer: PreTrainedTokenizer,
) -> Tuple[Dataset, Dataset]:
    """
    Loads the string data from the disk, cleans and preprocesses the entries, converts to a Dataset and applies the
    given tokenizer.
    """
    train_df = load_tweet_sentiment_csv_file(train_file_name)
    val_df = load_tweet_sentiment_csv_file(val_file_name)

    train_df = remove_leaked_training_samples(train_df, val_df)
    train_df = remove_augmented_training_samples(train_df)

    train_df = prepare_text_data(train_df)
    val_df = prepare_text_data(val_df)

    train_df = prepare_labels(train_df, label_definitions)
    val_df = prepare_labels(val_df, label_definitions)

    train_dataset = Dataset.from_pandas(train_df[["text", "label"]])
    val_dataset = Dataset.from_pandas(val_df[["text", "label"]])

    tokenizer_fn = lambda examples: tokenizer(examples["text"], truncation=True)
    tokenized_train = train_dataset.map(tokenizer_fn, batched=True)
    tokenized_test = val_dataset.map(tokenizer_fn, batched=True)
    return tokenized_train, tokenized_test
