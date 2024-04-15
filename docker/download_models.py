from transformers import pipeline

if __name__ == "__main__":
    model_names = [
        "distilbert/distilbert-base-uncased",
        "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        "christian-git-md/distilbert-base-uncased-finetuned-twitter-leak",
        "christian-git-md/distilbert-base-uncased-finetuned-twitter-noleak",
        "christian-git-md/distilbert-base-uncased-finetuned-twitter-noleak-noduplicates",
    ]

    for model_name in model_names:
        _ = pipeline(task="sentiment-analysis", model=model_name)
