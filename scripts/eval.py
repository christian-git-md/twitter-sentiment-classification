from model.evaluation import eval_model_summary

if __name__ == "__main__":
    model_names = [
        "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        "christian-git-md/distilbert-base-uncased-finetuned-twitter-leak",
        "christian-git-md/distilbert-base-uncased-finetuned-twitter-noleak",
        "christian-git-md/distilbert-base-uncased-finetuned-twitter-noleak-noduplicates",
    ]
    print(eval_model_summary(model_names))
