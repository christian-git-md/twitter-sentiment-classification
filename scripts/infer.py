from transformers import pipeline

sentiment_model = pipeline(
    task="sentiment-analysis",
    model="christian-git-md/distilbert-base-uncased-finetuned-twitter-noleak",
)
print(sentiment_model(["This game was great"]))
