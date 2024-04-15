from model.config import train_file_name, val_file_name, label_definitions, output_dir
from model.dataset import create_datasets
from model.metrics import compute_metrics
from transformers import (
    TrainingArguments,
    AutoModelForSequenceClassification,
    Trainer,
    DataCollatorWithPadding,
    AutoTokenizer,
)


def train():
    """
    Main training pipeline, load and pre-process the data, initialize the trainer from a pretrained model and start
    the training loop.
    """
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    train_dataset, val_dataset = create_datasets(
        train_file_name, val_file_name, label_definitions, tokenizer
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    id2label = {id: label for label, id in label_definitions.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=4,
        id2label=id2label,
        label2id=label_definitions,
        ignore_mismatched_sizes=True,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        save_strategy="epoch",
        push_to_hub=False,
        evaluation_strategy="epoch",
        eval_steps=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
