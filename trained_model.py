import numpy as np
import pandas as pd
import torch
from sklearn import metrics
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset


data = load_dataset(
    'csv',
    data_files={
        'train': '/Users/shubham/Desktop/Movie_Genere/train-00000-of-00001-b943ea66e0040b18.csv',
        'test': '/Users/shubham/Desktop/Movie_Genere/test-00000-of-00001-35e9a9274361daed.csv'
    },
    cache_dir='movie-dataset/data'
)


def clean_text(text: str):
    text = text.lower()
    text = text.strip()
    return text


def create_input(example: dict) -> dict:
    movie_name = example["movie_name"].lower()
    synopsis = example["synopsis"].lower()

    movie_name = clean_text(movie_name)
    synopsis = clean_text(synopsis)

    example['final_text'] = f'movie name - {movie_name}, synopsis - {synopsis}'

    return example


data = data.map(create_input)

data.save_to_disk('cleaned_data/')

data = data.class_encode_column('genre')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True, do_lower_case=True)
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(data['train'].features['genre']._int2str),
).to(device)


max_len = 0
for example in data['train']:
    input_ids = tokenizer.encode(example['final_text'], add_special_tokens=True)
    max_len = max(max_len, len(input_ids))

print(f'Max sentence length: {max_len}')


class ClassificationDataset:
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        text = str(self.data[item]["final_text"])
        target = int(self.data[item]["genre"])
        inputs = self.tokenizer(text, max_length=max_len, padding="max_length", truncation=True)

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "labels": torch.tensor(target, dtype=torch.long),
        }


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = metrics.accuracy_score(labels, predictions)
    return {"accuracy": accuracy}


def train(ds):
    ds_train = ds["train"]
    ds_test = ds["test"]
    temp_ds = ds_train.train_test_split(test_size=0.1, stratify_by_column="genre")
    ds_train = temp_ds["train"]
    ds_val = temp_ds["test"]
    
    train_dataset = ClassificationDataset(ds_train, tokenizer)
    valid_dataset = ClassificationDataset(ds_val, tokenizer)
    test_dataset = ClassificationDataset(ds_test, tokenizer)

    args = TrainingArguments(
        "model",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=1,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none",
        save_total_limit=1
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    
    preds = trainer.predict(test_dataset).predictions
    preds = np.argmax(preds, axis=1)
    
    submission = pd.DataFrame({"id": ds_test["id"], "genre": preds})
    submission.loc[:, "genre"] = submission.genre.apply(lambda x: ds_train.features["genre"].int2str(x))
    submission.to_csv(f"submission.csv", index=False)


train(data)
