import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    get_scheduler
)
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import evaluate
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from peft import PeftModel
from transformers import AutoModelForSequenceClassification

# Load dataset
dataset = load_dataset("dair-ai/emotion", "unsplit")
full_dataset = dataset["train"]

# Convert to pandas for easier splitting
df = full_dataset.to_pandas()

# Perform train-test split (80-20 split)
train_df, eval_df = train_test_split(
    df, 
    test_size=0.2, 
    random_state=42, 
    stratify=df['label']
)

from datasets import Dataset
train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(eval_df)

# Initialize tokenizer
model_checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Tokenization function
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

# Tokenize datasets
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_eval = eval_dataset.map(tokenize_function, batched=True)

# Convert to PyTorch format
tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
tokenized_eval.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Load model
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint,
    num_labels=6,
)

# Training arguments
training_args = TrainingArguments(
    output_dir="",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to=["none"],
)

# Compute metrics function
metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels, average='weighted')

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Save model to Hugging Face Hub (make sure to login first using HuggingFace CLI)
trainer.push_to_hub()

def emotion_classification(text):
    base_model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels = 6)
    model = PeftModel.from_pretrained(base_model, "katsuchi/bert-dair-ai-emotion")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    token = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        output = model(token['input_ids'])
        output.logits
        output = torch.softmax(output.logits, dim=-1)

    emotion = ['sadness','joy','love','anger','fear','surprise']

    idx = torch.argmax(output, dim=-1).item()
    classes = emotion[idx]
    return classes