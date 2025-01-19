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
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    PeftConfig,
    PeftModel
)
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

# Load base model
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint,
    num_labels=6,
)

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["query", "value"],
    bias="none",
    inference_mode=False,
)

# Apply LoRA config to the model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Training arguments
training_args = TrainingArguments(
    output_dir="bert-dair-ai-emotion", 
    num_train_epochs=3,
    gradient_accumulation_steps=16,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps= 100,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=20,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    push_to_hub=True,
    hub_model_id="katsuchi/bert-dair-ai-emotion"
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

# Save model and push to hub
trainer.model.save_pretrained("bert-dair-ai-emotion")
trainer.push_to_hub("katsuchi/bert-dair-ai-emotion")
