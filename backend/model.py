# Install required packages
#!pip install transformers torch datasets

# Import necessary libraries
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# Load a multilingual model (e.g., DistilBERT base)
model_name = "distilbert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Load your dataset (using a sample dataset for illustration)
dataset = load_dataset("xnli", "en")  # Replace with your curated educational content in multiple languages

# Tokenize the dataset
def preprocess(data):
    return tokenizer(data['premise'], padding="max_length", truncation=True)

tokenized_dataset = dataset.map(preprocess, batched=True)

# Fine-tune the model
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
)

trainer.train()
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")
