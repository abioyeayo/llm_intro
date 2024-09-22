# Load IMDb Dataset
from datasets import load_dataset
dataset = load_dataset('imdb')

# Preprocess Data (Tokenize)
from transformers import DistilBertTokenizerFast

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased',clean_up_tokenization_spaces=True)

def preprocess_data(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

encoded_dataset = dataset.map(preprocess_data, batched=True)
encoded_dataset = encoded_dataset.remove_columns(['text'])
encoded_dataset.set_format('torch')

train_dataset = encoded_dataset['train']
test_dataset = encoded_dataset['test']

# Fine-Tune DistilBERT
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    weight_decay=0.01
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()

# Save the model
model.save_pretrained('./fine-tuned-distilbert')
tokenizer.save_pretrained('./fine-tuned-distilbert')

