from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification


model = DistilBertForSequenceClassification.from_pretrained('./fine-tuned-distilbert')
tokenizer = DistilBertTokenizerFast.from_pretrained('./fine-tuned-distilbert')

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = logits.argmax().item()
    return 'positive' if predicted_class == 1 else 'negative'

# Test inference
text = "I absolutely love this movie!"
print(f"Sentiment: {predict_sentiment(text)}")

