from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
model_name = 'distilgpt2'
tokenizer = AutoTokenizer.from_pretrained(model_name,clean_up_tokenization_spaces=True)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Move model to the appropriate device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Define text generation function
def generate_text(prompt, max_length=50):
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    outputs = model.generate(**inputs, max_length=max_length,pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Run inference
# prompt = "What a day"
prompt = "a great story"
generated_text = generate_text(prompt)
print(generated_text)