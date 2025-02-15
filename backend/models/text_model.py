from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from loguru import logger

# Load pre-trained model for sentiment analysis
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    model = AutoModelForSequenceClassification.from_pretrained("assemblyai/distilbert-base-uncased-sst2")
    return tokenizer, model

# text analysis
def analyze_sentiment(text):
    logger.info(
        f'text prediction start for: {text}'
    )
    tokenizer, model = load_model_and_tokenizer()
    tokenized_text = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    input_ids = tokenized_text.input_ids
    attention_mask = tokenized_text.attention_mask

    # Get model predictions
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    probs = F.softmax(logits, dim=1).squeeze()
    print({"Negative probability": probs[0].item(), "Positive probability": probs[1].item()})
    return {"Negative probability": probs[0].item(), "Positive probability": probs[1].item()}


if __name__=="__main__":
    text="my name is vikash"
    print("started")
    analyze_sentiment(text)
    print("ended")