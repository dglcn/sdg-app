from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
from functools import lru_cache
import torch

app = FastAPI()
model_path = "degolcen/sdg-bert-model"

@lru_cache()
def load_model():
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

class InputData(BaseModel):
    text: str

@app.post("/")
async def predict(data: InputData):
    inputs = tokenizer(data.text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.sigmoid(outputs.logits).squeeze().tolist()
    labels = [f"SDG {i+1}" for i in range(len(probs))]
    topk = sorted(zip(labels, probs), key=lambda x: x[1], reverse=True)[:5]
    return {"predicted_sdgs": topk}
