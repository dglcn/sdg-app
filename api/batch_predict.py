from fastapi import FastAPI, UploadFile, File
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import torch
import io

app = FastAPI()

model = BertForSequenceClassification.from_pretrained("model/checkpoint-2885")
tokenizer = BertTokenizer.from_pretrained("model/checkpoint-2885")
model.eval()

@app.post("/")
async def batch_predict(file: UploadFile = File(...)):
    if file.filename.endswith(".csv"):
        df = pd.read_csv(file.file)
    elif file.filename.endswith(".xlsx"):
        df = pd.read_excel(file.file)
    else:
        return {"error": "File must be .csv or .xlsx"}

    if "text" not in df.columns:
        return {"error": "File must contain a 'text' column"}

    predictions = []
    for text in df["text"].astype(str).tolist():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits).squeeze().tolist()
        labels = [f"SDG {i+1}" for i in range(16)]
        topk = sorted(zip(labels, probs), key=lambda x: x[1], reverse=True)[:5]
        predictions.append(", ".join([sdg for sdg, _ in topk]))

    df["predicted_sdgs"] = predictions
    output = io.BytesIO()
    df.to_excel(output, index=False)
    output.seek(0)
    return {
        "filename": "hasil_prediksi.xlsx",
        "content": output.read().decode("latin1")  # untuk JSON serializable
    }
