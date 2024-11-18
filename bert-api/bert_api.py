import os
import joblib
from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
import torch

app = Flask(__name__)

model_dir = "./trained-bert"

# Load the model and tokenizer
tokenizer = BertTokenizer.from_pretrained(model_dir)
model = BertForSequenceClassification.from_pretrained(model_dir)

# Load the label encoder
label_encoder_path = os.path.join(model_dir, "label_encoder.pkl")
label_encoder = joblib.load(label_encoder_path)

@app.route('/')
def home():
    return "Flask is running!"

@app.route('/bert', methods=['POST'])
def bert():
    data = request.get_json()
    question = data['text']

    # Tokenize the input question
    inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=1).item()

    # Decode the predicted label into an answer
    predicted_label = label_encoder.inverse_transform([predicted_class_id])[0]

    print(f"Input Question: {question}", flush=True)
    print(f"Tokenized Input: {inputs}", flush=True)
    print(f"Logits: {logits}", flush=True)
    print(f"Predicted Class ID: {predicted_class_id}", flush=True)

    # print(f"Labels: {label_encoder.classes_}")
    
    return jsonify({"answer": predicted_label})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
