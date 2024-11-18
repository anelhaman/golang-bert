import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
import os
from datetime import datetime
import joblib
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim  # Import PyTorch's AdamW optimizer
import torch.nn as nn  # Import CrossEntropyLoss from torch.nn

# Step 1: Load the Dataset from Excel
file_path = './raw-data/EVme-KM-3.xlsx'
sheets = pd.read_excel(file_path, sheet_name=None, header=None)  # Load all sheets, no headers

questions = []
answers = []

# Iterate over all sheets and collect the data
for sheet_name, sheet_data in sheets.items():
    if sheet_data.shape[0] == 0 or sheet_data.empty or sheet_data.shape[1] < 2:  # Check for empty data or fewer than 2 columns
        print(f"Skipping sheet {sheet_name} due to insufficient rows or columns.")
        continue  # Skip the sheet if it doesn't meet the conditions

    questions.extend(sheet_data.iloc[:, 0].tolist())  # First column is questions
    answers.extend(sheet_data.iloc[:, 1].tolist())    # Second column is answers

# Step 2: Encode answers as labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(answers)  # Convert answers to integer labels

# Step 3: Tokenize the data
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=len(set(labels)))

def tokenize_data(question):
    return tokenizer(question, padding=True, truncation=True, max_length=512)

# Prepare data as a dataset
train_data = {
    "question": questions,
    "label": labels
}

train_dataset = Dataset.from_dict(train_data)
train_dataset = train_dataset.map(lambda x: tokenize_data(x['question']), batched=True)

# Compute class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

# Convert dataset to TensorDataset for PyTorch DataLoader
input_ids = torch.tensor(train_dataset['input_ids'])
attention_mask = torch.tensor(train_dataset['attention_mask'])
labels_tensor = torch.tensor(labels)

dataset = TensorDataset(input_ids, attention_mask, labels_tensor)
train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Step 4: Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=5e-5)  # Use PyTorch's AdamW optimizer
epochs = 5

# Loss function with class weights
loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))  # Pass class weights to the loss function

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_dataloader:
        batch_input_ids = batch[0].to(device)
        batch_attention_mask = batch[1].to(device)
        batch_labels = batch[2].to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch_input_ids, attention_mask=batch_attention_mask, labels=batch_labels)
        logits = outputs.logits

        # Compute loss with class weights
        loss = loss_fn(logits.view(-1, model.config.num_labels), batch_labels.view(-1))

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_dataloader)}")

# Step 5: Save the model with a timestamped directory name
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")  # Get current timestamp
model_dir = f"./trained-bert-{timestamp}"  # Append timestamp to the directory name

# Save model and tokenizer
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)

# Save the label encoder as a pickle file
label_encoder_path = os.path.join(model_dir, "label_encoder.pkl")
joblib.dump(label_encoder, label_encoder_path)
