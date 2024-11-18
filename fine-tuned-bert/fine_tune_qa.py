from transformers import BertTokenizer, BertForQuestionAnswering, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
import torch

# Step 1: Load your dataset
# Example format with questions, answers, and contexts
file_path = './raw-data/EVme-KM-3.xlsx'
sheets = pd.read_excel(file_path, sheet_name=None, header=None)  # Load all sheets, no headers

questions = []
answers = []
contexts = []

# Iterate over all sheets and collect the data (Assuming the format is [Question, Answer, Context])
for sheet_name, sheet_data in sheets.items():
    if sheet_data.shape[1] >= 3:
        questions.extend(sheet_data.iloc[:, 0].tolist())  # First column is questions
        answers.extend(sheet_data.iloc[:, 1].tolist())    # Second column is answers
        contexts.extend(sheet_data.iloc[:, 2].tolist())   # Third column is contexts

# Step 2: Prepare the data for BERT (Question-Answering)
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
model = BertForQuestionAnswering.from_pretrained("bert-base-multilingual-cased")

def create_features(question, context, answer):
    # Tokenize the question and context
    encoding = tokenizer(question, context, truncation=True, padding="max_length", max_length=512)

    # Find the start and end positions of the answer in the context
    start_position = context.find(answer)
    end_position = start_position + len(answer) - 1

    # Adjust positions if the answer is not found
    if start_position == -1:
        start_position = 0
        end_position = 0

    encoding['start_positions'] = start_position
    encoding['end_positions'] = end_position
    return encoding

# Apply tokenization and create features
train_data = {
    "question": questions,
    "context": contexts,
    "answer": answers
}

train_dataset = Dataset.from_dict(train_data)
train_dataset = train_dataset.map(lambda x: create_features(x['question'], x['context'], x['answer']), batched=True)

# Step 3: Fine-Tune the Model
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()

# Step 4: Save the model
model.save_pretrained("./trained-bert")
tokenizer.save_pretrained("./trained-bert")
