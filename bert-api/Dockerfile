# Base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the local files (including trained-bert directory) to the container
COPY . .

# # Ensure the model directory is copied into the container
# COPY ./trained-bert /app/trained-bert

# Command to run the BERT API
CMD ["python", "bert_api.py"]

# # Command to run the Flask API using Gunicorn
# CMD ["gunicorn", "-b", "0.0.0.0:5000", "bert_api:app"]
