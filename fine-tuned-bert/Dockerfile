# Base image with Python 3.9-slim
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the local files to the container
COPY . .

# Command to run the fine-tuning script
CMD ["python", "fine_tune_bert.py"]
