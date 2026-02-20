FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy backend source
COPY . .

# Start the API
CMD ["uvicorn", "nlp_api:app", "--host", "0.0.0.0", "--port", "8080"]